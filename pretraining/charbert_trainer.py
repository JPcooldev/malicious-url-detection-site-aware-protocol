import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from .trainer import PretrainingTrainerBase


class CharBERTTrainer(PretrainingTrainerBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
    
    def train_loop(self):
        # setup optimizer 
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # setup gradient scaler
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.args.num_epochs):

            for batch_idx, batch in enumerate(self.train_dataloader):
                start_time = time.time()
                
                self.model.train()
                if batch_idx > self.last_batch_idx:
                    break # early stopping
                
                char_input_ids, start_ids, end_ids, input_ids, mlm_labels, term_labels = batch

                char_input_ids = char_input_ids.to(self.device)
                start_ids = start_ids.to(self.device)
                end_ids = end_ids.to(self.device)
                input_ids = input_ids.to(self.device)
                mlm_labels = mlm_labels.to(self.device)
                term_labels = term_labels.to(self.device)

                # labels that do not contribute to the loss have label -1
                mlm_acc_labels = (mlm_labels != -1)
                term_acc_labels = (term_labels != -1)

                # get model outputs
                with torch.autocast(device_type=self.device_type):
                    output = self.model(
                        char_input_ids=char_input_ids, 
                        start_ids=start_ids, 
                        end_ids=end_ids, 
                        input_ids=input_ids, 
                        masked_lm_labels=mlm_labels, 
                        adv_labels=term_labels
                    )

                mlm_loss = output[0]
                term_loss = output[1]
                mlm_logits = output[2]
                term_logits = output[3]

                mlm_accuracy = (mlm_acc_labels * (torch.argmax(mlm_logits, dim=-1)==mlm_labels)).sum() / mlm_acc_labels.sum()
                term_accuracy = (term_acc_labels * (torch.argmax(term_logits, dim=-1)==term_labels)).sum() / term_acc_labels.sum()

                loss = mlm_loss + term_loss
                loss_accum = loss.detach().clone()
                mlm_loss_accum = mlm_loss.detach().clone()
                term_loss_accum = term_loss.detach().clone()

                mlm_accuracy_accum = mlm_accuracy.detach().clone()
                term_accuracy_accum = term_accuracy.detach().clone()

                self.optimizer_step(optimizer, scaler, loss, self.model.parameters())

                end_time = time.time()
                self.forw_back_times.append(end_time - start_time)

                torch.cuda.synchronize() # wait for the GPU to finish work

                num_batch_tokens = torch.prod(torch.tensor(input_ids.shape)).to(self.device)
                num_examples = torch.tensor(len(input_ids)).to(self.device)

                # inter-node communication between GPUs
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(mlm_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(term_loss_accum, op=dist.ReduceOp.AVG)

                dist.all_reduce(mlm_accuracy_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(term_accuracy_accum, op=dist.ReduceOp.AVG)

                dist.all_reduce(num_batch_tokens, op=dist.ReduceOp.SUM)
                self.total_num_training_tokens += num_batch_tokens
                dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
                self.total_num_examples += num_examples
                dist.all_reduce(self.grad_norm, op=dist.ReduceOp.AVG)

                step_idx = epoch * len(self.train_dataloader) + batch_idx

                # metrics logging
                metrics = {
                    "train_loss": loss_accum.item(),
                    "mlm_loss": mlm_loss_accum.item(),
                    "term_loss": term_loss_accum.item(),
                    "mlm_accuracy": mlm_accuracy_accum.item(),
                    "term_accuracy": term_accuracy_accum.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "grad_norm": self.grad_norm.item(),
                }
                if self.MASTER_PROCESS:
                    self.log_metrics_to_mlflow(metrics, step=step_idx)

                # checkpoint saving
                if ((batch_idx % self.args.saving_freq == 0 and batch_idx > 0) or batch_idx == self.last_batch_idx):
                    # this will create a nested run in training run and will host model checkpoint 
                    # along its hyperparameters
                    additional_params = {
                        "char_vocab_size": self.args.char_vocab_size,
                        "term_vocab_size": self.args.term_vocab_size,
                        "char_embedding_dim": self.args.char_embedding_size,
                        "char_maxlen_for_word": self.args.char_maxlen_for_word,
                        "char_vocab_path": os.path.abspath(self.args.char_vocab_path),
                        "term_vocab_path": os.path.abspath(self.args.term_vocab_path),
                    }
                    self.log_checkpoint_to_mlflow(
                        step_idx=step_idx, 
                        additional_params=additional_params, 
                        without_safe_serialization=True)
                    # evaluate model on validation set and log metrics to mlflow training run
                    # main training run holds training metrics and evaluation metrics
                    self.evaluate(checkpoint_idx=step_idx)


    def eval_loop(self, checkpoint_idx: int):
        # torch no_grad context applied 

        if self.MASTER_PROCESS:
            eval_losses = []
            eval_mlm_accuracies = []

        for eval_batch_idx, eval_batch in enumerate(self.val_dataloader):
            if eval_batch_idx == self.args.eval_steps:
                break

            char_input_ids, start_ids, end_ids, input_ids, mlm_labels, term_labels = eval_batch

            char_input_ids = char_input_ids.to(self.device)
            start_ids = start_ids.to(self.device)
            end_ids = end_ids.to(self.device)
            input_ids = input_ids.to(self.device)
            mlm_labels = mlm_labels.to(self.device)
            term_labels = term_labels.to(self.device)

            # labels that do not contribute to the loss have label -1
            mlm_acc_labels = (mlm_labels != -1)

            # get model outputs
            with torch.autocast(device_type=self.device_type):
                output = self.model(
                    char_input_ids=char_input_ids, 
                    start_ids=start_ids, 
                    end_ids=end_ids, 
                    input_ids=input_ids, 
                    masked_lm_labels=mlm_labels, 
                    adv_labels=term_labels
                )

            eval_mlm_loss = output[0]
            eval_term_loss = output[1]
            eval_mlm_logits = output[2]
            eval_term_logits = output[3]

            eval_loss = eval_mlm_loss + eval_term_loss
            eval_mlm_accuracy = (mlm_acc_labels * (torch.argmax(eval_mlm_logits, dim=-1)==mlm_labels)).sum() / mlm_acc_labels.sum()

            torch.cuda.synchronize() # wait for the GPU to finish work

            # inter-node communication between GPUs
            dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(eval_mlm_accuracy, op=dist.ReduceOp.AVG)

            # metrics logging
            metrics = {
                f"val_mlm_accuracy-checkpoint-{checkpoint_idx}": eval_mlm_accuracy.item(),
                f"val_loss-checkpoint-{checkpoint_idx}": eval_loss.item(),
            }
            self.log_metrics_to_mlflow(metrics, step=eval_batch_idx+1)

            if self.MASTER_PROCESS:
                eval_losses.append(eval_loss.cpu().item())
                eval_mlm_accuracies.append(eval_mlm_accuracy.cpu().item())

        if self.MASTER_PROCESS:
            avg_eval_loss = np.array(eval_losses).mean()
            avg_eval_mlm_accuracy = np.array(eval_mlm_accuracies).mean()
            eval_metrics = {
                "val_avg_mlm_accuracy": avg_eval_mlm_accuracy,
                "val_avg_loss": avg_eval_loss,
            }
            self.log_metrics_to_mlflow(eval_metrics, step=checkpoint_idx)