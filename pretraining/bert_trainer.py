import time
from typing import List
import argparse

import numpy as np
import torch
import torch.distributed as dist

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .trainer import PretrainingTrainerBase

class BERTTrainer(PretrainingTrainerBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
    
    def train_loop(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=self.args.mlm_prob, 
            return_tensors='pt'
        )

        # setup optimizer 
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        # setup gradient scaler
        scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.args.num_epochs):
            self.train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_dataloader):
                start_time = time.time()
                
                self.model.train()
                if batch_idx > self.last_batch_idx:
                    break # early stopping
                
                # tokenize batch
                encoded, labels, acc_labels = tokenize_batch(batch, self.tokenizer, data_collator)

                labels = labels.to(self.device)
                acc_labels = acc_labels.to(self.device)
                encoded = encoded.to(self.device)

                # get model outputs
                with torch.autocast(device_type=self.device_type):
                    output = self.model(**encoded, labels=labels)

                loss = output.loss
                loss_accum = loss.detach().clone()
                mlm_accuracy = (acc_labels * (torch.argmax(output.logits, dim=-1)==labels)).sum() / acc_labels.sum()
                
                self.optimizer_step(optimizer, scaler, loss, self.model.parameters())

                end_time = time.time()
                self.forw_back_times.append(end_time - start_time)

                torch.cuda.synchronize() # wait for the GPU to finish work

                num_batch_tokens = torch.prod(torch.tensor(encoded.input_ids.shape)).to(self.device)
                num_examples = torch.tensor(len(encoded.input_ids)).to(self.device)

                # inter-node communication between GPUs
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(num_batch_tokens, op=dist.ReduceOp.SUM)
                self.total_num_training_tokens += num_batch_tokens
                dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
                self.total_num_examples += num_examples
                dist.all_reduce(mlm_accuracy, op=dist.ReduceOp.AVG)
                dist.all_reduce(self.grad_norm, op=dist.ReduceOp.AVG)

                step_idx = epoch * len(self.train_dataloader) + batch_idx

                # metrics logging
                metrics = {
                    "mlm_accuracy": mlm_accuracy.item(),
                    "train_loss": loss_accum.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "grad_norm": self.grad_norm.item(),
                }
                self.log_metrics_to_mlflow(metrics, step=step_idx)

                # checkpoint saving
                if ((batch_idx % self.args.saving_freq == 0 and batch_idx > 0) or batch_idx == self.last_batch_idx):
                    # this will create a nested run in training run and will host model checkpoint 
                    # along its hyperparameters
                    self.log_checkpoint_to_mlflow(step_idx=step_idx)
                    # evaluate model on validation set and log metrics to mlflow training run
                    # main training run holds training metrics and evaluation metrics
                    self.evaluate(checkpoint_idx=step_idx)


    def eval_loop(self, checkpoint_idx: int):
        # torch no_grad context applied 
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=self.args.mlm_prob, 
            return_tensors='pt'
        )

        if self.MASTER_PROCESS:
            eval_losses = []
            eval_mlm_accuracies = []

        for eval_batch_idx, eval_batch in enumerate(self.val_dataloader):
            # tokenize batch
            encoded, labels, acc_labels = tokenize_batch(eval_batch, self.tokenizer, data_collator)

            encoded = encoded.to(self.device)
            labels = labels.to(self.device)
            acc_labels = acc_labels.to(self.device)

            # get model outputs
            with torch.autocast(device_type=self.device_type):
                output = self.model(**encoded, labels=labels)
            eval_loss = output.loss.detach().clone()
            eval_mlm_accuracy = (acc_labels * (torch.argmax(output.logits, dim=-1)==labels)).sum() / acc_labels.sum()

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


# tokenize batch on the fly
def tokenize_batch(batch: List[str], tokenizer: PreTrainedTokenizer, data_collator: DataCollatorForLanguageModeling):
    encoded = {
        'input_ids': [],
        'attention_mask': [],
    }
    tokenized_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    encoded['attention_mask'] = tokenized_batch.attention_mask

    encoded_mlm = data_collator(tokenized_batch.input_ids)
    encoded['input_ids'] = encoded_mlm["input_ids"]
    encoded = BatchEncoding(encoded)

    labels = encoded_mlm["labels"]
    acc_labels = (encoded_mlm["labels"] != -100)

    return encoded, labels, acc_labels