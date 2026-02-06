from typing import List
import numpy as np
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .trainer import PretrainingTrainerBase
from models.urlbert.virtual_adversarial_training import vat
from models.urlbert.contrastive_learning import sscl
from models.urlbert.token_manipulation import shuffle_tokens, replace_tokens


class URLBERTTrainer(PretrainingTrainerBase):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def train_loop(self):
        if self.args.stage == 1:
            if self.MASTER_PROCESS:
                print(f'[MASTER]: Stage 1 training')
            self.stage_1_train_loop()
        elif self.args.stage == 2:
            if self.MASTER_PROCESS:
                print(f'[MASTER]: Stage 2 training')
            self.stage_2_train_loop()
        else:
            raise ValueError(f"Stage {self.args.stage} not supported.")
    
    def stage_1_train_loop(self):
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

        hidden_size = self.model.module.args.hidden_size
        std_head = nn.Linear(hidden_size, 1).to(self.device)
        rtd_head = nn.Linear(hidden_size, 1).to(self.device)
        optimizer.add_param_group({'params': std_head.parameters()})
        optimizer.add_param_group({'params': rtd_head.parameters()})

        for epoch in range(self.args.num_epochs):
            self.train_sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(self.train_dataloader):
                start_time = time.time()

                self.model.train()
                if batch_idx > self.last_batch_idx:
                    break # early stopping
                
                # tokenize batch
                encoded, labels, acc_labels = tokenize_batch(batch, self.tokenizer, data_collator)

                input_ids = encoded.input_ids.to(self.device)
                attention_mask = encoded.attention_mask.to(self.device)
                token_type_ids = encoded.token_type_ids.to(self.device)
                mlm_labels = labels.to(self.device)
                acc_labels = acc_labels.to(self.device)

                mlm_input = input_ids.clone()
                std_input, std_labels = shuffle_tokens(input_ids.clone())
                rtd_input, rtd_labels = replace_tokens(input_ids.clone(), vocab_size=self.tokenizer.vocab_size)

                std_input = std_input.to(self.device)
                std_labels = std_labels.to(self.device)
                rtd_input = rtd_input.to(self.device)
                rtd_labels = rtd_labels.to(self.device)

                # Create position_ids to avoid using the shared buffer
                seq_length = input_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(input_ids.size(0), -1)

                # get model outputs
                with torch.autocast(device_type=self.device_type):
                    out_mlm = self.model(mlm_input,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=mlm_labels,
                            output_hidden_states=True)
                    mlm_loss = out_mlm[0]

                    # STD forward
                    out_std = self.model(std_input,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids.clone(),
                                    output_hidden_states=True)
                    std_logits = std_head(out_std.hidden_states[-1]).squeeze(-1)
                    std_loss = F.binary_cross_entropy_with_logits(std_logits, std_labels.float())

                    # RTD forward
                    out_rtd = self.model(rtd_input,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids.clone(),
                                    output_hidden_states=True)
                    rtd_logits = rtd_head(out_rtd.hidden_states[-1]).squeeze(-1)
                    rtd_loss = F.binary_cross_entropy_with_logits(rtd_logits, rtd_labels.float())




                loss = mlm_loss + self.args.std_coef * std_loss + self.args.rtd_coef * rtd_loss

                loss_accum = loss.detach().clone()
                mlm_accuracy = (acc_labels * (torch.argmax(out_mlm.logits, dim=-1)==mlm_labels)).sum() / acc_labels.sum()

                # clone loss values for logging into mlflow
                mlm_loss_accum = mlm_loss.detach().clone()
                std_loss_accum = std_loss.detach().clone()
                rtd_loss_accum = rtd_loss.detach().clone()

                params = list(self.model.parameters()) + list(std_head.parameters()) + list(rtd_head.parameters())
                self.optimizer_step(optimizer, scaler, loss, params)

                end_time = time.time()
                self.forw_back_times.append(end_time - start_time)

                torch.cuda.synchronize() # wait for the GPU to finish work

                num_batch_tokens = torch.prod(torch.tensor(encoded.input_ids.shape)).to(self.device)
                num_examples = torch.tensor(len(encoded.input_ids)).to(self.device)

                # inter-node communication between GPUs
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(mlm_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(std_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(rtd_loss_accum, op=dist.ReduceOp.AVG)
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
                    "mlm_loss": mlm_loss_accum.item(),
                    "std_loss": std_loss_accum.item(),
                    "rtd_loss": rtd_loss_accum.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "grad_norm": self.grad_norm.item(),
                }
                self.log_metrics_to_mlflow(metrics, step=step_idx)

                # checkpoint saving
                if ((batch_idx % self.args.saving_freq == 0 and batch_idx > 0) or batch_idx == self.last_batch_idx):
                    # this will create a nested run in training run and will host model checkpoint 
                    # along its hyperparameters
                    additional_params = {
                        "rtd_coef": self.args.rtd_coef,
                        "std_coef": self.args.std_coef,
                        "stage": 1,
                    }
                    self.log_checkpoint_to_mlflow(step_idx=step_idx, additional_params=additional_params)
                    # evaluate model on validation set and log metrics to mlflow training run
                    # main training run holds training metrics and evaluation metrics
                    self.evaluate(checkpoint_idx=step_idx)


    def stage_2_train_loop(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=self.args.mlm_prob, 
            return_tensors='pt'
        )

        # setup optimizer for VAT
        optimizer_vat = torch.optim.AdamW(self.model.parameters(), lr=self.args.vat_learning_rate, weight_decay=self.args.vat_weight_decay)
        # setup optimizer for SSCL
        optimizer_sscl = torch.optim.AdamW(self.model.parameters(), lr=self.args.sscl_learning_rate, weight_decay=self.args.sscl_weight_decay)

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

                input_ids = encoded.input_ids.to(self.device)
                attention_mask = encoded.attention_mask.to(self.device)
                token_type_ids = encoded.token_type_ids.to(self.device)
                labels = labels.to(self.device)
                acc_labels = acc_labels.to(self.device)

                # Create position_ids to avoid using the shared buffer
                seq_length = input_ids.size(1)
                position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device).unsqueeze(0).expand(input_ids.size(0), -1)

                # get model outputs
                with torch.autocast(device_type=self.device_type):
                    output = self.model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                            output_hidden_states=True)
                    loss = output.loss
                    logits = output.logits
                    embeddings = output.hidden_states[0]

                    # clone loss values for logging into mlflow
                    mlm_loss_accum = loss.detach().clone()

                    adv_loss = vat(self.model, embeddings, token_type_ids, attention_mask, logits, position_ids.clone())
                    if adv_loss:
                        loss = self.args.vat_coef * adv_loss + loss

                mlm_accuracy = (acc_labels * (torch.argmax(logits, dim=-1)==labels)).sum() / acc_labels.sum()

                # clone loss values for logging into mlflow
                adv_loss_accum = adv_loss.detach().clone() if adv_loss else -1.0
                
                self.optimizer_step(optimizer_vat, scaler, loss, self.model.parameters())
                grad_norm_vat = self.grad_norm
                
                with torch.autocast(device_type=self.device_type):
                    contrastive_loss = sscl(self.model, input_ids, token_type_ids, attention_mask, position_ids.clone(), device=self.device)

                if contrastive_loss:
                    self.optimizer_step(optimizer_sscl, scaler, contrastive_loss, self.model.parameters())
                    grad_norm_sscl = self.grad_norm
                    # clone loss values for logging into mlflow
                    loss_accum = (loss + self.args.sscl_coef * contrastive_loss).detach().clone()
                    contrastive_loss_accum = contrastive_loss.detach().clone()
                else:
                    # skip the SSCL optimizer step this batch
                    grad_norm_sscl = torch.tensor(0.0, device=self.device)
                    loss_accum = loss.detach().clone()
                    contrastive_loss_accum = torch.tensor(-1.0, device=self.device)

                end_time = time.time()
                self.forw_back_times.append(end_time - start_time)

                torch.cuda.synchronize() # wait for the GPU to finish work

                num_batch_tokens = torch.prod(torch.tensor(encoded.input_ids.shape)).to(self.device)
                num_examples = torch.tensor(len(encoded.input_ids)).to(self.device)

                # inter-node communication between GPUs
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(mlm_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(adv_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(contrastive_loss_accum, op=dist.ReduceOp.AVG)
                dist.all_reduce(num_batch_tokens, op=dist.ReduceOp.SUM)
                self.total_num_training_tokens += num_batch_tokens
                dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
                self.total_num_examples += num_examples
                dist.all_reduce(mlm_accuracy, op=dist.ReduceOp.AVG)
                dist.all_reduce(grad_norm_vat, op=dist.ReduceOp.AVG)
                dist.all_reduce(grad_norm_sscl, op=dist.ReduceOp.AVG)

                step_idx = epoch * len(self.train_dataloader) + batch_idx

                # metrics logging
                metrics = {
                    "mlm_accuracy": mlm_accuracy.item(),
                    "train_loss": loss_accum.item(),
                    "mlm_loss": mlm_loss_accum.item(),
                    "adv_loss": adv_loss_accum.item(),
                    "contrastive_loss": contrastive_loss_accum.item(),
                    "lr_vat": optimizer_vat.param_groups[0]['lr'],
                    "lr_sscl": optimizer_sscl.param_groups[0]['lr'],
                    "grad_norm_vat": grad_norm_vat.item(),
                    "grad_norm_sscl": grad_norm_sscl.item(),
                }
                self.log_metrics_to_mlflow(metrics, step=step_idx)

                # checkpoint saving
                if ((batch_idx % self.args.saving_freq == 0 and batch_idx > 0) or batch_idx == self.last_batch_idx):
                    # this will create a nested run in training run and will host model checkpoint 
                    # along its hyperparameters
                    additional_params = {
                        "lr_vat": self.args.vat_learning_rate,
                        "weight_decay_vat": self.args.vat_weight_decay,
                        "lr_sscl": self.args.sscl_learning_rate,
                        "weight_decay_sscl": self.args.sscl_weight_decay,
                        "vat_coef": self.args.vat_coef,
                        "sscl_coef": self.args.sscl_coef,
                        "stage": 2,
                        "learning_rate": None, # not used for stage 2
                        "weight_decay": None, # not used for stage 2
                    }
                    self.log_checkpoint_to_mlflow(step_idx=step_idx, additional_params=additional_params)
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
            eval_loss = output.loss.clone().detach()
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
    encoded['token_type_ids'] = tokenized_batch.token_type_ids

    encoded_mlm = data_collator(tokenized_batch.input_ids)
    encoded['input_ids'] = encoded_mlm["input_ids"]
    encoded = BatchEncoding(encoded)

    labels = encoded_mlm["labels"]
    acc_labels = (encoded_mlm["labels"] != -100)

    return encoded, labels, acc_labels