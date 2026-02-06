import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import PreTrainedTokenizer

import mlflow

from .utils import mlflow_setup, shorten_num_tokens
from .ddp_utils import ddp_setup, ddp_cleanup
from .data_processing.dataloader import generate_dataloader


class PretrainingTrainerBase(ABC):
    # DDP attributes
    device: str
    device_type: str
    DDP_WORLD_SIZE: int
    DDP_RANK: int
    DDP_LOCAL_RANK: int
    MASTER_PROCESS: bool

    # data attributes
    train_dataloader: DataLoader
    train_sampler: DistributedSampler
    val_dataloader: DataLoader
    val_sampler: DistributedSampler

    # model attributes
    model: nn.Module
    tokenizer: PreTrainedTokenizer

    # training attributes
    num_steps: int
    last_batch_idx: int
    total_num_examples: torch.Tensor
    total_num_training_tokens: torch.Tensor
    grad_norm: float

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = None
        self.model = None
        self.forw_back_times = []

    def setup_connection(self):
        # setup DDP
        ddp_args = ddp_setup(self.args.random_seed)
        self.device = ddp_args["device"]
        self.device_type = ddp_args["device_type"]
        self.DDP_WORLD_SIZE = ddp_args["ddp_world_size"]
        self.DDP_RANK = ddp_args["ddp_rank"]
        self.DDP_LOCAL_RANK = ddp_args["ddp_local_rank"]
        self.MASTER_PROCESS = ddp_args["master_process"]
        if self.MASTER_PROCESS:
            print('[MASTER]: DDP established')
        
        # setup mlflow connection
        mlflow_setup(self.args.experiment_path, self.args.db_token, self.args.db_host)
        if self.MASTER_PROCESS:
            print('[MASTER]: MLflow connection established')
        
    def train(self, model, tokenizer, df_train, df_val):
        self.setup_connection()
        self.tokenizer = tokenizer
        self.model = model
        self.df_train = df_train
        self.df_val = df_val

        # generate dataloaders
        self.train_dataloader, self.train_sampler = generate_dataloader(
            args=self.args, 
            df=self.df_train, 
            DDP_WORLD_SIZE=self.DDP_WORLD_SIZE, 
            DDP_RANK=self.DDP_RANK, 
            split="train", 
            tokenizer=self.tokenizer
        )

        # set tokenizer's max sequence length
        self.tokenizer.model_max_length = self.args.max_sequence_length

        # convert model to DDP
        self.model = DDP(self.model.to(self.device), device_ids=[self.DDP_LOCAL_RANK], find_unused_parameters=True)
        # enable gradient tracking for all parameters
        for _, param in self.model.named_parameters():
            param.requires_grad = True

        self.num_steps = len(self.train_dataloader)
        # specify last batch index (due to possible early stopping)
        self.last_batch_idx = self.num_steps - 1 if self.args.num_steps == -1 else self.args.num_steps - 1

        self.total_num_examples = torch.tensor(0).to(self.device)
        self.total_num_training_tokens = torch.tensor(0).to(self.device)
        run_name = self.get_run_name()

        if self.MASTER_PROCESS:
            print(f'[MASTER]: Run name established: {run_name}')
            print(f'[MASTER]: Training for {self.args.num_epochs} epochs')
            print(f'[MASTER]:   batch size = {self.args.worker_batch_size * self.DDP_WORLD_SIZE} (worker batch size: {self.args.worker_batch_size})')
            print(f'[MASTER]:   tokens per step = {self.args.worker_batch_size * self.DDP_WORLD_SIZE * tokenizer.model_max_length}')
            print(f'[MASTER]:   GPU:{self.DDP_RANK} model loaded! Memory allocated: {torch.cuda.memory_allocated(self.DDP_RANK) / 1024**2}Mb')
            print(f'[MASTER]: Data, tokenizer and model loaded. Going to training loop. Wish me luck!')

        self.mlflow_train_run = None
        
        if self.MASTER_PROCESS:
            self.mlflow_train_run = mlflow.start_run(run_name=run_name, log_system_metrics=True)
            mlflow.config.enable_system_metrics_logging()

        # training loop (each model architecture defines its own pre-training logic)
        self.train_loop()

        if self.MASTER_PROCESS:
            mlflow.end_run()

        # cleanup DDP
        ddp_cleanup()


    @abstractmethod
    def train_loop(self):
        pass

    def evaluate(self, checkpoint_idx: int):
        self.val_dataloader, self.val_sampler = generate_dataloader(
            args=self.args, 
            df=self.df_val, 
            DDP_WORLD_SIZE=self.DDP_WORLD_SIZE, 
            DDP_RANK=self.DDP_RANK, 
            split="val", 
            tokenizer=self.tokenizer
        )
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(0)
        self.model.eval()
        # evaluation
        with torch.no_grad():
            self.eval_loop(checkpoint_idx)

    @abstractmethod
    def eval_loop(self, checkpoint_idx: int):
        pass

    def optimizer_step(self, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss: torch.Tensor, params: List[torch.nn.Parameter]):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # unscale the gradients to clip the gradients properly
        scaler.unscale_(optimizer)

        # clip gradients
        max_norm = self.args.max_grad_norm if self.args.clip_gradients else float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.grad_norm = grad_norm.pow(2)
        
        scaler.step(optimizer)
        scaler.update()

    def log_params_to_mlflow(self, params_dict: Dict[str, Any]):
        if self.MASTER_PROCESS:
            mlflow.log_params(params_dict)

    def log_metrics_to_mlflow(self, metrics_dict: Dict[str, Any], step: int):
        if self.MASTER_PROCESS:
            mlflow.log_metrics(metrics_dict, step=step)

    def save_model_checkpoint(self, without_safe_serialization=False):
        save_dir = "pretrained_model"
        # 1. Save model and tokenizer manually with safe_serialization=False
        self.model.module.save_pretrained(os.path.join(save_dir, "model"), safe_serialization=without_safe_serialization)
        self.tokenizer.save_pretrained(os.path.join(save_dir, "components", "tokenizer"))
        mlflow.log_artifacts(save_dir, artifact_path="pretrained_model")

    def get_run_name(self):
        run_name = f"{self.args.model_type}-{self.args.model_size}"
        if self.args.run_name_postfix:
            run_name += f"-{self.args.run_name_postfix}"
        return run_name
    
    def log_checkpoint_to_mlflow(self, 
                                 step_idx: int, 
                                 additional_params: Dict[str, Any] = {}, 
                                 without_safe_serialization: bool = False):
        # log only on master GPU
        if self.MASTER_PROCESS:
            with mlflow.start_run(
                run_name=f"checkpoint-{step_idx}",
                nested=True,
                parent_run_id=self.mlflow_train_run.info.run_id
            ) as checkpoint_run:
                hyperparams = {
                    "model_type": self.args.model_type,
                    "model_size": self.args.model_size,
                    "dataset_path":  os.path.abspath(self.args.dataset_path),
                    "dataset_path_val": os.path.abspath(self.args.dataset_path_val),
                    "num_epochs": self.args.num_epochs,
                    "worker_batch_size": self.args.worker_batch_size,
                    "batch_size": self.args.worker_batch_size * self.DDP_WORLD_SIZE,
                    "learning_rate": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                    "max_sequence_length": self.args.max_sequence_length,
                    "num_steps": step_idx,
                    "max_grad_norm": self.args.max_grad_norm,
                    "num_training_tokens": shorten_num_tokens(self.total_num_training_tokens.item()),
                    "vocab_size": self.tokenizer.vocab_size,
                    "mlm_probability": self.args.mlm_prob,
                    "training_samples": self.total_num_examples.item(),
                    "avg_step_time": np.array(self.forw_back_times).mean(),
                }

                # update with additional params (adds new keys and overwrites existing ones)
                hyperparams.update(additional_params)
                
                mlflow.log_params(hyperparams)
                self.save_model_checkpoint(without_safe_serialization)
