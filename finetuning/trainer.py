import os
import argparse
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple, Dict, Any
import tqdm
import copy

import numpy as np
import mlflow
from transformers import PreTrainedTokenizerFast

import torch
from torch import nn
from torch.utils.data import DataLoader

from .utils.focal_loss import BinaryFocalLoss
from .data_processing.dataset import URLDataset
from .utils.utils import print_dict_level1_inline
from .utils.mlflow_utils import (
    log_evaluation_result_to_mlflow, 
    log_metrics_dict_recursively, 
    log_dict_json,
)
from .utils.metrics import (
    EvaluationResults, 
    find_best_threshold, 
    compute_metrics_default, 
    calculate_test_metrics
)


def get_loss_function(args: argparse.Namespace):
    if args.loss == "focal":
        return BinaryFocalLoss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
    elif args.loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function provided: {args.loss}")


def perf_time_synchronized(device: torch.device):
    """if you do model(batch) and then measure time using time.perf_counter right after that, synchronization error can occur
    model will still be calculated on GPU, but this will be called on CPU. This method ensures that CPU waits when command model(batch) is finished
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class TrainerBase(ABC):
    """
    Base class for all models (pytorch based, onnx based), which offers unified evaluation framework
    """

    args: argparse.Namespace
    # onnx needs pytorch device to calculate stuff with loss
    model: torch.nn.Module
    device: torch.device
    loss_fn: Callable
    log_to_mlflow: bool
    optimizer: torch.optim.Optimizer
    tokenizer: PreTrainedTokenizerFast

    def __init__(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module, 
        dataset: URLDataset, 
        log_to_mlflow: bool = True, 
        device: Optional[torch.device] = None,
        move_to_device: bool = True,
        is_evaluation: bool = False,
    ) -> None:
        # device created in parent if None
        self.args = args
        self.model = model
        self.dataset = dataset
        self.log_to_mlflow = log_to_mlflow
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if move_to_device:
            self.model.to(self.device)
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.optimizer = None
        self.is_evaluation = is_evaluation
        if not is_evaluation:
            self.loss_fn = get_loss_function(args)

    def evaluate(
        self, 
        data_loader: DataLoader, 
        decision_threshold: Optional[float] = None
    ) -> EvaluationResults:
        if decision_threshold is None:
            decision_threshold = self.args.decision_threshold
        n_samples = len(data_loader.dataset)
        true_labels = np.empty(n_samples, dtype=np.int64)
        class_probs = np.empty((n_samples, 2), dtype=np.float32)
        write_ptr = 0
        batch_losses = []

        self.model.eval()
        pbar = tqdm.tqdm(data_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = batch[0], batch[1]

                inputs = self._prepare_inputs(inputs)
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu()

                batch_size = labels.size(0)
                idx_end = write_ptr + batch_size

                loss = self.loss_fn(logits, labels.to(self.device)).cpu().item()

                class_probs[write_ptr:idx_end, :] = probs.numpy()
                true_labels[write_ptr:idx_end] = labels.cpu().numpy()
                batch_losses.append(loss)

                write_ptr = idx_end

        predictions = np.array(class_probs[:, 1] >= decision_threshold).astype(int)
        true_labels = np.array(true_labels)
        class_probs = np.array(class_probs)

        best_threshold = find_best_threshold(class_probs[:, 1], true_labels)
        alt_preds = (class_probs[:, 1] >= best_threshold).astype(int)
        alternative_metrics = compute_metrics_default(true_labels, alt_preds, class_probs)

        eval_results = EvaluationResults(
            metrics=compute_metrics_default(true_labels, predictions, class_probs),
            predictions=predictions,
            labels=true_labels,
            probabilities=class_probs,
            loss=float(np.mean(batch_losses)),
            best_threshold=best_threshold,
            alternative_metrics=alternative_metrics,
        )
        return eval_results

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for model - move to device"""
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared

    def measure_performance(
        self, 
        data_loader: DataLoader, 
        log_to_mlflow: bool = True, 
        enough_samples_to_measure: int = 5000, 
        prefix: str = ""
    ) -> Dict[str, float]:
        self.model.eval()
        device = self.device
        # --- Warmup Phase ---
        warmup_batches = 2
        print("[profiler] Warmup initiated")
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= warmup_batches:
                    break
                inputs = batch[0]
                inputs = self._prepare_inputs(inputs)
                logits = self.model(**inputs).logits
                _ = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
                # Synchronize to ensure warmup execution is complete.
                perf_time_synchronized(device)

        # --- Measurement Phase ---
        total_samples = 0
        n_samples_to_use = (
            min(enough_samples_to_measure, len(data_loader.dataset))
            if hasattr(data_loader, "dataset")
            else enough_samples_to_measure
        )
        print(f"[profiler] n_samples_to_use {n_samples_to_use}")
        progress = tqdm.tqdm(
            total=n_samples_to_use,
            desc="Samples processed",
            unit="sample",
            dynamic_ncols=False,  
            leave=False,  
            bar_format="{l_bar}{bar:20}{r_bar}",  
        )
        with_tokenizer_start = perf_time_synchronized(device)
        time_total_without_tokenizer = 0.0
        with torch.no_grad():
            for batch in data_loader:
                start_all = perf_time_synchronized(device)
                batch_size = batch[1].size(0)
                total_samples += batch_size
                progress.update(batch_size)

                inputs = batch[0]
                inputs = self._prepare_inputs(inputs)
                logits = self.model(**inputs).logits

                _ = torch.nn.functional.softmax(logits, dim=-1)

                time_total_without_tokenizer += perf_time_synchronized(device) - start_all
                if total_samples >= enough_samples_to_measure:
                    break
        progress.close()
        time_with_tokenizer = perf_time_synchronized(device) - with_tokenizer_start

        metrics = {
            "time_n_per_s_total_with_tokenizer": total_samples / time_with_tokenizer,
            "time_n_per_s_total_without_tokenizer": total_samples / time_total_without_tokenizer,
        }

        if log_to_mlflow:
            self._log_metrics_dict(metrics, synchronous=True)
            self._log_dict_json(metrics, path=f"{prefix}_performance.json")
        return metrics

    def _unwrap_model(self) -> torch.nn.Module:
        """This is necessary if model is wrapped in DataParallel and we need the model class"""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def create_train_dataloader(self) -> DataLoader:
        """Create training DataLoader with args parameterization"""
        return DataLoader(
            self.dataset.train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.prepare_batch,
            prefetch_factor=self.args.train_dataloader_prefetch_factor or None,
            persistent_workers=self.args.train_dataloader_persistent_workers or False,
            num_workers=self.args.train_dataloader_num_workers or 0,
        )

    def create_eval_dataloader(self) -> DataLoader:
        """Create evaluation DataLoader with args parameterization"""
        return DataLoader(
            self.dataset.eval_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.prepare_batch,
            prefetch_factor=self.args.eval_dataloader_prefetch_factor or None,
            persistent_workers=self.args.eval_dataloader_persistent_workers or False,
            num_workers=self.args.eval_dataloader_num_workers or 0,
        )

    def train_epoch(self, epoch: int, train_data_loader: DataLoader) -> None:
        print(f"[training] Starting epoch {epoch}")
        self.update_optimizer(epoch)
        pbar = tqdm.tqdm(train_data_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        pbar.set_description("Training batches")
        self.model.train()
        for batch_number, batch in enumerate(pbar):
            self.train_step(
                batch=batch,
                batch_number=batch_number,
            )

    def train(self, train_data_loader: DataLoader, eval_data_loader: DataLoader) -> None:
        eval_f1_history = []
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch, train_data_loader)

            eval_result = self.evaluate(eval_data_loader)
            f1 = eval_result.metrics["1"]["f1-score"]
            eval_f1_history.append(f1)
            self._log_metrics_dict(eval_result.metrics, prefix="eval", step=epoch, synchronous=True)
            print(f"[training] Evaluation result")
            print_dict_level1_inline(eval_result.metrics)
            if f1 >= max(eval_f1_history[:-1], default=0):
                print(f"[training] New best class f1-score {max(eval_f1_history)} achieved at epoch {epoch}.")
                self._log_best_epoch(eval_result)
                self.save_to_mlflow()

    def train_step(self, batch: Tuple[Dict[str, torch.Tensor], torch.Tensor], batch_number: int) -> None:
        inputs = batch[0]
        labels = batch[1]
        inputs = self._prepare_inputs(inputs)
        logits = self.model(**inputs).logits
        loss = self.loss_fn(logits, labels.to(self.device))
        class_1_probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1]
        predictions = (class_1_probs >= self.args.decision_threshold).int().detach().cpu()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if batch_number % 50 == 0:
            self._log_metrics_dict(
                {
                    "train_batch_average_loss": loss.item(),
                    "train_batch_accuracy": float(torch.sum(torch.eq(predictions, labels).long()).item() / len(labels)),
                },
                step=batch_number,
                synchronous=False,
            )

    def update_optimizer(self, epoch: int) -> None:
        """Initialize optimizer only once at the beginning of training"""
        if self.optimizer is None:
            print("[training] Initializing optimizer")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

    def run_training_and_evaluation(self) -> Dict[str, float]:
        """Core training and evaluation logic"""
        print(f"[runner] Loading dataset...")

        train_data_loader = self.create_train_dataloader()
        eval_data_loader = self.create_eval_dataloader()
        print()

        self._log_params(vars(self.args))
        if self.log_to_mlflow:
            self.dataset.log_to_mlflow(should_save=False)
        self.train(train_data_loader=train_data_loader, eval_data_loader=eval_data_loader)
        result = self.measure_performance(data_loader=eval_data_loader)
        print("[runner] Evaluating performance")
        print_dict_level1_inline(result)
        return result
    
    def run_evaluation(self) -> None:
        """Run for evaluation on a test set, without training"""
        print(f"[runner] Loading dataset...")

        eval_data_loader = self.create_eval_dataloader()
        print()

        self._log_params(vars(self.args))
        if self.log_to_mlflow:
            self.dataset.log_to_mlflow(should_save=False)
        eval_result = self.test_predictions(eval_data_loader)
        self._log_metrics_dict(eval_result, prefix="test", step=0, synchronous=True)
        print(f"[EVALUATION] Evaluation result")
        print_dict_level1_inline(eval_result)

    def test_predictions(self, data_loader: DataLoader, decision_threshold: Optional[float] = None) -> Dict[str, Any]:
        if decision_threshold is None:
            decision_threshold = self.args.decision_threshold
        n_samples = len(data_loader.dataset)
        true_labels = np.empty(n_samples, dtype=np.int64)
        class_probs = np.empty((n_samples, 2), dtype=np.float32)
        write_ptr = 0

        self.model.eval()
        pbar = tqdm.tqdm(data_loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = batch[0], batch[1]

                inputs = self._prepare_inputs(inputs)
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu()

                batch_size = labels.size(0)
                idx_end = write_ptr + batch_size

                class_probs[write_ptr:idx_end, :] = probs.numpy()
                true_labels[write_ptr:idx_end] = labels.cpu().numpy()

                write_ptr = idx_end

        predictions = np.array(class_probs[:, 1] >= decision_threshold).astype(int)
        true_labels = np.array(true_labels)

        test_metrics = calculate_test_metrics(true_labels, predictions)
        return test_metrics

    def _log_params(self, params_dict: Dict[str, Any]) -> None:
        if not self.log_to_mlflow:
            print("[logger] Logging params not successful - not enabled")
            return
        params_dict = self.filter_params_for_mlflow_logging(params_dict)
        print(f"[logger] Logged params {params_dict}")
        mlflow.log_params(params_dict)

    def _log_best_epoch(self, evaluation_result: EvaluationResults) -> None:
        if not self.log_to_mlflow:
            return
        log_evaluation_result_to_mlflow(evaluation_result)

    def _log_metrics_dict(
        self, 
        d: Dict[str, Any], 
        prefix: str = "", 
        step: Optional[int] = None, 
        synchronous: bool = True
    ) -> None:
        if not self.log_to_mlflow:
            return
        log_metrics_dict_recursively(d, prefix=prefix, step=step, synchronous=synchronous)

    def _log_dict_json(self, d: Dict[str, Any], path: str) -> None:
        if not self.log_to_mlflow:
            print("[logger] Logging json dict not successful - not enabled")
            return
        log_dict_json(d, path=path)   

    def save_to_mlflow(self):
        model = copy.deepcopy(self._unwrap_model())
        model.to("cpu")
        mlflow.transformers.log_model(
            transformers_model={
                "model": model,
                "tokenizer": self.tokenizer,
            },
            artifact_path="classification_model",
            input_example="duckduckgo.com/?t=h_&q=bacon&ia=web",
            task="text-classification",
        ) 

    @abstractmethod
    def prepare_batch(self, batch):
        """collate_fn for dataloader. Output format must be (anything model needs), labels"""
        pass

    def filter_params_for_mlflow_logging(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter params for MLflow logging"""
        params_to_log = {}
        general_params = [
            "seed", "model_type", "model_size", "decision_threshold", "max_sequence_length", 
            "dataset_path", "batch_size",
            "eval_dataloader_prefetch_factor", "eval_dataloader_persistent_workers", "eval_dataloader_num_workers",
        ]

        training_params = [
            "learning_rate", "weight_decay", "epochs", "loss", "focal_loss_gamma", "focal_loss_alpha", 
            "train_dataloader_prefetch_factor", "train_dataloader_persistent_workers", "train_dataloader_num_workers",
        ]

        charbert_params = [
            "char_vocab_path", "char_max_length_per_token",
        ]

        grambeddings_params = [
            "ngram_1", "ngram_2", "ngram_3", "max_features", "max_df", "min_df",
            "max_seq_len_grambeddings", "case_insensitive", "warm_start", "warm_mode", 
            "embed_dim", "attn_width", "rnn_cell_size", "char_embedding_dim", "retain_random_percentage",
        ]
        params_to_log_keys = general_params
        if self.args.model_type == "charbert" or self.args.model_type == "pmanet":
            params_to_log_keys = params_to_log_keys + charbert_params
        elif self.args.model_type == "grambeddings":
            params_to_log_keys = params_to_log_keys + grambeddings_params

        if not self.is_evaluation:
            params_to_log_keys = params_to_log_keys + training_params

        for param_name, param_value in params.items():
            if param_name in params_to_log_keys:
                if param_name in ["dataset_path", "char_vocab_path"]:
                    param_value = os.path.abspath(param_value)
                params_to_log[param_name] = param_value
        return params_to_log