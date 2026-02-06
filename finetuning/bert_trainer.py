import argparse
from typing import Optional, Tuple, List

from transformers import PreTrainedTokenizerFast
import torch

from .trainer import TrainerBase
from finetuning.data_processing.dataset import URLDataset


class BertTrainer(TrainerBase):
    """
    Trainer for BERT-based models
    """

    tokenizer: PreTrainedTokenizerFast

    def __init__(self, 
        args: argparse.Namespace, 
        model: torch.nn.Module, 
        tokenizer: PreTrainedTokenizerFast, 
        dataset: URLDataset, 
        device: Optional[torch.device] = None, 
        move_to_device: bool = True, 
        log_to_mlflow: bool = True,
        is_evaluation: bool = False):
        super().__init__(
            args, model, dataset, log_to_mlflow=log_to_mlflow, device=device, move_to_device=move_to_device, is_evaluation=is_evaluation
        )
        self.tokenizer = tokenizer

    def prepare_batch(self, batch: Tuple[List[str], List[int]]):
        urls, labels = zip(*batch)
        encoded_batch = self.tokenizer(
            urls, padding=True, truncation=True, max_length=self.args.max_sequence_length, return_tensors="pt"
        )
        input_dict = {
            "input_ids": encoded_batch["input_ids"],
            "attention_mask": encoded_batch["attention_mask"],
        }
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return input_dict, labels_tensor