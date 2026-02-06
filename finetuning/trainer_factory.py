import argparse
from typing import Union

import torch

from finetuning.data_processing.dataset import URLDataset
from finetuning.bert_trainer import BertTrainer
from finetuning.charbert_trainer import CharBertTrainer
from finetuning.grambeddings_trainer import GrambeddingsTrainer

from models.utils import (
    load_tokenizer, 
    load_pretrained_model, 
    mlflow_load_finetuned_model_and_tokenizer
)


def get_trainer(
    args: argparse.Namespace,
    dataset: URLDataset,
    device: torch.device = None,
    move_to_device: bool = True,
    log_to_mlflow: bool = True,
    is_evaluation: bool = False,
) -> Union[CharBertTrainer, BertTrainer, GrambeddingsTrainer]:
    """Get trainer for a given model type"""
    if args.model_type == "grambeddings":
        return GrambeddingsTrainer(args, dataset)

    tokenizer = load_tokenizer(args.model_type)
    model = load_pretrained_model(args.model_type, args.model_size, run_id=args.run_id)
    if hasattr(model, "config") and getattr(model.config, "num_labels", None) != 2:
        model.config.num_labels = 2
    
    if args.model_type == "charbert" or args.model_type == "pmanet":
        return CharBertTrainer(
            args=args, 
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            device=device, 
            move_to_device=move_to_device, 
            log_to_mlflow=log_to_mlflow,
            is_evaluation=is_evaluation
        )
    else:
        return BertTrainer(
            args=args, 
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            device=device, 
            move_to_device=move_to_device, 
            log_to_mlflow=log_to_mlflow,
            is_evaluation=is_evaluation
        )

# not implemented for grambeddings
def get_evaluator(
    args: argparse.Namespace,
    dataset: URLDataset,
    device: torch.device = None,
    move_to_device: bool = True,
    log_to_mlflow: bool = True,
    is_evaluation: bool = True,
):
    """Get evaluator for a given model type"""
    tokenizer = load_tokenizer(args.model_type)
    model = mlflow_load_finetuned_model_and_tokenizer(args.model_type, args.model_size)
    
    if args.model_type == "charbert" or args.model_type == "pmanet":
        return CharBertTrainer(
            args=args, 
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            device=device, 
            move_to_device=move_to_device, 
            log_to_mlflow=log_to_mlflow,
            is_evaluation=is_evaluation
        )
    else:
        return BertTrainer(
            args=args, 
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset, 
            device=device, 
            move_to_device=move_to_device, 
            log_to_mlflow=log_to_mlflow,
            is_evaluation=is_evaluation
        )