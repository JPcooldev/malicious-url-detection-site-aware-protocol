import argparse
from typing import Literal, Optional
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import BERTDatasetPretraining, CharBERTDatasetPretraining

def generate_dataloader(
    args: argparse.Namespace, 
    df: pd.DataFrame,
    DDP_WORLD_SIZE: int, 
    DDP_RANK: int, 
    split: Literal["train", "val"] = "train", 
    tokenizer = None
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    if args.model_type == "bert" or args.model_type == "urlbert":
        dataset = BERTDatasetPretraining(df)
        sampler = DistributedSampler(dataset, num_replicas=DDP_WORLD_SIZE, rank=DDP_RANK, shuffle=True if split == "train" else False)
        dataloader = DataLoader(
            dataset,
            batch_size=args.worker_batch_size,
            sampler=sampler,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader, sampler
    elif args.model_type == "charbert":
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for CharBERT pretraining dataloader.")
        dataset = CharBERTDatasetPretraining(
            tokenizer=tokenizer,
            char_vocab_path=args.char_vocab_path,
            term_vocab_path=args.term_vocab_path,
            dataset=df,
            seq_len=args.max_sequence_length,
            random_seed=args.random_seed,
            adv_probability=args.adv_prob if split == "train" else 0,
            mlm_probability=args.mlm_prob,
            char_maxlen_for_word=args.char_maxlen_for_word,
            world_size=DDP_WORLD_SIZE,
            world_rank=DDP_RANK,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.worker_batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        return dataloader, None
    else:
        raise ValueError(f"Model type {args.model_type} not supported.")
