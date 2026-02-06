# This file is modified from https://github.com/Davidup1/URLBERT
# Original code copyright (c) Davidup1, licensed under Apache License 2.0.

from typing import Tuple
import torch

def shuffle_tokens(input_ids: torch.Tensor, shuffle_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shuffle tokens in the input sequence.
    """
    labels = torch.zeros_like(input_ids)
    shuffled = input_ids.clone()
    # iterate over the batch
    for i in range(input_ids.size(0)):
        tokens = shuffled[i]
        # choose tokens to shuffle (based on shuffle_prob)
        mask = torch.rand(tokens.size(), device=tokens.device) < shuffle_prob
        # get indices of tokens to shuffle
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() > 1:
            # permute the indices of the tokens
            perm = idx[torch.randperm(idx.numel())]
            shuffled[i, idx] = tokens[perm]
            # set labels for the shuffled tokens
            labels[i, idx] = 1
    return shuffled, labels

def replace_tokens(input_ids: torch.Tensor, vocab_size: int, replace_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace tokens in the input sequence with a different tokens from the vocabulary.
    """
    labels = torch.zeros_like(input_ids)
    replaced = input_ids.clone()
    # choose tokens to replace (based on replace_prob)
    mask = torch.rand(input_ids.size(), device=input_ids.device) < replace_prob
    random_tokens = torch.randint(0, vocab_size, input_ids.size(), device=input_ids.device)
    # replace the tokens with random tokens from the vocabulary
    replaced[mask] = random_tokens[mask]
    # set labels for the replaced tokens
    labels[mask] = 1
    return replaced, labels