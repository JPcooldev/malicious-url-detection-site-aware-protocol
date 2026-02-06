import json
from typing import Dict
import numpy as np
import torch
import random


def print_dict_level1_inline(d: dict, float_precision=10):
    def format_value(v):
        if isinstance(v, float):
            return round(v, float_precision)
        if isinstance(v, dict) or isinstance(v, list):
            return json.dumps(v, indent=None, separators=(", ", ": "))
        return repr(v)

    for k, v in d.items():
        print(f"{repr(k)}: {format_value(v)}")


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "/") -> Dict:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif not isinstance(v, (list, tuple)):  # Skip lists and tuples
            items.append((new_key, v))
    return dict(items)


def setup_randomness(seed: int):
    """Setup random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # in case any standard library uses some random function
    random.seed(seed)