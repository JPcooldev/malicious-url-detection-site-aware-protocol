from datasets import load_dataset
import pandas as pd
import os

def load_english_wiki(num_samples: int = None, random_seed: int = 42, num_proc: int = 4):
    """
    Load the English Wikipedia dataset as an Arrow-backed Hugging Face Dataset, avoiding
    conversion to pandas to prevent driver OOM. Returns a shuffled map-style dataset
    (train split) with a single 'text' column.
    """
    dataset_name = "wikimedia/wikipedia"
    subset_name = "20231101.en"
    dataset = load_dataset(dataset_name, subset_name)
    dataset = dataset['train']

    # remove references section from the articles
    dataset = dataset.map(
        remove_references,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Removing References section from the articles (in batches)",
        load_from_cache_file=True,
    )

    # shuffle the dataset (Arrow-shuffle creates a new randomized order with an index mapping)
    dataset = dataset.shuffle(seed=random_seed)

    # select a subset of the dataset if requested
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])   

    return dataset

def remove_references(batch):
    processed_texts = []
    for text in batch['text']:
        parts = text.split("\nReferences")
        processed_texts.append(parts[0])
    batch['text'] = processed_texts
    return batch

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load dataset from path"""
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    df = pd.read_csv(dataset_path)
    assert "url" in df.columns, "Dataset must contain a column 'url'"
    return df