from collections import Counter
from pathlib import Path
import random
import json
import tempfile
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from .feature_extraction import get_sld


def count_slds(input_csv_path: str, output_json_path: str, url_column: str = "url", chunksize: int = 1_000_000) -> None:
    counts = Counter()
    input_path = Path(input_csv_path)
    reader_kwargs = {
        "filepath_or_buffer": input_path,
        "chunksize": chunksize,
        "dtype": {url_column or "url": "string"},
        "keep_default_na": False,
    }
    if not url_column:
        reader_kwargs.pop("dtype")
    total_rows = None
    try:
        total_rows = sum(1 for _ in open(input_path, "r", encoding="utf-8")) - 1
    except Exception:
        pass
    for chunk in tqdm(
        pd.read_csv(**reader_kwargs),
        total=(total_rows // chunksize + 1) if total_rows else None,
        desc="Processing chunks",
    ):
        if url_column and url_column not in chunk.columns:
            raise ValueError(f"Column '{url_column}' not found in input CSV")
        if not url_column:
            chunk.columns.values[0] = "url"
        slds = chunk["url"].map(get_sld)
        counts.update(slds.dropna().tolist())
    result = dict(counts)
    with open(output_json_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=2, ensure_ascii=False)


def gini_coefficient(array) -> float:
    """Calculate Gini coefficient for domain distribution"""
    # Fix: Check array length properly for numpy arrays
    if len(array) == 0:
        return 0.0
    counts = np.array(array)
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n


def load_blocked_domains_from_json(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return set(data.keys())
    if isinstance(data, list):
        return set(data)
    raise ValueError(f"{path} must be a JSON object or array")


def canonize_url(url: str) -> str:
    """
    Canonicalize a URL by:
      1. Lowercasing scheme & netloc
      2. Removing default ports (80 for http, 443 for https)
      3. Ensuring a non-empty path (defaults to '/')
      4. Sorting query parameters lexicographically
      5. Stripping fragment identifiers

    Examples:
      canonize_url("HTTP://Example.com:80/a?b=2&a=1#frag")
        -> "http://example.com/a?a=1&b=2"
    """
    p = urlparse(url)

    scheme = p.scheme.lower()
    netloc = p.hostname.lower() if p.hostname else ""
    port = p.port
    if port:
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
    if port:
        netloc = f"{netloc}:{port}"

    path = p.path or "/"

    query = ""
    if p.query:
        params = parse_qsl(p.query, keep_blank_values=True)
        params.sort()
        query = urlencode(params, doseq=True)

    canon = urlunparse((scheme, netloc, path, "", query, ""))
    return canon



def remove_protocol_and_www(url: str) -> str:
    """
    Remove leading http://, https://, ftp:// and leading 'www.'
    from a single URL string.
    """
    if not isinstance(url, str) or not url:
        return url

    u = url.strip()
    for proto in ("http://", "https://"):
        if u.lower().startswith(proto):
            u = u[len(proto) :]
            break

    if u.lower().startswith("www."):
        u = u[4:]

    return u


def external_shuffle_csv(input_path: str, output_path: str, num_buckets: int = 100, seed: int = 42):
    """
    Shuffle a large CSV file in a memory-efficient way, preserving the first row as header.

    Parameters:
      - input_path:  path to the original CSV
      - output_path: path for the shuffled CSV
      - num_buckets: how many temporary buckets to split into
      - seed:        RNG seed for reproducibility
    """
    random.seed(seed)

    with open(input_path, "r", newline="") as inp:
        header = inp.readline()

    with open(output_path, "w", newline="") as out:
        out.write(header)

    tmp_dir = tempfile.mkdtemp(prefix="csvshuffle_")
    bucket_paths = [os.path.join(tmp_dir, f"bucket_{i}.tmp") for i in range(num_buckets)]
    buckets = [open(p, "w", newline="") for p in bucket_paths]

    try:
        with open(input_path, "r", newline="") as inp:
            next(inp)
            for line in inp:
                idx = random.randrange(num_buckets)
                buckets[idx].write(line)
    finally:
        for f in buckets:
            f.close()

    order = list(range(num_buckets))
    random.shuffle(order)

    with open(output_path, "a", newline="") as out:
        for i in order:
            path = bucket_paths[i]
            with open(path, "r", newline="") as b:
                lines = b.readlines()
            random.shuffle(lines)
            out.writelines(lines)
            os.remove(path)

    os.rmdir(tmp_dir)
