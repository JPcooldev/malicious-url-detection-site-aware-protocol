import pandas as pd
from dataclasses import dataclass
import numpy as np
import heapq

from .feature_extraction import get_sld


def analyze_folds(df: pd.DataFrame, top_k: int = 3, max_url_len: int = None) -> pd.DataFrame:
    """
    Dynamically summarize each fold, including:
    - total sample count
    - count and % of each class in 'label'
    - number of unique SLDs
    - top-K domains by frequency (as separate columns)
    """
    all_labels = sorted(df["label"].unique())
    rows = []

    for fold, g in df.groupby("fold", sort=True):
        row = {
            "fold": fold,
            "total": len(g),
            "unique_domains": g["sld"].nunique(),
            "avg_url_len": (
                g["url"].str.len().mean() if max_url_len is None else g["url"].str.len().clip(upper=max_url_len).mean()
            ),
            "median_url_len": (
                g["url"].str.len().median()
                if max_url_len is None
                else g["url"].str.len().clip(upper=max_url_len).median()
            ),
            # Optional: you could also add min, max, or std if desired:
            # "min_url_len": g["url"].str.len().min(),
            # "max_url_len": g["url"].str.len().max(),
            "std_url_len": (
                g["url"].str.len().std() if max_url_len is None else g["url"].str.len().clip(upper=max_url_len).std()
            ),
        }

        label_counts = g["label"].value_counts()
        for label in all_labels:
            count = label_counts.get(label, 0)
            row[label] = count
            row[f"{label}_pct"] = count / len(g) * 100

        top = g["sld"].value_counts(normalize=True).head(top_k).mul(100).round(1)
        tops = [f"{domain} ({pct}%)" for domain, pct in top.items()]
        tops += [""] * (top_k - len(tops))
        for i, val in enumerate(tops, start=1):
            row[f"top_{i}"] = val

        rows.append(row)

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def get_domain_stats(df: pd.DataFrame) -> pd.DataFrame:
    assert "url_length" in df.columns, "url_length column is required"
    assert "sld" in df.columns, "sld column is required"
    df = df.copy()
    counts = (
        df.groupby("sld")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={"benign": "benign", "malicious": "malicious"})
    )
    avg_len = df.groupby("sld")["url_length"].mean().rename("avg_url_len")
    domain_stats = counts.join(avg_len).assign(total_records=lambda x: x["benign"] + x["malicious"]).reset_index()
    domain_stats["malicious_ratio"] = domain_stats["malicious"] / domain_stats["total_records"]
    domain_stats = domain_stats.sort_values(by="total_records", ascending=False).reset_index(drop=True)
    return domain_stats




@dataclass
class SamplerConfig:
    target_benign_balance: float = 0.9
    target_size: int = 3000000
    n_folds: int = 5
    seed: int = 42
    max_domain_share: float = 0.05
    length_cap: int = 255
    target_benign_per_fold: int = -1
    target_malicious_per_fold: int = -1
    target_benign_set: int = -1
    target_malicious_set: int = -1
    fold_size: int = -1
    max_urls_per_sld: int = -1
    print_fold_info: bool = False
    length_bins: list[tuple[int, int]] = (
        (0, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 120),
        (120, float("inf")),
    )

    def __post_init__(self):
        self.target_benign_set = self.target_benign_balance * self.target_size
        self.target_malicious_set = (1 - self.target_benign_balance) * self.target_size
        self.target_benign_per_fold = self.target_benign_set // self.n_folds
        self.target_malicious_per_fold = self.target_malicious_set // self.n_folds
        self.fold_size = self.target_size / self.n_folds
        self.max_urls_per_sld = self.fold_size * self.max_domain_share


def process_url_dataset(df, config: SamplerConfig):
    """
    Process URL dataset according to specifications:
    1. Remove URLs with conflicting labels
    2. Remove duplicates
    3. Assign folds first (on full clean data)
    4. Balance folds to achieve exact 9:1 ratio with length-aware sampling
    5. Final size: ~2.7M benign, ~300k malicious
    """
    np.random.seed(config.seed)
    print(f"Initial dataset size: {len(df):,}, Initial class distribution:\n{df['label'].value_counts()}")

    # Step 1: Extract SLD for each URL and calculate lengths
    print("Extracting SLDs and calculating URL lengths...")
    if "sld" not in df.columns:
        df["sld"] = df["url"].apply(get_sld)
    # cap at length_cap
    df["url_length"] = df["url"].str.len().clip(upper=config.length_cap)

    # Print initial length statistics
    print(
        f"\nInitial URL length statistics: mean ({df['url_length'].mean():.1f}), median ({df['url_length'].median():.1f}), % above 90: {(df['url_length'] > 90).mean() * 100:.1f}%"
    )

    # Step 2: Remove URLs with conflicting labels
    print("\nRemoving URLs with conflicting labels...")
    url_label_counts = df.groupby("url")["label"].nunique()
    conflicting_urls = url_label_counts[url_label_counts > 1].index
    print(f"Found {len(conflicting_urls):,} URLs with conflicting labels")
    df_clean = df[~df["url"].isin(conflicting_urls)].copy()
    print(
        f"Removed {len(conflicting_urls):,} URLs with conflicting labels ({(len(conflicting_urls)/len(df)*100):.1f}%)"
    )

    # Step 3: Remove duplicates
    print("\nRemoving duplicates...")
    df_clean = df_clean.drop_duplicates(subset=["url"], keep="first")
    print(f"Removed {len(df_clean) - len(df):,} duplicates ({((len(df_clean) - len(df))/len(df)*100):.1f}%)")
    print(f"Class distribution:\n{df_clean['label'].value_counts()}\n")

    # Step 3.5: Remove urls with excessive sld counts
    print("Removing urls with excessive sld counts...")
    grouped = df_clean.groupby("sld")["url"].count()
    df_clean, removed_urls = get_dataset_without_excessive_sld_counts(df_clean, config)
    print(
        f"Removed {len(removed_urls):,} urls with excessive sld counts ({((len(removed_urls))/len(df_clean)*100):.1f}%)"
    )
    print(f"Class distribution:\n{df_clean['label'].value_counts()}\n")

    # Step 4: Get domain statistics on full clean data
    print("Calculating domain statistics...")
    domain_stats = get_domain_stats(df_clean)
    print(f"Number of unique domains: {len(domain_stats):,}")

    # Step 5: Assign folds on full clean data
    print(f"\nAssigning {config.n_folds} folds...")
    df_with_folds = assign_folds(df_clean, domain_stats, n_folds=config.n_folds, seed=config.seed)

    # Step 6: Analyze initial fold distribution
    print("\nInitial fold analysis:")
    fold_analysis = analyze_folds(df_with_folds)
    print(fold_analysis.to_string())

    # Step 7: Balance folds with length-preserving sampling
    print("\nBalancing folds with length-preserving sampling...")
    df_balanced = balance_folds_preserve_length(df_with_folds, config)

    # Step 8: Final analysis
    print("\nFinal fold analysis:")
    fold_analysis = analyze_folds(df_balanced)
    print(fold_analysis.to_string())
    print(f"\n{'='*60}")
    print(f"Final dataset size: {len(df_balanced):,}")
    print(f"Final class distribution:\n{df_balanced['label'].value_counts()}")
    final_ratio = df_balanced["label"].value_counts()["benign"] / df_balanced["label"].value_counts()["malicious"]
    print(f"Final overall ratio: {final_ratio:.2f}:1")

    final_benign_mean = df_balanced[df_balanced["label"] == "benign"]["url_length"].mean()
    final_malicious_mean = df_balanced[df_balanced["label"] == "malicious"]["url_length"].mean()
    print(f"\nFinal mean lengths - Benign: {final_benign_mean:.1f}, Malicious: {final_malicious_mean:.1f}")
    print(f"Mean difference: {abs(final_benign_mean - final_malicious_mean):.1f}")

    return df_balanced


def get_dataset_without_excessive_sld_counts(df: pd.DataFrame, config: SamplerConfig):
    """for each sld that has lower or equal number of urls than max_urls_per_sld, retain them all. For every other sld, retain max_urls_per_sld urls. Remove them by url length - retain the longest urls. Return retained and removed urls"""
    url_counts = df.groupby("sld")["url"].count()
    retained_urls = df[df["sld"].isin(url_counts[url_counts <= config.max_urls_per_sld].index)]
    # for SLDs with more than max_urls_per_sld urls, retain the longest urls
    df = df[~df["sld"].isin(url_counts[url_counts > config.max_urls_per_sld].index)]
    # retain the longest urls
    df = df.sort_values("url_length", ascending=False).groupby("sld").head(config.max_urls_per_sld)
    removed_urls = df[~df["sld"].isin(retained_urls["sld"])]
    return retained_urls, removed_urls


def assign_folds(dataset, domain_stats: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Greedy assignment that balances total_records first, malicious count second.
    Complexity: O(D log F) where D = #domains, F = #folds.
    """
    np.random.seed(seed)
    shuffled = domain_stats.sample(frac=1, random_state=seed)
    # (total - first criterion, malicious_count - second criterion, i)
    heap = [(0, 0, i) for i in range(n_folds)]
    heapq.heapify(heap)

    domain, fold = [], []
    for d, total, malicious in shuffled[["sld", "total_records", "malicious"]].to_numpy():
        total_f, malicious_f, fid = heapq.heappop(heap)
        domain.append(d)
        fold.append(fid)
        heapq.heappush(heap, (total_f + total, malicious_f + malicious, fid))

    assigned = pd.DataFrame({"sld": domain, "fold": fold})
    assigned = assigned.merge(domain_stats[["sld", "total_records"]], on="sld", how="left")

    sld_fold_mapping = assigned[["sld", "fold"]]
    mapped_dataset = dataset.merge(sld_fold_mapping, on="sld", how="left", validate="many_to_one")
    return mapped_dataset


def balance_folds_preserve_length(df, config: SamplerConfig):
    """
    Balance folds to maintain exact 9:1 ratio while strongly preserving longer URLs.
    Ensures benign and malicious URLs end up with the same mean length.
    """
    balanced_folds = []

    for fold_id in sorted(df["fold"].unique()):
        print(f"Balancing fold {fold_id}")
        fold_df = df[df["fold"] == fold_id].copy()
        fold_balanced = balance_fold_preserve_length(fold_df, config)
        balanced_folds.append(fold_balanced)
    df_final = pd.concat(balanced_folds, ignore_index=True)
    return df_final


def balance_fold_preserve_length(df, config: SamplerConfig):
    """
    Jointly sample benign and malicious URLs to ensure matching mean lengths.
    Removes samples from both classes simultaneously to maintain balance and same mean.
    """
    df = df.copy()
    benign_df = df[df["label"] == "benign"].copy()
    malicious_df = df[df["label"] == "malicious"].copy()

    # If we already have less than target, return what we have
    if len(benign_df) <= config.target_benign_per_fold and len(malicious_df) <= config.target_malicious_per_fold:
        return df

    benign_sampled = do_bin_stratification(benign_df, config, config.target_benign_per_fold * 1.25)
    malicious_sampled = do_bin_stratification(malicious_df, config, config.target_malicious_per_fold * 1.25)
    combined_sampled = pd.concat([benign_sampled, malicious_sampled])
    if (
        len(benign_sampled) != config.target_benign_per_fold
        or len(malicious_sampled) != config.target_malicious_per_fold
    ):
        combined_sampled = final_adjustment(combined_sampled, config)

    return combined_sampled


import math


def do_bin_stratification(df, config, target_size):
    df = df.copy()

    def assign_bin(length):
        for i, (low, high) in enumerate(config.length_bins):
            if low <= length < high:
                return i
        return None

    df["length_bin"] = df["url_length"].apply(assign_bin)

    bin_groups = df.groupby("length_bin")
    bin_sizes = bin_groups.size().to_dict()

    remaining_target = target_size
    remaining_bins = set(bin_sizes.keys())
    allocations = {b: 0 for b in bin_sizes}

    while remaining_target > 0 and remaining_bins:
        # Fair share per bin
        share = math.ceil(remaining_target / len(remaining_bins))
        filled = set()

        for b in list(remaining_bins):
            capacity = bin_sizes[b] - allocations[b]
            take = min(share, capacity)
            allocations[b] += take
            remaining_target -= take

            if allocations[b] == bin_sizes[b]:  # bin is exhausted
                filled.add(b)

        remaining_bins -= filled

    balanced_parts = []
    for b, n in allocations.items():
        if n > 0:
            part = bin_groups.get_group(b).sample(n=n, random_state=config.seed)
            balanced_parts.append(part)

    balanced_dataset = pd.concat(balanced_parts, ignore_index=True)
    return balanced_dataset


def _achievable_mean_interval(lengths: np.ndarray, r: int):
    """
    Given vector of lengths (float) and an exact removal count r,
    return the *achievable* interval [mu_min, mu_max] of the mean
    after removing exactly r items (by removing largest vs smallest).
    """
    n = len(lengths)
    if r <= 0:
        m = float(np.mean(lengths)) if n > 0 else np.nan
        return m, m

    if r >= n:
        # Degenerate; after removing all, mean undefined. Constrain tightly to current mean.
        m = float(np.mean(lengths)) if n > 0 else np.nan
        return m, m

    s = float(np.sum(lengths))
    # sums of r smallest / largest
    sort_asc = np.sort(lengths)
    sum_smallest = float(np.sum(sort_asc[:r]))
    sum_largest = float(np.sum(sort_asc[-r:]))

    mu_inc = (s - sum_smallest) / (n - r)  # remove r smallest -> mean increases (max achievable mean)
    mu_dec = (s - sum_largest) / (n - r)  # remove r largest  -> mean decreases (min achievable mean)
    mu_min = min(mu_inc, mu_dec)
    mu_max = max(mu_inc, mu_dec)
    return mu_min, mu_max


def _choose_target_mean(interval_b, interval_m, fixed_mean_b=None, fixed_mean_m=None):
    """
    Pick a common target mean mu* for both classes after removal.
    - If one class doesn't remove, fix mu* to its current mean (clipped to the other's interval).
    - If both remove, pick any value in the intersection; if empty, pick the midpoint between
      the nearest ends to minimize total deviation.
    """
    b_min, b_max = interval_b
    m_min, m_max = interval_m

    # Case: only benign fixed
    if fixed_mean_b is not None and fixed_mean_m is None:
        mu = fixed_mean_b
        # Clip into malicious achievable interval
        return min(max(mu, m_min), m_max)
    # Case: only malicious fixed
    if fixed_mean_m is not None and fixed_mean_b is None:
        mu = fixed_mean_m
        # Clip into benign achievable interval
        return min(max(mu, b_min), b_max)

    # Both remove: try intersection
    inter_min = max(b_min, m_min)
    inter_max = min(b_max, m_max)
    if inter_min <= inter_max:
        # Any point in intersection works; pick middle to be neutral
        return 0.5 * (inter_min + inter_max)

    # No intersection: pick midpoint between closest ends
    # (minimize symmetric deviation to both intervals)
    # Distance between intervals:
    if b_max < m_min:
        return 0.5 * (b_max + m_min)
    else:
        return 0.5 * (m_max + b_min)


def _probabilistic_remove(
    df_class: pd.DataFrame,
    target_mean: float,
    remove_count: int,
    length_col="url_length",
    alpha: float = 2.0,
    eps: float = 1e-12,
    rng: np.random.Generator | None = None,
):
    """
    Remove exactly `remove_count` rows from df_class by *random sampling without replacement*.
    Sampling probabilities are shaped to move the mean toward `target_mean`:
    - If current mean < target: prioritize removing *shorter* items.
    - If current mean > target: prioritize removing *longer* items.
    alpha (>1) sharpens the preference; eps ensures support everywhere.
    """
    if remove_count <= 0 or df_class.empty:
        return df_class

    lengths = df_class[length_col].to_numpy(dtype=float)
    cur_mean = float(np.mean(lengths)) if len(lengths) else target_mean
    if rng is None:
        rng = np.random.default_rng()

    if cur_mean < target_mean:
        # Need to increase mean -> remove shorter items
        weights = np.maximum(0.0, target_mean - lengths) ** alpha + eps
    elif cur_mean > target_mean:
        # Need to decrease mean -> remove longer items
        weights = np.maximum(0.0, lengths - target_mean) ** alpha + eps
    else:
        # Already on target -> uniform removal
        weights = np.full_like(lengths, 1.0, dtype=float)

    p = weights / weights.sum()
    remove_count = int(min(remove_count, len(df_class)))
    # sample without replacement, respecting probabilities
    idx = rng.choice(len(df_class), size=remove_count, replace=False, p=p)
    mask = np.ones(len(df_class), dtype=bool)
    mask[idx] = False
    return df_class.iloc[mask]


def final_adjustment(df: pd.DataFrame, config: SamplerConfig):
    """
    Final adjustment to hit exact targets while maintaining *equalized* class means.
    - Removes exactly the excess from each class.
    - Chooses a feasible common target mean and samples removals probabilistically to move toward it.
    """
    rng = np.random.default_rng(getattr(config, "seed", None))

    # Split by class
    benign_df = df[df["label"] == "benign"].copy()
    malicious_df = df[df["label"] == "malicious"].copy()

    n_b, n_m = len(benign_df), len(malicious_df)
    tgt_b = int(getattr(config, "target_benign_per_fold"))
    tgt_m = int(getattr(config, "target_malicious_per_fold"))

    r_b = max(0, n_b - tgt_b)
    r_m = max(0, n_m - tgt_m)

    # Quick exits: nothing to remove
    if r_b == 0 and r_m == 0:
        return df

    # Compute achievable mean intervals after removing exactly r items
    b_lengths = benign_df["url_length"].to_numpy(dtype=float)
    m_lengths = malicious_df["url_length"].to_numpy(dtype=float)

    b_interval = _achievable_mean_interval(b_lengths, r_b)
    m_interval = _achievable_mean_interval(m_lengths, r_m)

    fixed_b = None if r_b > 0 else (float(np.mean(b_lengths)) if n_b else None)
    fixed_m = None if r_m > 0 else (float(np.mean(m_lengths)) if n_m else None)

    # Choose feasible common target mean
    mu_star = _choose_target_mean(b_interval, m_interval, fixed_mean_b=fixed_b, fixed_mean_m=fixed_m)

    # Probabilistic removals toward mu_star
    if r_b > 0:
        benign_df = _probabilistic_remove(
            benign_df, target_mean=mu_star, remove_count=r_b, length_col="url_length", rng=rng
        )
    if r_m > 0:
        malicious_df = _probabilistic_remove(
            malicious_df, target_mean=mu_star, remove_count=r_m, length_col="url_length", rng=rng
        )

    # Sanity: enforce exact sizes
    # (due to numerical issues with probabilities, double-check counts)
    if len(benign_df) > tgt_b:
        # remove uniformly at random the tiny extra
        extra = len(benign_df) - tgt_b
        keep_idx = rng.choice(len(benign_df), size=tgt_b, replace=False)
        benign_df = benign_df.iloc[keep_idx]
    if len(malicious_df) > tgt_m:
        extra = len(malicious_df) - tgt_m
        keep_idx = rng.choice(len(malicious_df), size=tgt_m, replace=False)
        malicious_df = malicious_df.iloc[keep_idx]

    out = pd.concat([benign_df, malicious_df], axis=0)
    # Optional: shuffle for cleanliness
    out = out.sample(frac=1.0, random_state=getattr(config, "seed", None)).reset_index(drop=True)
    return out