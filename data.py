from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PARQUET_FILE = "data/processed/MoNA.parquet"


def load_dataset(parquet_file: str = DEFAULT_PARQUET_FILE) -> pd.DataFrame:
    """Load the preprocessed dataset parquet file."""
    return pd.read_parquet(parquet_file)


def _resolve_split_sizes(
    total: int,
    train_size: Optional[int],
    val_size: Optional[int],
    test_size: Optional[int],
    val_ratio: float,
    test_ratio: float,
) -> dict:
    """Decide how many molecules go to each split when manual sizes are requested."""
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    for name, val in sizes.items():
        if val is not None and val < 0:
            raise ValueError(f"{name}_size must be >= 0, got {val}")

    specified_sum = sum(v for v in sizes.values() if v is not None)
    if specified_sum > total:
        raise ValueError(f"Requested split sizes exceed dataset size: requested={specified_sum}, available={total}")

    missing = [k for k, v in sizes.items() if v is None]
    remaining = total - specified_sum
    if missing:
        default_ratios = {
            "train": max(0.0, 1.0 - val_ratio - test_ratio),
            "val": val_ratio,
            "test": test_ratio,
        }
        ratio_sum = sum(default_ratios[m] for m in missing)
        if ratio_sum <= 0:
            ratio_sum = len(missing)
            default_ratios = {m: 1.0 for m in missing}

        for m in missing:
            sizes[m] = int(round(remaining * default_ratios[m] / ratio_sum))

        # Adjust rounding by giving/taking the difference to the train split
        final_sum = sum(sizes.values())
        if final_sum < total:
            sizes["train"] += total - final_sum
        elif final_sum > total:
            sizes["train"] = max(0, sizes["train"] - (final_sum - total))

    return sizes


def split_dataset(
    seed: int,
    flag_column: str,
    domain_split: bool,
    n_pos: Optional[int],
    n_neg: Optional[int],
    train_size: Optional[int],
    val_size: Optional[int],
    test_size: Optional[int],
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    parquet_file: str = DEFAULT_PARQUET_FILE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test DataFrames according to the existing split strategy.

    - If domain_split is False *and* any of train_size/val_size/test_size is given,
      the dataset is shuffled and split by the requested molecule counts (n_pos/n_neg are ignored).
    - Otherwise, perform the original flag-aware stratified split and downsample train by n_pos/n_neg.
    """
    df = load_dataset(parquet_file)
    if flag_column not in df.columns:
        raise ValueError(f"flag_column '{flag_column}' not found in dataframe")

    manual_split_requested = (not domain_split) and any(sz is not None for sz in (train_size, val_size, test_size))
    if domain_split and manual_split_requested:
        raise ValueError("Manual split sizes are only supported when domain_split=False.")

    if manual_split_requested:
        print("Using simple random split by molecule counts (domain_split=False). n_pos/n_neg are ignored.")
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_sizes = _resolve_split_sizes(len(df_shuffled), train_size, val_size, test_size, val_ratio, test_ratio)

        train_end = split_sizes["train"]
        val_end = train_end + split_sizes["val"]
        test_end = val_end + split_sizes["test"]

        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:test_end]
        print(f"Split sizes (molecules): train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df[flag_column],
    )

    adj_val_ratio = val_ratio / (1.0 - test_ratio)
    train_full_df, val_df = train_test_split(
        train_val_df,
        test_size=adj_val_ratio,
        random_state=seed,
        stratify=train_val_df[flag_column],
    )

    flag_series = train_full_df[flag_column].astype(bool)
    pool_pos = train_full_df[flag_series]
    pool_neg = train_full_df[~flag_series]

    if n_pos > len(pool_pos):
        raise ValueError(f"Requested n_pos={n_pos}, but only {len(pool_pos)} positives available in train pool")
    if n_neg > len(pool_neg):
        raise ValueError(f"Requested n_neg={n_neg}, but only {len(pool_neg)} negatives available in train pool")

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(len(pool_pos), size=n_pos, replace=False)
    neg_idx = rng.choice(len(pool_neg), size=n_neg, replace=False)

    train_df = pd.concat(
        [
            pool_pos.iloc[pos_idx],
            pool_neg.iloc[neg_idx],
        ],
        axis=0,
    )
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df
