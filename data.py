from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 作成したNPZファイルを指定
DEFAULT_DATA_FILE = "processed/MoNA_features.npz"

def load_dataset(data_file: str = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """Load the dataset from NPZ and convert to DataFrame."""
    try:
        data = np.load(data_file, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_file}. Please run merge_npz.py first.")

    df_dict = {}
    n_samples = len(data["smiles"]) if "smiles" in data else None
    
    # 1. 通常のメタデータ/ラベル (1次元配列) とスペクトル(2次元)の格納
    exclude_keys = {'rdkit_2d', 'rdkit_3d', 'bert_features', 'dim_rdkit_2d', 'dim_rdkit_3d', 'dim_bert'}
    for key in data.files:
        if key in exclude_keys:
            continue
        arr = data[key]
        if arr.ndim == 1 and n_samples is not None and len(arr) == n_samples:
            df_dict[key] = arr
        elif key == "spectrum" and arr.ndim >= 2 and n_samples is not None and arr.shape[0] == n_samples:
            # スペクトルは (N, num_bins) をリストとして保存
            df_dict["spectrum"] = list(arr)
    
    # 2. 多次元配列の特徴量を「リストの列」として格納
    # これにより、train_test_split で行と一緒に安全に分割できます
    
    # BERT
    if 'bert_features' in data:
        # np.array(N, 768) -> list of arrays
        df_dict['bert_features'] = list(data['bert_features'])
        
    # RDKit 2D
    if 'rdkit_2d' in data:
        df_dict['rdkit_2d'] = list(data['rdkit_2d'])
        
    # RDKit 3D
    if 'rdkit_3d' in data:
        df_dict['rdkit_3d'] = list(data['rdkit_3d'])

    df = pd.DataFrame(df_dict)
    
    # SMILESのデコード処理 (バイト列の場合)
    if 'smiles' in df.columns and isinstance(df['smiles'].iloc[0], bytes):
        df['smiles'] = df['smiles'].str.decode('utf-8')
        
    return df


def _resolve_split_sizes(
    total: int,
    train_size: Optional[int],
    val_size: Optional[int],
    test_size: Optional[int],
    val_ratio: float,
    test_ratio: float,
) -> dict:
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
    data_file: str = DEFAULT_DATA_FILE, # 引数名 data_file
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    df = load_dataset(data_file)
    
    if flag_column not in df.columns:
        raise ValueError(f"flag_column '{flag_column}' not found in dataframe")

    manual_split_requested = (not domain_split) and any(sz is not None for sz in (train_size, val_size, test_size))
    
    if manual_split_requested:
        print("Using simple random split by molecule counts (domain_split=False).")
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_sizes = _resolve_split_sizes(len(df_shuffled), train_size, val_size, test_size, val_ratio, test_ratio)

        train_end = split_sizes["train"]
        val_end = train_end + split_sizes["val"]
        test_end = val_end + split_sizes["test"]

        train_df = df_shuffled.iloc[:train_end]
        val_df = df_shuffled.iloc[train_end:val_end]
        test_df = df_shuffled.iloc[val_end:test_end]
        print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    # 従来の層化抽出ロジック（省略せずに記述）
    train_val_df, test_df = train_test_split(
        df, test_size=test_ratio, random_state=seed, stratify=df[flag_column]
    )
    adj_val_ratio = val_ratio / (1.0 - test_ratio)
    train_full_df, val_df = train_test_split(
        train_val_df, test_size=adj_val_ratio, random_state=seed, stratify=train_val_df[flag_column]
    )

    flag_series = train_full_df[flag_column].astype(bool)
    pool_pos = train_full_df[flag_series]
    pool_neg = train_full_df[~flag_series]

    # n_pos/n_neg が指定されている場合のサンプリング
    rng = np.random.default_rng(seed)
    
    # n_pos
    if n_pos is not None:
        if n_pos > len(pool_pos):
             raise ValueError(f"Requested n_pos={n_pos} > available {len(pool_pos)}")
        pos_idx = rng.choice(len(pool_pos), size=n_pos, replace=False)
        pos_df = pool_pos.iloc[pos_idx]
    else:
        pos_df = pool_pos
    
    # n_neg
    if n_neg is not None:
        if n_neg > len(pool_neg):
             raise ValueError(f"Requested n_neg={n_neg} > available {len(pool_neg)}")
        neg_idx = rng.choice(len(pool_neg), size=n_neg, replace=False)
        neg_df = pool_neg.iloc[neg_idx]
    else:
        neg_df = pool_neg

    train_df = pd.concat([pos_df, neg_df], axis=0)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df
