import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# features.py からクラスをインポート
# ※このスクリプトと同じ階層に features.py がある前提です
try:
    from features import ChemBERTaFeatureExtractor
except ImportError:
    print("Error: 'features.py' not found. Please ensure it is in the same directory.")
    exit(1)

# ファイルパス（環境に合わせて調整してください）
INPUT_MONA_PATH = "processed/MoNA.npz"
INPUT_FEAT_PATH = "processed/MoNA_Rdkit_feature.npz"
OUTPUT_PATH = "processed/MoNA_features.npz"

def main():
    print(f"Loading {INPUT_MONA_PATH} ...")
    try:
        data_mona = np.load(INPUT_MONA_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_MONA_PATH}")
        return

    print(f"Loading {INPUT_FEAT_PATH} ...")
    try:
        data_feat = np.load(INPUT_FEAT_PATH, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FEAT_PATH}")
        return

    # 1. SMILESの一致確認
    smi_mona = data_mona['smiles']
    smi_feat = data_feat['smiles']

    # 型変換して比較用に揃える
    smi_mona_str = smi_mona.astype(str)
    smi_feat_str = smi_feat.astype(str)

    if not np.array_equal(smi_mona_str, smi_feat_str):
        raise ValueError("Error: SMILES in the two NPZ files do not match! The order or content is different.")
    
    print("SMILES alignment verified. Proceeding to merge.")

    # 2. 3D計算成功フラグの取得
    success_mask = data_feat['success'].astype(bool)
    total_count = len(success_mask)
    valid_count = success_mask.sum()
    drop_count = total_count - valid_count

    print(f"Total molecules: {total_count}")
    print(f"3D Calc Success: {valid_count} (Dropping {drop_count} molecules)")

    # 3. データの結合とフィルタリング
    save_dict = {}

    # (A) MoNA.npz の全データを mask でフィルタリングして格納
    for key in data_mona.files:
        original_arr = data_mona[key]
        if len(original_arr) == total_count:
            save_dict[key] = original_arr[success_mask]
        else:
            print(f"Warning: Key '{key}' has length {len(original_arr)} != {total_count}. Keeping as is.")
            save_dict[key] = original_arr

    # (B) MoNA_feature.npz から記述子を追加（ここもフィルタリング）
    save_dict['rdkit_2d'] = data_feat['descriptors_2d'][success_mask]
    save_dict['rdkit_3d'] = data_feat['descriptors_3d'][success_mask]

    # 次元情報の保存
    dim_2d = save_dict['rdkit_2d'].shape[1]
    dim_3d = save_dict['rdkit_3d'].shape[1]
    save_dict['dim_rdkit_2d'] = dim_2d
    save_dict['dim_rdkit_3d'] = dim_3d

    print(f"RDKit Features added: 2D shape={save_dict['rdkit_2d'].shape}, 3D shape={save_dict['rdkit_3d'].shape}")

    # ---------------------------------------------------------
    # (C) BERT特徴量の計算と追加
    # ---------------------------------------------------------
    print("\n--- Starting BERT Feature Calculation ---")
    
    # GPUチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extractorの初期化
    extractor = ChemBERTaFeatureExtractor(device=device)
    
    # 対象となるSMILESを取得（フィルタリング済みのもの）
    target_smiles = save_dict['smiles'].astype(str)
    n_samples = len(target_smiles)

    # 埋め込み次元の取得 (ダミー計算)
    dummy_emb = extractor("C", as_numpy=True)
    dim_bert = dummy_emb.shape[0]
    print(f"BERT Embedding dim: {dim_bert}")

    # 結果格納用配列
    bert_features = np.zeros((n_samples, dim_bert), dtype=np.float32)

    print(f"Processing {n_samples} molecules...")
    for i, smi in enumerate(tqdm(target_smiles, desc="BERT Encoding")):
        # features.py の仕様に合わせて1つずつ計算
        # (GPUに乗せているので推論モードならそこそこ速いです)
        emb = extractor(smi, as_numpy=True)
        bert_features[i] = emb

    # 辞書に追加
    save_dict['bert_features'] = bert_features
    save_dict['dim_bert'] = dim_bert
    
    print(f"BERT Features added: shape={bert_features.shape}")

    # 4. 保存
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path} ...")
    np.savez_compressed(output_path, **save_dict)
    print("Done. All features (RDKit 2D/3D + BERT) are merged and saved.")

if __name__ == "__main__":
    main()