import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 自作モジュールのインポート
from models import build_model, ModelConfig
from train import (
    DEFAULT_FLAG_COLUMNS,
    MassSpecDataset,
    split_dataset,
    apply_feature_scaling,
)

# --- 設定 (実験環境に合わせて変更してください) ---
MODEL_PATH = "model/mass_spec_model.pth"       # 学習済みモデルのパス
DATA_FILE = "data/processed/MoNA_features.npz" # データファイルのパス
FEATURE_TYPE = "ecfp+bert+rdkit2d+rdkit3d+flag" # 学習時と同じ特徴量設定
ECFP_BITS = 1024
FLAG_COLUMN = "has_F"
BATCH_SIZE = 128
SEEDS = 1  # データ分割の再現用

def main():
    # 1. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. データの準備 (train.pyと同様の手順でテストデータを取得)
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    print("Loading and splitting dataset...")
    # split_datasetを使って、学習時と同じテストセットを復元します
    # (scalingを行うため、train_dfも必要です)
    train_df, _, test_df = split_dataset(
        seed=SEEDS,
        flag_column=FLAG_COLUMN,
        domain_split=False, # 実験に合わせて変更
        n_pos=None, n_neg=None, # サイズ指定分割の場合はNoneでOK
        data_file=DATA_FILE,
        train_size=7000,
        val_size=500,
        test_size=1000,
    )
    
    # スケーリングの適用 (RdKit記述子などがある場合必須)
    component_order = FEATURE_TYPE.split('+')
    scaling_cols = []
    if "rdkit2d" in component_order: scaling_cols.append("rdkit_2d")
    if "rdkit3d" in component_order: scaling_cols.append("rdkit_3d")
    
    if scaling_cols:
        print(f"Applying feature scaling to: {scaling_cols}")
        train_df, _, test_df = apply_feature_scaling(train_df, test_df.copy(), test_df, scaling_cols)

    # Dataset作成 (次元数取得のためTrainも一瞬作る)
    dummy_train_ds = MassSpecDataset(
        train_df.head(1),
        feature_type=FEATURE_TYPE,
        ecfp_bits=ECFP_BITS,
        flag_column=FLAG_COLUMN,
        structural_flag_columns=DEFAULT_FLAG_COLUMNS,
    )
    test_dataset = MassSpecDataset(
        test_df,
        feature_type=FEATURE_TYPE,
        ecfp_bits=ECFP_BITS,
        flag_column=FLAG_COLUMN,
        structural_flag_columns=DEFAULT_FLAG_COLUMNS,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # 解析なのでシャッフル不要
        num_workers=0
    )

    # 3. モデル設定の復元
    # Datasetから各特徴量の次元数を取得
    feature_shapes = dummy_train_ds.component_dims
    # 順序を保証
    ordered_shapes = {k: feature_shapes[k] for k in dummy_train_ds.component_order}
    total_input_dim = dummy_train_ds.features.shape[1]

    print(f"Feature Shapes: {ordered_shapes}")
    
    config = ModelConfig(
        fp_length=total_input_dim,
        max_mass_spec_peak_loc=test_dataset.spec_len,
        feature_type=FEATURE_TYPE,
        feature_shapes=ordered_shapes,
        fusion_dim=512, # 学習時の設定に合わせてください
        device=device
    )

    # 4. モデルのロード
    print(f"Loading model from {MODEL_PATH}...")
    model = build_model("attention_mlp", config=config)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Check if MODEL_PATH matches the saved 'attention_mlp' model.")
        return

    # 5. 推論とAttention重みの収集
    print("Running inference to collect attention weights...")
    
    all_weights = []
    all_flags = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mw = batch['mw'].to(device)
            
            # 推論実行 (ここで model.last_attn_weights が更新される)
            model(features, mw)
            
            # 直前のバッチの重みを取得 (Batch, Num_Features)
            # ※ models.py の修正で .cpu() されている前提
            w = model.last_attn_weights
            
            # フラグ情報も取得 (F含有かどうかで色分けするため)
            if 'flag' in batch:
                f = batch['flag'].squeeze(-1) # (Batch,)
            else:
                # フラグがない場合は全部0扱い
                f = torch.zeros(features.shape[0])
            
            all_weights.append(w)
            all_flags.append(f)

    # データを結合
    weights_tensor = torch.cat(all_weights, dim=0).numpy() # (N_samples, N_features)
    flags_tensor = torch.cat(all_flags, dim=0).numpy()     # (N_samples,)
    
    feature_names = list(ordered_shapes.keys())

    # 6. 可視化 (Seaborn Boxplot)
    # DataFrameに整形: [Weight, FeatureName, Has_F]
    rows = []
    for i in range(weights_tensor.shape[0]):
        has_f = "F-containing" if flags_tensor[i] >= 0.5 else "General"
        for j, feat_name in enumerate(feature_names):
            rows.append({
                "Attention Weight": weights_tensor[i, j],
                "Feature": feat_name,
                "Group": has_f
            })
            
    df_plot = pd.DataFrame(rows)

    print("\n--- Visualization ---")
    plt.figure(figsize=(12, 6))

    groups = ["General", "F-containing"]
    colors = {"General": "skyblue", "F-containing": "salmon"}

    box_data = []
    positions = []
    for idx, feat in enumerate(feature_names):
        for j, grp in enumerate(groups):
            vals = df_plot[(df_plot["Feature"] == feat) & (df_plot["Group"] == grp)]["Attention Weight"].values
            box_data.append(vals)
            positions.append(idx * 3 + j)

    bp = plt.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )

    for patch, pos in zip(bp["boxes"], positions):
        grp = groups[pos % len(groups)]
        patch.set_facecolor(colors.get(grp, "lightgray"))

    centers = [(idx * 3 + 0.5) for idx in range(len(feature_names))]
    plt.xticks(centers, feature_names)
    plt.xlim(-1, positions[-1] + 1)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Attention Weight")
    plt.title(f"Attention Weights Distribution by Molecule Type\n(Model: {MODEL_PATH})", fontsize=14)

    legend_handles = [
        plt.Line2D([0], [0], color=colors[g], lw=6, label=g)
        for g in groups
    ]
    plt.legend(handles=legend_handles, title="Molecule Type")

    plt.tight_layout()
    
    save_path = "tmpattn_fusion_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

    # (オプション) 数値での集計表示
    print("\n--- Mean Attention Weights ---")
    summary = df_plot.groupby(["Group", "Feature"])["Attention Weight"].mean().unstack()
    print(summary)

if __name__ == "__main__":
    main()