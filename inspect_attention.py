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

DATA_FILE = "data/processed/MoNA_features.npz" # データファイルのパス
FEATURE_TYPE = "ecfp+bert+rdkit2d" # 学習時と同じ特徴量設定
ECFP_BITS = 1024
FLAG_COLUMN = "has_F"
BATCH_SIZE = 128
SEEDS = 0  # データ分割の再現用
OUTPUT_CSV = "tmp/attention_fusion/attn_fusion_weights_3features.csv"
# データ分割設定（学習時に合わせて切り替える）
DOMAIN_SPLIT = True
# domain_split=False の場合に使うサイズ指定
TRAIN_SIZE = 7000
VAL_SIZE = 500
TEST_SIZE = 1000
# domain_split=True で n_pos/n_neg サンプリングを使う場合に指定（未指定なら元データの比率）
N_POS = [0, 1, 2, 5, 10, 20, 50, 100, 120, 170]

# 学習済みモデルのパス一覧（複数指定可）
MODEL_PATHS = []
for n_pos in N_POS:
    model_path = f"model/attn_fusion_model_f{n_pos}_3features.pth"
    MODEL_PATHS.append(model_path)

def main():
    # 1. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return

    component_order = FEATURE_TYPE.split('+')
    scaling_cols = []
    if "rdkit2d" in component_order:
        scaling_cols.append("rdkit_2d")
    if "rdkit3d" in component_order:
        scaling_cols.append("rdkit_3d")

    results_rows = []
    feature_key_map = {
        "bert": "bert",
        "ecfp": "ecfp",
        "rdkit2d": "rdkit2d",
    }

    for n_pos, MODEL_PATH in zip(N_POS, MODEL_PATHS):
        print("\n============================")
        print(f"Preparing data for n_pos={n_pos} ...")
        print("Loading and splitting dataset...")
        train_df, _, test_df = split_dataset(
            seed=SEEDS,
            flag_column=FLAG_COLUMN,
            domain_split=DOMAIN_SPLIT,
            n_pos=n_pos,
            n_neg=7000 - n_pos,
            data_file=DATA_FILE,
            train_size=TRAIN_SIZE,
            val_size=VAL_SIZE,
            test_size=TEST_SIZE,
        )

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

        feature_names = list(ordered_shapes.keys())

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
            continue

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

        # 6. 可視化 (Seaborn Boxplot)
        # DataFrameに整形: [Weight, FeatureName, Has_F]
        rows = []
        for i in range(weights_tensor.shape[0]):
            has_f = "F-containing" if flags_tensor[i] >= 0.5 else "Non-F"
            for j, feat_name in enumerate(feature_names):
                rows.append({
                    "Attention Weight": weights_tensor[i, j],
                    "Feature": feat_name,
                    "Group": has_f
                })

        df_plot = pd.DataFrame(rows)

        print("\n--- Visualization ---")
        plt.figure(figsize=(12, 6))

        groups = ["Non-F", "F-containing"]
        colors = {"Non-F": "skyblue", "F-containing": "salmon"}

        box_data = []
        positions = []
        for idx, feat in enumerate(feature_names):
            for j, grp in enumerate(groups):
                vals = df_plot[(df_plot["Feature"] == feat) & (df_plot["Group"] == grp)]["Attention Weight"].values
                box_data.append(vals)
                positions.append(idx * 3 + j)

        # # (オプション) 数値での集計表示
        print(f"loaded weight: {MODEL_PATH}")
        print("\n--- Mean Attention Weights ---")
        summary_mean = df_plot.groupby(["Group", "Feature"])["Attention Weight"].mean().unstack()
        summary_std = df_plot.groupby(["Group", "Feature"])["Attention Weight"].std().unstack()
        print(summary_mean)

        for group_label in summary_mean.index:
            split_name = "f-containing" if group_label.lower().startswith("f") else "non-f"
            row = {"n_pos": n_pos, "split": split_name}
            for feat_key, out_key in feature_key_map.items():
                mean_val = summary_mean.at[group_label, feat_key] if feat_key in summary_mean.columns else np.nan
                std_val = summary_std.at[group_label, feat_key] if feat_key in summary_std.columns else np.nan
                row[f"{out_key}_mean"] = mean_val
                row[f"{out_key}_std"] = std_val
            results_rows.append(row)

    if results_rows:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_out = pd.DataFrame(results_rows)
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved summary to {OUTPUT_CSV}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main()
