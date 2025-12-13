import pandas as pd
from tqdm import tqdm
from train import run_training
import itertools
import os  # osモジュールを追加

# --- Experiment settings ---
FLAG_COLUMN = "has_F"
VAL_METRIC = "val_loss"
ECFP_BITS = 1024
SEEDS = [0]
DOMAIN_SPLIT = True  # Trueにするとdomain splitモードを使用

# スケーリング用データサイズ (domain_split=False のとき)
TRAIN_NUM = [500, 7000]
VAL_NUM = 500
TEST_NUM = 1000

# domain_split=Trueのときの設定
POS_COUNTS = [0, 1, 2, 5, 10, 20, 50, 100, 120, 170]
TOTAL_SAMPLES = 7000

# 結果保存先
OUTPUT_CSV = f"tmp/attn_fusion_scaling_has-F_3features.csv"
OUTPUT_COLUMNS = [
    "split_mode",
    "feature_type",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "n_pos_req",
    "n_neg_req",
    "pos_metric",
    "neg_metric",
    "overall_metric",
    "best_epoch",
]

# hparams dict
hparams_vanilla = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3544,
    "epochs": 500,
    "patience": 10,
}
hparams = hparams_vanilla

# 基本の特徴量リスト
ALL_FEATURES = ["ecfp", "rdkit2d", "rdkit3d", "bert", "flag"]

# 組み合わせの自動生成
# experiments = []
# for r in range(1, len(ALL_FEATURES) + 1):
#     for combo in itertools.combinations(ALL_FEATURES, r):
#         experiments.append(list(combo))

# print(f"Total experiment configurations: {len(experiments)}")

experiments =[
    # "ecfp+bert+flag",
    # "ecfp+rdkit2d+flag",
    # "ecfp+rdkit2d+rdkit3d",
    # "ecfp+rdkit2d+flag+bert",
    # "ecfp+rdkit2d+rdkit3d+bert+flag"
    'ecfp+bert+rdkit2d'
]


def _append_record(record: dict):
    """単一の実験結果をCSVに追記する。"""
    df_row = pd.DataFrame([record]).reindex(columns=OUTPUT_COLUMNS)
    if not os.path.isfile(OUTPUT_CSV):
        df_row.to_csv(OUTPUT_CSV, index=False, mode='w', header=True)
    else:
        df_row.to_csv(OUTPUT_CSV, index=False, mode='a', header=False)


def experiment_random_split():
    """domain_split=Falseでtrain/val/testサイズを指定して回す実験。"""
    for exp_features in experiments:
        # used_features_str = "+".join(exp_features)
        used_features_str = exp_features
        print(f"Running experiment (random split): {used_features_str}")
        
        for n_train in tqdm(TRAIN_NUM, desc="train size sweep"):
            for seed in tqdm(SEEDS, desc="seeds", leave=False):
                try:
                    res = run_training(
                        n_pos=None,
                        n_neg=None,
                        seed=seed,
                        flag_column=FLAG_COLUMN,
                        learning_rate=hparams["learning_rate"],  
                        weight_decay=hparams["weight_decay"],  
                        dropout_rate=hparams["dropout_rate"], 
                        epochs=hparams["epochs"],
                        patience=hparams["patience"],
                        early_stopping_metric=VAL_METRIC, 
                        domain_split=False,
                        train_size=n_train,
                        val_size=VAL_NUM,
                        test_size=TEST_NUM,
                        feature_type=used_features_str,
                        ecfp_bits=ECFP_BITS,
                    )
                except ValueError as e:
                    print(f"Skipped {used_features_str}, n={n_train}, seed={seed}: {e}")
                    continue

                record = {
                    "split_mode": "random",
                    "feature_type": used_features_str,
                    "n_train": n_train,
                    "n_val": VAL_NUM,
                    "n_test": TEST_NUM,
                    "seed": seed,
                    "n_pos_req": None,
                    "n_neg_req": None,
                    "pos_metric": res.get("pos"),
                    "neg_metric": res.get("neg"),
                    "overall_metric": res.get("overall"),
                    "best_epoch": res.get("best_epoch"),
                }
                
                _append_record(record)


def experiment_domain_split():
    """domain_split=Trueで、POS_COUNTS/TOTAL_SAMPLESを使って回す実験。"""
    for exp_features in experiments:
        # used_features_str = "+".join(exp_features)
        used_features_str = exp_features
        print(f"Running experiment (domain split): {used_features_str}")

        for n_pos in tqdm(POS_COUNTS, desc="n_pos sweep"):
            n_neg = TOTAL_SAMPLES - n_pos
            if n_neg <= 0:
                print(f"Skipped {used_features_str}, n_pos={n_pos}: TOTAL_SAMPLES={TOTAL_SAMPLES} too small")
                continue

            for seed in tqdm(SEEDS, desc="seeds", leave=False):
                try:
                    res = run_training(
                        n_pos=n_pos,
                        n_neg=n_neg,
                        seed=seed,
                        flag_column=FLAG_COLUMN,
                        learning_rate=hparams["learning_rate"],  
                        weight_decay=hparams["weight_decay"],  
                        dropout_rate=hparams["dropout_rate"], 
                        epochs=hparams["epochs"],
                        patience=hparams["patience"],
                        early_stopping_metric=VAL_METRIC, 
                        domain_split=True,
                        train_size=None,
                        val_size=None,
                        test_size=None,
                        feature_type=used_features_str,
                        ecfp_bits=ECFP_BITS,
                        model_type="attention_mlp",
                    )
                except ValueError as e:
                    print(f"Skipped {used_features_str}, n_pos={n_pos}, seed={seed}: {e}")
                    continue

                record = {
                    "split_mode": "domain",
                    "feature_type": used_features_str,
                    "n_train": None,
                    "n_val": None,
                    "n_test": None,
                    "seed": seed,
                    "n_pos_req": n_pos,
                    "n_neg_req": n_neg,
                    "pos_metric": res.get("pos"),
                    "neg_metric": res.get("neg"),
                    "overall_metric": res.get("overall"),
                    "best_epoch": res.get("best_epoch"),
                }

                _append_record(record)


def main():
    # ディレクトリの作成をループの前に行う
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # もし既存のファイルをリセット（削除）したい場合はここで os.remove(OUTPUT_CSV) を行いますが、
    # 安全のため、ここでは手動削除か「追記」を前提とします。

    if DOMAIN_SPLIT:
        experiment_domain_split()
    else:
        experiment_random_split()

    print(f"All experiments completed. Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
