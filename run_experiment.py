import pandas as pd
from tqdm import tqdm


from train import run_training


# Experiment settings
FLAG_COLUMN = "has_F"
VAL_METRIC = "val_loss"  # {"val_loss", "val_pos_loss"}
USED_FEATURES = "ecfp+bert+rdkit2d+rdkit3d+flag"  # ecfp, bert, rdkit2d, rdkit3d, flag
ECFP_BITS = 1024
SEEDS = [0, 1, 2, 3, 4]
POS_COUNTS = [10, 100, 120]
TOTAL_SAMPLES = 5000
OUTPUT_CSV = f"tmp/scaling/scaling-{FLAG_COLUMN}-{USED_FEATURES}_{ECFP_BITS}.csv"
# OUTPUT_FIG = "scaling_error_bars.png"

DOMAIN_SPLIT = False  # Falseならtrain/val/testをドメイン分割で行う
TRAIN_NUM = [500, 1000, 2000, 5000, 7000]
VAL_NUM = 500     # DOMAIN_SPLIT=Falseのときのvalサイズ
TEST_NUM = 1000   # DOMAIN_SPLIT=Falseのときのtestサイズ


# hparams dict
hparams_vanilla = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3544,
    "epochs": 100,
    "patience": 10,
}

hparams = hparams_vanilla


def main():
    records = []
    if DOMAIN_SPLIT:
        for n_pos in tqdm(POS_COUNTS, desc="n_pos sweep"):
            n_neg = TOTAL_SAMPLES - n_pos
            if n_neg <= 0:
                print(f"Skip n_pos={n_pos}: TOTAL_SAMPLES={TOTAL_SAMPLES} not enough for neg")
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
                        feature_type=USED_FEATURES,  
                        ecfp_bits=ECFP_BITS,
                        )
                except ValueError as e:
                    # 例: プールに十分なデータがない場合
                    print(f"Skip n_pos={n_pos}, seed={seed}: {e}")
                    continue

                records.append(
                    {
                        "n_pos_req": n_pos,
                        "n_neg_req": n_neg,
                        "seed": seed,
                        "pos_acc": res.get("pos"),
                        "neg_acc": res.get("neg"),
                        "overall_acc": res.get("overall"),
                        "best_epoch": res.get("best_epoch"),
                    }
                )
    else:
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
                        feature_type=USED_FEATURES,  
                        ecfp_bits=ECFP_BITS,
                    )
                except ValueError as e:
                    print(f"Skip n_train={n_train}, seed={seed}: {e}")
                    continue

                records.append(
                    {
                        "n_train": n_train,
                        "n_val": VAL_NUM,
                        "n_test": TEST_NUM,
                        "seed": seed,
                        "pos_acc": res.get("pos"),
                        "neg_acc": res.get("neg"),
                        "overall_acc": res.get("overall"),
                        "best_epoch": res.get("best_epoch"),
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV} (rows={len(df)})")

    if len(df) == 0:
        print("No data to plot")
        return



if __name__ == "__main__":
    main()
