import pandas as pd
from tqdm import tqdm


from train import run_training


# Experiment settings
FLAG_COLUMN = "has_F"
TRAIN_MODE = "vanilla"  # "vanilla" or "class_balanced_loss"
VAL_METRIC = "val_loss"  # {"val_loss", "val_pos_loss"}
# CB_BETA = 0.999
FLAG_BOOST = 1.0
SEEDS = [0, 1, 2, 3, 4]
POS_COUNTS = [10, 100, 120]
TOTAL_SAMPLES = 5000
OUTPUT_CSV = f"tmp/{FLAG_COLUMN}-{TRAIN_MODE}-scaling_ecfp+bert+flag.csv"
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

hparams_class_balanced_loss = {
    "learning_rate": 0.00018579356614638177,
    "weight_decay": 0.00010558384297715928,
    "dropout_rate": 0.34145869213527447,
    "epochs": 118,
    "patience": 30,
    "cb_beta": 0.990110725960859,
    "flag_boost": 1.768852367763028,
}

if TRAIN_MODE == "vanilla":
    hparams = hparams_vanilla
else:
    hparams = hparams_class_balanced_loss


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
                        train_mode=TRAIN_MODE,
                        # cb_beta=hparams["cb_beta"],
                        # flag_boost=hparams["flag_boost"],
                        # weighted_sampler=False,
                        learning_rate=hparams["learning_rate"],  
                        weight_decay=hparams["weight_decay"],  
                        dropout_rate=hparams["dropout_rate"], 
                        epochs=hparams["epochs"],
                        patience=hparams["patience"],
                        early_stopping_metric=VAL_METRIC, 
                        domain_split=True,
                        feature_type="ecfp+bert+flag",  # ecfpのみ
                        ecfp_bits=1024,
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
                        train_mode=TRAIN_MODE,
                        # cb_beta=hparams["cb_beta"],
                        # flag_boost=hparams["flag_boost"],
                        # weighted_sampler=False,
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
