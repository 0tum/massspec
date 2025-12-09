import os
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner

from train import run_training

# 固定設定（通常学習用）
N_POS = 100
N_NEG = 4900
SEED = 42
FLAG_COLUMN = "has_F"
TRAIN_MODE = "vanilla"  # "vanilla" or "class_balanced_loss"

STORAGE = "sqlite:///output/optuna_study.db"
STUDY_NAME = "massspec_cb_optuna"
OUTPUT_CSV = "output/csv/optuna_class-balanced.csv"


def objective(trial: optuna.Trial) -> float:
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_units = trial.suggest_categorical("hidden_units", [512, 1024, 2000])
    ecfp_bits = trial.suggest_categorical("ecfp_bits", [512, 1024, 2048, 4096])
    # epochs = trial.suggest_int("epochs", 20, 120)
    # patience = trial.suggest_int("patience", 3, 30)
    # cb_beta = trial.suggest_float("cb_beta", 0.99, 0.9999, log=True)
    # flag_boost = trial.suggest_float("flag_boost", 1.0, 3.0)

    res = run_training(
        n_pos=N_POS,
        n_neg=N_NEG,
        seed=SEED,
        flag_column=FLAG_COLUMN,
        train_mode=TRAIN_MODE,
        weighted_sampler=True,
        compile_model=False,
        pretrained_path=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout_rate=dropout_rate,
        hidden_units=hidden_units,
        ecfp_bits=ecfp_bits,
        epochs=100,
        patience=10,
        model_save_path=None,
        early_stopping_metric="val_loss",
        domain_split=False,
        train_size=7000,
        val_size=500,
        test_size=1000,
    )

    val_loss = res.get("val_loss")
    if val_loss is None:
        # 監視指標が取れなかった場合は Trial をスキップ
        raise optuna.TrialPruned("val_loss is None")
    return float(val_loss)


def main():
    Path("output").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    df = study.trials_dataframe()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved trials to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
