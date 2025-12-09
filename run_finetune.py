import pandas as pd
from pathlib import Path

from train import run_training

# Experiment settings
IS_FREEZE = True
if IS_FREEZE:
    file_suffix = "freezed"
else:
    file_suffix = "unfreezed"

SEEDS = [0, 1, 2]
POS_COUNTS = [100, 120]
PRETRAIN_SAMPLES = {"n_pos": 0, "n_neg": 5000}
FT_LR = 2e-5
OUTPUT_CSV = f"output/csv/finetune_results_{file_suffix}.csv"





def main():
    Path("output/csv").mkdir(parents=True, exist_ok=True)
    Path("model/finetune").mkdir(parents=True, exist_ok=True)

    records = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        pretrained_path = f"model/finetune/pretrained_seed_{seed}.pth"

        print(f"[Seed {seed}] Phase 1: Pre-training (n_pos={PRETRAIN_SAMPLES['n_pos']}, n_neg={PRETRAIN_SAMPLES['n_neg']})")
        run_training(
            n_pos=PRETRAIN_SAMPLES["n_pos"],
            n_neg=PRETRAIN_SAMPLES["n_neg"],
            seed=seed,
            model_save_path=pretrained_path,
        )

        print(f"[Seed {seed}] Phase 2: Fine-tuning sweep")
        for n_pos in POS_COUNTS:
            try:
                res = run_training(
                    n_pos=n_pos,
                    n_neg=0,
                    seed=seed,
                    pretrained_path=pretrained_path,
                    learning_rate=FT_LR,
                    early_stopping_metric="val_pos_loss",
                    model_save_path=f"model/finetune/finetune_seed_{seed}_pos_{n_pos}_{file_suffix}.pth",
                    freeze=IS_FREEZE,
                )
            except ValueError as e:
                print(f"Skip seed={seed}, n_pos={n_pos}: {e}")
                continue

            records.append(
                {
                    "seed": seed,
                    "n_pos_req": n_pos,
                    "pos_acc": res.get("pos"),
                    "neg_acc": res.get("neg"),
                    "overall_acc": res.get("overall"),
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV} (rows={len(df)})")


if __name__ == "__main__":
    main()
