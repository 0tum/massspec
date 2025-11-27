import pandas as pd
from tqdm import tqdm


from train import run_training


# Experiment settings
SEEDS = [0, 1, 2, 3, 4]
POS_COUNTS = [0, 1, 2, 5, 10, 20, 50, 80, 100, 120]
TOTAL_SAMPLES = 5000
OUTPUT_CSV = "output/experiment_results_multi_seed.csv"
# OUTPUT_FIG = "scaling_error_bars.png"


def main():
    records = []

    for n_pos in tqdm(POS_COUNTS, desc="n_pos sweep"):
        n_neg = TOTAL_SAMPLES - n_pos
        if n_neg <= 0:
            print(f"Skip n_pos={n_pos}: TOTAL_SAMPLES={TOTAL_SAMPLES} not enough for neg")
            continue

        for seed in tqdm(SEEDS, desc="seeds", leave=False):
            try:
                res = run_training(n_pos=n_pos, n_neg=n_neg, seed=seed)
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

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV} (rows={len(df)})")

    if len(df) == 0:
        print("No data to plot")
        return

    # sns.set(style="whitegrid")
    # plt.figure(figsize=(9, 5))

    # sns.lineplot(data=df, x="n_pos_req", y="pos_acc", errorbar="sd", label="Pos Accuracy", color="red")
    # sns.lineplot(data=df, x="n_pos_req", y="neg_acc", errorbar="sd", label="Neg Accuracy", color="blue")
    # sns.lineplot(data=df, x="n_pos_req", y="overall_acc", errorbar="sd", label="Overall Accuracy", color="gray")

    # plt.title("Scaling Law with Error Bars (5 seeds)")
    # plt.xlabel("n_pos_req")
    # plt.ylabel("Cosine Similarity")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(OUTPUT_FIG)
    # print(f"Saved plot to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
