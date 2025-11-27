import pandas as pd
import matplotlib.pyplot as plt
from train import run_training

# --- 実験設定 ---
# 実験したい n_pos のリスト (0から始めて、在庫限界の120付近まで対数的に増やすイメージ)
POS_COUNTS = [0, 1, 2, 5, 10, 20, 50, 80, 100, 120]
TOTAL_SAMPLES = 5000 # 学習データの総数 (n_pos + n_neg)

results = []

print(f"=== Starting Experiment (Total Fixed at {TOTAL_SAMPLES}) ===")

for n_pos in POS_COUNTS:
    n_neg = TOTAL_SAMPLES - n_pos
    
    # 万が一 n_neg がマイナスになったらスキップ
    if n_neg < 0:
        print(f"Skipping n_pos={n_pos} (Total exceeded)")
        continue

    print(f"\n>>> Running: n_pos={n_pos}, n_neg={n_neg} (Ratio: {n_pos/TOTAL_SAMPLES:.2%})")
    
    try:
        # train.py の関数を呼び出し
        res = run_training(
            n_pos=n_pos,
            n_neg=n_neg,
            seed=42,           # シード固定＝Test/Val固定
            test_ratio=0.15,
            val_ratio=0.15,
            flag_column="has_F"
        )
        
        # 結果リストに追加
        res['n_pos_req'] = n_pos # リクエスト値を記録
        results.append(res)
        
    except ValueError as e:
        print(f"Error at n_pos={n_pos}: {e}")
        # 在庫不足(Pool枯渇)のエラーが出たら、それ以降も無理なのでループを抜ける
        if "available" in str(e):
            print("Stopping experiment due to data exhaustion.")
            break

# --- 結果の保存とプロット ---
df_res = pd.DataFrame(results)
df_res.to_csv("output/experiment_results_fixed_total.csv", index=False)
print("\nResults saved to output/experiment_results_fixed_total.csv")
