import numpy as np

# 読み込み
data = np.load("data/processed/MoNA_features.npz")

# カラム一覧
for key in data.files:
    print(f"{key}, Shape: {data[key].shape}")
