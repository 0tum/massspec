import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# models.py から必要なクラスをインポート
from models import build_model, ModelConfig

# --- 設定 ---
PARQUET_FILE = "data/processed/MoNA.parquet"
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 50

# --- 1. Datasetクラス (変更なし) ---
class MassSpecDataset(Dataset):
    def __init__(self, dataframe, mode='train'):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode

        first_spec = self.df.iloc[0]['spectrum']
        self.spec_len = len(first_spec)

        print(f"[{mode}] Loaded {len(self.df)} spectra.")
        if mode == 'train': # ログがうるさくならないようにtrain時のみ表示
            print(f"   - Feature: spectrum length = {self.spec_len}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spectrum = row['spectrum'].astype(np.float32)
        nonzero_indices = np.nonzero(spectrum)[0]
        approx_mw = float(nonzero_indices.max()) if len(nonzero_indices) > 0 else 0.0

        return {
            'features': torch.tensor(spectrum, dtype=torch.float32),
            'mw': torch.tensor([approx_mw], dtype=torch.float32),
            'spectrum': torch.tensor(spectrum, dtype=torch.float32)
        }

# --- 2. 損失関数 (変更なし) ---
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, y_pred, y_true):
        score = self.cosine(y_pred, y_true)
        return torch.mean(1.0 - score)

# --- 3. 学習ループ (修正版) ---
def run_training():
    print("Loading data...")
    df = pd.read_parquet(PARQUET_FILE)
    
    # 【修正1】データの3分割 (Train: 80%, Val: 10%, Test: 10%)
    # まず全体からTest(10%)を切り出す
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    # 残りのTrain+ValからVal(元の全体の10%になるよう、約11%を指定)を切り出す
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)
    
    # Dataset作成
    train_dataset = MassSpecDataset(train_df, mode='train')
    val_dataset = MassSpecDataset(val_df, mode='val')
    test_dataset = MassSpecDataset(test_df, mode='test') # 【修正2】Test Dataset追加
    
    # DataLoader作成
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 【修正2】Test Loader追加
    
    # --- Model Config設定 ---
    total_input_dim = train_dataset.spec_len
    
    config = ModelConfig(
        fp_length=total_input_dim,
        max_mass_spec_peak_loc=train_dataset.spec_len,
        hidden_units=2000,
        num_hidden_layers=3,
        dropout_rate=0.2,
        bidirectional_prediction=True,
        gate_bidirectional_predictions=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print(f"Initializing model on {config.device}...")
    model = build_model("mlp", config=config)
    model.to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = CosineLoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("\nStart Training ...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(config.device)
            mw = batch['mw'].to(config.device)
            target = batch['spectrum'].to(config.device)
            
            optimizer.zero_grad()
            prediction, _ = model(features, mw)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config.device)
                mw = batch['mw'].to(config.device)
                target = batch['spectrum'].to(config.device)
                prediction, _ = model(features, mw)
                loss = criterion(prediction, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Sim: {1-avg_train_loss:.4f} | "
              f"Val Sim: {1-avg_val_loss:.4f}")

    # モデル保存
    save_path = "mass_spec_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 学習曲線の表示
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss (1 - Cosine Similarity)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # --- 【修正3】Testデータでの最終評価 ---
    print("\n--- Final Evaluation on Test Set ---")
    
    # (オプション) 学習後のモデル状態を確実にロードする場合
    # model.load_state_dict(torch.load(save_path))
    
    model.eval()
    test_loss = 0
    test_sim_accum = 0 # サンプルごとの厳密な平均を出したい場合用
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(config.device)
            mw = batch['mw'].to(config.device)
            target = batch['spectrum'].to(config.device)
            
            prediction, _ = model(features, mw)
            loss = criterion(prediction, target)
            test_loss += loss.item()
            
            # バッチ平均ではなく、より直感的な類似度を表示用
            # CosineLossは平均(1-sim)を返しているので、ここでの Sim = 1 - Loss
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_sim = 1.0 - avg_test_loss
    
    print(f"Test Loss (1-CosSim): {avg_test_loss:.4f}")
    print(f"Test Similarity     : {avg_test_sim:.4f}")
    print("------------------------------------")

if __name__ == "__main__":
    run_training()