import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from torch.optim.lr_scheduler import CosineAnnealingLR # 追加

from models import build_model, ModelConfig

# --- 設定 ---
PARQUET_FILE = "data/processed/MoNA.parquet"
BATCH_SIZE = 64
LR = 0.00020942790883545764  # 少し上げてもいいかもしれません default: 1e-3
WEIGHT_DECAY = 8e-5  # L2正則化
EPOCHS = 50
FP_BITS = 4096
# 【重要】モデルの想定に合わせる
INTENSITY_POWER = 0.5 

# --- 1. Datasetクラス (修正版) ---
class MassSpecDataset(Dataset):
    def __init__(self, dataframe, mode='train', intensity_power=0.5):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        self.intensity_power = intensity_power  # 保存

        self.fp_radius = 2
        self.fp_bits = FP_BITS
        self.mfgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fp_radius, 
            fpSize=self.fp_bits,
            includeChirality=True 
        )

        first_spec = self.df.iloc[0]['spectrum']
        self.spec_len = len(first_spec)

        print(f"[{mode}] Loaded {len(self.df)} spectra.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        # X: Features
        # 【修正1】Exact Mass (モノアイソトピック質量) を使用する
        # 双方向予測のアンカー位置を正確にするため必須
        mw = Descriptors.ExactMolWt(mol)
        # 【修正2】Count Fingerprint を使用する
        # GetFingerprintAsNumPy -> GetCountFingerprintAsNumPy
        # uint32などが返るため、float32に変換
        fp_arr = self.mfgen.GetCountFingerprintAsNumPy(mol).astype(np.float32)

        # Y: Spectrum
        raw_spectrum = row['spectrum'].astype(np.float32)
        
        # 【重要】強度のスケーリング (ルート変換)
        # 0に近い値の勾配消失を防ぎ、微小ピークを強調する
        if self.intensity_power != 1.0:
            spectrum = np.power(raw_spectrum, self.intensity_power)
        else:
            spectrum = raw_spectrum

        return {
            'features': torch.tensor(fp_arr, dtype=torch.float32),
            'mw': torch.tensor([mw], dtype=torch.float32),
            'spectrum': torch.tensor(spectrum, dtype=torch.float32)
        }
    
# --- 2. 損失関数 (Cosine Loss) ---
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, y_pred, y_true):
        # 学習はルート空間(変形された空間)同士で行うのが最も収束が良い
        score = self.cosine(y_pred, y_true)
        return torch.mean(1.0 - score)

# --- 3. 学習ループ ---
def run_training():
    print("Loading data...")
    df = pd.read_parquet(PARQUET_FILE)
    
    # 分割
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    # Datasetに intensity_power を渡す
    train_dataset = MassSpecDataset(train_df, mode='train', intensity_power=INTENSITY_POWER)
    val_dataset = MassSpecDataset(val_df, mode='val', intensity_power=INTENSITY_POWER)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Model Config ---
    total_input_dim = FP_BITS

    config = ModelConfig(
        fp_length=total_input_dim,
        max_mass_spec_peak_loc=train_dataset.spec_len,
        hidden_units=2000,
        num_hidden_layers=2,
        dropout_rate=0.25,
        bidirectional_prediction=True,
        gate_bidirectional_predictions=True,
        
        # Configにも記録しておく (models.pyの内部ロジックでは使われないが整合性のため)
        intensity_power=INTENSITY_POWER,
        
        # Loss指定: models.pyのforward出力は ReLU(raw) になる
        # CosineLossを使うならReLUで0以上にクリップされているのは好都合
        loss="normalized_generalized_mse", 
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print(f"Initializing model on {config.device} with Intensity Power {INTENSITY_POWER}...")
    model = build_model("mlp", config=config)
    model.to(config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) # 追加
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
            
            # models.py の forward は (prediction, logits) を返す
            # prediction は Config.loss != 'cross_entropy' なので ReLU が掛かった値
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Sim: {1-avg_train_loss:.4f} | Val Sim: {1-avg_val_loss:.4f}")
        scheduler.step()  # 追加

    torch.save(model.state_dict(), "mass_spec_model.pth")
    print("Model saved to mass_spec_model.pth")

    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss (1 - Cosine Similarity)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_training()
