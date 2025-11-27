from typing import Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LR = 0.00020942790883545764  # default: 1e-3
WEIGHT_DECAY = 8e-5  # L2正則化
EPOCHS = 100
FP_BITS = 4096
# 【重要】モデルの想定に合わせる
INTENSITY_POWER = 0.5 
PATIENCE = 15  # early stopping の猶予エポック数

# --- 1. Datasetクラス (修正版) ---
class MassSpecDataset(Dataset):
    def __init__(self, dataframe, mode='train', intensity_power=0.5, flag_column: Optional[str] = None):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        self.intensity_power = intensity_power  # 保存
        self.flag_column = flag_column

        self.fp_radius = 2
        self.fp_bits = FP_BITS
        self.mfgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fp_radius, 
            fpSize=self.fp_bits,
            includeChirality=True 
        )

        # 事前計算してキャッシュ
        features_list = []
        mw_list = []
        spectrum_list = []
        flag_list = [] if (flag_column and flag_column in self.df.columns) else None

        for _, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            mw = Descriptors.ExactMolWt(mol)
            fp_arr = self.mfgen.GetCountFingerprintAsNumPy(mol).astype(np.float32)

            raw_spectrum = np.array(row['spectrum'], dtype=np.float32)
            if self.intensity_power != 1.0:
                spectrum = np.power(raw_spectrum, self.intensity_power)
            else:
                spectrum = raw_spectrum

            features_list.append(torch.from_numpy(fp_arr))
            mw_list.append(torch.tensor([mw], dtype=torch.float32))
            spectrum_list.append(torch.from_numpy(spectrum))

            if flag_list is not None:
                flag_list.append(torch.tensor([row[flag_column]], dtype=torch.float32))

        self.features = torch.stack(features_list)
        self.mw = torch.stack(mw_list)
        self.spectra = torch.stack(spectrum_list)
        self.flags = torch.stack(flag_list) if flag_list is not None else None

        self.spec_len = self.spectra.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            'features': self.features[idx],
            'mw': self.mw[idx],
            'spectrum': self.spectra[idx],
        }
        if self.flags is not None:
            item['flag'] = self.flags[idx]
        return item
    
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
def run_training(
    n_pos: int,
    n_neg: int,
    seed: int = 42,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    flag_column: str = "has_F",
    compile_model: bool = False,
)-> dict:
    """Train model with fixed stratified Test/Val and downsampled Train.

    Returns dict with cosine similarities on the fixed test set.
    """

    # --- Step 0: load and basic checks ---
    df = pd.read_parquet(PARQUET_FILE)
    if flag_column not in df.columns:
        raise ValueError(f"flag_column '{flag_column}' not found in dataframe")

    # --- Step 1: stratified Test split ---
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=df[flag_column],
    )

    # --- Step 2: stratified Val split from remaining ---
    adj_val_ratio = val_ratio / (1.0 - test_ratio)
    train_full_df, val_df = train_test_split(
        train_val_df,
        test_size=adj_val_ratio,
        random_state=seed,
        stratify=train_val_df[flag_column],
    )

    # --- Step 3: downsample Train_Full to requested n_pos/n_neg ---
    flag_series = train_full_df[flag_column].astype(bool)
    pool_pos = train_full_df[flag_series]
    pool_neg = train_full_df[~flag_series]

    if n_pos > len(pool_pos):
        raise ValueError(f"Requested n_pos={n_pos}, but only {len(pool_pos)} positives available in train pool")
    if n_neg > len(pool_neg):
        raise ValueError(f"Requested n_neg={n_neg}, but only {len(pool_neg)} negatives available in train pool")

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(len(pool_pos), size=n_pos, replace=False)
    neg_idx = rng.choice(len(pool_neg), size=n_neg, replace=False)

    train_df = pd.concat([
        pool_pos.iloc[pos_idx],
        pool_neg.iloc[neg_idx],
    ], axis=0)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def _count_pos_neg(df_sub: pd.DataFrame, name: str):
        pos = int((df_sub[flag_column] >= 0.5).sum())
        neg = int((df_sub[flag_column] < 0.5).sum())
        print(f"{name}: n_pos={pos}, n_neg={neg}, total={len(df_sub)}")

    print("Dataset sizes (by flag)")
    _count_pos_neg(train_df, "Train (sampled)")
    _count_pos_neg(val_df, "Val (fixed)")
    _count_pos_neg(test_df, "Test (fixed)")
    
    # Datasetに intensity_power を渡す
    train_dataset = MassSpecDataset(train_df, mode='train', intensity_power=INTENSITY_POWER, flag_column=flag_column)
    val_dataset = MassSpecDataset(val_df, mode='val', intensity_power=INTENSITY_POWER, flag_column=flag_column)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)
    
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

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed, continuing without compilation: {e}")
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) # 追加
    criterion = CosineLoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    
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

        # Early Stopping: 検証損失が更新されない場合は学習を打ち切る
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch}, best val loss: {best_val_loss:.6f})")
            break

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = EPOCHS

    # ベストモデルで評価・保存
    model.load_state_dict(best_state)
    torch.save(best_state, "mass_spec_model.pth")
    print(f"Best model (epoch {best_epoch}) saved to mass_spec_model.pth")

    # --- Test Evaluation (overall / pos / neg) ---
    model.eval()

    def evaluate_df(df: pd.DataFrame, label: str):
        if len(df) == 0:
            print(f"{label}: N/A (n=0)")
            return None, 0

        dataset = MassSpecDataset(df, mode=f"test-{label}", intensity_power=INTENSITY_POWER, flag_column=flag_column)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        sims = []
        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(config.device)
                mw = batch['mw'].to(config.device)
                target = batch['spectrum'].to(config.device)

                prediction, _ = model(features, mw)
                batch_sims = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
                sims.extend(batch_sims.detach().cpu().numpy().tolist())

        mean_sim = float(np.mean(sims)) if len(sims) > 0 else None
        if mean_sim is None:
            print(f"{label}: N/A (n=0)")
        else:
            print(f"{label}: {mean_sim:.4f} | n={len(sims)}")
        return mean_sim, len(sims)

    print("--- Test Set Cosine Similarity ---")
    overall_sim, overall_n = evaluate_df(test_df, "Overall")
    pos_df = test_df[test_df[flag_column] >= 0.5]
    neg_df = test_df[test_df[flag_column] < 0.5]
    pos_sim, pos_n = evaluate_df(pos_df, f"Pos ({flag_column}=1)")
    neg_sim, neg_n = evaluate_df(neg_df, f"Neg ({flag_column}=0)")

    results = {
        "overall": overall_sim,
        "pos": pos_sim,
        "neg": neg_sim,
        "n_overall": overall_n,
        "n_pos": pos_n,
        "n_neg": neg_n,
        "best_epoch": best_epoch,
    }

    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss (1 - Cosine Similarity)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig("tmp/lc/training_loss.png")
    plt.show()

    return results

if __name__ == "__main__":
    run_training(
        n_pos=100,
        n_neg=4900,
        seed=42,
        test_ratio=0.15,
        val_ratio=0.15,
        flag_column="has_F",
        compile_model=True,
    )
