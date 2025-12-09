from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem # 追加
from rdkit.Chem import rdFingerprintGenerator # 追加

from torch.optim.lr_scheduler import CosineAnnealingLR # 追加

from data import DEFAULT_PARQUET_FILE, split_dataset
from features import ChemBERTaFeatureExtractor, get_hybrid_features
from models import build_model, ModelConfig

# --- 設定 ---
PARQUET_FILE = DEFAULT_PARQUET_FILE 
BATCH_SIZE = 64
INTENSITY_POWER = 0.5 

# # hparams dict
hparams_vanilla = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout_rate": 0.3544,
    "epochs": 100,
    "patience": 10,
    "cb_beta": None,
    "flag_boost": None,
}
hparams = hparams_vanilla

# --- Hard-coded run configuration (no CLI needed) ---
RUN_CONFIG = {
    # Split/seed settings
    "n_pos": 100,
    "n_neg": 4900,
    "seed": 0,
    "flag_column": "has_F",
    "domain_split": False,      # Falseで下のtrain/val/test_sizeが有効
    "train_size": 8000,         # 例: 4000
    "val_size": 500,           # 例: 500
    "test_size": 1000,          # 例: 500

    # Training mode & loss
    "train_mode": "vanilla", # "vanilla" or "class_balanced_loss"
    "cb_beta": hparams["cb_beta"],
    "flag_boost": hparams["flag_boost"],
    "early_stopping_metric": "val_loss",  # "val_loss", "val_pos_loss"

    # Optimization
    "learning_rate": hparams["learning_rate"],
    "weight_decay": hparams["weight_decay"],
    "dropout_rate": hparams["dropout_rate"],
    "epochs": hparams["epochs"],
    "patience": hparams["patience"],
    "weighted_sampler": False,
    "compile_model": False,

    # Files / model options
    "model_save_path": "model/mass_spec_model.pth",
    "pretrained_path": None,
    "freeze": False,

    # Features
    "feature_type": "ecfp+bert+flag",
    "structural_flag_columns": None,
    "feature_extractor": None,
    "inspect_input": False,
}

# --- 1. Datasetクラス (修正版) ---
class MassSpecDataset(Dataset):
    def __init__(
        self,
        dataframe,
        mode: str = "train",
        intensity_power: float = 0.5,
        flag_column: Optional[str] = None,
        structural_flag_columns: Optional[list] = None,
        feature_extractor: Optional[ChemBERTaFeatureExtractor] = None,
        feature_device: Optional[torch.device] = None,
        feature_type: str = "ecfp+bert+flag",  # {"ecfp", "ecfp+bert+flag"}
        ecfp_radius: int = 2,
        ecfp_bits: int = 4096,     #fearure_typeの変更のとき注意
    ):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        self.intensity_power = intensity_power
        self.flag_column = flag_column
        self.structural_flag_columns = structural_flag_columns or []
        valid_feature_types = {"ecfp", "ecfp+bert+flag"}
        if feature_type not in valid_feature_types:
            raise ValueError(f"feature_type must be one of {valid_feature_types}, got '{feature_type}'")
        self.feature_type = feature_type
        self.use_bert = feature_type == "ecfp+bert+flag"
        self.feature_extractor = None
        if self.use_bert:
            self.feature_extractor = feature_extractor or ChemBERTaFeatureExtractor(device=feature_device)

        features_list = []
        mw_list = []
        spectrum_list = []
        flag_list = [] if (flag_column and flag_column in self.df.columns) else None

        # 【追加】Generatorをここで作成（ループの外で作るのが推奨です）
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=ecfp_radius, fpSize=ecfp_bits
        )

        for _, row in self.df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue # あるいはエラーハンドリング
            # 【修正】正規化されたSMILESを取得
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            mw = Descriptors.ExactMolWt(mol)

            raw_spectrum = np.array(row["spectrum"], dtype=np.float32)
            if self.intensity_power != 1.0:
                spectrum = np.power(raw_spectrum, self.intensity_power)
            else:
                spectrum = raw_spectrum

            feature_parts = []

            if self.use_bert:
                structural_flags = self._collect_structural_flags(row)
                hybrid_features = get_hybrid_features(
                    smiles=canonical_smiles,
                    flags=structural_flags,
                    extractor=self.feature_extractor,
                    device=self.feature_extractor.device,
                    as_numpy=True,
                )
                feature_parts.append(hybrid_features)

            # 2. ECFPの計算 (追加)
            ecfp_array = self.morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            feature_parts.append(ecfp_array)

            if len(feature_parts) == 1:
                final_features = feature_parts[0]
            else:
                final_features = np.concatenate(feature_parts, axis=-1)

            # Tensor化してリストに追加
            features_list.append(torch.tensor(final_features, dtype=torch.float32))
            mw_list.append(torch.tensor([mw], dtype=torch.float32))
            spectrum_list.append(torch.from_numpy(spectrum))

            if flag_list is not None:
                flag_list.append(torch.tensor([row[flag_column]], dtype=torch.float32))

        self.features = torch.stack(features_list)
        self.mw = torch.stack(mw_list)
        self.spectra = torch.stack(spectrum_list)
        self.flags = torch.stack(flag_list) if flag_list is not None else None

        self.spec_len = self.spectra.shape[1]

    def _collect_structural_flags(self, row) -> np.ndarray:
        """Gather structural flags from a list-like column or specified columns."""
        if "structural_flags" in row and row["structural_flags"] is not None:
            flags = row["structural_flags"]
        elif self.structural_flag_columns:
            flags = [row[col] for col in self.structural_flag_columns]
        else:
            flags = []
        return np.asarray(flags, dtype=np.float32)

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
    val_ratio: float = 0.10,
    flag_column: str = "has_F",
    structural_flag_columns: Optional[List[str]] = None,
    train_mode: str = "vanilla",  # "vanilla" | "class_balanced_loss"
    cb_beta: float = 0.99,
    flag_boost: float = 1.0,
    weighted_sampler: bool = False,
    compile_model: bool = False,
    pretrained_path: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
    dropout_rate: Optional[float] = None,
    hidden_units: int = 2000,
    num_hidden_layers: int = 2,
    ecfp_bits: int = 1024,
    feature_type: str = "ecfp+bert+flag",  # {"ecfp", "ecfp+bert+flag"}
    epochs: int = None,
    patience: int = None,
    model_save_path: Optional[str] = "model/mass_spec_model.pth",
    early_stopping_metric: str = "val_loss",  # "val_loss", "val_pos_loss"
    freeze: bool = False,
    feature_extractor: Optional[ChemBERTaFeatureExtractor] = None,
    inspect_input: bool = False,
    domain_split: bool = False,
    train_size: Optional[int] = 7000,
    val_size: Optional[int] = 500,
    test_size: Optional[int] = 1000,
)-> dict:
    """Train model with either a stratified flag-aware split or a simple size-based split.

    - If domain_split is False *and* any of train_size/val_size/test_size is given,
      the dataset is shuffled and split by the requested molecule counts (n_pos/n_neg are ignored).
    - Otherwise, perform the original flag-aware stratified split and downsample train by n_pos/n_neg.

    Returns dict with cosine similarities on the fixed test set.
    """

    if train_mode not in ("vanilla", "class_balanced_loss"):
        raise ValueError(f"Unsupported train_mode '{train_mode}'. Use 'vanilla' or 'class_balanced_loss'.")

    # --- Step 0: load and split data ---
    cb_beta_val = cb_beta if cb_beta is not None else 0.99
    flag_boost_val = flag_boost if flag_boost is not None else 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df, val_df, test_df = split_dataset(
        seed=seed,
        flag_column=flag_column,
        domain_split=domain_split,
        n_pos=n_pos,
        n_neg=n_neg,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        parquet_file=PARQUET_FILE,
    )

    # クラスバランス重みの計算（train_df上で実施）
    use_cb_loss = train_mode == "class_balanced_loss"
    cb_weights = None
    pos_count = int((train_df[flag_column] >= 0.5).sum())
    neg_count = int((train_df[flag_column] < 0.5).sum())
    if use_cb_loss:

        def _cb_weight(count: int, beta: float) -> float:
            if count <= 0:
                return 0.0
            return (1.0 - beta) / (1.0 - (beta ** count))

        cb_weights = {
            "pos": _cb_weight(pos_count, cb_beta_val),
            "neg": _cb_weight(neg_count, cb_beta_val),
            "pos_count": pos_count,
            "neg_count": neg_count,
        }
        print(f"[CB-Loss] beta={cb_beta_val}, flag_boost={flag_boost_val} | w_pos={cb_weights['pos']:.6f} (n={pos_count}), w_neg={cb_weights['neg']:.6f} (n={neg_count})")

    def _count_pos_neg(df_sub: pd.DataFrame, name: str):
        pos = int((df_sub[flag_column] >= 0.5).sum())
        neg = int((df_sub[flag_column] < 0.5).sum())
        print(f"{name}: n_pos={pos}, n_neg={neg}, total={len(df_sub)}")

    print("Dataset sizes (by flag)")
    _count_pos_neg(train_df, "Train (sampled)")
    _count_pos_neg(val_df, "Val (fixed)")
    _count_pos_neg(test_df, "Test (fixed)")

    flag_cols = structural_flag_columns or []
    if feature_type == "ecfp+bert+flag":
        feature_extractor = feature_extractor or ChemBERTaFeatureExtractor()
    else:
        feature_extractor = None
    
    # Datasetに intensity_power を渡す
    train_dataset = MassSpecDataset(
        train_df,
        mode="train",
        intensity_power=INTENSITY_POWER,
        flag_column=flag_column,
        structural_flag_columns=flag_cols,
        feature_extractor=feature_extractor,
        feature_type=feature_type,
        ecfp_bits=ecfp_bits,
    )
    val_dataset = MassSpecDataset(
        val_df,
        mode="val",
        intensity_power=INTENSITY_POWER,
        flag_column=flag_column,
        structural_flag_columns=flag_cols,
        feature_extractor=feature_extractor,
        feature_type=feature_type,
        ecfp_bits=ecfp_bits,
    )

    pin_memory = torch.cuda.is_available()
    # WeightedRandomSamplerで各バッチに正例が含まれやすくする（pos/negで同じ確率にする）
    use_sampler = weighted_sampler and pos_count > 0 and neg_count > 0
    print(f"Using weighted sampler: {use_sampler}")
    if use_sampler:
        flags_tensor = train_dataset.flags.squeeze(-1)
        pos_mask = flags_tensor >= 0.5
        pos_w = 1.0 / pos_count
        neg_w = 1.0 / neg_count
        sample_weights = torch.where(pos_mask, torch.full_like(flags_tensor, pos_w), torch.full_like(flags_tensor, neg_w)).double()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        print("Using WeightedRandomSampler for train loader (balanced pos/neg sampling).")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory)

    if inspect_input:
        batch = next(iter(train_loader))
        features = batch["features"]
        print("# 入力の分布を確認")
        if feature_type == "ecfp+bert+flag":
            bert_dim = 768
            bert_part = features[:, :bert_dim]
            remaining = features[:, bert_dim:]
            flag_dim = max(0, remaining.shape[1] - ecfp_bits)
            flag_part = remaining[:, :flag_dim] if flag_dim > 0 else None
            ecfp_part = remaining[:, flag_dim:]

            print(f"BERT Mean: {bert_part.mean():.4f}, Std: {bert_part.std():.4f}")
            print(f"BERT Min:  {bert_part.min():.4f},  Max: {bert_part.max():.4f}")
            print("-" * 30)
            if flag_part is not None and flag_part.numel() > 0:
                print(f"Flag Mean: {flag_part.mean():.4f}, Std: {flag_part.std():.4f}")
                print(f"Flag Min:  {flag_part.min():.4f},  Max: {flag_part.max():.4f}")
                print("-" * 30)
            print(f"ECFP Mean: {ecfp_part.mean():.4f}, Std: {ecfp_part.std():.4f}")
            print(f"ECFP Min:  {ecfp_part.min():.4f},  Max: {ecfp_part.max():.4f}")
        else:
            ecfp_part = features
            print(f"ECFP Mean: {ecfp_part.mean():.4f}, Std: {ecfp_part.std():.4f}")
            print(f"ECFP Min:  {ecfp_part.min():.4f},  Max: {ecfp_part.max():.4f}")

        return {
            "overall": None,
            "pos": None,
            "neg": None,
            "n_overall": None,
            "n_pos": None,
            "n_neg": None,
            "best_epoch": None,
            "val_loss": None,
            "val_pos_loss": None,
            "val_neg_loss": None,
        }
    
    # --- Model Config ---
    total_input_dim = train_dataset.features.shape[1]

    config = ModelConfig(
        fp_length=total_input_dim,
        max_mass_spec_peak_loc=train_dataset.spec_len,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate if dropout_rate is not None else 0.25,
        bidirectional_prediction=True,
        gate_bidirectional_predictions=True,
        
        # Configにも記録しておく (models.pyの内部ロジックでは使われないが整合性のため)
        intensity_power=INTENSITY_POWER,
        feature_type=feature_type,
        
        # Loss指定: models.pyのforward出力は ReLU(raw) になる
        # CosineLossを使うならReLUで0以上にクリップされているのは好都合
        loss="normalized_generalized_mse", 
        device=device
    )
    
    print(f"Initializing model on {config.device} with Intensity Power {INTENSITY_POWER}...")
    print(f"Selected Training Mode: {train_mode}")
    model = build_model("mlp", config=config)
    model.to(config.device)

    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location=config.device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")

    if freeze:
        print("Freezing backbone parameters...")
        freeze_modules = []
        # ここで指定した名前がモデルに存在するかを確認
        target_attrs = ["input_layer", "residual_blocks", "final_bn"]
        
        frozen_count = 0
        for attr in target_attrs:
            if hasattr(model, attr):
                module = getattr(model, attr)
                freeze_modules.append(module)
                # 実際にフラグを折る
                for p in module.parameters():
                    p.requires_grad = False
                print(f"  - Frozen layer: {attr}")
                frozen_count += 1
            else:
                print(f"  - Warning: Layer '{attr}' not found in model. Skipped.")
        
        if frozen_count == 0:
            print("  - WARNING: No modules were frozen! Check model attribute names.")

    lr = learning_rate if learning_rate is not None else LR
    
    # 【追加】実際に学習されるパラメータ名を表示して確認する
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    # print(f"Trainable parameters ({len(trainable_names)} tensors):")
    # # 数が多い場合は先頭だけ表示するなど調整
    # for name in trainable_names[:5]: 
    #     print(f"  - {name}")
    # if len(trainable_names) > 5:
    #     print(f"  - ... and {len(trainable_names)-5} others.")

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Set freeze=False or unfreeze some layers.")
    
    # 定数の上書き処理
    lr = learning_rate if learning_rate is not None else LR
    # 【追加】
    wd = weight_decay if weight_decay is not None else WEIGHT_DECAY
    pat = patience if patience is not None else PATIENCE
    max_epochs = epochs if epochs is not None else EPOCHS

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=wd)
    
    # ... (後略)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6) # 追加
    
    history = {'train_loss': [], 'val_loss': [], 'val_pos_loss': [], 'val_neg_loss': []}
    best_score = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    best_metrics_at_best = {}
    
    print("\nStart Training ...")
    cb_w_pos = cb_w_neg = flag_boost_tensor = None
    if use_cb_loss and cb_weights is not None:
        cb_w_pos = torch.tensor(cb_weights["pos"], device=config.device)
        cb_w_neg = torch.tensor(cb_weights["neg"], device=config.device)
        flag_boost_tensor = torch.tensor(flag_boost_val, device=config.device)
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(config.device)
            mw = batch['mw'].to(config.device)
            target = batch['spectrum'].to(config.device)
            flags = batch['flag'].to(config.device).squeeze(-1)
            
            optimizer.zero_grad()
            
            # models.py の forward は (prediction, logits) を返す
            # prediction は Config.loss != 'cross_entropy' なので ReLU が掛かった値
            prediction, _ = model(features, mw)
            
            sims = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
            losses = 1.0 - sims  # per-sample loss

            if use_cb_loss and cb_w_pos is not None and cb_w_neg is not None:
                pos_mask = flags >= 0.5
                class_weights = torch.where(pos_mask, cb_w_pos, cb_w_neg)
                if flag_boost_val != 1.0:
                    flag_weights = torch.where(pos_mask, flag_boost_tensor, torch.ones_like(class_weights))
                    sample_weights = class_weights * flag_weights
                else:
                    sample_weights = class_weights
                weight_sum = torch.clamp(sample_weights.sum(), min=1e-12)
                loss = (losses * sample_weights).sum() / weight_sum
            else:
                loss = losses.mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_pos_loss_sum = 0.0
        val_pos_count = 0
        val_neg_loss_sum = 0.0
        val_neg_count = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(config.device)
                mw = batch['mw'].to(config.device)
                target = batch['spectrum'].to(config.device)
                
                prediction, _ = model(features, mw)
                sims = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
                losses = 1.0 - sims  # per-sample loss

                val_loss_sum += losses.sum().item()
                val_count += losses.numel()

                flags = batch['flag'].to(config.device).squeeze(-1)
                pos_mask = flags >= 0.5
                neg_mask = flags < 0.5

                if pos_mask.any():
                    val_pos_loss_sum += losses[pos_mask].sum().item()
                    val_pos_count += pos_mask.sum().item()
                if neg_mask.any():
                    val_neg_loss_sum += losses[neg_mask].sum().item()
                    val_neg_count += neg_mask.sum().item()

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else None
        avg_val_pos_loss = val_pos_loss_sum / val_pos_count if val_pos_count > 0 else None
        avg_val_neg_loss = val_neg_loss_sum / val_neg_count if val_neg_count > 0 else None

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_pos_loss'].append(avg_val_pos_loss)
        history['val_neg_loss'].append(avg_val_neg_loss)
        
        pos_disp = f"{avg_val_pos_loss:.4f}" if avg_val_pos_loss is not None else "N/A"
        neg_disp = f"{avg_val_neg_loss:.4f}" if avg_val_neg_loss is not None else "N/A"
        overall_disp = f"{avg_val_loss:.4f}" if avg_val_loss is not None else "N/A"
        if epoch % 1 == 0 or epoch == max_epochs - 1:
            print(
                f"Epoch {epoch+1}/{max_epochs} | Train Sim: {1-avg_train_loss:.4f} | "
                f"Val Sim: {1-avg_val_loss:.4f} | "
                f"Val Loss: {overall_disp} | Val Pos Loss: {pos_disp} | Val Neg Loss: {neg_disp}"
            )
        scheduler.step()  # 追加

        metrics = {
            "val_loss": avg_val_loss,
            "val_pos_loss": avg_val_pos_loss,
            "val_neg_loss": avg_val_neg_loss,
        }
        monitor_value = metrics.get(early_stopping_metric)
        if monitor_value is None:
            monitor_value = avg_val_loss
            print(f"Early stopping metric '{early_stopping_metric}' unavailable; fallback to val_loss.")

        # Early Stopping: 監視指標が改善しない場合は学習を打ち切る
        if monitor_value is not None and monitor_value < best_score:
            best_score = monitor_value
            best_state = model.state_dict()
            epochs_no_improve = 0
            best_epoch = epoch + 1
            best_metrics_at_best = {
                "val_loss": avg_val_loss,
                "val_pos_loss": avg_val_pos_loss,
                "val_neg_loss": avg_val_neg_loss,
            }
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= pat:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch}, best val loss: {best_score:.6f})")
            break

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = EPOCHS

    # ベストモデルで評価・保存
    model.load_state_dict(best_state)
    if model_save_path:
        torch.save(best_state, model_save_path)
        print(f"Best model (epoch {best_epoch}) saved to {model_save_path}")

    # --- Test Evaluation (overall / pos / neg) ---
    model.eval()

    def evaluate_df(df: pd.DataFrame, label: str):
        if len(df) == 0:
            print(f"{label}: N/A (n=0)")
            return None, 0

        dataset = MassSpecDataset(
            df,
            mode=f"test-{label}",
            intensity_power=INTENSITY_POWER,
            flag_column=flag_column,
            structural_flag_columns=flag_cols,
            feature_extractor=feature_extractor,
            feature_type=feature_type,
            ecfp_bits=ecfp_bits,
        )
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        sims = []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(config.device)
                mw = batch["mw"].to(config.device)
                target = batch["spectrum"].to(config.device)

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
        "val_loss": best_metrics_at_best.get("val_loss"),
        "val_pos_loss": best_metrics_at_best.get("val_pos_loss"),
        "val_neg_loss": best_metrics_at_best.get("val_neg_loss"),
    }

    return results

def _main_cli():
    res = run_training(
        **RUN_CONFIG,
    )

    def _fmt(val: Optional[float]) -> str:
        return "N/A" if val is None else f"{val:.4f}"

    print("\n=== Test Cosine Similarity ===")
    print(f"Overall: {_fmt(res.get('overall'))} (n={res.get('n_overall', 'N/A')})")
    print(f"Pos ({RUN_CONFIG['flag_column']}=1): {_fmt(res.get('pos'))} (n={res.get('n_pos', 'N/A')})")
    print(f"Neg ({RUN_CONFIG['flag_column']}=0): {_fmt(res.get('neg'))} (n={res.get('n_neg', 'N/A')})")
    print(f"Best epoch: {res.get('best_epoch', 'N/A')} | Val loss: {_fmt(res.get('val_loss'))}")


if __name__ == "__main__":
    _main_cli()
