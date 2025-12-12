"""Core PyTorch model architecture for mass spectrum prediction.

This module mirrors the abstractions in the original TensorFlow code but uses
PyTorch modules so that the rest of the pipeline can be rewritten around Torch.
The goal for this first step is to provide model definitions, feature encoders,
and helper utilities without binding to any I/O or training loop details.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hyper-parameter container
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration values that control the model architecture.

    The defaults are chosen to match the TensorFlow implementation so that we
    can perform like-for-like comparisons once the training loop is migrated.
    """

    fp_length: int = 4096
    radius: int = 2
    use_counting_fp: bool = True
    max_mass_spec_peak_loc: int = 1000
    max_prediction_above_molecule_mass: int = 5

    hidden_units: int = 2000
    num_hidden_layers: int = 2  
    dropout_rate: float = 0.3420465041712287
    activation: str = "relu"
    resnet_bottleneck_factor: float = 0.5

    bidirectional_prediction: bool = True # 双方向予測を行うかどうか
    gate_bidirectional_predictions: bool = True    # 双方向予測のゲーティングを行うかどうか
    reverse_prediction: bool = True

    loss: str = "normalized_generalized_mse"  # {"generalized_mse", "normalized_generalized_mse", "cross_entropy"}
    intensity_power: float = 0.5
    mass_power: float = 1.0

    # Input feature selection
    feature_type: str = "ecfp"  # {"ecfp", "ecfp+bert+flag"}

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = torch.float32

    # --- Attention Fusion用の設定 (追加) ---
    use_attention_fusion: bool = False
    fusion_dim: int = 512  # 各特徴量をこの次元に揃える
    # 各特徴量の次元数マップ (例: {'ecfp': 1024, 'rdkit2d': 200})
    # ※ dataclassのfieldとして定義するため、初期値は空辞書にする
    feature_shapes: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


MODEL_REGISTRY: Dict[str, Type["MassSpectraModel"]] = {}


def register_model(model_type: str) -> Callable[[Type["MassSpectraModel"]], Type["MassSpectraModel"]]:
    """Decorator that registers a model class under a string key."""

    def _decorator(cls: Type["MassSpectraModel"]) -> Type["MassSpectraModel"]:
        if model_type in MODEL_REGISTRY:
            raise ValueError(f"Model type '{model_type}' is already registered")
        MODEL_REGISTRY[model_type] = cls
        cls.model_type = model_type  # type: ignore[attr-defined]
        return cls

    return _decorator


def build_model(model_type: str, config: Optional[ModelConfig] = None) -> "MassSpectraModel":
    """Instantiate a registered model with the provided configuration."""

    if model_type not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model type '{model_type}'. Registered: {sorted(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_type]
    return cls(config=config)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "elu":
        return F.elu
    if name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    raise ValueError(f"Unsupported activation '{name}'")


def _apply_mass_mask(
    prediction: torch.Tensor,
    molecule_weight: torch.Tensor,
    max_prediction_above_mass: int,
) -> torch.Tensor:
    """Zero out prediction bins to the right of (mass + margin)."""

    if prediction.dim() != 2:
        raise ValueError("Prediction tensor must have shape [batch, num_bins]")

    total_mass = torch.round(molecule_weight.squeeze(-1)).to(torch.long)
    length = prediction.shape[1]
    indices = torch.arange(length, device=prediction.device).unsqueeze(0)
    threshold = total_mass.unsqueeze(1) + max_prediction_above_mass
    mask = indices <= threshold
    return prediction * mask.to(prediction.dtype)


def _reverse_prediction(
    prediction: torch.Tensor,
    molecule_weight: torch.Tensor,
    max_prediction_above_mass: int,
) -> torch.Tensor:
    """Align the prediction such that mass becomes the anchor index.

    This mirrors util.scatter_by_anchor_indices from the TensorFlow code.
    """

    if prediction.dim() != 2:
        raise ValueError("Prediction tensor must have shape [batch, num_bins]")

    batch_size, length = prediction.shape
    total_mass = torch.round(molecule_weight.squeeze(-1)).to(torch.long)
    indices = torch.arange(length, device=prediction.device).view(1, length)

    dest = total_mass.view(batch_size, 1) - indices + max_prediction_above_mass
    valid = (dest >= 0) & (dest < length)

    output = torch.zeros_like(prediction)
    if not valid.any():
        return output

    dest_clamped = dest.clamp(0, length - 1)
    values = torch.where(valid, prediction, torch.zeros_like(prediction))
    output.scatter_add_(1, dest_clamped, values)
    # Explicitly zero out entries that came from invalid positions (already zero)
    return output


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class MassSpectraModel(nn.Module):
    """Abstract base class encapsulating common prediction helpers."""

    model_type: str = "unknown"

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.activation = _get_activation(self.config.activation)

    # -- abstract-ish API -------------------------------------------------

    def encode_features(self, fingerprint: torch.Tensor) -> torch.Tensor:
        """Map raw fingerprint features to a learned representation."""
        raise NotImplementedError

    def forward(
        self,
        fingerprint: torch.Tensor,
        molecule_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (prediction, logits/raw_prediction)."""

        features = self.encode_features(fingerprint)
        return self.predict_from_encoded(features, molecule_weight)

    def predict_from_encoded(
        self,
        features: torch.Tensor,
        molecule_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_prediction = self._make_prediction(features, molecule_weight)

        if self.config.loss == "cross_entropy":
            prediction = F.softmax(raw_prediction, dim=-1)
        else:
            prediction = F.relu(raw_prediction)
        return prediction, raw_prediction

    def get_feature_dim(self) -> int:
        attr = getattr(self, "feature_dim", None)
        if isinstance(attr, int):
            return int(attr)

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                dummy = torch.zeros(
                    1,
                    self.config.fp_length,
                    device=self.config.device,
                    dtype=self.config.dtype,
                )
                features = self.encode_features(dummy)
        finally:
            if was_training:
                self.train()
        if features.ndim == 0:
            raise ValueError("encode_features must return a tensor with at least one dimension")
        return int(features.shape[-1])

    # -- helpers ----------------------------------------------------------

    def _make_prediction(self, features: torch.Tensor, molecule_weight: torch.Tensor) -> torch.Tensor:
        if self.config.bidirectional_prediction:
            forward = self.forward_head(features)
            forward = _apply_mass_mask(forward, molecule_weight, self.config.max_prediction_above_molecule_mass)

            backward = self.backward_head(features)
            backward = _reverse_prediction(backward, molecule_weight, self.config.max_prediction_above_molecule_mass)

            if self.config.gate_bidirectional_predictions:
                gate = torch.sigmoid(self.gate_head(features))
                raw = gate * forward + (1.0 - gate) * backward
            else:
                raw = forward + backward
        else:
            raw = self.output_head(features)
            if self.config.reverse_prediction:
                raw = _reverse_prediction(raw, molecule_weight, self.config.max_prediction_above_molecule_mass)
            else:
                raw = _apply_mass_mask(raw, molecule_weight, self.config.max_prediction_above_molecule_mass)
        return raw


# ---------------------------------------------------------------------------
# MLP backbone
# ---------------------------------------------------------------------------


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that gracefully handles batch size 1 by reusing running stats."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training and input.dim() >= 2 and input.shape[0] == 1:
            return F.batch_norm(
                input,
                self.running_mean if self.track_running_stats else None,
                self.running_var if self.track_running_stats else None,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float, activation: Callable[[torch.Tensor], torch.Tensor], bottleneck_factor: float) -> None:
        super().__init__()
        bottleneck_size = max(1, int(hidden_units * bottleneck_factor))
        self.bn1 = SafeBatchNorm1d(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_units, bottleneck_size)
        self.bn2 = SafeBatchNorm1d(bottleneck_size)
        self.fc2 = nn.Linear(bottleneck_size, hidden_units)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bn1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.fc2(out)
        return residual + out


@register_model("mlp")
class MLPSpectraModel(MassSpectraModel):
    """Multi-layer perceptron with ResNet-style residual blocks."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config=config)
        cfg = self.config

        # --- 【追加】入力データを正規化する層 ---
        # SafeBatchNorm1d はこのファイル内で定義されているのでそのまま使えます
        self.input_bn = SafeBatchNorm1d(cfg.fp_length) 
        # ------------------------------------

        self.feature_dim = cfg.hidden_units if cfg.num_hidden_layers > 0 else cfg.fp_length
        self.input_layer = (
            nn.Linear(cfg.fp_length, cfg.hidden_units)
            if cfg.num_hidden_layers > 0
            else nn.Identity()
        )

        self.residual_blocks = nn.ModuleList(
            ResidualBlock(
                hidden_units=self.feature_dim,
                dropout_rate=cfg.dropout_rate,
                activation=self.activation,
                bottleneck_factor=cfg.resnet_bottleneck_factor,
            )
            for _ in range(max(0, cfg.num_hidden_layers))
        )

        self.final_bn = SafeBatchNorm1d(self.feature_dim)

        # Heads for different prediction modes
        self.forward_head = nn.Linear(self.feature_dim, cfg.max_mass_spec_peak_loc)
        self.backward_head = nn.Linear(self.feature_dim, cfg.max_mass_spec_peak_loc)
        self.gate_head = nn.Linear(self.feature_dim, cfg.max_mass_spec_peak_loc)
        self.output_head = nn.Linear(self.feature_dim, cfg.max_mass_spec_peak_loc)

        self.to(self.config.device, self.config.dtype)

    def encode_features(self, fingerprint: torch.Tensor) -> torch.Tensor:
        x = fingerprint.to(device=self.config.device, dtype=self.config.dtype)
        # --- 【追加】最初に正規化！ ---
        x = self.input_bn(x)
        # ---------------------------
        x = self.input_layer(x)
        if not isinstance(self.input_layer, nn.Identity):
            x = self.activation(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.final_bn(x)
        x = self.activation(x)
        return x

# attention fusionの実装
# 「F分子ならRDKitのスコアを上げる」 という動作を担う核心部分
class AttentionFusionLayer(nn.Module):
    def __init__(self, feature_shapes: Dict[str, int], fusion_dim: int = 512):
        super().__init__()
        self.feature_shapes = feature_shapes
        self.fusion_dim = fusion_dim
        self.feature_names = list(feature_shapes.keys()) # 順序固定のためリスト化
        self.total_input_dim = sum(feature_shapes.values())
        if self.total_input_dim <= 0:
            raise ValueError("AttentionFusionLayer requires at least one feature with positive dimension.")

        # 1. Projectors: 各特徴量を同じ次元(fusion_dim)に変換
        self.projectors = nn.ModuleDict()
        for name, input_dim in feature_shapes.items():
            self.projectors[name] = nn.Sequential(
                nn.Linear(input_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            )

        # 2. Attention Scorer: 各特徴量の重要度(スカラー)を算出
        # 入力: fusion_dim -> 出力: 1 (logit)
        self.attn_fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.Tanh(),
            nn.Linear(fusion_dim // 2, 1) # スカラーを出力
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 結合された巨大な1次元テンソル [Batch, Total_Dim]
        Returns:
            fused_vec: 融合後のベクトル [Batch, fusion_dim]
            attn_weights: Attentionの重み [Batch, Num_Features] (可視化用)
        """
        if x.shape[1] != self.total_input_dim:
            raise ValueError(
                f"AttentionFusionLayer expected input dim {self.total_input_dim} "
                f"from feature_shapes, but got {x.shape[1]}"
            )

        projected_list = []
        
        # 1. 巨大なテンソルを元の特徴量ごとにスライスして射影
        start_idx = 0
        for name in self.feature_names:
            dim = self.feature_shapes[name]
            # スライス
            feat_raw = x[:, start_idx : start_idx + dim]
            start_idx += dim
            
            # 射影 (Batch, fusion_dim)
            proj = self.projectors[name](feat_raw)
            projected_list.append(proj)

        # スタック -> (Batch, Num_Features, fusion_dim)
        H = torch.stack(projected_list, dim=1)

        # 2. Attentionスコア計算
        # (Batch, Num_Features, fusion_dim) -> (Batch, Num_Features, 1)
        scores = self.attn_fc(H)
        
        # Softmaxで確率化 (合計1.0にする)
        alpha = F.softmax(scores, dim=1) 

        # 3. 重み付け和 (Context Vector)
        # alpha: (B, N, 1) * H: (B, N, D) -> sum(dim=1) -> (B, D)
        fused_vec = torch.sum(alpha * H, dim=1)
        
        return fused_vec, alpha.squeeze(-1)

@register_model("attention_mlp")
class AttentionFusionSpectraModel(MassSpectraModel):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config=config)
        cfg = self.config

        # 入力の正規化 (結合された状態で行う)
        self.input_bn = SafeBatchNorm1d(cfg.fp_length)

        # ★ ここが変更点: Attention Fusion層の初期化
        if not cfg.feature_shapes:
            raise ValueError("Attention Fusion requires 'feature_shapes' in config.")
        
        self.fusion_layer = AttentionFusionLayer(
            feature_shapes=cfg.feature_shapes,
            fusion_dim=cfg.fusion_dim
        )
        if self.fusion_layer.total_input_dim != cfg.fp_length:
            raise ValueError(
                f"Sum of feature_shapes ({self.fusion_layer.total_input_dim}) must match fp_length ({cfg.fp_length}) for Attention Fusion."
            )

        # Backboneへの入力次元は fusion_dim になる
        self.feature_dim = cfg.hidden_units
        # 最初のLinear層の入力次元を fusion_dim に変更
        self.backbone_input = nn.Linear(cfg.fusion_dim, cfg.hidden_units)

        # 残りは既存のMLPと同じResidual Blocks
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(
                hidden_units=cfg.hidden_units,
                dropout_rate=cfg.dropout_rate,
                activation=self.activation,
                bottleneck_factor=cfg.resnet_bottleneck_factor,
            )
            for _ in range(max(0, cfg.num_hidden_layers))
        )
        self.final_bn = SafeBatchNorm1d(cfg.hidden_units)
        
        # Heads (既存と同じ)
        self.forward_head = nn.Linear(cfg.hidden_units, cfg.max_mass_spec_peak_loc)
        self.backward_head = nn.Linear(cfg.hidden_units, cfg.max_mass_spec_peak_loc)
        self.gate_head = nn.Linear(cfg.hidden_units, cfg.max_mass_spec_peak_loc)
        self.output_head = nn.Linear(cfg.hidden_units, cfg.max_mass_spec_peak_loc)

        self.to(self.config.device, self.config.dtype)

    def encode_features(self, fingerprint: torch.Tensor) -> torch.Tensor:
        x = fingerprint.to(device=self.config.device, dtype=self.config.dtype)
        
        # 1. 全体正規化
        x = self.input_bn(x)
        
        # 2. Attention Fusion (ここで分離・結合が行われる)
        fused_x, attn_weights = self.fusion_layer(x)
        
        # ※ attn_weights は解析用に保持したいが、encode_featuresの戻り値定義上
        # ここではグラフを切った形で最後の重みのみ保持してメモリ圧迫を避ける
        self.last_attn_weights = attn_weights.detach().cpu()

        # 3. Backbone (MLP)
        out = self.backbone_input(fused_x)
        out = self.activation(out)
        
        for block in self.residual_blocks:
            out = block(out)
        out = self.final_bn(out)
        out = self.activation(out)
        
        return out