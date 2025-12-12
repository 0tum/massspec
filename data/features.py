"""Feature extraction utilities for NEIMS based on ChemBERTa embeddings."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


CHEMBERTA_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaFeatureExtractor:
    """Wraps a frozen ChemBERTa model to produce [CLS] embeddings."""

    def __init__(self, model_name: str = CHEMBERTA_MODEL_NAME, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim: int = int(self.model.config.hidden_size)

    @torch.no_grad()
    def __call__(self, smiles: str, as_numpy: bool = False) -> Union[torch.Tensor, np.ndarray]:
        encoded = self.tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        cls_token = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        cls_flat = cls_token.squeeze(0).detach()
        if as_numpy:
            return cls_flat.cpu().numpy()
        return cls_flat


_DEFAULT_EXTRACTOR: Optional[ChemBERTaFeatureExtractor] = None


def get_default_extractor(device: Optional[torch.device] = None) -> ChemBERTaFeatureExtractor:
    """Create or return a cached ChemBERTa feature extractor."""

    global _DEFAULT_EXTRACTOR
    needs_new = _DEFAULT_EXTRACTOR is None or (device is not None and _DEFAULT_EXTRACTOR.device != device)
    if needs_new:
        _DEFAULT_EXTRACTOR = ChemBERTaFeatureExtractor(device=device)
    return _DEFAULT_EXTRACTOR


def get_hybrid_features(
    smiles: str,
    flags: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]],
    extractor: Optional[ChemBERTaFeatureExtractor] = None,
    device: Optional[torch.device] = None,
    as_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Return ChemBERTa [CLS] embedding concatenated with structural flags."""

    extractor = extractor or get_default_extractor(device=device)
    emb = extractor(smiles, as_numpy=False)
    target_device = device or emb.device

    if flags is None:
        flags = []

    flag_tensor = torch.as_tensor(flags, dtype=torch.float32, device=target_device)
    if flag_tensor.dim() == 0:
        flag_tensor = flag_tensor.unsqueeze(0)

    hybrid = torch.cat([emb.to(target_device), flag_tensor], dim=-1)
    if as_numpy:
        return hybrid.detach().cpu().numpy()
    return hybrid.detach()
