"""Configurable PyTorch Multi-Layer Perceptron for candidate road-segment scoring."""

from __future__ import annotations

import os
import random
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_global_seed(seed: int) -> None:
    """Fix all relevant RNG seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


class PathSnapMLP(nn.Module):
    """Configurable feed-forward MLP binary classifier over candidate road segments.

    Outputs a single logit per candidate row; pass through `sigmoid` to obtain
    P(match | observation, candidate_arc) for Top-1 ranking and HMM/Viterbi emissions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64, 32),
        dropout: float = 0.20,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims: Tuple[int, ...] = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.use_batch_norm = bool(use_batch_norm)

        layers = []
        in_features = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_features = h_dim

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch_size,)."""
        return self.network(x).squeeze(-1)

    @torch.no_grad()
    def predict_proba_numpy(
        self, x_scaled: np.ndarray, batch_size: int = 8192, device: str = "cpu"
    ) -> np.ndarray:
        """Run batched inference and return P(match=1) as float64 numpy array."""
        self.eval()
        self.to(device)
        n = len(x_scaled)
        if n == 0:
            return np.empty(0, dtype=np.float64)
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(np.ascontiguousarray(x_scaled[start:end], dtype=np.float32)).to(device)
            logits = self.forward(xb)
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
        return out


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_probs_per_observation(
    probs: np.ndarray, obs_keys: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-observation normalized emission probabilities and log-probabilities.

    Args:
        probs: Raw sigmoid probabilities P(match=1) for each candidate row.
        obs_keys: Integer or hashable observation group key per row (same within an obs).
        eps: Numerical floor for log-probabilities.

    Returns:
        norm_probs: Probabilities summing to 1.0 across candidates of each observation.
        log_probs: log(clip(norm_probs, eps, 1.0)) ready for Viterbi decoding.
    """
    n = len(probs)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    _, inv, counts = np.unique(obs_keys, return_inverse=True, return_counts=True)
    clipped = np.clip(probs, eps, 1.0)
    sums = np.bincount(inv, weights=clipped)
    norm_probs = clipped / np.maximum(sums[inv], eps)
    log_probs = np.log(np.clip(norm_probs, eps, 1.0))
    return norm_probs, log_probs
