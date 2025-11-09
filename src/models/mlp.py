# src/models/mlp.py
import torch
import torch.nn as nn
from typing import Iterable, Union

class EEGFeatureMLP(nn.Module):
    """
    Flexible MLP for EEG feature vectors.
    Accepts either:
      - hidden_dim=<int> (two layers of the same size), or
      - hidden_dims=[h1, h2, ...] / tuple / int
    """
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        hidden_dim: int = 64,                 # kept for backward-compat
        hidden_dims: Union[int, Iterable[int]] = None,  # new flexible arg
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        # Normalize hidden_dims
        if hidden_dims is None:
            # fall back to 2 layers of hidden_dim (your previous default)
            hidden_dims = [hidden_dim, hidden_dim]
        elif isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]       # single hidden layer

        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(n_classes)]

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Backwards-friendly alias
MLP = EEGFeatureMLP
__all__ = ["EEGFeatureMLP", "MLP"]
