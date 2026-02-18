from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


ActivationName = Literal["tanh", "relu", "gelu", "silu"]


def _make_activation(name: ActivationName) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """
    Простая MLP: (x,y,z,t) -> (ux,uy,uz,T)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        depth: int,
        activation: ActivationName = "tanh",
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2 (input->...->output)")

        act = _make_activation(activation)

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(act)

        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(act)

        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        # Инициализация под tanh/relu: Xavier works OK as baseline
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(cfg: dict, device: torch.device, dtype: torch.dtype) -> nn.Module:
    mcfg = cfg["model"]
    if mcfg.get("type", "mlp") != "mlp":
        raise ValueError("Only model.type=mlp is supported for now")

    model = MLP(
        in_dim=int(mcfg["in_dim"]),
        out_dim=int(mcfg["out_dim"]),
        width=int(mcfg["width"]),
        depth=int(mcfg["depth"]),
        activation=str(mcfg.get("activation", "tanh")).lower(),  # type: ignore[arg-type]
    )
    return model.to(device=device, dtype=dtype)
