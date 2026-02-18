from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import torch

from pinn_te.sampling.collocation import Domain


@dataclass(frozen=True)
class Sensors:
    xyz: torch.Tensor  # (R,3)
    # при необходимости можно хранить индексы/названия


def _circle_receivers(
    center_xy: Tuple[float, float],
    radius: float,
    z_m: float,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    cx, cy = center_xy
    angles = torch.linspace(0.0, 2.0 * math.pi, steps=n + 1, device=device, dtype=dtype)[:-1]
    x = cx + radius * torch.cos(angles)
    y = cy + radius * torch.sin(angles)
    z = torch.full((n,), float(z_m), device=device, dtype=dtype)
    return torch.stack([x, y, z], dim=1)  # (n,3)


def build_sensors(cfg: dict, domain: Domain, device: torch.device, dtype: torch.dtype) -> Sensors | None:
    scfg = cfg["sampling"].get("sensors", {})
    if not scfg or not bool(scfg.get("enabled", False)):
        return None

    layout = str(scfg.get("receiver_layout", "circle"))
    n = int(scfg.get("n_receivers", 12))
    z_m = float(scfg.get("z_m", (domain.z[0] + domain.z[1]) * 0.5))

    # центр домена в XY
    cx = 0.5 * (domain.x[0] + domain.x[1])
    cy = 0.5 * (domain.y[0] + domain.y[1])

    if layout == "circle":
        radius = float(scfg.get("radius_m", 0.25 * min(domain.x[1] - domain.x[0], domain.y[1] - domain.y[0])))
        xyz = _circle_receivers((cx, cy), radius, z_m, n, device, dtype)
        return Sensors(xyz=xyz)

    raise ValueError(f"Unsupported receiver_layout: {layout}")
