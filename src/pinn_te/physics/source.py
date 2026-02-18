from __future__ import annotations

from typing import Dict, Tuple

import torch


def ricker(t: torch.Tensor, f0: float, t0: float) -> torch.Tensor:
    """
    Ricker wavelet (Mexican hat)
    t: (N,1)
    """
    pi = torch.tensor(torch.pi, device=t.device, dtype=t.dtype)
    a = pi * f0 * (t - t0)
    return (1.0 - 2.0 * a**2) * torch.exp(-a**2)


def gaussian_spatial(xyz: torch.Tensor, center: Tuple[float, float, float], sigma_m: float) -> torch.Tensor:
    """
    isotropic Gaussian in space
    xyz: (N,3)
    """
    c = torch.tensor(center, device=xyz.device, dtype=xyz.dtype).view(1, 3)
    r2 = torch.sum((xyz - c) ** 2, dim=1, keepdim=True)  # (N,1)
    return torch.exp(-0.5 * r2 / (sigma_m**2))


def body_force_from_cfg(xyzt: torch.Tensor, cfg_source: Dict) -> torch.Tensor:
    """
    f(x,t) = A * spatial(x) * ricker(t) * direction
    returns: (N,3)
    """
    xyz = xyzt[:, :3]
    t = xyzt[:, 3:4]

    f0 = float(cfg_source["f0_hz"])
    t0 = float(cfg_source["t0"])
    amp = float(cfg_source.get("amplitude", 1.0))
    center = tuple(float(v) for v in cfg_source["location"])
    direction = torch.tensor(cfg_source.get("direction", [1.0, 0.0, 0.0]),
                             device=xyzt.device, dtype=xyzt.dtype).view(1, 3)

    # нормируем direction
    direction = direction / (torch.norm(direction, dim=1, keepdim=True) + 1e-12)

    sigma_m = float(cfg_source.get("sigma_m", 50.0))  # ширина источника в метрах

    w_t = ricker(t, f0=f0, t0=t0)             # (N,1)
    w_x = gaussian_spatial(xyz, center, sigma_m=sigma_m)  # (N,1)

    f = amp * w_x * w_t * direction           # (N,3)
    return f
