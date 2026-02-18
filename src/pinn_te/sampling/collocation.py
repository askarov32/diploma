from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class Domain:
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    t: Tuple[float, float]


def _rand_uniform(n: int, lo: float, hi: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (lo + (hi - lo) * torch.rand((n, 1), device=device, dtype=dtype))


def sample_interior(domain: Domain, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = _rand_uniform(n, domain.x[0], domain.x[1], device, dtype)
    y = _rand_uniform(n, domain.y[0], domain.y[1], device, dtype)
    z = _rand_uniform(n, domain.z[0], domain.z[1], device, dtype)
    t = _rand_uniform(n, domain.t[0], domain.t[1], device, dtype)
    return torch.cat([x, y, z, t], dim=1)  # (N,4)


def sample_initial(domain: Domain, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # t = t0 (обычно 0)
    x = _rand_uniform(n, domain.x[0], domain.x[1], device, dtype)
    y = _rand_uniform(n, domain.y[0], domain.y[1], device, dtype)
    z = _rand_uniform(n, domain.z[0], domain.z[1], device, dtype)
    t0 = torch.full((n, 1), float(domain.t[0]), device=device, dtype=dtype)
    return torch.cat([x, y, z, t0], dim=1)


def sample_boundary(domain: Domain, n: int, bc_at: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    bc_at: строка типа "z=0" или "x=0" или "y=2000" и т.д.
    Пока поддерживаем только одну плоскость.
    """
    bc_at = bc_at.replace(" ", "")
    if "=" not in bc_at:
        raise ValueError(f"Invalid bc_at: {bc_at}, expected like 'z=0'")

    axis, val_s = bc_at.split("=", 1)
    val = float(val_s)

    x = _rand_uniform(n, domain.x[0], domain.x[1], device, dtype)
    y = _rand_uniform(n, domain.y[0], domain.y[1], device, dtype)
    z = _rand_uniform(n, domain.z[0], domain.z[1], device, dtype)
    t = _rand_uniform(n, domain.t[0], domain.t[1], device, dtype)

    if axis == "x":
        x = torch.full((n, 1), val, device=device, dtype=dtype)
    elif axis == "y":
        y = torch.full((n, 1), val, device=device, dtype=dtype)
    elif axis == "z":
        z = torch.full((n, 1), val, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown axis in bc_at: {axis}")

    return torch.cat([x, y, z, t], dim=1)


def domain_from_cfg(cfg: dict) -> Domain:
    d = cfg["domain"]
    return Domain(
        x=(float(d["x"][0]), float(d["x"][1])),
        y=(float(d["y"][0]), float(d["y"][1])),
        z=(float(d["z"][0]), float(d["z"][1])),
        t=(float(d["t"][0]), float(d["t"][1])),
    )


def build_collocation_batches(cfg: dict, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    domain = domain_from_cfg(cfg)
    samp = cfg["sampling"]

    interior = sample_interior(domain, int(samp["interior"]["n"]), device, dtype)

    # boundary (пока один bc, как в cfg.boundary_conditions.mechanical.at)
    bc_at = str(cfg["boundary_conditions"]["mechanical"]["at"])
    boundary = sample_boundary(domain, int(samp["boundary"]["n"]), bc_at, device, dtype)

    initial = sample_initial(domain, int(samp["initial"]["n"]), device, dtype)

    return {"interior": interior, "boundary": boundary, "initial": initial}
