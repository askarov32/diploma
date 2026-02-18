from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    pde_motion: torch.Tensor
    pde_heat: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor


def mse(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x * x)


def ic_loss(model: torch.nn.Module, xyzt0: torch.Tensor, ic_cfg: Dict) -> torch.Tensor:
    xyzt0 = xyzt0.requires_grad_(True)
    pred = model(xyzt0)
    u = pred[:, 0:3]
    T = pred[:, 3:4]

    u0 = torch.tensor(ic_cfg["u"], device=u.device, dtype=u.dtype).view(1, 3)
    T0 = torch.tensor([float(ic_cfg["T"])], device=T.device, dtype=T.dtype).view(1, 1)

    return mse(u - u0) + mse(T - T0)


def bc_loss_free_surface(res_boundary: Dict[str, torch.Tensor]) -> torch.Tensor:
    # n=(0,0,1) => traction = sigma[:,:,2]
    sigma = res_boundary["sigma"]      # (N,3,3)
    traction = sigma[:, :, 2]          # (N,3)
    return mse(traction)


def compute_pinn_loss(
    res_interior: Dict[str, torch.Tensor],
    res_boundary: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    batches: Dict[str, torch.Tensor],
    cfg: Dict,
) -> LossBreakdown:
    w = cfg["training"]["weights"]

    L_motion = mse(res_interior["r_motion"])
    L_heat = mse(res_interior["r_heat"])
    L_ic = ic_loss(model, batches["initial"], cfg["initial_conditions"])
    L_bc = bc_loss_free_surface(res_boundary)

    total = (
        float(w["pde_motion"]) * L_motion
        + float(w["pde_heat"]) * L_heat
        + float(w["ic"]) * L_ic
        + float(w["bc"]) * L_bc
    )

    return LossBreakdown(
        total=total,
        pde_motion=L_motion,
        pde_heat=L_heat,
        ic=L_ic,
        bc=L_bc,
    )
