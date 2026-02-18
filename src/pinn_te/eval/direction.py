from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from pinn_te.physics.residuals import thermoelastic_residuals


@dataclass(frozen=True)
class DirectionResult:
    # (K,3) направление на каждом t_k (агрегированное по сенсорам)
    dir_t: torch.Tensor
    # (K,1) энергия (норма S) на каждом t_k (агрегированная)
    energy_t: torch.Tensor
    # итоговое направление (3,)
    dir_final: torch.Tensor
    # угол ошибки (градусы) если задан true_dir
    angle_deg: float | None


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + eps)
    return v / n


def energy_flux(sigma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    sigma: (N,3,3), v: (N,3) -> S: (N,3)
    S = - sigma · v
    """
    return -(sigma @ v.unsqueeze(-1)).squeeze(-1)


def angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a,b: (3,) torch
    """
    a = _normalize(a.view(1, 3)).view(3)
    b = _normalize(b.view(1, 3)).view(3)
    cos = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    ang = torch.arccos(cos) * (180.0 / torch.pi)
    return float(ang.detach().cpu().item())


@torch.no_grad()
def _predict_fields_no_grad(model, xyzt: torch.Tensor) -> torch.Tensor:
    # Нельзя использовать no_grad для производных, поэтому эта функция тут не используется.
    return model(xyzt)


def directional_prediction_energy_flux(
    model: torch.nn.Module,
    material,
    cfg: Dict,
    receivers_xyz: torch.Tensor,
    t_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    true_dir: torch.Tensor | None = None,
) -> DirectionResult:
    """
    receivers_xyz: (R,3)
    t_grid: (K,) или (K,1)
    Возвращает направление распространения на основе energy flux.
    """
    model.eval()

    if t_grid.ndim == 1:
        t_grid = t_grid.view(-1, 1)  # (K,1)

    R = receivers_xyz.shape[0]
    K = t_grid.shape[0]

    # будем агрегировать по сенсорам: mean(S) и mean(|S|)
    dirs = []
    energies = []

    for k in range(K):
        t = t_grid[k:k+1].repeat(R, 1)  # (R,1)
        xyzt = torch.cat([receivers_xyz, t], dim=1).to(device=device, dtype=dtype)  # (R,4)

        # Важно: здесь нужны градиенты по времени/пространству -> no_grad нельзя
        out = thermoelastic_residuals(
            model=model,
            xyzt=xyzt,
            material=material,
            heat_cfg=cfg["physics"]["heat"],
            coupling_cfg=cfg["physics"]["coupling"],
            body_force=None,
        )

        sigma = out["sigma"]   # (R,3,3)
        v = out["u_t"]         # (R,3)

        S = energy_flux(sigma, v)          # (R,3)
        S_mean = torch.mean(S, dim=0)      # (3,)
        e_mean = torch.mean(torch.norm(S, dim=1, keepdim=True), dim=0)  # (1,)

        dirs.append(_normalize(S_mean))
        energies.append(e_mean)

    dir_t = torch.stack(dirs, dim=0)         # (K,3)
    energy_t = torch.stack(energies, dim=0)  # (K,1)

    # Итог: берём направление в момент максимальной энергии
    k_max = int(torch.argmax(energy_t[:, 0]).detach().cpu().item())
    dir_final = dir_t[k_max, :]

    ang = None
    if true_dir is not None:
        ang = angle_deg(dir_final, true_dir.to(device=device, dtype=dtype))

    return DirectionResult(
        dir_t=dir_t.detach(),
        energy_t=energy_t.detach(),
        dir_final=dir_final.detach(),
        angle_deg=ang,
    )
