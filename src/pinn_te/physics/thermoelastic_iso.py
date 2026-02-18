from __future__ import annotations

import torch

from pinn_te.physics.kinematics import trace_3x3
from pinn_te.physics.kinematics import jacobian

def stress_iso_thermo(
    eps: torch.Tensor,
    T: torch.Tensor,
    lame_lambda: torch.Tensor,
    lame_mu: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    σ_ij = λ δ_ij tr(ε) + 2μ ε_ij - γ δ_ij T
    eps: (N,3,3)
    T: (N,1)
    params: (N,1) each or broadcastable
    """
    if eps.ndim != 3 or eps.shape[1:] != (3, 3):
        raise ValueError("eps must have shape (N,3,3)")
    if T.ndim != 2 or T.shape[1] != 1:
        raise ValueError("T must have shape (N,1)")

    tr = trace_3x3(eps)  # (N,1)
    I = torch.eye(3, device=eps.device, dtype=eps.dtype).unsqueeze(0)  # (1,3,3)

    lam = lame_lambda.view(-1, 1, 1)
    mu = lame_mu.view(-1, 1, 1)
    gam = gamma.view(-1, 1, 1)
    tr_ = tr.view(-1, 1, 1)
    T_ = T.view(-1, 1, 1)

    sigma = lam * tr_ * I + 2.0 * mu * eps - gam * T_ * I
    return sigma


def div_stress(sigma: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """
    (∇·σ)_i = ∂σ_ij/∂x_j
    sigma: (N,3,3), xyz: (N,3) -> (N,3)
    """
    if sigma.ndim != 3 or sigma.shape[1:] != (3, 3):
        raise ValueError("sigma must have shape (N,3,3)")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N,3)")

    out = []
    # для каждого i берём строку sigma[i,:] и считаем дивергенцию
    for i in range(3):
        # sigma_i: (N,3) компоненты по j
        sigma_i = sigma[:, i, :]  # (N,3)
        # производные по x,y,z: получим (N,3,3)
        # d(sigma_i_j)/d(x_k)
        from pinn_te.physics.kinematics import jacobian  # локальный импорт чтобы избежать циклов
        J = jacobian(sigma_i, xyz)  # (N,3,3)
        div_i = (J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]).unsqueeze(1)  # (N,1)
        out.append(div_i)

    return torch.cat(out, dim=1)  # (N,3)

def div_stress_from_xyzt(sigma: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
    """
    (∇·σ)_i = ∂σ_ij/∂x_j, j in {x,y,z}
    sigma: (N,3,3), xyzt: (N,4) -> (N,3)
    """
    out = []
    for i in range(3):
        sigma_i = sigma[:, i, :]          # (N,3)
        J = jacobian(sigma_i, xyzt)[:, :, :3]  # (N,3,3) по x,y,z
        div_i = (J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]).unsqueeze(1)
        out.append(div_i)
    return torch.cat(out, dim=1)          # (N,3)