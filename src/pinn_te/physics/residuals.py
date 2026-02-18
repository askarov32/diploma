from __future__ import annotations

from typing import Dict, Tuple

import torch

from pinn_te.physics.kinematics import jacobian, strain_tensor_from_xyzt, trace_3x3, laplacian_scalar_from_xyzt
from pinn_te.physics.thermoelastic_iso import stress_iso_thermo, div_stress_from_xyzt


def compute_time_derivatives_u(u: torch.Tensor, xyzt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    u: (N,3), xyzt: (N,4) -> u_t: (N,3), u_tt: (N,3)
    """
    Ju = jacobian(u, xyzt)          # (N,3,4)
    u_t = Ju[:, :, 3]               # (N,3)
    Ju_t = jacobian(u_t, xyzt)      # (N,3,4)
    u_tt = Ju_t[:, :, 3]            # (N,3)
    return u_t, u_tt


def compute_time_derivative_scalar(y: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
    """
    y: (N,1) -> y_t: (N,1)
    """
    Jy = jacobian(y, xyzt)          # (N,1,4)
    return Jy[:, :, 3]              # (N,1)


def thermoelastic_residuals(
    model: torch.nn.Module,
    xyzt: torch.Tensor,
    material,
    heat_cfg: Dict,
coupling_cfg: Dict,
    body_force: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Возвращает:
      r_motion: (N,3)  rho u_tt - div(sigma) - f
      r_heat  : (N,1)  k lap(T) - C T_t - gamma*T0 * d/dt(tr(eps))
      plus intermediate tensors for post-processing
    """
    xyzt = xyzt.requires_grad_(True)
    xyz = xyzt[:, :3]

    pred = model(xyzt)
    u = pred[:, 0:3]
    T = pred[:, 3:4]

    mat = material.eval(xyz)
    rho = mat["rho"]               # (N,1)
    lam = mat["lame_lambda"]       # (N,1)
    mu = mat["lame_mu"]            # (N,1)

    # тепло/связь пока константами из конфига (позже можно тоже по слоям)
    k = torch.full_like(T, float(heat_cfg["k_W_mK"]))
    C = torch.full_like(T, float(heat_cfg["C_J_m3K"]))

    gamma = torch.full_like(T, float(coupling_cfg["gamma_Pa_K"]))
    T0 = torch.full_like(T, float(coupling_cfg.get("T0_K", 293.15)))

    # mechanics
    u_t, u_tt = compute_time_derivatives_u(u, xyzt)
    eps = strain_tensor_from_xyzt(u, xyzt)
    sigma = stress_iso_thermo(eps, T, lam, mu, gamma)
    div_sig = div_stress_from_xyzt(sigma, xyzt)
    lapT = laplacian_scalar_from_xyzt(T, xyzt)

    if body_force is None:
        f = torch.zeros_like(u)  # (N,3)
    else:
        f = body_force

    r_motion = rho * u_tt - div_sig - f

    # heat equation
    T_t = compute_time_derivative_scalar(T, xyzt)       # (N,1)
    lapT = laplacian_scalar_from_xyzt(T, xyzt)                     # (N,1)

    tr_eps = trace_3x3(eps)                             # (N,1)
    tr_eps_t = compute_time_derivative_scalar(tr_eps, xyzt)  # (N,1)

    r_heat = k * lapT - C * T_t - gamma * T0 * tr_eps_t

    return {
        "r_motion": r_motion,
        "r_heat": r_heat,
        "u": u,
        "T": T,
        "u_t": u_t,
        "sigma": sigma,
        "rho": rho,
        "lambda": lam,
        "mu": mu,
    }
