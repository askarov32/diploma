from __future__ import annotations

import torch


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: (N,1), x: (N,D) -> dy/dx: (N,D)
    """
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must have shape (N,1)")
    if x.ndim != 2:
        raise ValueError("x must be 2D (N,D)")

    g = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return g


def jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: (N,M), x: (N,D) -> J: (N,M,D)
    """
    if y.ndim != 2:
        raise ValueError("y must be 2D (N,M)")
    cols = []
    for i in range(y.shape[1]):
        gi = grad_scalar(y[:, i:i+1], x)  # (N,D)
        cols.append(gi.unsqueeze(1))      # (N,1,D)
    return torch.cat(cols, dim=1)         # (N,M,D)


def strain_tensor_from_xyzt(u: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
    """
    u: (N,3), xyzt: (N,4) -> eps: (N,3,3)
    eps_ij = 0.5 (du_i/dx_j + du_j/dx_i), j in {x,y,z}
    """
    if u.shape[1] != 3 or xyzt.shape[1] != 4:
        raise ValueError("u must be (N,3) and xyzt must be (N,4)")

    J = jacobian(u, xyzt)[:, :, :3]  # (N,3,3) по x,y,z
    return 0.5 * (J + J.transpose(1, 2))


def trace_3x3(A: torch.Tensor) -> torch.Tensor:
    return (A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]).unsqueeze(1)


def laplacian_scalar_from_xyzt(y: torch.Tensor, xyzt: torch.Tensor) -> torch.Tensor:
    """
    y: (N,1), xyzt: (N,4) -> ∆y по x,y,z: (N,1)
    """
    g = grad_scalar(y, xyzt)[:, :3]  # (N,3) только x,y,z
    lap = 0.0
    for j in range(3):
        # производная g_j по x_j (через xyzt, но берём ту же компоненту j)
        lap_j = grad_scalar(g[:, j:j+1], xyzt)[:, j:j+1]
        lap = lap + lap_j
    return lap
