import torch

def grad(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

def residual(model, xyzt, material, scenario):
    xyzt.requires_grad_(True)
    pred = model(xyzt)

    u = pred[:, :3]
    T = pred[:, 3:4]

    mat = material.eval(xyzt[:, :3])
    rho = mat["rho"]

    u_t = grad(u, xyzt)[:, :, 3]
    u_tt = grad(u_t, xyzt)[:, :, 3]

    f = scenario.body_force(xyzt)

    r_motion = rho * u_tt - f

    return r_motion
