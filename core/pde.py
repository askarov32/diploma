import torch

def grad_scalar(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True
    )[0]

def grad_vector(y, x):
    grads = []
    for i in range(y.shape[1]):
        gi = torch.autograd.grad(
            y[:, i],
            x,
            grad_outputs=torch.ones_like(y[:, i]),
            create_graph=True
        )[0]
        grads.append(gi.unsqueeze(1))
    return torch.cat(grads, dim=1)  # (N,3,4)

def residual(model, xyzt, material, scenario):
    xyzt.requires_grad_(True)
    pred = model(xyzt)

    u = pred[:, :3]      # (N,3)
    T = pred[:, 3:4]     # (N,1)

    mat = material.eval(xyzt[:, :3])
    rho = mat["rho"]

    du = grad_vector(u, xyzt)      # (N,3,4)
    u_t = du[:, :, 3]              # (N,3)

    du_t = grad_vector(u_t, xyzt)  # (N,3,4)
    u_tt = du_t[:, :, 3]           # (N,3)

    f = scenario.body_force(xyzt)

    r_motion = rho * u_tt - f

    return r_motion
