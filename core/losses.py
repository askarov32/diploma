import torch

def motion_loss(residual):
    return torch.mean(residual**2)
