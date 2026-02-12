import torch

class ConstantMaterial:
    def __init__(self, params: dict):
        self.params = params

    def eval(self, xyz):
        N = xyz.shape[0]
        device = xyz.device
        return {
            k: torch.full((N,1), float(v), device=device)
            for k, v in self.params.items()
        }
