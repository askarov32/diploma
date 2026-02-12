import torch
import math

class Scenario:

    def __init__(self, cfg):
        self.cfg = cfg

    def body_force(self, xyzt):
        mech = self.cfg.get("mechanics", None)
        if not mech or mech["type"] != "body_force":
            return torch.zeros((xyzt.shape[0],3), device=xyzt.device)

        xyz = xyzt[:, :3]
        t = xyzt[:, 3:4]

        center = torch.tensor(mech["spatial"]["center"], device=xyzt.device)
        r = mech["spatial"]["radius"]

        dist2 = torch.sum((xyz - center)**2, dim=1, keepdim=True)
        spatial = torch.exp(-dist2/(2*r*r))

        f0 = mech["temporal"]["f0"]
        t0 = mech["temporal"]["t0"]
        tau = t - t0
        temporal = (1 - 2*(math.pi*f0*tau)**2) * torch.exp(-(math.pi*f0*tau)**2)

        direction = torch.tensor(mech["vector"]["value"], device=xyzt.device)
        amplitude = mech["amplitude"]

        return amplitude * spatial * temporal * direction
