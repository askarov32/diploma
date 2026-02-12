import torch

class Sampler:
    def __init__(self, domain, device):
        self.domain = domain
        self.device = device

    def interior(self, n):
        x = torch.rand((n,1), device=self.device)*(self.domain["x"][1]-self.domain["x"][0])+self.domain["x"][0]
        y = torch.rand((n,1), device=self.device)*(self.domain["y"][1]-self.domain["y"][0])+self.domain["y"][0]
        z = torch.rand((n,1), device=self.device)*(self.domain["z"][1]-self.domain["z"][0])+self.domain["z"][0]
        t = torch.rand((n,1), device=self.device)*(self.domain["t1"]-self.domain["t0"])+self.domain["t0"]
        return torch.cat([x,y,z,t], dim=1)

    def initial(self, n):
        x = torch.rand((n,1), device=self.device)*(self.domain["x"][1]-self.domain["x"][0])+self.domain["x"][0]
        y = torch.rand((n,1), device=self.device)*(self.domain["y"][1]-self.domain["y"][0])+self.domain["y"][0]
        z = torch.rand((n,1), device=self.device)*(self.domain["z"][1]-self.domain["z"][0])+self.domain["z"][0]
        t = torch.full((n,1), self.domain["t0"], device=self.device)
        return torch.cat([x,y,z,t], dim=1)
