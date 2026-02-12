import torch
from core.model import MLP
from core.material import ConstantMaterial
from core.scenario import Scenario
from core.sampler import Sampler
from core.pde import residual
from core.losses import motion_loss

def train(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain = cfg["domain"]["bounds"]
    domain["t0"] = cfg["domain"]["time"]["t0"]
    domain["t1"] = cfg["domain"]["time"]["t1"]

    model = MLP().to(device)

    material = ConstantMaterial(cfg["material"]["constants"])
    scenario = Scenario(cfg["sources"]["scenarios"][0])

    sampler = Sampler(domain, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10000):

        optimizer.zero_grad()

        x = sampler.interior(5000)

        r = residual(model, x, material, scenario)

        loss = motion_loss(r)

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())
