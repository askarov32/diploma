import yaml
import torch

from pinn_te.models.mlp import build_model
from pinn_te.sampling.collocation import build_collocation_batches, domain_from_cfg
from pinn_te.sampling.sensors import build_sensors

cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))

device = torch.device("cpu")
dtype = torch.float32

model = build_model(cfg, device, dtype)

batches = build_collocation_batches(cfg, device, dtype)
print({k: v.shape for k, v in batches.items()})

domain = domain_from_cfg(cfg)
sensors = build_sensors(cfg, domain, device, dtype)
print("sensors:", None if sensors is None else sensors.xyz.shape)

# прогон модели
x = batches["interior"][:8]
y = model(x)
print("model out:", y.shape)
