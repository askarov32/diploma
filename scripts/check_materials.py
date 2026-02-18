import torch
import yaml

from pinn_te.materials.layers import LayersMaterial

cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))
device = torch.device("cpu")
dtype = torch.float32

mat = LayersMaterial.from_config(cfg, device, dtype)

xyz = torch.tensor([
    [0.0, 0.0, 100.0],
    [0.0, 0.0, 900.0],
], dtype=dtype)

out = mat.eval(xyz)
print(out)
