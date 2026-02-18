import yaml
import torch

from pinn_te.models.mlp import build_model
from pinn_te.materials.layers import LayersMaterial
from pinn_te.sampling.collocation import build_collocation_batches
from pinn_te.physics.residuals import thermoelastic_residuals

cfg = yaml.safe_load(open("configs/base.yaml", "r", encoding="utf-8"))

device = torch.device("cpu")
dtype = torch.float32

model = build_model(cfg, device, dtype)
material = LayersMaterial.from_config(cfg, device, dtype)

b = build_collocation_batches(cfg, device, dtype)
xyzt = b["interior"][:64].clone().requires_grad_(True)

out = thermoelastic_residuals(
    model=model,
    xyzt=xyzt,
    material=material,
    heat_cfg=cfg["physics"]["heat"],
    coupling_cfg=cfg["physics"]["coupling"],
)

print("r_motion:", out["r_motion"].shape)
print("r_heat  :", out["r_heat"].shape)
print("sigma   :", out["sigma"].shape)
