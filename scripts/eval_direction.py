from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from pinn_te.materials.layers import LayersMaterial
from pinn_te.models.mlp import build_model
from pinn_te.sampling.collocation import domain_from_cfg
from pinn_te.sampling.sensors import build_sensors
from pinn_te.eval.direction import directional_prediction_energy_flux


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="Run dir, e.g. runs/demo/2026.../ ")
    ap.add_argument("--checkpoint", type=str, default="", help="Checkpoint filename. Default: latest in run dir")
    ap.add_argument("--K", type=int, default=80, help="Number of time samples")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg_path = run_dir / "config.resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # device/dtype
    proj = cfg.get("project", {})
    device_str = str(proj.get("device", "cpu")).lower()
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float32 if str(proj.get("dtype", "float32")).lower() == "float32" else torch.float64

    # model
    model = build_model(cfg, device, dtype)

    # checkpoint
    if args.checkpoint:
        ckpt_path = run_dir / args.checkpoint
    else:
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found in run dir.")
        ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {ckpt_path.name} (step={ckpt.get('step')})")

    # material + sensors
    material = LayersMaterial.from_config(cfg, device, dtype)

    domain = domain_from_cfg(cfg)
    sensors = build_sensors(cfg, domain, device, dtype)
    if sensors is None:
        raise ValueError("Sensors are disabled in config. Enable sampling.sensors.enabled=true")

    # time grid
    t0, t1 = float(cfg["domain"]["t"][0]), float(cfg["domain"]["t"][1])
    t_grid = torch.linspace(t0, t1, steps=args.K, device=device, dtype=dtype)

    true_dir = torch.tensor(cfg["source"]["direction"], device=device, dtype=dtype)

    res = directional_prediction_energy_flux(
        model=model,
        material=material,
        cfg=cfg,
        receivers_xyz=sensors.xyz.to(device=device, dtype=dtype),
        t_grid=t_grid,
        device=device,
        dtype=dtype,
        true_dir=true_dir,
    )

    print("Predicted direction (unit):", res.dir_final.detach().cpu().numpy())
    if res.angle_deg is not None:
        print(f"Angle error (deg): {res.angle_deg:.3f}")

    # Дополнительно: покажем момент пика энергии
    k_max = int(torch.argmax(res.energy_t[:, 0]).detach().cpu().item())
    print(f"Peak energy at t={float(t_grid[k_max].detach().cpu().item()):.4f} s")


if __name__ == "__main__":
    main()
