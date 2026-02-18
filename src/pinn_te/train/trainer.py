from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from rich.console import Console

from pinn_te.materials.layers import LayersMaterial
from pinn_te.models.mlp import build_model
from pinn_te.physics.residuals import thermoelastic_residuals
from pinn_te.sampling.collocation import build_collocation_batches, domain_from_cfg
from pinn_te.sampling.sensors import build_sensors
from pinn_te.losses.pinn_loss import compute_pinn_loss
from pinn_te.physics.source import body_force_from_cfg

console = Console()


@dataclass
class Trainer:
    cfg: Dict
    device: torch.device
    dtype: torch.dtype
    run_dir: Path

    def __post_init__(self) -> None:
        self.model = build_model(self.cfg, self.device, self.dtype)
        self.material = LayersMaterial.from_config(self.cfg, self.device, self.dtype)

        self.domain = domain_from_cfg(self.cfg)
        self.sensors = build_sensors(self.cfg, self.domain, self.device, self.dtype)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg["training"]["lr"]))

    def step(self, batches: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        f_int = body_force_from_cfg(batches["interior"], self.cfg["source"])

        res_int = thermoelastic_residuals(
            model=self.model,
            xyzt=batches["interior"],
            material=self.material,
            heat_cfg=self.cfg["physics"]["heat"],
            coupling_cfg=self.cfg["physics"]["coupling"],
            body_force=f_int,
        )

        res_bc = thermoelastic_residuals(
            model=self.model,
            xyzt=batches["boundary"],
            material=self.material,
            heat_cfg=self.cfg["physics"]["heat"],
            coupling_cfg=self.cfg["physics"]["coupling"],
            body_force=None,
        )

        loss = compute_pinn_loss(res_int, res_bc, self.model, batches, self.cfg)

        loss.total.backward()
        self.opt.step()

        return {
            "loss_total": float(loss.total.detach().cpu().item()),
            "loss_motion": float(loss.pde_motion.detach().cpu().item()),
            "loss_heat": float(loss.pde_heat.detach().cpu().item()),
            "loss_ic": float(loss.ic.detach().cpu().item()),
            "loss_bc": float(loss.bc.detach().cpu().item()),
        }

    def train(self) -> None:
        steps = int(self.cfg["training"]["steps"])
        eval_every = int(self.cfg["outputs"].get("eval_every", 2000))

        for step in range(1, steps + 1):
            batches = build_collocation_batches(self.cfg, self.device, self.dtype)
            metrics = self.step(batches)

            if step % 50 == 0 or step == 1:
                console.print(
                    f"[step {step}/{steps}] "
                    f"total={metrics['loss_total']:.3e} "
                    f"motion={metrics['loss_motion']:.3e} "
                    f"heat={metrics['loss_heat']:.3e} "
                    f"ic={metrics['loss_ic']:.3e}"
                )

            if step % eval_every == 0:
                self.save_checkpoint(step)

        self.save_checkpoint(steps)

    def save_checkpoint(self, step: int) -> None:
        ckpt = {
            "step": step,
            "model_state": self.model.state_dict(),
            "cfg": self.cfg,
        }
        path = self.run_dir / f"checkpoint_{step:07d}.pt"
        torch.save(ckpt, path)
        console.print(f"[cyan]Saved[/cyan] {path}")
