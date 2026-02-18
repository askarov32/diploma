from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from rich.console import Console
from rich.pretty import pretty_repr

from pinn_te.utils.seeding import seed_all
from pinn_te.train.trainer import Trainer

console = Console()

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict).")
    return data


def _resolve_device(device_str: str) -> torch.device:
    d = device_str.lower().strip()

    if d == "cuda":
        if not torch.cuda.is_available():
            console.print("[yellow]CUDA requested but not available; falling back to CPU[/yellow]")
            return torch.device("cpu")
        return torch.device("cuda")

    if d == "mps":
        if not torch.backends.mps.is_available():
            console.print("[yellow]MPS requested but not available; falling back to CPU[/yellow]")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


def _ensure_run_dir(base_run_dir: str | Path) -> Path:
    run_dir = Path(base_run_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    full = run_dir / stamp
    full.mkdir(parents=True, exist_ok=False)
    return full


def _save_run_artifacts(run_dir: Path, cfg: Dict[str, Any]) -> None:
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    meta = {
        "created_at": datetime.now().isoformat(),
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

# Runtime

@dataclass(frozen=True)
class Runtime:
    cfg: Dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    run_dir: Path


def _prepare_runtime(cfg_path: str) -> Runtime:
    cfg = _load_yaml(cfg_path)

    # Seed
    project = cfg.get("project", {})
    seed = int(project.get("seed", 42))
    seed_all(seed)

    # Device
    device_str = str(project.get("device", "cpu"))
    device = _resolve_device(device_str)

    # Dtype
    dtype_str = str(project.get("dtype", "float32")).lower()
    dtype = torch.float32 if dtype_str == "float32" else torch.float64

    # Run directory
    outputs = cfg.get("outputs", {})
    base_run_dir = outputs.get("run_dir", "runs/exp")
    run_dir = _ensure_run_dir(base_run_dir)

    cfg.setdefault("outputs", {})
    cfg["outputs"]["run_dir_resolved"] = str(run_dir)

    _save_run_artifacts(run_dir, cfg)

    console.print("[green]Runtime prepared[/green]")
    console.print(f"  device: {device}")
    console.print(f"  dtype : {dtype}")
    console.print(f"  run_dir: {run_dir}")

    return Runtime(cfg=cfg, device=device, dtype=dtype, run_dir=run_dir)

# Train entry

def train_entry() -> None:
    parser = argparse.ArgumentParser(description="Train PINN thermoelastic model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g. configs/base.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate/print config and create run dir",
    )

    args = parser.parse_args()

    rt = _prepare_runtime(args.config)

    if args.dry_run:
        console.print("\n[bold]Resolved config:[/bold]")
        console.print(pretty_repr(rt.cfg))
        console.print("\n[yellow]Dry-run: no training executed.[/yellow]")
        return

    console.print("[cyan]Starting training...[/cyan]")

    trainer = Trainer(
        cfg=rt.cfg,
        device=rt.device,
        dtype=rt.dtype,
        run_dir=rt.run_dir,
    )

    trainer.train()

    console.print("[green]Training finished.[/green]")

# Direction evaluation entry

def eval_direction_entry() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate directional prediction from a trained run"
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Path to run directory (e.g. runs/demo/20260218-... )",
    )

    args = parser.parse_args()

    run_dir = Path(args.run)
    cfg_path = run_dir / "config.resolved.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Resolved config not found in run dir: {cfg_path}")

    cfg = _load_yaml(cfg_path)

    console.print(f"[cyan]Loaded run:[/cyan] {run_dir}")
    console.print(pretty_repr(cfg))

    # Пока заглушка — позже подключим energy-flux directional prediction
    console.print("[yellow]Directional evaluation not implemented yet.[/yellow]")
