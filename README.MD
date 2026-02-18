# AI Directional Prediction of Thermoelastic Wave Propagation in Geological Media (PINN)

This repository contains a reproducible research codebase for Physics-Informed Neural Networks (PINN)
applied to coupled thermoelastic wave dynamics in heterogeneous geological media.

## What the program does
1. Solves a coupled PDE system (linear isotropic thermoelasticity + heat equation) using PINNs.
2. Supports layered media defined from rock-property tables (ρ, Vp, Vs → λ, μ, ρ).
3. Provides **directional prediction**:
   - (A) inverse mode: estimate the unknown source direction from sparse sensor data
   - (B) field mode: compute local propagation direction using the energy-flux vector

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train.py --config configs/base.yaml
python scripts/eval_direction.py --run runs/demo
