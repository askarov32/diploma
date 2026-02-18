from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from pinn_te.materials.database import MaterialsDB
from pinn_te.materials.derive_params import derive_elastic_si


@dataclass(frozen=True)
class LayerSpec:
    name: str
    z_min: float
    z_max: float
    # входные “геологические” значения (могут быть из базы или overrides)
    props: Dict[str, float]
    # вычисленные SI параметры
    elastic_si: Dict[str, float]


class LayersMaterial:
    """
    Слоистая среда по z:
      - на вход: xyz (N,3) torch
      - на выход: параметры rho, lame_lambda, lame_mu (torch)
    Теплопараметры и coupling добавим чуть позже (аналогично).
    """

    def __init__(self, layers: List[LayerSpec], device: torch.device, dtype: torch.dtype) -> None:
        self.layers = layers
        self.device = device
        self.dtype = dtype

    @staticmethod
    def from_config(cfg: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> "LayersMaterial":
        mats_cfg = cfg["materials"]
        db_path = mats_cfg["database"]
        db = MaterialsDB.from_yaml(db_path)

        layers_in = mats_cfg["layers"]
        layers: List[LayerSpec] = []

        for layer in layers_in:
            lname = str(layer["name"])
            z_min = float(layer["z_min"])
            z_max = float(layer["z_max"])
            overrides = layer.get("props", {}) or {}

            # Если lname совпадает с ключом базы — используем базу, иначе считаем что overrides полные
            if lname in db.list_names():
                props = db.resolve_props(lname, overrides=overrides)
            else:
                # ожидание: overrides содержит rho_g_cm3, Vp_km_s, Vs_km_s
                props = {k: float(v) for k, v in overrides.items()}

            elastic = derive_elastic_si(props).as_dict()

            layers.append(
                LayerSpec(
                    name=lname,
                    z_min=z_min,
                    z_max=z_max,
                    props=props,
                    elastic_si=elastic,
                )
            )

        # проверка непересечения/сортировки
        layers = sorted(layers, key=lambda L: L.z_min)
        for i in range(1, len(layers)):
            if layers[i].z_min < layers[i - 1].z_max:
                raise ValueError("Layer intervals overlap. Fix z_min/z_max.")

        return LayersMaterial(layers=layers, device=device, dtype=dtype)

    def eval(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        xyz: (N,3) in meters
        returns dict of tensors (N,1): rho, lame_lambda, lame_mu
        """
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N,3)")

        z = xyz[:, 2:3]  # (N,1)

        # Инициализируем нулями, потом заполняем по маскам
        rho = torch.zeros_like(z, device=self.device, dtype=self.dtype)
        lam = torch.zeros_like(z, device=self.device, dtype=self.dtype)
        mu = torch.zeros_like(z, device=self.device, dtype=self.dtype)

        for L in self.layers:
            mask = (z >= L.z_min) & (z < L.z_max)
            if torch.any(mask):
                rho_val = torch.tensor(L.elastic_si["rho"], device=self.device, dtype=self.dtype)
                lam_val = torch.tensor(L.elastic_si["lame_lambda"], device=self.device, dtype=self.dtype)
                mu_val = torch.tensor(L.elastic_si["lame_mu"], device=self.device, dtype=self.dtype)

                rho = torch.where(mask, rho_val, rho)
                lam = torch.where(mask, lam_val, lam)
                mu = torch.where(mask, mu_val, mu)

        # sanity: если какие-то z не попали ни в один слой
        if torch.any(rho == 0):
            raise ValueError("Some points are outside defined layers. Extend z_min/z_max ranges.")

        return {
            "rho": rho,
            "lame_lambda": lam,
            "lame_mu": mu,
        }
