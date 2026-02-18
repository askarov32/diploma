from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml


Number = Union[int, float]
Range = Tuple[float, float]


def _as_range(v: Any) -> Optional[Range]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (float(v[0]), float(v[1]))
    raise ValueError(f"Expected [min, max] or null, got: {v!r}")


def _median(r: Range) -> float:
    return 0.5 * (r[0] + r[1])


@dataclass(frozen=True)
class RockProperty:
    # храним диапазоны (если заданы)
    rho_g_cm3: Range
    Vp_km_s: Range
    Vs_km_s: Range
    pt: Optional[Range] = None
    pe: Optional[Range] = None

    def median_values(self) -> Dict[str, float]:
        return {
            "rho_g_cm3": _median(self.rho_g_cm3),
            "Vp_km_s": _median(self.Vp_km_s),
            "Vs_km_s": _median(self.Vs_km_s),
            "pt": _median(self.pt) if self.pt else float("nan"),
            "pe": _median(self.pe) if self.pe else float("nan"),
        }


class MaterialsDB:
    def __init__(self, materials: Dict[str, RockProperty]) -> None:
        self._materials = materials

    @staticmethod
    def from_yaml(path: str | Path) -> "MaterialsDB":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"materials.yaml not found: {p.resolve()}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "materials" not in data:
            raise ValueError("materials.yaml must contain top-level key: materials")
        mats = data["materials"]
        if not isinstance(mats, dict):
            raise ValueError("materials must be a mapping")

        parsed: Dict[str, RockProperty] = {}
        for name, props in mats.items():
            if not isinstance(props, dict):
                raise ValueError(f"Material {name} must be a mapping")
            rp = RockProperty(
                rho_g_cm3=_as_range(props.get("rho_g_cm3")),
                Vp_km_s=_as_range(props.get("Vp_km_s")),
                Vs_km_s=_as_range(props.get("Vs_km_s")),
                pt=_as_range(props.get("pt")),
                pe=_as_range(props.get("pe")),
            )
            parsed[str(name)] = rp

        return MaterialsDB(parsed)

    def list_names(self) -> list[str]:
        return sorted(self._materials.keys())

    def get(self, name: str) -> RockProperty:
        if name not in self._materials:
            raise KeyError(f"Unknown material: {name}. Available: {self.list_names()}")
        return self._materials[name]

    def resolve_props(self, name: str, overrides: Optional[Dict[str, Number]] = None) -> Dict[str, float]:
        """
        Возвращает одно значение для rho/Vp/Vs:
        - берём median из базы (диапазона)
        - применяем overrides (если в base.yaml заданы конкретные числа)
        """
        base = self.get(name).median_values()
        if overrides:
            for k, v in overrides.items():
                if v is None:
                    continue
                base[k] = float(v)
        return base
