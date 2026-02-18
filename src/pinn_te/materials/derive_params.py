from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def rho_g_cm3_to_kg_m3(rho_g_cm3: float) -> float:
    # 1 г/см3 = 1000 кг/м3
    return float(rho_g_cm3) * 1000.0


def km_s_to_m_s(v_km_s: float) -> float:
    return float(v_km_s) * 1000.0


@dataclass(frozen=True)
class ElasticSI:
    rho: float       # kg/m3
    lame_lambda: float  # Pa
    lame_mu: float      # Pa
    Vp: float        # m/s
    Vs: float        # m/s

    def as_dict(self) -> Dict[str, float]:
        return {
            "rho": self.rho,
            "lame_lambda": self.lame_lambda,
            "lame_mu": self.lame_mu,
            "Vp": self.Vp,
            "Vs": self.Vs,
        }


def derive_lame_from_vp_vs(rho_kg_m3: float, vp_m_s: float, vs_m_s: float) -> ElasticSI:
    """
    Из изотропной линейной упругости:
      mu = rho * Vs^2
      lambda = rho * (Vp^2 - 2*Vs^2)
    """
    rho = float(rho_kg_m3)
    vp = float(vp_m_s)
    vs = float(vs_m_s)

    mu = rho * (vs ** 2)
    lam = rho * (vp ** 2 - 2.0 * vs ** 2)

    if mu <= 0 or lam <= 0:
        raise ValueError(
            f"Non-physical Lamé parameters derived: lambda={lam:.3e}, mu={mu:.3e}. "
            f"Check rho/Vp/Vs."
        )

    return ElasticSI(rho=rho, lame_lambda=lam, lame_mu=mu, Vp=vp, Vs=vs)


def derive_elastic_si(props: Dict[str, float]) -> ElasticSI:
    """
    props expects:
      rho_g_cm3, Vp_km_s, Vs_km_s
    """
    rho = rho_g_cm3_to_kg_m3(props["rho_g_cm3"])
    vp = km_s_to_m_s(props["Vp_km_s"])
    vs = km_s_to_m_s(props["Vs_km_s"])
    return derive_lame_from_vp_vs(rho, vp, vs)
