## Поднять проект

1) `pyproject.toml` + установка пакета  
2) `src/pinn_te/utils/seeding.py` и `cli.py` (загрузка yaml, device, seed)  
3) `materials/database.py` (читать configs/materials.yaml)  
4) `materials/derive_params.py` (ρ,Vp,Vs → λ,μ + конверсия единиц)  
5) `materials/layers.py` (слоистая среда: по z выдаёт параметры)  
6) `models/mlp.py`  
7) `sampling/collocation.py` (interior/bc/ic) + `sampling/sensors.py`  
8) `physics/*` (strain, stress, div, heat, residuals)  
9) `losses/pinn_loss.py`  
10) `train/trainer.py`  
11) `eval/direction.py` + `scripts/eval_direction.py`  
12) tests минимум 2–3 штуки

## Важно про табличные данные
Из таблицы (ρ, Vp, Vs) мы можем **строго вычислить** параметры Ламе:
- \(\mu = \rho V_s^2\)
- \(\lambda = \rho (V_p^2 - 2V_s^2)\)

Это идеальный “мост” между геологией и моделью: таблица → физические параметры → PDE → PINN.