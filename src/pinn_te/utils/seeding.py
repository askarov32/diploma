from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """
    Фиксируем все источники случайности для воспроизводимости:
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # для детерминированности (может немного замедлить обучение)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
