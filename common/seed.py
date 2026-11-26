"""Reproducibility helpers. The seed used for every run is logged with the run."""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU + CUDA). Returns the seed so it can be logged.

    deterministic=True also forces deterministic cuDNN kernels. That is slower and is
    only needed when checking bit-exact reproducibility, not for normal runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass
    return seed
