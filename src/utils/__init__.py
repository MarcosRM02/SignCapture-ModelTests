"""Reproducibility utilities."""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
