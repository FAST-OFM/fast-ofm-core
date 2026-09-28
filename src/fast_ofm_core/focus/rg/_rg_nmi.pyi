"""Typed interface to the optional exhaustive NMI extension."""

import numpy as np

def candidates(
    first: np.ndarray,
    second: np.ndarray,
    support: np.ndarray,
    bins: int,
    maximum_x: int,
    maximum_y: int,
    required_fraction: float,
) -> list[tuple[float, int, int]]: ...
