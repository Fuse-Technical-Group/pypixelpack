"""Frames the layout suites share (§spec:layouts).

One recipe for a random frame, so the fact that v210 identity needs
chroma to agree within each horizontal pair is encoded once, and the
per-depth dtype rule with it.
"""

from __future__ import annotations

import numpy as np

from pypixelpack import LAYOUTS, SUBSAMPLED_422


def frame(layout: str, height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random ``(height, width, 3)`` frame of in-range values for ``layout``."""
    bits = LAYOUTS[layout][2]
    rng = np.random.default_rng((seed, sum(map(ord, layout))))
    dtype = np.uint8 if bits == 8 else np.uint16
    px = rng.integers(0, 1 << bits, size=(height, width, 3)).astype(dtype)
    if layout in SUBSAMPLED_422:
        pairs = 2 * (width // 2)  # an odd trailing column has no partner
        px[:, 1:pairs:2, 1:] = px[:, 0:pairs:2, 1:]
    return px


def rgb_frame(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random ``(height, width, 3)`` float32 frame in [0, 1).

    float32 is load-bearing: it is what makes the host and device paths
    produce identical codes, so no test builds one any other way.
    """
    return np.random.default_rng((seed, height, width)).random(
        (height, width, 3), dtype=np.float32
    )
