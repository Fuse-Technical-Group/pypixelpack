"""Frames the layout suites share (§spec:layouts).

One recipe for a random frame, so the fact that v210 identity needs
chroma to agree within each horizontal pair is encoded once, and the
per-depth dtype rule with it.
"""

from __future__ import annotations

import os

import numpy as np
from hypothesis import settings

from pypixelpack import LAYOUTS, SUBSAMPLED_422, channels

# Hypothesis profiles for tests/test_properties.py. The default keeps CI
# within seconds; HYPOTHESIS_PROFILE=thorough draws a hundred times as many
# examples, for a local search before a release or after a codec change.
# Neither has a deadline: these properties check values, not speed, and on
# a loaded host a timed-out example would fail a correct codec.
settings.register_profile("default", max_examples=100, deadline=None, print_blob=True)
settings.register_profile("thorough", max_examples=10_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def frame(layout: str, height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random ``(height, width, channels)`` frame of in-range values for
    ``layout``: three channels, or four for a layout that carries alpha."""
    bits = LAYOUTS[layout][2]
    rng = np.random.default_rng((seed, sum(map(ord, layout))))
    dtype = np.uint8 if bits == 8 else np.uint16
    px = rng.integers(0, 1 << bits, size=(height, width, channels(layout))).astype(
        dtype
    )
    if layout in SUBSAMPLED_422:
        pairs = 2 * (width // 2)  # an odd trailing column has no partner
        px[:, 1:pairs:2, 1:3] = px[:, 0:pairs:2, 1:3]  # chroma only; alpha is per pixel
    return px


def rgb_frame(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random ``(height, width, 3)`` float32 frame in [0, 1).

    float32 is load-bearing: it is what makes the host and device paths
    produce identical codes, so no test builds one any other way.
    """
    return np.random.default_rng((seed, height, width)).random(
        (height, width, 3), dtype=np.float32
    )
