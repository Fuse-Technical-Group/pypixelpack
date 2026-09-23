"""Pixel layouts and wire encodings, device-free (§spec:problem).

numpy is the reference backend and the only import; any other array
namespace is the caller's to supply (§spec:package-shape).
"""

from pypixelpack.encoding import MATRICES, decode, encode, encoding_for, legal_codes
from pypixelpack.layouts import (
    ALPHA_LAYOUTS,
    ENCODINGS,
    LAYOUTS,
    SUBSAMPLED_422,
    channels,
    pack,
    row_bytes,
    unpack,
)

__all__ = [
    "ALPHA_LAYOUTS",
    "ENCODINGS",
    "LAYOUTS",
    "MATRICES",
    "SUBSAMPLED_422",
    "channels",
    "decode",
    "encode",
    "encoding_for",
    "legal_codes",
    "pack",
    "row_bytes",
    "unpack",
]
