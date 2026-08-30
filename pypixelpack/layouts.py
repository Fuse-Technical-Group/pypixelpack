"""Pack integer RGB/YUV pixel values into wire layouts (§spec:layouts).

``pack(pixels, layout, row_bytes)`` returns a 1-D ``uint8`` buffer of
``height * row_bytes``. ``unpack(data, layout, width, height, row_bytes)``
recovers pixel values from such a buffer. ``unpack(pack(x)) == x`` for
every layout.

Pixel arrays are ``(height, width, 3)`` integer arrays. For RGB layouts
the channels are ``[R, G, B]``; the alpha channel of ARGB/BGRA is written
at peak on pack and dropped on unpack. For the 4:2:2 YUV layouts v210
and 2vuy the channels are ``[Y, Cb, Cr]``; chroma is sampled from even
columns on pack and replicated across each pair on unpack, so round-trip
identity holds when chroma is equal within each horizontal pair.

Both functions take the array namespace as ``xp`` — numpy by default,
torch for a frame that stays on its device — and every array operation
is whole-array: no branch reads a pixel, so the torch path traces under
``torch.compile`` (§spec:backend).

Each layout is data: its geometry in ``LAYOUTS``, and one bit map that
both ``pack`` and ``unpack`` read, so the two directions cannot disagree.
Layouts follow the DeckLink SDK 15.3 manual section 3.4.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from pypixelpack._backend import astype, contiguous, is_compiling

__all__ = ["ENCODINGS", "LAYOUTS", "SUBSAMPLED_422", "pack", "row_bytes", "unpack"]

if sys.byteorder != "little":  # pragma: no cover
    raise ImportError(
        "pypixelpack serialises words by memory view on little-endian hosts"
    )

# Channel indices within a pixel triple.
_R, _G, _B = 0, 1, 2

# (group_pixels, group_bytes, bit_depth) per layout. bit_depth is not
# derivable from the group size (argb and r210 share (1, 4) but pack 8 vs
# 10 bits), so it is carried explicitly.
LAYOUTS: dict[str, tuple[int, int, int]] = {
    "argb": (1, 4, 8),
    "bgra": (1, 4, 8),
    "r210": (1, 4, 10),
    "r10b": (1, 4, 10),
    "r10l": (1, 4, 10),
    "v210": (6, 16, 10),
    "2vuy": (2, 4, 8),
    "r12b": (8, 36, 12),
    "r12l": (8, 36, 12),
}

# The component encoding a layout carries — (bits, subsampling) — for the
# layouts that carry one; RGB layouts take code values as given.
ENCODINGS: dict[str, tuple[int, str]] = {
    "v210": (10, "422"),
    "2vuy": (8, "422"),
}

# Layouts whose chroma is shared across each horizontal pair.
SUBSAMPLED_422: frozenset[str] = frozenset(
    name for name, (_, subsampling) in ENCODINGS.items() if subsampling == "422"
)

# Memory order of a group's bytes for the 8-bit layouts: (pixel in
# group, channel), or ``None`` for the alpha byte. On unpack a component
# with no byte of its own reads pixel 0's, which is the 4:2:2 rule.
_Y, _CB, _CR = 0, 1, 2
_BYTE_ORDER: dict[str, tuple[tuple[int, int] | None, ...]] = {
    "argb": (None, (0, _R), (0, _G), (0, _B)),
    "bgra": ((0, _B), (0, _G), (0, _R), None),
    "2vuy": ((0, _CB), (0, _Y), (0, _CR), (1, _Y)),
}

# Bit offset of R, G, B within the 32-bit word, and whether the word is
# stored big-endian. r210 packs 2:10:10:10; r10b/r10l pack 10:10:10:2.
_RGB10: dict[str, tuple[tuple[int, int, int], bool]] = {
    "r210": ((20, 10, 0), True),
    "r10b": ((22, 12, 2), True),
    "r10l": ((22, 12, 2), False),
}

# 12-bit RGB is a plain 12-bit little-endian bitstream of R0 G0 B0 R1 …,
# three bytes per pair of components; r12b stores each 32-bit word
# byte-swapped. Component pairs per 8-pixel group.
_R12_PAIRS = 12


def _layout(layout: str) -> tuple[int, int, int]:
    spec = LAYOUTS.get(layout)
    if spec is None:
        raise ValueError(f"unknown layout: {layout!r}")
    return spec


def row_bytes(layout: str, width: int) -> int:
    """The smallest ``row_bytes`` that holds ``width`` pixels of ``layout``."""
    group_px, group_bytes, _ = _layout(layout)
    return ((width + group_px - 1) // group_px) * group_bytes


_min_row_bytes = row_bytes  # `row_bytes` is also a parameter name below


def pack(pixels: Any, layout: str, row_bytes: int, *, xp: Any = np) -> Any:
    """Pack ``(height, width, 3)`` integer pixel values into ``layout``.

    Returns a 1-D ``uint8`` array of length ``height * row_bytes`` on
    ``xp``, on the input's device. ``row_bytes`` must be at least the
    packed active-line size; extra bytes are zero padding.
    """
    group_px, _, bits = _layout(layout)
    arr = xp.asarray(pixels)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"pixels must have shape (height, width, 3), got {tuple(arr.shape)}"
        )
    height, width, _ = arr.shape

    min_row = _min_row_bytes(layout, width)
    if row_bytes < min_row:
        raise ValueError(
            f"row_bytes={row_bytes} too small for width={width} "
            f"({layout!r} needs at least {min_row})"
        )

    # A host-side assertion on the unwidened input: a compiler cannot
    # trace it (see _backend). torch has no uint16 reduction, so that
    # one dtype widens to int32 for the read.
    if not is_compiling(xp) and height and width:
        probe = astype(arr, xp.int32) if arr.dtype == xp.uint16 else arr
        if int(probe.max()) > (1 << bits) - 1:
            raise ValueError(f"pixel value exceeds {bits}-bit range for {layout!r}")

    # 8-bit layouts shuffle bytes and never shift; the rest work in int64.
    src = astype(arr, xp.uint8 if bits == 8 else xp.int64)

    padded_w = -(-width // group_px) * group_px
    if padded_w != width:
        pad = xp.zeros(
            (height, padded_w - width, 3), dtype=src.dtype, device=src.device
        )
        src = xp.concatenate([src, pad], axis=1)

    packer, _ = _CODECS[layout]
    group_data = packer(xp, layout, src)  # (height, min_row)

    if row_bytes != min_row:
        pad = xp.zeros((height, row_bytes - min_row), dtype=xp.uint8, device=src.device)
        group_data = xp.concatenate([group_data, pad], axis=1)
    # A DMA consumer reads from the raw pointer, so the buffer is
    # contiguous by contract, not by the luck of a copy having happened
    # upstream: a one-word big-endian frame reaches here as a flipped view.
    return contiguous(xp, group_data.reshape(-1))


def unpack(
    data: Any,
    layout: str,
    width: int,
    height: int,
    row_bytes: int,
    *,
    xp: Any = np,
) -> Any:
    """Recover ``(height, width, 3)`` pixel values from a ``layout`` buffer.

    Inverse of :func:`pack`. Returns ``uint8`` values for 8-bit layouts and
    ``uint16`` for 10/12-bit layouts, on ``xp``, on the input's device.
    """
    buf = astype(xp.asarray(data), xp.uint8).reshape(-1)
    if buf.shape[0] < height * row_bytes:
        raise ValueError(
            f"data too small: got {buf.shape[0]} bytes, need {height * row_bytes}"
        )
    rows = buf[: height * row_bytes].reshape(height, row_bytes)
    group_data = rows[:, : _min_row_bytes(layout, width)]

    _, unpacker = _CODECS[layout]
    return unpacker(xp, layout, group_data)[:, :width, :]


# --- word serialisation -----------------------------------------------------


def _swap_word_bytes(xp: Any, b: Any) -> Any:
    """Reverse the four bytes of every 32-bit word in a byte array."""
    words = b.reshape(*b.shape[:-1], -1, 4)
    return xp.flip(words, (-1,)).reshape(b.shape)


def _words_to_bytes(xp: Any, words: Any, big_endian: bool) -> Any:
    """(..., N) int64 words -> (..., N*4) uint8, by memory view."""
    w32 = astype(words, xp.int32)  # wraps bit 31 identically on both backends
    out = w32.reshape(*w32.shape, 1).view(xp.uint8).reshape(*w32.shape[:-1], -1)
    return _swap_word_bytes(xp, out) if big_endian else out


def _bytes_to_words(xp: Any, data: Any, big_endian: bool) -> Any:
    """(..., N*4) uint8 -> (..., N) int64 words, by memory view."""
    b = _swap_word_bytes(xp, data) if big_endian else data
    quads = contiguous(xp, b.reshape(*b.shape[:-1], -1, 4))
    w32 = quads.view(xp.int32).reshape(quads.shape[:-1])
    return astype(w32, xp.int64) & 0xFFFFFFFF  # undo the sign extension


# --- per-layout codecs ------------------------------------------------------
# Every packer takes (xp, layout, src) with src (height, padded_width, 3)
# and returns (height, min_row) uint8; every unpacker is its inverse on
# (height, min_row) and returns (height, padded_width, 3).


def _pack_bytes(xp: Any, layout: str, src: Any) -> Any:
    """Every 8-bit layout is a byte shuffle read straight off the order table."""
    height = src.shape[0]
    g = src.reshape(height, -1, LAYOUTS[layout][0], 3)
    alpha = xp.full_like(g[..., 0, 0], 0xFF)
    planes = [alpha if e is None else g[..., e[0], e[1]] for e in _BYTE_ORDER[layout]]
    return xp.stack(planes, axis=-1).reshape(height, -1)


def _unpack_bytes(xp: Any, layout: str, data: Any) -> Any:
    height = data.shape[0]
    group_px, group_bytes, _ = LAYOUTS[layout]
    order = _BYTE_ORDER[layout]
    q = data.reshape(height, -1, group_bytes)
    # One stack of views, (pixel, channel) in raster order, each read from
    # its own byte or, for shared chroma, from pixel 0's.
    planes = [
        q[..., order.index((p, c) if (p, c) in order else (0, c))]
        for p in range(group_px)
        for c in (_R, _G, _B)
    ]
    return xp.stack(planes, axis=-1).reshape(height, -1, 3)


def _pack_10bit_rgb(xp: Any, layout: str, src: Any) -> Any:
    (r_lo, g_lo, b_lo), big_endian = _RGB10[layout]
    words = (src[..., _R] << r_lo) | (src[..., _G] << g_lo) | (src[..., _B] << b_lo)
    return _words_to_bytes(xp, words, big_endian)


def _unpack_10bit_rgb(xp: Any, layout: str, data: Any) -> Any:
    shifts, big_endian = _RGB10[layout]
    words = _bytes_to_words(xp, data, big_endian)
    return xp.stack(
        [astype((words >> lo) & 0x3FF, xp.uint16) for lo in shifts], axis=-1
    )


def _pack_v210(xp: Any, layout: str, src: Any) -> Any:  # noqa: ARG001 — one codec signature
    height = src.shape[0]
    g = src.reshape(height, -1, 6, 3)
    y, cb, cr = g[..., 0], g[..., 1], g[..., 2]  # chroma read at even pixels
    words = xp.stack(
        (
            cb[..., 0] | (y[..., 0] << 10) | (cr[..., 0] << 20),
            y[..., 1] | (cb[..., 2] << 10) | (y[..., 2] << 20),
            cr[..., 2] | (y[..., 3] << 10) | (cb[..., 4] << 20),
            y[..., 4] | (cr[..., 4] << 10) | (y[..., 5] << 20),
        ),
        axis=-1,
    )
    return _words_to_bytes(xp, words, False).reshape(height, -1)


def _unpack_v210(xp: Any, layout: str, data: Any) -> Any:  # noqa: ARG001 — one codec signature
    height = data.shape[0]
    words = _bytes_to_words(xp, data.reshape(height, -1, 16), False)
    w0, w1, w2, w3 = words[..., 0], words[..., 1], words[..., 2], words[..., 3]

    def field(word: Any, lo: int) -> Any:
        return astype((word >> lo) & 0x3FF, xp.uint16)

    cb0, cb2, cb4 = field(w0, 0), field(w1, 10), field(w2, 20)
    cr0, cr2, cr4 = field(w0, 20), field(w2, 0), field(w3, 10)
    # (pixel, channel) in raster order with chroma shared per pair: one
    # stack of eighteen views rather than a stack of stacks, which copies
    # every plane twice.
    out = xp.stack(
        (
            field(w0, 10), cb0, cr0, field(w1, 0), cb0, cr0,
            field(w1, 20), cb2, cr2, field(w2, 10), cb2, cr2,
            field(w3, 0), cb4, cr4, field(w3, 20), cb4, cr4,
        ),
        axis=-1,
    )  # fmt: skip
    return out.reshape(height, -1, 3)


def _pack_12bit(xp: Any, layout: str, src: Any) -> Any:
    height = src.shape[0]
    pairs = src.reshape(height, -1, _R12_PAIRS, 2)
    c0, c1 = pairs[..., 0], pairs[..., 1]
    out = xp.stack(
        (
            astype(c0 & 0xFF, xp.uint8),
            astype(((c0 >> 8) & 0xF) | ((c1 & 0xF) << 4), xp.uint8),
            astype(c1 >> 4, xp.uint8),
        ),
        axis=-1,
    ).reshape(height, -1)
    return _swap_word_bytes(xp, out) if layout == "r12b" else out


def _unpack_12bit(xp: Any, layout: str, data: Any) -> Any:
    height = data.shape[0]
    b = _swap_word_bytes(xp, data) if layout == "r12b" else data
    triples = astype(b.reshape(height, -1, _R12_PAIRS, 3), xp.int64)
    b0, b1, b2 = triples[..., 0], triples[..., 1], triples[..., 2]
    c0 = astype(b0 | ((b1 & 0xF) << 8), xp.uint16)
    c1 = astype((b1 >> 4) | (b2 << 4), xp.uint16)
    return xp.stack((c0, c1), axis=-1).reshape(height, -1, 3)


_CODECS: dict[str, tuple[Any, Any]] = {
    "argb": (_pack_bytes, _unpack_bytes),
    "bgra": (_pack_bytes, _unpack_bytes),
    "r210": (_pack_10bit_rgb, _unpack_10bit_rgb),
    "r10b": (_pack_10bit_rgb, _unpack_10bit_rgb),
    "r10l": (_pack_10bit_rgb, _unpack_10bit_rgb),
    "v210": (_pack_v210, _unpack_v210),
    "2vuy": (_pack_bytes, _unpack_bytes),
    "r12b": (_pack_12bit, _unpack_12bit),
    "r12l": (_pack_12bit, _unpack_12bit),
}
assert set(_CODECS) == set(LAYOUTS)
