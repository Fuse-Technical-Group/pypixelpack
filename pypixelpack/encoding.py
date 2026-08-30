"""RGB to component samples and back (§spec:encoding).

``encode(rgb)`` takes ``(height, width, 3)`` float RGB in [0, 1] and
returns ``(height, width, 3)`` ``uint16`` ``[Y, Cb, Cr]`` codes;
``decode(ycbcr)`` returns ``float32`` RGB clamped to [0, 1]. An encoding
is a colour matrix, a level range (``narrow``, ``full``), a chroma
subsampling (``444``, or ``422`` by pair average) and a bit depth; a
layout name selects the depth and subsampling a wire format expects.
Both directions take the array namespace as ``xp`` (§spec:backend), and
every array operation is whole-array, so the torch path traces under
``torch.compile``.

Rounding is half to even and the arithmetic is float32 on both
backends, so host and device produce the same codes; the operation
order is the GPU render pipeline's this was seeded from, which a
byte-identity check pins.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from pypixelpack._backend import astype
from pypixelpack.layouts import ENCODINGS

__all__ = [
    "MATRICES",
    "LegalCodes",
    "decode",
    "encode",
    "encoding_for",
    "legal_codes",
]

# Luma coefficients (KR, KB) per matrix; KG is what remains.
MATRICES: dict[str, tuple[float, float]] = {
    "bt709": (0.2126, 0.0722),  # ITU-R BT.709-6 Table 3
    "bt2020": (0.2627, 0.0593),  # ITU-R BT.2020-2 Table 4
}

# Narrow range is defined from 8 bits up (BT.709-6 section 4.4: luma 16
# to 235, chroma 16 to 240 about 128), shifted up by ``bits - 8``. Full
# range is every code, chroma about ``2^(bits-1)`` (H.273 with
# VideoFullRangeFlag set). Codes are carried as uint16.
_MIN_BITS, _MAX_BITS = 8, 16
_NARROW_8BIT = {"luma": (16, 235), "chroma": (16, 240), "chroma_mid": 128}
_LEVELS = ("narrow", "full")
_DEFAULT_BITS, _DEFAULT_SUBSAMPLING = 10, "444"


class LegalCodes(NamedTuple):
    """The integer codes a level range can represent at a depth."""

    luma: range
    chroma: range
    chroma_mid: int
    """The code carrying no colour difference — a neutral patch's Cb and Cr.

    Exposed because it cannot be derived from `chroma` alone: narrow
    10-bit chroma runs 64..960, whose midpoint is 512 rather than the 512
    a caller would get from the range's own bounds. A consumer driving
    neutral codes onto the wire needs it, and re-deriving `1 << (bits - 1)`
    outside this library puts the wire's arithmetic somewhere that does
    not own it.
    """


class _Span(NamedTuple):
    """Legal luma and chroma codes, and the achromatic chroma code."""

    luma: range
    chroma: range
    chroma_mid: int


def _matrix(name: str) -> tuple[float, float, float]:
    coefficients = MATRICES.get(name)
    if coefficients is None:
        raise ValueError(f"unknown matrix: {name!r}")
    kr, kb = coefficients
    return kr, 1.0 - kr - kb, kb


def _span(levels: str, bits: int) -> _Span:
    if levels not in _LEVELS:
        raise ValueError(f"unknown levels: {levels!r}")
    if not _MIN_BITS <= bits <= _MAX_BITS:
        raise ValueError(f"bits must be within {_MIN_BITS}..{_MAX_BITS}, got {bits}")
    if levels == "narrow":
        shift = bits - _MIN_BITS
        (y_lo, y_hi), (c_lo, c_hi) = _NARROW_8BIT["luma"], _NARROW_8BIT["chroma"]
        return _Span(
            range(y_lo << shift, (y_hi << shift) + 1),
            range(c_lo << shift, (c_hi << shift) + 1),
            _NARROW_8BIT["chroma_mid"] << shift,
        )
    every = range(1 << bits)
    return _Span(every, every, 1 << (bits - 1))


def legal_codes(*, levels: str = "narrow", bits: int = _DEFAULT_BITS) -> LegalCodes:
    """The luma and chroma codes ``levels`` represents at ``bits``.

    ``len(legal_codes().luma)`` is 877: narrow range cannot represent
    every 10-bit code, and a caller driving exact values needs to know
    which ones survive.
    """
    span = _span(levels, bits)
    return LegalCodes(span.luma, span.chroma, span.chroma_mid)


def encoding_for(layout: str) -> tuple[int, str]:
    """The ``(bits, subsampling)`` a component layout carries."""
    encoding = ENCODINGS.get(layout)
    if encoding is None:
        raise ValueError(f"{layout!r} carries no component encoding")
    return encoding


def _resolve(
    layout: str | None, bits: int | None, subsampling: str | None
) -> tuple[int, str]:
    """Depth and subsampling from a layout, explicit keywords, or defaults.

    A layout names what the wire expects; a keyword that contradicts it
    is refused rather than silently overridden.
    """
    if layout is not None:
        l_bits, l_sub = encoding_for(layout)
        if bits not in (None, l_bits) or subsampling not in (None, l_sub):
            raise ValueError(
                f"{layout!r} carries {l_bits}-bit {l_sub}; "
                f"bits={bits!r}, subsampling={subsampling!r} contradict it"
            )
        return l_bits, l_sub
    return (
        _DEFAULT_BITS if bits is None else bits,
        _DEFAULT_SUBSAMPLING if subsampling is None else subsampling,
    )


def _channels(xp: Any, pixels: Any) -> tuple[Any, Any, Any]:
    arr = xp.asarray(pixels)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"pixels must have shape (height, width, 3), got {tuple(arr.shape)}"
        )
    arr = astype(arr, xp.float32)
    return arr[..., 0], arr[..., 1], arr[..., 2]


def _quantise(xp: Any, normalised: Any, span: range, offset: int) -> Any:
    """A normalised component to its code: scale, offset, round, clamp.

    In place after the first product — four full-frame temporaries
    become one, and the operation order is unchanged.
    """
    t = normalised * (len(span) - 1)
    t += offset
    xp.round(t, out=t)
    xp.clip(t, span.start, span[-1], out=t)
    return t


def _chroma(component: Any, y_n: Any, k: float) -> Any:
    """``0.5 * (component - y) / (1 - k)`` in bm's operation order, in place."""
    t = component - y_n
    t *= 0.5
    t /= 1.0 - k
    return t


def _codes_444(xp: Any, y: Any, cb: Any, cr: Any, span: _Span) -> Any:
    return xp.stack(
        (
            _quantise(xp, y, span.luma, span.luma.start),
            _quantise(xp, cb, span.chroma, span.chroma_mid),
            _quantise(xp, cr, span.chroma, span.chroma_mid),
        ),
        axis=-1,
    )


def _codes_422(xp: Any, y: Any, cb: Any, cr: Any, span: _Span) -> Any:
    """Chroma averaged over each horizontal pair and written to both pixels.

    Averaged on the normalised value, quantised at half width, then
    broadcast into pair space beside per-pixel luma, so the chroma
    arithmetic runs once per pair rather than once per pixel. An odd
    trailing column is its own pair.
    """
    height, width = y.shape
    if width % 2:
        y, cb, cr = (xp.concatenate((p, p[:, -1:]), axis=1) for p in (y, cb, cr))
    pairs = y.shape[1] // 2

    def halve(plane: Any) -> Any:
        p = plane.reshape(height, pairs, 2)
        return (p[..., 0] + p[..., 1]) * 0.5

    y_q = _quantise(xp, y, span.luma, span.luma.start).reshape(height, pairs, 2)
    cb_q = _quantise(xp, halve(cb), span.chroma, span.chroma_mid)
    cr_q = _quantise(xp, halve(cr), span.chroma, span.chroma_mid)
    shape = (height, pairs, 2)
    codes = xp.stack(
        (
            y_q,
            xp.broadcast_to(cb_q[..., None], shape),
            xp.broadcast_to(cr_q[..., None], shape),
        ),
        axis=-1,
    )
    return codes.reshape(height, pairs * 2, 3)[:, :width]


_ASSEMBLE = {"444": _codes_444, "422": _codes_422}


def encode(
    rgb: Any,
    *,
    matrix: str = "bt709",
    levels: str = "narrow",
    layout: str | None = None,
    bits: int | None = None,
    subsampling: str | None = None,
    xp: Any = np,
) -> Any:
    """``(height, width, 3)`` float RGB in [0, 1] to ``uint16`` ``[Y, Cb, Cr]``.

    ``layout`` selects the depth and subsampling a wire format expects
    (``"v210"`` is 10-bit 4:2:2, ``"2vuy"`` 8-bit 4:2:2); otherwise
    ``bits`` and ``subsampling`` default to 10 and ``"444"``. Codes clamp to the level range's span.
    On ``xp``, on the input's device.
    """
    kr, kg, kb = _matrix(matrix)
    bits, subsampling = _resolve(layout, bits, subsampling)
    span = _span(levels, bits)
    assemble = _ASSEMBLE.get(subsampling)
    if assemble is None:
        raise ValueError(f"unknown subsampling: {subsampling!r}")
    r, g, b = _channels(xp, rgb)

    y_n = kr * r + kg * g + kb * b
    codes = assemble(xp, y_n, _chroma(b, y_n, kb), _chroma(r, y_n, kr), span)
    return astype(codes, xp.uint16)


def decode(
    ycbcr: Any,
    *,
    matrix: str = "bt709",
    levels: str = "narrow",
    layout: str | None = None,
    bits: int | None = None,
    xp: Any = np,
) -> Any:
    """``(height, width, 3)`` integer ``[Y, Cb, Cr]`` to ``float32`` RGB in [0, 1].

    Inverse of :func:`encode` to within half a code per component. Per
    pixel: a 4:2:2 array as ``unpack`` or ``encode`` produce already has
    its chroma in both pixels of each pair. Any integer dtype is accepted.
    """
    kr, kg, kb = _matrix(matrix)
    bits, _ = _resolve(layout, bits, None)
    span = _span(levels, bits)
    y, cb, cr = _channels(xp, ycbcr)

    y_n = (y - span.luma.start) / (len(span.luma) - 1)
    cb_n = (cb - span.chroma_mid) / (len(span.chroma) - 1)
    cr_n = (cr - span.chroma_mid) / (len(span.chroma) - 1)

    # Inverse matrix coefficients, YCbCr -> RGB; accumulated in place
    # with the addition order kept.
    red = (2.0 * (1.0 - kr)) * cr_n
    red += y_n
    green = (-2.0 * kb * (1.0 - kb) / kg) * cb_n
    green += y_n
    green += (-2.0 * kr * (1.0 - kr) / kg) * cr_n
    blue = (2.0 * (1.0 - kb)) * cb_n
    blue += y_n

    rgb = xp.stack((red, green, blue), axis=-1)
    xp.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb
