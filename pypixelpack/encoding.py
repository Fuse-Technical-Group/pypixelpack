"""RGB to component samples and back (§spec:encoding).

``encode(rgb)`` takes ``(height, width, 3)`` float RGB in [0, 1] and
returns ``(height, width, 3)`` ``uint16`` ``[Y, Cb, Cr]`` codes;
``decode(ycbcr)`` returns ``float32`` RGB clamped to [0, 1]. An encoding
is a colour matrix, a level range (``narrow``, ``full``,
``fullprotect``), a chroma subsampling (``444``, ``422`` or ``420``,
co-sited with even samples or centred) and a bit depth; a layout name selects the depth and
subsampling a wire format expects. ``encode_rgb`` and ``decode_rgb`` map
the same float RGB to ``[R, G, B]`` codes at a level range and depth,
with no matrix.
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
    "decode_rgb",
    "encode",
    "encode_rgb",
    "encoding_for",
    "legal_codes",
]

# Luma coefficients (KR, KB) per matrix; KG is what remains.
MATRICES: dict[str, tuple[float, float]] = {
    "bt601": (0.299, 0.114),  # ITU-R BT.601-7 section 2.5.1
    "bt709": (0.2126, 0.0722),  # ITU-R BT.709-6 Table 3
    "bt2020": (0.2627, 0.0593),  # ITU-R BT.2020-2 Table 4
}

# Narrow range is defined from 8 bits up (BT.709-6 items 3.4 and 4.6:
# R, G, B and Y 16 to 235, chroma 16 to 240 about 128), shifted up by
# ``bits - 8``; R, G and B share the luma span. Full range is every
# code, chroma about ``2^(bits-1)`` (H.273 with VideoFullRangeFlag set).
# FULLPROTECT is full range clamped off the codes an SDI-derived
# interface reserves for timing (ST 2110-20:2022 section 7.3, SMPTE RP
# 2077:2013 section 5). Codes are carried as uint16.
_MIN_BITS, _MAX_BITS = 8, 16
_NARROW_8BIT = {"luma": (16, 235), "chroma": (16, 240), "chroma_mid": 128}
_LEVELS = ("narrow", "full", "fullprotect")
# The depths BT.709-6 item 4.7 and BT.2100-3 Table 9 state the protected
# codes at. No standard states them at another depth, and a guessed span
# would clamp a real source's picture wrongly, so other depths are refused.
_FULLPROTECT_BITS = (8, 10, 12)
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
    """Legal codes, and the scale that maps a normalised value onto them.

    The scale is separate from the legal span because FULLPROTECT clamps
    full range without rescaling it: its codes run 4..1019 at 10 bits
    while black and white stay at 0 and 1023.
    """

    luma: range
    chroma: range
    chroma_mid: int
    black: int
    luma_scale: int
    chroma_scale: int


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
    shift = bits - _MIN_BITS
    if levels == "narrow":
        (y_lo, y_hi), (c_lo, c_hi) = _NARROW_8BIT["luma"], _NARROW_8BIT["chroma"]
        luma = range(y_lo << shift, (y_hi << shift) + 1)
        chroma = range(c_lo << shift, (c_hi << shift) + 1)
        return _Span(
            luma,
            chroma,
            _NARROW_8BIT["chroma_mid"] << shift,
            luma.start,
            len(luma) - 1,
            len(chroma) - 1,
        )
    peak = (1 << bits) - 1
    legal = range(1 << bits)
    if levels == "fullprotect":
        if bits not in _FULLPROTECT_BITS:
            raise ValueError(
                f"fullprotect is defined at {_FULLPROTECT_BITS} bits, got {bits}"
            )
        # The protected codes are those whose top eight bits are all zero
        # or all one: 0 and 255 at 8 bits, 0-3 and 1020-1023 at 10, 0-15
        # and 4080-4095 at 12 (BT.709-6 item 4.7, BT.2100-3 Table 9).
        reserved = 1 << shift
        legal = range(reserved, (1 << bits) - reserved)
    return _Span(legal, legal, 1 << (bits - 1), 0, peak, peak)


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


def _quantise(xp: Any, normalised: Any, scale: int, offset: int, legal: range) -> Any:
    """Maps a normalised component to its code: scale, offset, round, clamp.

    In place after the first product — four full-frame temporaries
    become one, and the operation order is unchanged.
    """
    t = normalised * scale
    t += offset
    xp.round(t, out=t)
    xp.clip(t, legal.start, legal[-1], out=t)
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
            _quantise(xp, y, span.luma_scale, span.black, span.luma),
            _quantise(xp, cb, span.chroma_scale, span.chroma_mid, span.chroma),
            _quantise(xp, cr, span.chroma_scale, span.chroma_mid, span.chroma),
        ),
        axis=-1,
    )


def _pad_even(xp: Any, plane: Any, axis: int) -> Any:
    """Repeats the last row or column when the axis is odd, so it pairs."""
    if plane.shape[axis] % 2 == 0:
        return plane
    edge = plane[-1:] if axis == 0 else plane[:, -1:]
    return xp.concatenate((plane, edge), axis=axis)


def _along(axis: int, part: slice) -> tuple[slice, ...]:
    """The index taking ``part`` of a plane's ``axis`` and all of the other."""
    return (part, slice(None)) if axis == 0 else (slice(None), part)


_EVEN, _ODD = slice(0, None, 2), slice(1, None, 2)


def _centred(xp: Any, plane: Any, axis: int) -> Any:  # noqa: ARG001
    """Each pair's mean along ``axis``, sited between its two samples."""
    return (plane[_along(axis, _EVEN)] + plane[_along(axis, _ODD)]) * 0.5


def _cosited(xp: Any, plane: Any, axis: int) -> Any:
    """[1 2 1] / 4 about each even sample along ``axis``, sited on it.

    The sample before the first is the first again. ``(left + right) *
    0.25 + centre * 0.5`` rounds twice, at the sum and the last add; the
    products by powers of two are exact, so a fused multiply-add on a GPU
    gives the same result.
    """
    even, odd = plane[_along(axis, _EVEN)], plane[_along(axis, _ODD)]
    first = plane[_along(axis, slice(0, 1))]
    left = xp.concatenate((first, odd[_along(axis, slice(0, -1))]), axis=axis)
    t = left + odd
    t *= 0.25
    t += even * 0.5
    return t


def _interleave(xp: Any, at: Any, between: Any, axis: int, size: int) -> Any:
    """``at`` on the even positions of ``axis`` and ``between`` on the odd,
    cut to ``size``."""
    both = xp.stack((at, between), axis=axis + 1)
    shape = list(at.shape)
    shape[axis] *= 2
    return both.reshape(shape)[_along(axis, slice(0, size))]


def _repeat_up(xp: Any, samples: Any, axis: int, size: int) -> Any:
    """Each sample written to both positions of its pair along ``axis``."""
    return _interleave(xp, samples, samples, axis, size)


def _interpolate_up(xp: Any, samples: Any, axis: int, size: int) -> Any:
    """Each sample on its even position and the mean of its neighbours on
    the odd position between; the odd position past the last sample takes
    the last."""
    rest = samples[_along(axis, slice(1, None))]
    last = samples[_along(axis, slice(-1, None))]
    after = xp.concatenate((rest, last), axis=axis)
    return _interleave(xp, samples, (samples + after) * 0.5, axis, size)


_SITINGS = {"cosited": _cosited, "centred": _centred}
_UPSAMPLE = {"cosited": _interpolate_up, "centred": _repeat_up}
_DEFAULT_SITING = "cosited"


def _broadcast_chroma(
    xp: Any, y_q: Any, cb_q: Any, cr_q: Any, rows: int, columns: int
) -> Any:
    """Codes at full shape, each subsampled chroma sample written to the
    ``rows`` x ``columns`` block of pixels it covers."""
    height, width = y_q.shape
    shape = (height // rows, rows, width // columns, columns)

    def spread(plane: Any) -> Any:
        return xp.broadcast_to(plane[:, None, :, None], shape).reshape(height, width)

    return xp.stack((y_q, spread(cb_q), spread(cr_q)), axis=-1)


def _subsampled(rows: int) -> Any:
    """An assembler halving chroma along each row and, for ``rows`` of 2,
    down each column too, in the siting ``subsample`` places it.

    Chroma is filtered on the normalised value along the row, then down the
    column, quantised once at reduced size and broadcast back beside
    per-pixel luma, so its arithmetic runs once a sample rather than once a
    pixel. An odd trailing row or column is repeated, so it filters as its
    own pair.
    """

    def assemble(xp: Any, y: Any, cb: Any, cr: Any, span: _Span, subsample: Any) -> Any:
        height, width = y.shape
        y = _pad_even(xp, y, 1)
        if rows > 1:
            y = _pad_even(xp, y, 0)

        def chroma(plane: Any) -> Any:
            half = subsample(xp, _pad_even(xp, plane, 1), 1)
            if rows > 1:
                half = subsample(xp, _pad_even(xp, half, 0), 0)
            return _quantise(xp, half, span.chroma_scale, span.chroma_mid, span.chroma)

        y_q = _quantise(xp, y, span.luma_scale, span.black, span.luma)
        codes = _broadcast_chroma(xp, y_q, chroma(cb), chroma(cr), rows, 2)
        return codes[:height, :width]

    return assemble


def _codes_444_any(
    xp: Any,
    y: Any,
    cb: Any,
    cr: Any,
    span: _Span,
    subsample: Any,  # noqa: ARG001 -- 4:4:4 has no chroma to place
) -> Any:
    return _codes_444(xp, y, cb, cr, span)


_ASSEMBLE = {"444": _codes_444_any, "422": _subsampled(1), "420": _subsampled(2)}


def encode(
    rgb: Any,
    *,
    matrix: str = "bt709",
    levels: str = "narrow",
    layout: str | None = None,
    bits: int | None = None,
    subsampling: str | None = None,
    siting: str = _DEFAULT_SITING,
    xp: Any = np,
) -> Any:
    """``(height, width, 3)`` float RGB in [0, 1] to ``uint16`` ``[Y, Cb, Cr]``.

    ``layout`` selects the depth and subsampling a wire format expects
    (``"v210"`` is 10-bit 4:2:2, ``"2vuy"`` 8-bit 4:2:2); otherwise
    ``bits`` and ``subsampling`` default to 10 and ``"444"``. ``"422"``
    and ``"420"`` place chroma where ``siting`` says: ``"cosited"`` on the
    even samples (and lines), as ST 2036-1 and BT.709 site it, or
    ``"centred"`` between them by plain average. Codes clamp to the level
    range's span. On ``xp``, on the input's device.
    """
    kr, kg, kb = _matrix(matrix)
    bits, subsampling = _resolve(layout, bits, subsampling)
    span = _span(levels, bits)
    assemble = _ASSEMBLE.get(subsampling)
    if assemble is None:
        raise ValueError(f"unknown subsampling: {subsampling!r}")
    subsample = _SITINGS.get(siting)
    if subsample is None:
        raise ValueError(f"unknown siting: {siting!r}")
    r, g, b = _channels(xp, rgb)

    y_n = kr * r + kg * g + kb * b
    codes = assemble(xp, y_n, _chroma(b, y_n, kb), _chroma(r, y_n, kr), span, subsample)
    return astype(codes, xp.uint16)


def decode(
    ycbcr: Any,
    *,
    matrix: str = "bt709",
    levels: str = "narrow",
    layout: str | None = None,
    bits: int | None = None,
    subsampling: str | None = None,
    siting: str = _DEFAULT_SITING,
    xp: Any = np,
) -> Any:
    """``(height, width, 3)`` integer ``[Y, Cb, Cr]`` to ``float32`` RGB in [0, 1].

    Inverse of :func:`encode` to within half a code per component. A
    subsampled array as ``unpack`` or ``encode`` produce it carries each
    chroma sample in every pixel of its pair (and row pair, for ``"420"``);
    decode reads the sample from the pair's first pixel and places it as
    ``siting`` says: ``"cosited"`` interpolates the pixels between samples,
    ``"centred"`` repeats each sample over its pixels. ``"444"``, the
    default without a layout, reads every pixel's own chroma. Any integer
    dtype is accepted.
    """
    kr, kg, kb = _matrix(matrix)
    bits, subsampling = _resolve(layout, bits, subsampling)
    if subsampling not in _ASSEMBLE:
        raise ValueError(f"unknown subsampling: {subsampling!r}")
    upsample = _UPSAMPLE.get(siting)
    if upsample is None:
        raise ValueError(f"unknown siting: {siting!r}")
    span = _span(levels, bits)
    y, cb, cr = _channels(xp, ycbcr)

    y_n = (y - span.black) / span.luma_scale
    cb_n = (cb - span.chroma_mid) / span.chroma_scale
    cr_n = (cr - span.chroma_mid) / span.chroma_scale
    if subsampling != "444":
        height, width = y.shape

        def place(plane: Any) -> Any:
            plane = upsample(xp, plane[:, 0::2], 1, width)
            if subsampling == "420":
                plane = upsample(xp, plane[0::2], 0, height)
            return plane

        cb_n, cr_n = place(cb_n), place(cr_n)

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


def encode_rgb(
    rgb: Any, *, levels: str = "narrow", bits: int = _DEFAULT_BITS, xp: Any = np
) -> Any:
    """Maps ``(height, width, 3)`` float RGB in [0, 1] to ``uint16`` ``[R, G, B]``.

    Every component takes the luma span of ``levels`` at ``bits``: 64..940
    for 10-bit narrow range, every code for full range. Codes clamp to
    that span. On ``xp``, on the input's device.
    """
    span = _span(levels, bits)
    codes = xp.stack(
        tuple(
            _quantise(xp, c, span.luma_scale, span.black, span.luma)
            for c in _channels(xp, rgb)
        ),
        axis=-1,
    )
    return astype(codes, xp.uint16)


def decode_rgb(
    codes: Any, *, levels: str = "narrow", bits: int = _DEFAULT_BITS, xp: Any = np
) -> Any:
    """Maps ``(height, width, 3)`` integer ``[R, G, B]`` to ``float32`` RGB in [0, 1].

    Inverse of :func:`encode_rgb` to within half a code per component;
    sub-black and super-white codes clamp to 0 and 1. Any integer dtype
    is accepted.
    """
    span = _span(levels, bits)
    rgb = xp.stack(
        tuple((c - span.black) / span.luma_scale for c in _channels(xp, codes)),
        axis=-1,
    )
    xp.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb
