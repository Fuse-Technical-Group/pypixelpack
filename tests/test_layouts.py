"""Reference vectors and round-trip identity for every layout (§spec:layouts).

Byte-exact golden vectors are hand-computed from the DeckLink SDK 15.3
section 3.4 pixel-format layout tables. Each layout is also checked for
``unpack(pack(x)) == x``. The 12-bit R12B / R12L layouts are exercised
across the 8-pixel / 36-byte group boundary (the historically
error-prone case).

Ported from pydecklink's test_packing.py; only the keys changed from the
SDK enum to layout names.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import frame

from pypixelpack import LAYOUTS, pack, row_bytes, unpack

# --- Surface ----------------------------------------------------------------


def test_layout_table_is_public() -> None:
    """The table names every layout with (group pixels, group bytes, bits)."""
    assert set(LAYOUTS) == {
        "argb",
        "bgra",
        "r210",
        "r10b",
        "r10l",
        "v210",
        "2vuy",
        "r12b",
        "r12l",
    }
    assert LAYOUTS["v210"] == (6, 16, 10)
    assert LAYOUTS["2vuy"] == (2, 4, 8)
    assert LAYOUTS["r12b"] == (8, 36, 12)


# --- 8-bit RGB golden vectors ----------------------------------------------


def test_argb_byte_exact() -> None:
    """ARGB memory order is A, R, G, B with alpha at peak (SDK 3.4)."""
    px = np.array([[[0x12, 0x34, 0x56]]], dtype=np.uint8)  # R, G, B
    out = pack(px, "argb", row_bytes=4)
    assert out.tolist() == [0xFF, 0x12, 0x34, 0x56]


def test_bgra_byte_exact() -> None:
    """BGRA memory order is B, G, R, A with alpha at peak (SDK 3.4)."""
    px = np.array([[[0x12, 0x34, 0x56]]], dtype=np.uint8)  # R, G, B
    out = pack(px, "bgra", row_bytes=4)
    assert out.tolist() == [0x56, 0x34, 0x12, 0xFF]


# --- 10-bit RGB golden vectors ---------------------------------------------

_RGB10 = np.array([[[768, 512, 256]]], dtype=np.uint16)  # R, G, B


def test_r210_byte_exact() -> None:
    """r210: word = (R<<20)|(G<<10)|B, big-endian (SDK 3.4)."""
    out = pack(_RGB10, "r210", row_bytes=4)
    assert out.tolist() == [0x30, 0x08, 0x01, 0x00]


def test_r10b_byte_exact() -> None:
    """R10b: word = (R<<22)|(G<<12)|(B<<2), big-endian (SDK 3.4)."""
    out = pack(_RGB10, "r10b", row_bytes=4)
    assert out.tolist() == [0xC0, 0x20, 0x04, 0x00]


def test_r10l_byte_exact() -> None:
    """R10l: same word as R10b, little-endian (SDK 3.4)."""
    out = pack(_RGB10, "r10l", row_bytes=4)
    assert out.tolist() == [0x00, 0x04, 0x20, 0xC0]


# --- 10-bit YUV (v210) golden vector ---------------------------------------


def test_v210_byte_exact() -> None:
    """v210: 6 pixels in 4 little-endian words; chroma from even pixels (SDK 3.4)."""
    y = [64, 65, 66, 67, 68, 69]
    cb = [100, 0, 101, 0, 102, 0]  # sampled at even pixels 0, 2, 4
    cr = [200, 0, 201, 0, 202, 0]
    px = np.array([[[y[i], cb[i], cr[i]] for i in range(6)]], dtype=np.uint16)
    out = pack(px, "v210", row_bytes=16)
    assert out.tolist() == [
        0x64,
        0x00,
        0x81,
        0x0C,
        0x41,
        0x94,
        0x21,
        0x04,
        0xC9,
        0x0C,
        0x61,
        0x06,
        0x44,
        0x28,
        0x53,
        0x04,
    ]


# --- 8-bit YUV (2vuy) golden vector ----------------------------------------


def test_2vuy_byte_exact() -> None:
    """2vuy: 2 pixels in 4 bytes, memory order Cb0 Y0 Cr0 Y1; chroma from the
    even pixel (SDK 3.4; CoreVideo kCVPixelFormatType_422YpCbCr8)."""
    y = [0x12, 0x78]
    cb = [0x34, 0]  # sampled at even pixel 0
    cr = [0x56, 0]
    px = np.array([[[y[i], cb[i], cr[i]] for i in range(2)]], dtype=np.uint8)
    out = pack(px, "2vuy", row_bytes=4)
    assert out.dtype == np.uint8
    assert out.tolist() == [0x34, 0x12, 0x56, 0x78]


# --- 12-bit RGB golden vectors ---------------------------------------------


def _group_pixels_first_only() -> np.ndarray:
    """One 8-pixel group: pixel 0 distinctive, pixels 1-7 zero."""
    px = np.zeros((1, 8, 3), dtype=np.uint16)
    px[0, 0] = [0xABC, 0xDEF, 0x123]  # R, G, B
    return px


def test_r12b_byte_exact() -> None:
    """R12B big-endian nibble packing for pixel 0 (SDK 3.4 table)."""
    out = pack(_group_pixels_first_only(), "r12b", row_bytes=36)
    expected = [0x00] * 36
    expected[0:4] = [0x23, 0xDE, 0xFA, 0xBC]
    expected[7] = 0x01  # B0[11:8] in low nibble of byte 7
    assert out.tolist() == expected


def test_r12l_byte_exact() -> None:
    """R12L == R12B with each 4-byte word byte-reversed (SDK 3.4)."""
    out = pack(_group_pixels_first_only(), "r12l", row_bytes=36)
    expected = [0x00] * 36
    expected[0:4] = [0xBC, 0xFA, 0xDE, 0x23]
    expected[4:8] = [0x01, 0x00, 0x00, 0x00]
    assert out.tolist() == expected


# --- Round-trip identity ----------------------------------------------------


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_round_trip(layout: str) -> None:
    """unpack(pack(x)) == x for every layout over a random frame, on the
    input's device, at the depth's dtype."""
    height, width = 5, 17  # non-multiple of any group size
    px = frame(layout, height, width, seed=1234)
    rb = row_bytes(layout, width)
    packed = pack(px, layout, row_bytes=rb, xp=np)
    assert packed.dtype == np.uint8
    assert packed.shape == (height * rb,)
    assert packed.device == px.device
    out = unpack(packed, layout, width=width, height=height, row_bytes=rb)
    assert out.dtype == px.dtype
    assert out.device == packed.device
    np.testing.assert_array_equal(out, px)


def test_row_bytes_is_the_minimum_pack_accepts() -> None:
    for layout in LAYOUTS:
        rb = row_bytes(layout, 17)
        pack(frame(layout, 1, 17), layout, row_bytes=rb)
        with pytest.raises(ValueError, match="row_bytes"):
            pack(frame(layout, 1, 17), layout, row_bytes=rb - 1)


def test_every_layout_has_a_codec() -> None:
    from pypixelpack.layouts import _CODECS

    assert set(_CODECS) == set(LAYOUTS)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_pack_output_is_contiguous(layout: str) -> None:
    """The buffer is DMA-ready: a consumer memcpy's from its raw pointer.

    A 1x1 big-endian frame with no padding used to come back as a
    negative-stride view of the flipped word.
    """
    for height, width in ((1, 1), (2, 7)):
        out = pack(frame(layout, height, width), layout, row_bytes(layout, width))
        assert out.flags.c_contiguous
        assert out.strides == (1,)


def test_row_padding_is_zero() -> None:
    """Bytes past the packed active line are zero padding."""
    px = np.full((2, 1, 3), 0xFF, dtype=np.uint8)
    out = pack(px, "argb", row_bytes=8).reshape(2, 8)
    assert out[:, 4:].tolist() == [[0] * 4] * 2


# --- 12-bit group-boundary stress ------------------------------------------

# The SDK's own description of R12B, kept as an oracle: each output byte
# is a list of placements taking ``nbits`` bits from bit ``src_lo`` of
# component ``(pixel, channel)`` into bit ``dst_lo`` of the byte. The
# library packs the same layout as a 12-bit little-endian bitstream in
# three whole-array expressions; this table is what proves it may.
_R, _G, _B = 0, 1, 2
_R12B_MAP: list[list[tuple[int, int, int, int, int]]] = [
    [(0, _B, 0, 8, 0)],
    [(0, _G, 4, 8, 0)],
    [(0, _G, 0, 4, 4), (0, _R, 8, 4, 0)],
    [(0, _R, 0, 8, 0)],
    [(1, _B, 0, 4, 4), (1, _G, 8, 4, 0)],
    [(1, _G, 0, 8, 0)],
    [(1, _R, 4, 8, 0)],
    [(1, _R, 0, 4, 4), (0, _B, 8, 4, 0)],
    [(2, _G, 4, 8, 0)],
    [(2, _G, 0, 4, 4), (2, _R, 8, 4, 0)],
    [(2, _R, 0, 8, 0)],
    [(1, _B, 4, 8, 0)],
    [(3, _G, 0, 8, 0)],
    [(3, _R, 4, 8, 0)],
    [(3, _R, 0, 4, 4), (2, _B, 8, 4, 0)],
    [(2, _B, 0, 8, 0)],
    [(4, _G, 0, 4, 4), (4, _R, 8, 4, 0)],
    [(4, _R, 0, 8, 0)],
    [(3, _B, 4, 8, 0)],
    [(3, _B, 0, 4, 4), (3, _G, 8, 4, 0)],
    [(5, _R, 4, 8, 0)],
    [(5, _R, 0, 4, 4), (4, _B, 8, 4, 0)],
    [(4, _B, 0, 8, 0)],
    [(4, _G, 4, 8, 0)],
    [(6, _R, 0, 8, 0)],
    [(5, _B, 4, 8, 0)],
    [(5, _B, 0, 4, 4), (5, _G, 8, 4, 0)],
    [(5, _G, 0, 8, 0)],
    [(7, _R, 0, 4, 4), (6, _B, 8, 4, 0)],
    [(6, _B, 0, 8, 0)],
    [(6, _G, 4, 8, 0)],
    [(6, _G, 0, 4, 4), (6, _R, 8, 4, 0)],
    [(7, _B, 4, 8, 0)],
    [(7, _B, 0, 4, 4), (7, _G, 8, 4, 0)],
    [(7, _G, 0, 8, 0)],
    [(7, _R, 4, 8, 0)],
]


def _pack_r12b_by_map(px: np.ndarray) -> np.ndarray:
    height, width, _ = px.shape
    g = px.astype(np.int64).reshape(height, width // 8, 8, 3)
    out = []
    for placements in _R12B_MAP:
        byte = np.zeros(g.shape[:2], dtype=np.int64)
        for pixel, channel, src_lo, nbits, dst_lo in placements:
            byte |= ((g[..., pixel, channel] >> src_lo) & ((1 << nbits) - 1)) << dst_lo
        out.append(byte)
    return np.stack(out, axis=-1).astype(np.uint8).reshape(-1)


@pytest.mark.parametrize("width", [8, 16, 3840])
def test_r12b_matches_the_sdk_placement_map(width: int) -> None:
    px = frame("r12b", 2, width, seed=width)
    np.testing.assert_array_equal(
        pack(px, "r12b", row_bytes=row_bytes("r12b", width)), _pack_r12b_by_map(px)
    )


@pytest.mark.parametrize("layout", ["r12b", "r12l"])
@pytest.mark.parametrize("width", [9, 15, 17, 23])  # span >= 2 groups, non-mult-of-8
def test_r12_group_boundary_round_trip(layout: str, width: int) -> None:
    """R12B / R12L round-trip across the 8-pixel / 36-byte group boundary."""
    px = frame(layout, 2, width, seed=width)
    rb = row_bytes(layout, width)
    np.testing.assert_array_equal(
        unpack(
            pack(px, layout, row_bytes=rb), layout, width=width, height=2, row_bytes=rb
        ),
        px,
    )


# --- Error handling ---------------------------------------------------------


def test_unknown_layout_raises() -> None:
    with pytest.raises(ValueError, match="yuv2"):
        pack(np.zeros((1, 1, 3), np.uint8), "yuv2", row_bytes=4)
    with pytest.raises(ValueError, match="yuv2"):
        unpack(np.zeros(4, np.uint8), "yuv2", width=1, height=1, row_bytes=4)


def test_bad_pixel_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        pack(np.zeros((4, 3), np.uint8), "argb", row_bytes=16)


def test_row_bytes_too_small_raises() -> None:
    with pytest.raises(ValueError, match="row_bytes"):
        pack(np.zeros((1, 4, 3), np.uint8), "argb", row_bytes=4)


def test_value_out_of_range_raises() -> None:
    px = np.array([[[4096, 0, 0]]], dtype=np.uint16)  # exceeds 12-bit range
    with pytest.raises(ValueError, match="12-bit"):
        pack(px, "r12b", row_bytes=36)


def test_data_too_small_raises() -> None:
    with pytest.raises(ValueError, match="too small"):
        unpack(np.zeros(4, np.uint8), "argb", width=2, height=1, row_bytes=8)
