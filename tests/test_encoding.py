"""RGB to component samples and back (§spec:encoding).

Golden codes are hand-computed from the BT.709-6 coefficients at 10-bit
narrow range, the values the seed implementation produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import rgb_frame

from pypixelpack import ENCODINGS, SUBSAMPLED_422, pack, row_bytes, unpack
from pypixelpack.encoding import decode, encode, encoding_for, legal_codes

# Half a code in each component, mapped through the widest inverse
# coefficient (Cb into B, 1.8556): 0.5/876 + 1.8556 * 0.5/896 < 2e-3.
_ROUND_TRIP_10BIT = 2e-3

_RED, _GREEN, _BLUE = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def _rgb(*pixels: tuple[float, float, float]) -> np.ndarray:
    return np.array([pixels], dtype=np.float32)  # one row


def _gradient() -> np.ndarray:
    """The seed implementation's reference case: smooth content, so 4:2:2
    averaging leaves only quantisation."""
    x = np.linspace(0.2, 0.8, 12, dtype=np.float32)
    planes = (
        np.broadcast_to(x, (4, 12)),
        np.broadcast_to(x[::-1], (4, 12)),
        np.full((4, 12), 0.5),
    )
    return np.stack(planes, axis=-1).astype(np.float32)


# --- BT.709 narrow-range golden codes ----------------------------------------


@pytest.mark.parametrize(
    ("rgb", "ycbcr"),
    [
        ((0.0, 0.0, 0.0), (64, 512, 512)),
        ((1.0, 1.0, 1.0), (940, 512, 512)),
        ((0.5, 0.5, 0.5), (502, 512, 512)),
        (_RED, (250, 409, 960)),
        (_GREEN, (691, 167, 105)),
        (_BLUE, (127, 960, 471)),
    ],
)
def test_bt709_narrow_golden_codes(
    rgb: tuple[float, float, float], ycbcr: tuple[int, int, int]
) -> None:
    out = encode(_rgb(rgb))
    assert out.dtype == np.uint16
    assert out.shape == (1, 1, 3)
    assert out[0, 0].tolist() == list(ycbcr)


def test_encode_clamps_to_the_legal_span() -> None:
    """A scene-referred frame runs past 1.0 and narrow range has nowhere
    to put it; the codes stop at the span's ends rather than wrapping."""
    out = encode(_rgb((2.0, 2.0, 2.0), (-1.0, 2.0, -1.0), (2.0, -1.0, 2.0)))
    assert out[0].tolist() == [[940, 512, 512], [940, 64, 64], [64, 960, 960]]


def test_rounding_is_half_to_even() -> None:
    """Luma 0.5/876 sits exactly on 64.5: half-even gives 64, half-up 65."""
    assert encode(_rgb((0.5 / 876, 0.5 / 876, 0.5 / 876)))[0, 0, 0] == 64


def test_decode_returns_float32_rgb_in_unit_range() -> None:
    codes = np.array([[[64, 512, 512], [940, 512, 512], [1023, 0, 0]]], np.uint16)
    out = decode(codes)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, :2], [[0.0] * 3, [1.0] * 3], atol=1e-6)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


def test_decode_accepts_any_integer_dtype() -> None:
    codes = np.array([[[502, 512, 512]]])
    for dtype in (np.uint16, np.int32, np.int64):
        np.testing.assert_allclose(decode(codes.astype(dtype))[0, 0], 0.5, atol=1e-3)


def test_encode_rejects_a_frame_without_three_channels() -> None:
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        encode(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        decode(np.zeros((2, 2), dtype=np.uint16))


# --- matrix, levels and depth -----------------------------------------------


@pytest.mark.parametrize(
    ("rgb", "ycbcr"),
    [(_RED, (294, 387, 960)), (_GREEN, (658, 189, 100)), (_BLUE, (116, 960, 476))],
)
def test_bt2020_narrow_golden_codes(
    rgb: tuple[float, float, float], ycbcr: tuple[int, int, int]
) -> None:
    """BT.2020 KR = 0.2627, KB = 0.0593 (ITU-R BT.2020-2 Table 4)."""
    assert encode(_rgb(rgb), matrix="bt2020")[0, 0].tolist() == list(ycbcr)


@pytest.mark.parametrize(
    ("rgb", "ycbcr"),
    [
        ((0.0, 0.0, 0.0), (0, 512, 512)),
        ((1.0, 1.0, 1.0), (1023, 512, 512)),
        (_RED, (217, 395, 1023)),  # Cr lands on 1023.5 and clips
        (_BLUE, (74, 1023, 465)),
    ],
)
def test_bt709_full_golden_codes(
    rgb: tuple[float, float, float], ycbcr: tuple[int, int, int]
) -> None:
    """Full range spans every code: luma 0 to 2^b - 1, chroma about 2^(b-1)."""
    assert encode(_rgb(rgb), levels="full")[0, 0].tolist() == list(ycbcr)


@pytest.mark.parametrize(
    ("bits", "levels", "white", "black"),
    [
        (8, "narrow", (235, 128, 128), (16, 128, 128)),
        (10, "narrow", (940, 512, 512), (64, 512, 512)),
        (12, "narrow", (3760, 2048, 2048), (256, 2048, 2048)),
        (12, "full", (4095, 2048, 2048), (0, 2048, 2048)),
        (16, "narrow", (60160, 32768, 32768), (4096, 32768, 32768)),
    ],
)
def test_depth_scales_the_span(
    bits: int, levels: str, white: tuple[int, ...], black: tuple[int, ...]
) -> None:
    """Narrow range is the 8-bit span shifted up by ``bits - 8``."""
    out = encode(_rgb((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), levels=levels, bits=bits)
    assert out.dtype == np.uint16
    assert out[0].tolist() == [list(white), list(black)]


@pytest.mark.parametrize("matrix", ["bt709", "bt2020"])
@pytest.mark.parametrize("levels", ["narrow", "full"])
@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("subsampling", ["444", "422"])
def test_decode_inverts_encode_for_every_parameter(
    matrix: str, levels: str, bits: int, subsampling: str
) -> None:
    rgb = _gradient() if subsampling == "422" else rgb_frame(4, 12, seed=bits)
    kw = {"matrix": matrix, "levels": levels, "bits": bits}
    back = decode(encode(rgb, subsampling=subsampling, **kw), **kw)
    # Half a code through the widest inverse coefficient (1.8814 for
    # BT.2020), scaled by depth. 4:2:2 adds the pair average's own error,
    # which is content, not quantisation, and does not shrink with depth;
    # The seed implementation's bound for that case is 0.06.
    bound = 0.06 if subsampling == "422" else _ROUND_TRIP_10BIT * 2 ** (10 - bits)
    assert float(np.abs(back - rgb).max()) < bound


def test_unknown_parameters_are_refused() -> None:
    rgb = _rgb((0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="matrix"):
        encode(rgb, matrix="bt601")
    with pytest.raises(ValueError, match="levels"):
        encode(rgb, levels="limited")
    for bits in (7, 17):
        with pytest.raises(ValueError, match="bits"):
            encode(rgb, bits=bits)
    with pytest.raises(ValueError, match="subsampling"):
        encode(rgb, subsampling="420")
    with pytest.raises(ValueError, match="matrix"):
        decode(np.zeros((1, 1, 3), np.uint16), matrix="bt601")


# --- the representable set --------------------------------------------------


@pytest.mark.parametrize(
    ("levels", "bits", "luma", "chroma"),
    [
        ("narrow", 10, range(64, 941), range(64, 961)),  # 877 luma levels
        ("narrow", 8, range(16, 236), range(16, 241)),
        ("narrow", 12, range(256, 3761), range(256, 3841)),
        ("full", 10, range(0, 1024), range(0, 1024)),
        ("full", 8, range(0, 256), range(0, 256)),
    ],
)
def test_legal_codes_per_levels_and_depth(
    levels: str, bits: int, luma: range, chroma: range
) -> None:
    assert legal_codes(levels=levels, bits=bits) == (luma, chroma)


def test_legal_codes_is_what_encode_produces() -> None:
    """Every code encode emits is legal, and the ends are reachable: black
    and white for luma; blue and red (Cb, Cr = +0.5), yellow and cyan
    (Cb, Cr = -0.5) for chroma."""
    rgb = _rgb(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), _BLUE, _RED, (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)
    )
    for levels in ("narrow", "full"):
        codes = legal_codes(levels=levels)
        out = encode(rgb, levels=levels)
        luma, chroma = out[..., 0].ravel().tolist(), out[..., 1:].ravel().tolist()
        assert set(luma) <= set(codes.luma) and set(chroma) <= set(codes.chroma)
        assert {codes.luma[0], codes.luma[-1]} <= set(luma)
        assert {codes.chroma[0], codes.chroma[-1]} <= set(chroma)


# --- chroma subsampling, and the layout that names it ----------------------


def test_422_averages_chroma_over_each_pair_and_keeps_luma() -> None:
    """Pair average happens on the normalised chroma before quantisation:
    Cb over (red, blue) is (-0.1146 + 0.5) / 2 -> 685, Cr (0.5 - 0.0458)
    / 2 -> 715. Luma stays per pixel."""
    out = encode(_rgb(_RED, _BLUE), subsampling="422")
    assert out[0].tolist() == [[250, 685, 715], [127, 685, 715]]


def test_422_odd_trailing_column_is_its_own_pair() -> None:
    out = encode(_rgb(_RED, _BLUE, _GREEN), subsampling="422")
    assert out.shape == (1, 3, 3)
    assert out[0, 2].tolist() == [691, 167, 105]  # the 4:4:4 code for green


def test_444_is_the_default() -> None:
    np.testing.assert_array_equal(
        encode(_rgb(_RED, _BLUE)), encode(_rgb(_RED, _BLUE), subsampling="444")
    )


@pytest.mark.parametrize("layout", sorted(ENCODINGS))
@pytest.mark.parametrize("width", [18, 17])
def test_layout_selects_the_encoding_the_wire_expects(layout: str, width: int) -> None:
    """`encode(layout=)` is the (bits, subsampling) the layout packs, and the
    result round-trips through that layout exactly."""
    rgb = rgb_frame(3, width, seed=width)
    bits, subsampling = encoding_for(layout)
    codes = encode(rgb, layout=layout)
    np.testing.assert_array_equal(
        codes, encode(rgb, bits=bits, subsampling=subsampling)
    )
    if layout in SUBSAMPLED_422:
        np.testing.assert_array_equal(
            codes[:, 1::2, 1:], codes[:, 0 : 2 * (width // 2) : 2, 1:]
        )
    rb = row_bytes(layout, width)
    back = unpack(pack(codes, layout, rb), layout, width, 3, rb)
    np.testing.assert_array_equal(back, codes)
    assert (
        float(np.abs(decode(back, layout=layout) - decode(codes, bits=bits)).max()) == 0
    )


def test_2vuy_carries_8bit_422() -> None:
    """The 8-bit container takes 8-bit narrow codes: white is 235, not 940."""
    out = encode(_rgb((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), layout="2vuy")
    assert out[0].tolist() == [[235, 128, 128], [16, 128, 128]]


def test_a_keyword_that_contradicts_the_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="contradict"):
        encode(_rgb(_RED), layout="v210", bits=12)
    with pytest.raises(ValueError, match="contradict"):
        encode(_rgb(_RED), layout="v210", subsampling="444")
    with pytest.raises(ValueError, match="no component encoding"):
        encode(_rgb(_RED), layout="r210")
