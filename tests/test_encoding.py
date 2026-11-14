"""RGB to component samples and back (§spec:encoding).

Golden codes are hand-computed from the BT.709-6 coefficients at 10-bit
narrow range, the values the seed implementation produced.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import rgb_frame

from pypixelpack import ENCODINGS, SUBSAMPLED_422, channels, pack, row_bytes, unpack
from pypixelpack.encoding import (
    decode,
    decode_rgb,
    encode,
    encode_rgb,
    encoding_for,
    legal_codes,
)

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
    [(_RED, (326, 361, 960)), (_GREEN, (578, 215, 137)), (_BLUE, (164, 960, 439))],
)
def test_bt601_narrow_golden_codes(
    rgb: tuple[float, float, float], ycbcr: tuple[int, int, int]
) -> None:
    """BT.601 KR = 0.299, KB = 0.114 (ITU-R BT.601-7 section 2.5.1)."""
    assert encode(_rgb(rgb), matrix="bt601")[0, 0].tolist() == list(ycbcr)


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
    ("rgb", "ycbcr"),
    [
        ((0.0, 0.0, 0.0), (4, 512, 512)),
        ((1.0, 1.0, 1.0), (1019, 512, 512)),
        (_RED, (217, 395, 1019)),
        (_BLUE, (74, 1019, 465)),
    ],
)
def test_bt709_fullprotect_golden_codes(
    rgb: tuple[float, float, float], ycbcr: tuple[int, int, int]
) -> None:
    """FULLPROTECT is full range clamped to 4..1019 (SMPTE RP 2077:2013
    section 5): the full-range codes above, with the protected ends taken
    out and nothing rescaled."""
    assert encode(_rgb(rgb), levels="fullprotect")[0, 0].tolist() == list(ycbcr)


@pytest.mark.parametrize("bits", [8, 10, 12])
def test_fullprotect_is_full_clamped_and_decodes_as_full(bits: int) -> None:
    """RP 2077:2013 equation (1) clamps a full-range code into the permitted
    span; equation (2) carries a permitted code back unchanged, so decode
    reads it on full range's scale."""
    rgb = rgb_frame(4, 12, seed=bits)
    span = legal_codes(levels="fullprotect", bits=bits).luma
    full = encode(rgb, levels="full", bits=bits)
    protected = encode(rgb, levels="fullprotect", bits=bits)
    np.testing.assert_array_equal(protected, np.clip(full, span[0], span[-1]))
    np.testing.assert_array_equal(
        decode(protected, levels="fullprotect", bits=bits),
        decode(protected, levels="full", bits=bits),
    )


@pytest.mark.parametrize(
    ("bits", "levels", "white", "black"),
    [
        (8, "narrow", (235, 128, 128), (16, 128, 128)),
        (10, "narrow", (940, 512, 512), (64, 512, 512)),
        (12, "narrow", (3760, 2048, 2048), (256, 2048, 2048)),
        (12, "full", (4095, 2048, 2048), (0, 2048, 2048)),
        (12, "fullprotect", (4079, 2048, 2048), (16, 2048, 2048)),
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


@pytest.mark.parametrize("matrix", ["bt601", "bt709", "bt2020"])
@pytest.mark.parametrize("levels", ["narrow", "full", "fullprotect"])
@pytest.mark.parametrize("bits", [8, 10, 12])
@pytest.mark.parametrize("subsampling", ["444", "422"])
def test_decode_inverts_encode_for_every_parameter(
    matrix: str, levels: str, bits: int, subsampling: str
) -> None:
    rgb = _gradient() if subsampling == "422" else rgb_frame(4, 12, seed=bits)
    if levels == "fullprotect":
        # FULLPROTECT clamps the codes nearest black and white away; held
        # inside [0.1, 0.9], the frame keeps every component off them.
        rgb = (0.1 + 0.8 * rgb).astype(np.float32)
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
        encode(rgb, matrix="bt470")
    with pytest.raises(ValueError, match="levels"):
        encode(rgb, levels="limited")
    for bits in (7, 17):
        with pytest.raises(ValueError, match="bits"):
            encode(rgb, bits=bits)
    with pytest.raises(ValueError, match="subsampling"):
        encode(rgb, subsampling="420")
    with pytest.raises(ValueError, match="matrix"):
        decode(np.zeros((1, 1, 3), np.uint16), matrix="bt470")


@pytest.mark.parametrize("bits", [9, 11, 14, 16])
def test_fullprotect_is_refused_where_no_standard_defines_it(bits: int) -> None:
    """BT.709-6 item 4.7 and BT.2100-3 Table 9 state the protected codes at
    8, 10 and 12 bits only, so any other depth is refused, not inferred."""
    rgb = _rgb((0.5, 0.5, 0.5))
    codes = np.zeros((1, 1, 3), np.uint16)
    for call in (
        lambda: legal_codes(levels="fullprotect", bits=bits),
        lambda: encode(rgb, levels="fullprotect", bits=bits),
        lambda: decode(codes, levels="fullprotect", bits=bits),
        lambda: encode_rgb(rgb, levels="fullprotect", bits=bits),
        lambda: decode_rgb(codes, levels="fullprotect", bits=bits),
    ):
        with pytest.raises(ValueError, match="fullprotect"):
            call()


# --- the representable set --------------------------------------------------


@pytest.mark.parametrize(
    ("levels", "bits", "luma", "chroma"),
    [
        ("narrow", 10, range(64, 941), range(64, 961)),  # 877 luma levels
        ("narrow", 8, range(16, 236), range(16, 241)),
        ("narrow", 12, range(256, 3761), range(256, 3841)),
        ("full", 10, range(0, 1024), range(0, 1024)),
        ("full", 8, range(0, 256), range(0, 256)),
        ("fullprotect", 8, range(1, 255), range(1, 255)),
        ("fullprotect", 10, range(4, 1020), range(4, 1020)),
        ("fullprotect", 12, range(16, 4080), range(16, 4080)),
    ],
)
def test_legal_codes_per_levels_and_depth(
    levels: str, bits: int, luma: range, chroma: range
) -> None:
    codes = legal_codes(levels=levels, bits=bits)
    assert (codes.luma, codes.chroma) == (luma, chroma)


def test_legal_codes_is_what_encode_produces() -> None:
    """Every code encode emits is legal, and the ends are reachable: black
    and white for luma; blue and red (Cb, Cr = +0.5), yellow and cyan
    (Cb, Cr = -0.5) for chroma."""
    rgb = _rgb(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), _BLUE, _RED, (1.0, 1.0, 0.0), (0.0, 1.0, 1.0)
    )
    for levels in ("narrow", "full", "fullprotect"):
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
    samples = codes
    if channels(layout) == 4:
        # ``encode`` stops at the colour samples; a layout that carries
        # alpha takes the caller's own plane as its fourth channel.
        alpha = np.full((*codes.shape[:2], 1), (1 << bits) - 1, dtype=codes.dtype)
        samples = np.concatenate((codes, alpha), axis=-1)
    back = unpack(pack(samples, layout, rb), layout, width, 3, rb)
    np.testing.assert_array_equal(back, samples)
    back = back[..., :3]
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


def test_chroma_mid_is_the_code_a_neutral_patch_carries() -> None:
    """Not derivable from the chroma span, which is why it is exposed.

    Narrow 10-bit chroma runs 64..960; its own midpoint is 512 only by
    coincidence of the span being symmetric, and at other depths a
    caller working it out from the bounds gets it wrong.
    """
    for bits in (8, 10, 12):
        for levels in ("narrow", "full", "fullprotect"):
            assert legal_codes(levels=levels, bits=bits).chroma_mid == 1 << (bits - 1)
    neutral = np.full((1, 1, 3), 0.5, dtype=np.float32)
    codes = encode(neutral, bits=10, levels="narrow", subsampling="444")
    mid = legal_codes(levels="narrow", bits=10).chroma_mid
    assert int(codes[0, 0, 1]) == mid and int(codes[0, 0, 2]) == mid


# --- RGB codes --------------------------------------------------------------

_LEVELS = ("narrow", "full", "fullprotect")
# Every level range at every depth it is defined at: fullprotect stops at 12.
_LEVEL_DEPTHS = [
    (levels, bits)
    for levels in _LEVELS
    for bits in (8, 10, 12, 16)
    if not (levels == "fullprotect" and bits == 16)
]


@pytest.mark.parametrize(
    ("levels", "bits", "codes"),
    [
        ("narrow", 10, (64, 502, 940)),  # 64 + 0.5 * 876
        ("narrow", 8, (16, 126, 235)),  # 16 + 0.5 * 219 = 125.5, half to even
        ("narrow", 12, (256, 2008, 3760)),
        ("full", 10, (0, 512, 1023)),  # 0.5 * 1023 = 511.5, half to even
        ("fullprotect", 10, (4, 512, 1019)),  # full range, protected ends clamped
        ("fullprotect", 8, (1, 128, 254)),
    ],
)
def test_encode_rgb_golden_codes(
    levels: str, bits: int, codes: tuple[int, int, int]
) -> None:
    """Every RGB component takes the luma span: 0, 0.5 and 1 land on that
    span's black, midpoint and white."""
    out = encode_rgb(_rgb((0.0, 0.5, 1.0)), levels=levels, bits=bits)
    assert out.dtype == np.uint16
    assert out.shape == (1, 1, 3)
    assert out[0, 0].tolist() == list(codes)


def test_encode_rgb_defaults_to_10bit_narrow() -> None:
    rgb = rgb_frame(3, 5)
    np.testing.assert_array_equal(
        encode_rgb(rgb), encode_rgb(rgb, levels="narrow", bits=10)
    )


def test_encode_rgb_clamps_to_the_legal_span() -> None:
    out = encode_rgb(_rgb((-1.0, 2.0, 0.0)))
    assert out[0, 0].tolist() == [64, 940, 64]


def test_decode_rgb_returns_float32_in_unit_range() -> None:
    codes = np.array([[[64, 940, 502], [0, 1023, 64]]], np.uint16)
    out = decode_rgb(codes)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.0, 1.0, 438 / 876], atol=1e-7)
    # Sub-black and super-white codes clamp to [0, 1].
    assert out[0, 1].tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(("levels", "bits"), _LEVEL_DEPTHS)
def test_decode_rgb_inverts_encode_rgb_within_half_a_code(
    levels: str, bits: int
) -> None:
    """Within half a code of the input, once the input is clamped to the
    values the span represents."""
    rgb = rgb_frame(4, 64, seed=bits)
    rgb[0, :3] = [[0.0] * 3, [1.0] * 3, [0.5] * 3]
    span = legal_codes(levels=levels, bits=bits).luma
    full_scale = (1 << bits) - 1
    black, scale = (0, full_scale) if levels != "narrow" else (span[0], len(span) - 1)
    lo, hi = (span[0] - black) / scale, (span[-1] - black) / scale
    back = decode_rgb(
        encode_rgb(rgb, levels=levels, bits=bits), levels=levels, bits=bits
    )
    error = np.abs(back - np.clip(rgb, lo, hi))
    assert float(error.max()) <= 0.5 / scale + 1e-6


@pytest.mark.parametrize(("levels", "bits"), _LEVEL_DEPTHS)
def test_encode_rgb_inverts_decode_rgb_for_every_legal_code(
    levels: str, bits: int
) -> None:
    """Exhaustive below 16 bits; at 16, every 61st code and both ends.

    Each channel carries the codes in a different order, so a channel
    swap would not pass.
    """
    span = legal_codes(levels=levels, bits=bits).luma
    codes = np.array(span[::61] if bits == 16 else span, dtype=np.int64)
    if bits == 16:
        codes = np.append(codes, span[-1])
    planes = (codes, codes[::-1], np.roll(codes, len(codes) // 3))
    c = np.stack(planes, axis=-1)[None].astype(np.uint16)
    back = encode_rgb(decode_rgb(c, levels=levels, bits=bits), levels=levels, bits=bits)
    np.testing.assert_array_equal(back, c)


def test_rgb_codes_refuse_bad_parameters() -> None:
    rgb = _rgb((0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="levels"):
        encode_rgb(rgb, levels="limited")
    with pytest.raises(ValueError, match="levels"):
        decode_rgb(np.zeros((1, 1, 3), np.uint16), levels="limited")
    for bits in (7, 17):
        with pytest.raises(ValueError, match="bits"):
            encode_rgb(rgb, bits=bits)
        with pytest.raises(ValueError, match="bits"):
            decode_rgb(np.zeros((1, 1, 3), np.uint16), bits=bits)
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        encode_rgb(np.zeros((2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\(height, width, 3\)"):
        decode_rgb(np.zeros((2, 2), dtype=np.uint16))
