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

# One step of ``_gradient`` (0.6 / 11) through the widest inverse
# coefficient, 1.8814, is about 0.10.
_EDGE_ERROR = 0.11

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
@pytest.mark.parametrize("subsampling", ["444", "422", "420"])
def test_decode_inverts_encode_for_every_parameter(
    matrix: str, levels: str, bits: int, subsampling: str
) -> None:
    rgb = _gradient() if subsampling != "444" else rgb_frame(4, 12, seed=bits)
    if levels == "fullprotect":
        # FULLPROTECT clamps the codes nearest black and white away; held
        # inside [0.1, 0.9], the frame keeps every component off them.
        rgb = (0.1 + 0.8 * rgb).astype(np.float32)
    kw = {"matrix": matrix, "levels": levels, "bits": bits, "subsampling": subsampling}
    back = decode(encode(rgb, **kw), **kw)
    error = np.abs(back - rgb)
    # Half a code through the widest inverse coefficient (1.8814 for
    # BT.2020), scaled by depth.
    quantisation = _ROUND_TRIP_10BIT * 2 ** (10 - bits)
    if subsampling == "444":
        assert float(error.max()) < quantisation
        return
    # Co-sited chroma on a linear gradient comes back exact up to
    # quantisation inside the frame. The last column has no sample after
    # it and repeats the one before, so it is off by the gradient's step
    # through the inverse matrix. The first sample's filter repeats the
    # edge and is off by a quarter step, which the second column inherits
    # by half.
    interior = error[:, 2:-1]
    assert float(interior.max()) < 2 * quantisation
    assert float(error.max()) < _EDGE_ERROR


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
        encode(rgb, subsampling="411")
    with pytest.raises(ValueError, match="unknown subsampling: '411'"):
        decode(encode(rgb), subsampling="411")
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


_GREY = (0.5, 0.5, 0.5)


def _impulse(width: int, at: int, height: int = 1, row: int = 0) -> np.ndarray:
    """A grey frame with one blue pixel: its chroma is the filter's response."""
    frame = np.full((height, width, 3), 0.5, dtype=np.float32)
    frame[row, at] = _BLUE
    return frame


def _pair_chroma(codes: np.ndarray, row: int = 0) -> list[list[int]]:
    """The Cb and Cr each pair carries, read from its even pixel."""
    return codes[row, 0::2, 1:].tolist()


# Blue's colour differences scaled by a filter weight, at 10-bit narrow
# range: Cb 0.5 and Cr -0.0458 times 896 about 512, rounded half to even.
_HALF, _QUARTER, _SIXTEENTH = [736, 491], [624, 502], [540, 509]
_THREE_QUARTERS, _NEUTRAL = [848, 481], [512, 512]


def test_422_sites_chroma_on_even_samples_by_default() -> None:
    """ST 2036-1 section 7.7 and BT.709 co-site 4:2:2 chroma with even luma
    samples. Each pair's chroma is [1 2 1] / 4 about its even sample, so an
    odd pixel feeds both neighbouring pairs a quarter."""
    out = encode(_impulse(8, at=3), subsampling="422")
    assert _pair_chroma(out) == [_NEUTRAL, _QUARTER, _QUARTER, _NEUTRAL]
    assert out[0, :, 0].tolist() == encode(_impulse(8, at=3))[0, :, 0].tolist()


def test_422_an_even_pixel_weighs_half_in_its_own_pair_alone() -> None:
    out = encode(_impulse(8, at=4), subsampling="422")
    assert _pair_chroma(out) == [_NEUTRAL, _NEUTRAL, _HALF, _NEUTRAL]


@pytest.mark.parametrize(("width", "at", "pair"), [(8, 0, 0), (5, 4, 2), (6, 5, 2)])
def test_422_the_filter_repeats_the_edge_sample(width: int, at: int, pair: int) -> None:
    """Off either end the filter reads the edge sample again, so a blue edge
    pixel weighs three quarters at an even edge; an odd trailing column is
    its own pair, centred on itself."""
    out = encode(_impulse(width, at=at), subsampling="422")
    expected = _THREE_QUARTERS if at % 2 == 0 else _QUARTER
    assert _pair_chroma(out)[pair] == expected
    assert out.shape == (1, width, 3)


def test_422_centred_averages_chroma_over_each_pair_and_keeps_luma() -> None:
    """``siting="centred"`` is the pair average, sited between the two
    pixels: Cb over (red, blue) is (-0.1146 + 0.5) / 2 -> 685, Cr (0.5 -
    0.0458) / 2 -> 715. Luma stays per pixel."""
    out = encode(_rgb(_RED, _BLUE), subsampling="422", siting="centred")
    assert out[0].tolist() == [[250, 685, 715], [127, 685, 715]]
    assert _pair_chroma(
        encode(_impulse(8, at=3), subsampling="422", siting="centred")
    ) == [
        _NEUTRAL,
        _HALF,
        _NEUTRAL,
        _NEUTRAL,
    ]


def test_422_centred_odd_trailing_column_is_its_own_pair() -> None:
    out = encode(_rgb(_RED, _BLUE, _GREEN), subsampling="422", siting="centred")
    assert out.shape == (1, 3, 3)
    assert out[0, 2].tolist() == [691, 167, 105]  # the 4:4:4 code for green


def test_420_sites_chroma_on_even_samples_and_lines_by_default() -> None:
    """ST 2036-1 section 7.7 co-sites 4:2:0 chroma with even samples and even
    lines: [1 2 1] / 4 along the row, then down the column. A pixel on an
    odd line and column weighs a sixteenth in the block to its upper left."""
    out = encode(_impulse(4, at=3, height=4, row=3), subsampling="420")
    assert _pair_chroma(out, row=0) == [_NEUTRAL, _NEUTRAL]
    assert _pair_chroma(out, row=2) == [_NEUTRAL, _SIXTEENTH]
    even = encode(_impulse(4, at=2, height=4, row=2), subsampling="420")
    assert _pair_chroma(even, row=2) == [_NEUTRAL, _QUARTER]
    assert _pair_chroma(even, row=0) == [_NEUTRAL, _NEUTRAL]


def test_420_centred_averages_each_two_by_two_block() -> None:
    """Pairs along each row, then the two rows, as the GPU send's first
    4:2:0 encode did."""
    out = encode(
        _impulse(4, at=3, height=4, row=3), subsampling="420", siting="centred"
    )
    assert _pair_chroma(out, row=2) == [_NEUTRAL, _QUARTER]


@pytest.mark.parametrize("siting", ["cosited", "centred"])
@pytest.mark.parametrize(("height", "width"), [(4, 6), (5, 7)])
def test_420_writes_each_blocks_chroma_to_all_four_pixels(
    siting: str, height: int, width: int
) -> None:
    """The full ``(H, W, 3)`` shape, as 4:2:2 keeps it; an odd last row or
    column is its own block."""
    out = encode(rgb_frame(height, width, seed=width), subsampling="420", siting=siting)
    assert out.shape == (height, width, 3)
    chroma = out[..., 1:]
    for y in range(height):
        for x in range(width):
            np.testing.assert_array_equal(chroma[y, x], chroma[y - y % 2, x - x % 2])
    np.testing.assert_array_equal(
        out[..., 0], encode(rgb_frame(height, width, seed=width))[..., 0]
    )


def test_unknown_siting_is_refused() -> None:
    with pytest.raises(ValueError, match="siting"):
        encode(_rgb(_RED, _BLUE), subsampling="422", siting="left")
    with pytest.raises(ValueError, match="siting"):
        decode(np.zeros((1, 2, 3), np.uint16), subsampling="422", siting="left")


def _codes(cb_rows: list[list[int]], repeat_rows: int = 1) -> np.ndarray:
    """10-bit codes at mid grey, each listed Cb written to its pair (and,
    for 4:2:0, its row pair); Cr stays neutral."""
    cb = np.repeat(np.repeat(np.array(cb_rows), 2, axis=1), repeat_rows, axis=0)
    codes = np.full((*cb.shape, 3), 512, dtype=np.uint16)
    codes[..., 1] = cb
    return codes


def test_422_cosited_decode_interpolates_the_odd_pixels() -> None:
    """A co-sited sample belongs to its even pixel; the odd pixel between
    two samples takes their mean, and one past the last takes the last."""
    got = decode(_codes([[512, 960, 512]]), subsampling="422")
    per_pixel = _codes([[512, 960, 512]])
    per_pixel[0, :, 1] = [512, 736, 960, 736, 512, 512]
    np.testing.assert_array_equal(got, decode(per_pixel))


def test_420_cosited_decode_interpolates_odd_lines_too() -> None:
    got = decode(_codes([[512, 960], [512, 512]], repeat_rows=2), subsampling="420")
    per_pixel = _codes([[512, 512], [512, 512]], repeat_rows=2)
    per_pixel[..., 1] = [
        [512, 736, 960, 960],
        [512, 624, 736, 736],
        [512, 512, 512, 512],
        [512, 512, 512, 512],
    ]
    np.testing.assert_array_equal(got, decode(per_pixel))


@pytest.mark.parametrize("subsampling", ["422", "420"])
def test_centred_decode_repeats_each_sample_over_its_block(subsampling: str) -> None:
    codes = _codes([[512, 960, 512], [960, 512, 512]], repeat_rows=2)
    np.testing.assert_array_equal(
        decode(codes, subsampling=subsampling, siting="centred"), decode(codes)
    )


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
        float(
            np.abs(
                decode(back, layout=layout)
                - decode(codes, bits=bits, subsampling=subsampling)
            ).max()
        )
        == 0
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
