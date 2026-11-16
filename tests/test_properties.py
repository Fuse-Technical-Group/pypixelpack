"""Properties of the layouts and encodings over generated inputs
(§spec:layouts, §spec:encoding).

The example suites pin bytes against the SDK manual at shapes chosen by
hand. These draw the shape, the row padding and the values, so a defect at
a group boundary, an odd width or the top of a component's range is found
and shrunk to the smallest frame that shows it. CI runs the default
profile; ``HYPOTHESIS_PROFILE=thorough`` searches a hundred times longer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from test_layouts import _pack_r12b_by_map

from pypixelpack import (
    ALPHA_LAYOUTS,
    LAYOUTS,
    MATRICES,
    SUBSAMPLED_422,
    channels,
    decode,
    decode_rgb,
    encode,
    encode_rgb,
    legal_codes,
    pack,
    row_bytes,
    unpack,
)

_LAYOUT_NAMES = sorted(LAYOUTS)
# Wide enough to span two groups of every layout but ay10's 64 pixels, and
# ay10's boundary with them; tall enough that a row-indexing slip shows.
_MAX_WIDTH, _MAX_HEIGHT = 140, 3
_MAX_EXTRA_ROW_BYTES = 40
_LEVELS = ("narrow", "full", "fullprotect")
_DEPTHS = (8, 10, 12, 16)
_FULLPROTECT_DEPTHS = (8, 10, 12)
_SUBSAMPLINGS = ("444", "422", "420")
_SITINGS = ("cosited", "centred")
# float32 arithmetic in the encode and decode paths: the slack a
# round trip may carry beyond its half-code quantisation.
_FLOAT_SLACK = 1e-5


def _bits(layout: str) -> int:
    return LAYOUTS[layout][2]


def _dtype(layout: str) -> type[np.unsignedinteger[Any]]:
    return np.uint8 if _bits(layout) == 8 else np.uint16


def _pair_chroma(layout: str, px: np.ndarray) -> np.ndarray:
    """Chroma copied from each even column to the odd one beside it, the
    frame a 4:2:2 layout round-trips exactly."""
    if layout in SUBSAMPLED_422:
        pairs = 2 * (px.shape[1] // 2)
        px[:, 1:pairs:2, 1:3] = px[:, 0:pairs:2, 1:3]
    return px


@st.composite
def frames(
    draw: st.DrawFn, layout: str, min_height: int = 1, min_width: int = 1
) -> np.ndarray:
    """A frame of in-range values for ``layout``, its shape drawn too."""
    height = draw(st.integers(min_height, _MAX_HEIGHT))
    width = draw(st.integers(min_width, _MAX_WIDTH))
    px = draw(
        hnp.arrays(
            _dtype(layout),
            (height, width, channels(layout)),
            elements=st.integers(0, (1 << _bits(layout)) - 1),
        )
    )
    return _pair_chroma(layout, px)


@st.composite
def layout_frames(draw: st.DrawFn) -> tuple[str, np.ndarray, int]:
    """A layout, a frame for it, and a row size at or past the minimum."""
    layout = draw(st.sampled_from(_LAYOUT_NAMES))
    px = draw(frames(layout))
    extra = draw(st.integers(0, _MAX_EXTRA_ROW_BYTES))
    return layout, px, row_bytes(layout, px.shape[1]) + extra


# --- layouts --------------------------------------------------------------


@given(layout_frames())
def test_unpack_inverts_pack(case: tuple[str, np.ndarray, int]) -> None:
    layout, px, rb = case
    height, width, _ = px.shape
    out = unpack(pack(px, layout, rb), layout, width=width, height=height, row_bytes=rb)
    assert out.dtype == px.dtype
    np.testing.assert_array_equal(out, px)


@given(layout_frames())
def test_a_packed_frame_is_whole_rows_of_zero_padded_bytes(
    case: tuple[str, np.ndarray, int],
) -> None:
    """Every row is its active bytes then zeros, the buffer contiguous."""
    layout, px, rb = case
    height, width, _ = px.shape
    out = pack(px, layout, rb)
    assert out.dtype == np.uint8
    assert out.shape == (height * rb,)
    assert out.flags.c_contiguous
    rows = out.reshape(height, rb)
    assert not rows[:, row_bytes(layout, width) :].any()


@given(layout_frames())
def test_the_active_bytes_do_not_depend_on_the_padding(
    case: tuple[str, np.ndarray, int],
) -> None:
    layout, px, rb = case
    height, width, _ = px.shape
    active = row_bytes(layout, width)
    tight = pack(px, layout, active).reshape(height, active)
    padded = pack(px, layout, rb).reshape(height, rb)
    np.testing.assert_array_equal(padded[:, :active], tight)


@given(layout_frames(), st.data())
def test_unpack_reads_nothing_past_the_active_line(
    case: tuple[str, np.ndarray, int], data: st.DataObject
) -> None:
    """Whatever a sender leaves in a row's padding, the pixels are the same."""
    layout, px, rb = case
    height, width, _ = px.shape
    active = row_bytes(layout, width)
    rows = pack(px, layout, rb).reshape(height, rb).copy()
    rows[:, active:] = data.draw(
        hnp.arrays(np.uint8, (height, rb - active)), label="padding"
    )
    out = unpack(rows.reshape(-1), layout, width=width, height=height, row_bytes=rb)
    np.testing.assert_array_equal(out, px)


@given(st.sampled_from(_LAYOUT_NAMES), st.data())
def test_a_change_to_one_pixel_stays_in_its_group(
    layout: str, data: st.DataObject
) -> None:
    """One component changed alters only the bytes of the group holding its
    pixel: no field of the codec spills into a neighbour's."""
    px = data.draw(frames(layout, min_width=2), label="frame")
    height, width, nch = px.shape
    row = data.draw(st.integers(0, height - 1), label="row")
    column = data.draw(st.integers(0, width - 1), label="column")
    channel = data.draw(st.integers(0, nch - 1), label="channel")
    value = data.draw(st.integers(0, (1 << _bits(layout)) - 1), label="value")
    assume(px[row, column, channel] != value)
    changed = px.copy()
    changed[row, column, channel] = value
    if layout in SUBSAMPLED_422 and channel in (1, 2):
        changed = _pair_chroma(layout, changed)
        partner = column ^ 1
        if partner < width:
            changed[row, partner, channel] = value
    rb = row_bytes(layout, width)
    before = pack(px, layout, rb).reshape(height, rb)
    after = pack(changed, layout, rb).reshape(height, rb)
    group_px, group_bytes, _ = LAYOUTS[layout]
    first = (column // group_px) * group_bytes
    differs = np.flatnonzero((before != after).reshape(-1))
    assert differs.size
    assert set(differs) <= set(range(row * rb + first, row * rb + first + group_bytes))


@given(st.integers(1, 3), st.integers(1, 24).map(lambda groups: 8 * groups), st.data())
def test_r12b_follows_the_sdk_placement_map_at_any_group_count(
    height: int, width: int, data: st.DataObject
) -> None:
    px = data.draw(
        hnp.arrays(np.uint16, (height, width, 3), elements=st.integers(0, 4095))
    )
    np.testing.assert_array_equal(
        pack(px, "r12b", row_bytes("r12b", width)), _pack_r12b_by_map(px)
    )


@given(st.sampled_from(_LAYOUT_NAMES), st.data())
def test_a_row_one_byte_short_is_refused_both_ways(
    layout: str, data: st.DataObject
) -> None:
    px = data.draw(frames(layout), label="frame")
    height, width, _ = px.shape
    short = row_bytes(layout, width) - 1
    with pytest.raises(ValueError, match="row_bytes"):
        pack(px, layout, short)
    with pytest.raises(ValueError, match="row_bytes"):
        unpack(
            np.zeros(height * (short + 1), np.uint8),
            layout,
            width=width,
            height=height,
            row_bytes=short,
        )


@given(st.sampled_from(_LAYOUT_NAMES), st.data())
def test_any_value_outside_the_depth_is_refused(
    layout: str, data: st.DataObject
) -> None:
    """One component past either end of the range, wherever it sits,
    refuses the frame."""
    px = data.draw(frames(layout), label="frame").astype(np.int32)
    height, width, nch = px.shape
    peak = (1 << _bits(layout)) - 1
    where = (
        data.draw(st.integers(0, height - 1), label="row"),
        data.draw(st.integers(0, width - 1), label="column"),
        data.draw(st.integers(0, nch - 1), label="channel"),
    )
    px[where] = data.draw(
        st.one_of(st.integers(-(1 << 20), -1), st.integers(peak + 1, 1 << 20)),
        label="value",
    )
    with pytest.raises(ValueError, match=rf"{_bits(layout)}-bit range 0\.\.{peak} "):
        pack(px, layout, row_bytes(layout, width))


@given(st.sampled_from(sorted(ALPHA_LAYOUTS)), st.data())
def test_alpha_travels_per_pixel(layout: str, data: st.DataObject) -> None:
    """Each pixel's alpha comes back as its own, with no pairing rule."""
    px = data.draw(frames(layout), label="frame")
    height, width, _ = px.shape
    rb = row_bytes(layout, width)
    out = unpack(pack(px, layout, rb), layout, width=width, height=height, row_bytes=rb)
    np.testing.assert_array_equal(out[..., 3], px[..., 3])


# --- encodings ------------------------------------------------------------


@st.composite
def level_depths(draw: st.DrawFn) -> tuple[str, int]:
    levels = draw(st.sampled_from(_LEVELS))
    depths = _FULLPROTECT_DEPTHS if levels == "fullprotect" else _DEPTHS
    return levels, draw(st.sampled_from(depths))


def _rgb(
    min_value: float = 0.0, max_value: float = 1.0
) -> st.SearchStrategy[np.ndarray]:
    shapes = st.tuples(st.integers(1, 6), st.integers(1, 9), st.just(3))
    return shapes.flatmap(
        lambda shape: hnp.arrays(
            np.float32,
            shape,
            elements=st.floats(min_value, max_value, width=32, allow_nan=False),
        )
    )


@given(
    _rgb(-0.5, 1.5),
    st.sampled_from(sorted(MATRICES)),
    level_depths(),
    st.sampled_from(_SUBSAMPLINGS),
    st.sampled_from(_SITINGS),
)
def test_every_code_is_legal_whatever_the_input(
    rgb: np.ndarray,
    matrix: str,
    level_depth: tuple[str, int],
    subsampling: str,
    siting: str,
) -> None:
    """Out-of-range RGB clamps onto the span, never past it."""
    levels, bits = level_depth
    codes = encode(
        rgb,
        matrix=matrix,
        levels=levels,
        bits=bits,
        subsampling=subsampling,
        siting=siting,
    )
    legal = legal_codes(levels=levels, bits=bits)
    assert codes.dtype == np.uint16
    assert codes.shape == rgb.shape
    assert codes[..., 0].min() >= legal.luma.start
    assert codes[..., 0].max() <= legal.luma[-1]
    assert codes[..., 1:].min() >= legal.chroma.start
    assert codes[..., 1:].max() <= legal.chroma[-1]


@given(_rgb(), st.sampled_from(sorted(MATRICES)), level_depths())
def test_decode_inverts_encode_within_half_a_code(
    rgb: np.ndarray, matrix: str, level_depth: tuple[str, int]
) -> None:
    """Half a code of error in each of Y, Cb and Cr, carried through the
    inverse matrix, bounds each RGB component's error."""
    levels, bits = level_depth
    assume(levels != "fullprotect")  # its clamp moves codes by more than half
    kr, kb = MATRICES[matrix]
    kg = 1.0 - kr - kb
    legal = legal_codes(levels=levels, bits=bits)
    e_y = 0.5 / (len(legal.luma) - 1)
    e_c = 0.5 / (len(legal.chroma) - 1)
    bound = np.array(
        [
            e_y + 2 * (1 - kr) * e_c,
            e_y + (2 * kb * (1 - kb) + 2 * kr * (1 - kr)) / kg * e_c,
            e_y + 2 * (1 - kb) * e_c,
        ]
    )
    codes = encode(rgb, matrix=matrix, levels=levels, bits=bits)
    back = decode(codes, matrix=matrix, levels=levels, bits=bits)
    assert back.dtype == np.float32
    assert (np.abs(back - rgb) <= bound + _FLOAT_SLACK).all()


@given(
    st.integers(1, 6),
    st.integers(1, 9),
    st.floats(0, 1, width=32),
    st.sampled_from(sorted(MATRICES)),
    level_depths(),
)
def test_a_grey_carries_no_colour_difference(
    height: int,
    width: int,
    value: float,
    matrix: str,
    level_depth: tuple[str, int],
) -> None:
    levels, bits = level_depth
    rgb = np.full((height, width, 3), value, np.float32)
    codes = encode(rgb, matrix=matrix, levels=levels, bits=bits)
    mid = legal_codes(levels=levels, bits=bits).chroma_mid
    assert (codes[..., 1:] == mid).all()


@given(
    st.integers(1, 6),
    st.integers(1, 9),
    st.lists(st.floats(0, 1, width=32), min_size=3, max_size=3),
    st.sampled_from(sorted(MATRICES)),
    st.sampled_from(("422", "420")),
    st.sampled_from(_SITINGS),
)
def test_a_flat_frame_subsamples_to_its_full_resolution_codes(
    height: int,
    width: int,
    colour: list[float],
    matrix: str,
    subsampling: str,
    siting: str,
) -> None:
    """A filter of equal samples is that sample, at any width or height."""
    rgb = np.broadcast_to(np.array(colour, np.float32), (height, width, 3)).copy()
    full = encode(rgb, matrix=matrix)
    reduced = encode(rgb, matrix=matrix, subsampling=subsampling, siting=siting)
    np.testing.assert_array_equal(reduced, full)


@given(_rgb(), st.sampled_from(("422", "420")), st.sampled_from(_SITINGS))
def test_subsampled_chroma_is_one_sample_per_block(
    rgb: np.ndarray, subsampling: str, siting: str
) -> None:
    """Luma is the full-resolution luma; chroma repeats across each pair,
    and down each row pair for 4:2:0, odd edges included."""
    codes = encode(rgb, subsampling=subsampling, siting=siting)
    np.testing.assert_array_equal(codes[..., 0], encode(rgb)[..., 0])
    chroma = codes[..., 1:]
    width = chroma.shape[1]
    np.testing.assert_array_equal(
        chroma[:, 1 : 2 * (width // 2) : 2], chroma[:, 0 : 2 * (width // 2) : 2]
    )
    if subsampling == "420":
        height = chroma.shape[0]
        np.testing.assert_array_equal(
            chroma[1 : 2 * (height // 2) : 2], chroma[0 : 2 * (height // 2) : 2]
        )


@given(_rgb(), st.sampled_from(("422", "420")), st.sampled_from(_SITINGS), st.data())
def test_decode_reads_chroma_from_the_first_pixel_of_each_block(
    rgb: np.ndarray, subsampling: str, siting: str, data: st.DataObject
) -> None:
    """Whatever the rest of a pair (or, for 4:2:0, of a two-row block)
    carries as chroma is not read, and the picture keeps its shape."""
    codes = encode(rgb, subsampling=subsampling, siting=siting)
    noisy = codes.copy()
    unread = np.ones(codes.shape[:2], bool)
    rows = slice(0, None, 2) if subsampling == "420" else slice(None)
    unread[rows, 0::2] = False
    chroma = legal_codes().chroma
    noise = data.draw(
        hnp.arrays(
            np.uint16,
            (int(unread.sum()), 2),
            elements=st.integers(chroma.start, chroma[-1]),
        ),
        label="unread chroma",
    )
    noisy[unread, 1:] = noise
    clean = decode(codes, subsampling=subsampling, siting=siting)
    assert clean.shape == rgb.shape
    np.testing.assert_array_equal(
        decode(noisy, subsampling=subsampling, siting=siting), clean
    )


@given(st.sampled_from(sorted(MATRICES)), level_depths(), st.data())
def test_decode_clamps_any_legal_code_into_the_rgb_cube(
    matrix: str, level_depth: tuple[str, int], data: st.DataObject
) -> None:
    """Y, Cb and Cr can each be legal while their combination lies outside
    the RGB cube; decode clamps it onto the cube's surface."""
    levels, bits = level_depth
    legal = legal_codes(levels=levels, bits=bits)
    shape = data.draw(st.tuples(st.integers(1, 4), st.integers(1, 6)), label="shape")
    luma = data.draw(
        hnp.arrays(
            np.uint16, shape, elements=st.integers(legal.luma.start, legal.luma[-1])
        )
    )
    chroma = data.draw(
        hnp.arrays(
            np.uint16,
            (*shape, 2),
            elements=st.integers(legal.chroma.start, legal.chroma[-1]),
        )
    )
    codes = np.concatenate((luma[..., None], chroma), axis=-1)
    rgb = decode(codes, matrix=matrix, levels=levels, bits=bits)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0


@given(level_depths(), st.data())
def test_encode_rgb_inverts_decode_rgb_on_every_legal_code(
    level_depth: tuple[str, int], data: st.DataObject
) -> None:
    levels, bits = level_depth
    luma = legal_codes(levels=levels, bits=bits).luma
    codes = data.draw(
        hnp.arrays(
            np.uint16,
            st.tuples(st.integers(1, 4), st.integers(1, 6), st.just(3)),
            elements=st.integers(luma.start, luma[-1]),
        )
    )
    back = encode_rgb(
        decode_rgb(codes, levels=levels, bits=bits), levels=levels, bits=bits
    )
    np.testing.assert_array_equal(back, codes)


@given(_rgb(), level_depths())
def test_decode_rgb_inverts_encode_rgb_within_half_a_code(
    rgb: np.ndarray, level_depth: tuple[str, int]
) -> None:
    levels, bits = level_depth
    assume(levels != "fullprotect")  # its clamp moves codes by more than half
    scale = len(legal_codes(levels=levels, bits=bits).luma) - 1
    back = decode_rgb(
        encode_rgb(rgb, levels=levels, bits=bits), levels=levels, bits=bits
    )
    assert (np.abs(back - rgb) <= 0.5 / scale + _FLOAT_SLACK).all()


@given(level_depths())
def test_legal_codes_are_the_standard_spans(level_depth: tuple[str, int]) -> None:
    """Narrow range is the 8-bit span shifted up; full is every code; the
    protected codes are those whose top eight bits are all zeros or ones."""
    levels, bits = level_depth
    legal = legal_codes(levels=levels, bits=bits)
    shift = bits - 8
    if levels == "narrow":
        assert legal.luma == range(16 << shift, (235 << shift) + 1)
        assert legal.chroma == range(16 << shift, (240 << shift) + 1)
    elif levels == "full":
        assert legal.luma == legal.chroma == range(1 << bits)
    else:
        assert (
            legal.luma == legal.chroma == range(1 << shift, (1 << bits) - (1 << shift))
        )
    expected_mid = 128 << shift if levels == "narrow" else 1 << (bits - 1)
    assert legal.chroma_mid == expected_mid


def test_legal_codes_default_to_ten_bit_narrow_range() -> None:
    assert legal_codes() == legal_codes(levels="narrow", bits=10)


# --- backends -------------------------------------------------------------


@pytest.fixture(scope="module")
def torch() -> Any:
    """torch, or a skip where it is absent; CI installs it."""
    return pytest.importorskip("torch")


@given(layout_frames())
def test_torch_packs_and_unpacks_the_bytes_numpy_does(
    torch: Any, case: tuple[str, np.ndarray, int]
) -> None:
    layout, px, rb = case
    height, width, _ = px.shape
    on_torch = torch.from_numpy(px.astype(np.int32)).to(
        torch.uint8 if px.dtype == np.uint8 else torch.int32
    )
    packed = pack(on_torch, layout, rb, xp=torch)
    np.testing.assert_array_equal(packed.numpy(), pack(px, layout, rb))
    out = unpack(packed, layout, width=width, height=height, row_bytes=rb, xp=torch)
    np.testing.assert_array_equal(out.to(torch.int32).numpy(), px)


@given(
    _rgb(),
    st.sampled_from(sorted(MATRICES)),
    level_depths(),
    st.sampled_from(_SUBSAMPLINGS),
)
def test_torch_encodes_the_codes_numpy_does(
    torch: Any,
    rgb: np.ndarray,
    matrix: str,
    level_depth: tuple[str, int],
    subsampling: str,
) -> None:
    levels, bits = level_depth
    options = {
        "matrix": matrix,
        "levels": levels,
        "bits": bits,
        "subsampling": subsampling,
    }
    on_torch = encode(torch.from_numpy(rgb), xp=torch, **options)
    np.testing.assert_array_equal(
        on_torch.to(torch.int32).numpy(), encode(rgb, **options).astype(np.int32)
    )
