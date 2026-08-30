"""numpy and torch pack identical bytes and encode identical codes, and
the torch path compiles (§spec:backend).

Fusion is asserted, not hoped for: every public function is traced
with ``fullgraph=True``, so a graph break is an error rather than a
silent fall to eager, and a counting backend confirms a graph was
compiled and run. Skips cleanly when torch is absent; CI installs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from conftest import frame, rgb_frame

from pypixelpack import LAYOUTS, pack, row_bytes, unpack
from pypixelpack.encoding import decode, encode

if TYPE_CHECKING:
    from collections.abc import Callable

torch = pytest.importorskip("torch")

_HEIGHT, _WIDTH = 3, 18  # 18 spans 3 v210 groups and 3 r12 groups with padding
_EXTRA_ROW_BYTES = 8  # exercise the zero padding past the active line


def _row_bytes(layout: str) -> int:
    return row_bytes(layout, _WIDTH) + _EXTRA_ROW_BYTES


def _frame(layout: str) -> np.ndarray:
    return frame(layout, _HEIGHT, _WIDTH)


class _CountingBackend:
    """A torch.compile backend that runs the traced graph and counts them."""

    def __init__(self) -> None:
        self.graphs = 0

    def __call__(self, gm: Any, example_inputs: Any) -> Callable[..., Any]:
        self.graphs += 1
        return gm.forward


def _compiled(fn: Callable[..., Any]) -> tuple[Callable[..., Any], _CountingBackend]:
    torch.compiler.reset()  # each parametrisation gets its own cache
    backend = _CountingBackend()
    return torch.compile(fn, backend=backend, fullgraph=True), backend


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_pack_bytes_agree_eager_and_compiled(layout: str) -> None:
    px = _frame(layout)
    rb = _row_bytes(layout)
    expected = pack(px, layout, rb)

    tensor = torch.from_numpy(px)
    eager = pack(tensor, layout, rb, xp=torch)
    assert eager.dtype == torch.uint8
    assert eager.device == tensor.device
    np.testing.assert_array_equal(eager.numpy(), expected)

    compiled, backend = _compiled(lambda x: pack(x, layout, rb, xp=torch))
    fused = compiled(tensor)
    assert backend.graphs == 1
    np.testing.assert_array_equal(fused.numpy(), expected)


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_unpack_round_trips_eager_and_compiled(layout: str) -> None:
    px = _frame(layout)
    rb = _row_bytes(layout)
    packed = torch.from_numpy(pack(px, layout, rb))

    eager = unpack(packed, layout, _WIDTH, _HEIGHT, rb, xp=torch)
    assert eager.dtype == (torch.uint8 if LAYOUTS[layout][2] == 8 else torch.uint16)
    assert eager.device == packed.device
    np.testing.assert_array_equal(eager.to(torch.int64).numpy(), px)

    compiled, backend = _compiled(
        lambda d: unpack(d, layout, _WIDTH, _HEIGHT, rb, xp=torch)
    )
    fused = compiled(packed)
    assert backend.graphs == 1
    np.testing.assert_array_equal(fused.to(torch.int64).numpy(), px)


def test_eager_torch_keeps_the_range_check() -> None:
    px = torch.tensor([[[4096, 0, 0]]], dtype=torch.int32)  # exceeds 12-bit range
    with pytest.raises(ValueError, match="12-bit"):
        pack(px, "r12b", row_bytes=36, xp=torch)


# --- encoding ---------------------------------------------------------------


def _assert_parity(
    fn: Callable[..., Any], array: np.ndarray, expected: np.ndarray, dtype: Any
) -> None:
    """torch eager matches numpy; the compiled path runs one graph and matches."""
    tensor = torch.from_numpy(array)
    eager = fn(tensor)
    assert eager.dtype == dtype
    assert eager.device == tensor.device
    np.testing.assert_array_equal(
        eager.to(torch.int64).numpy() if dtype != torch.float32 else eager.numpy(),
        expected,
    )
    compiled, backend = _compiled(fn)
    fused = compiled(tensor)
    assert backend.graphs == 1
    np.testing.assert_array_equal(
        fused.to(torch.int64).numpy() if dtype != torch.float32 else fused.numpy(),
        expected,
    )


@pytest.mark.parametrize("matrix", ["bt709", "bt2020"])
@pytest.mark.parametrize("levels", ["narrow", "full"])
def test_encode_codes_agree_eager(matrix: str, levels: str) -> None:
    """The float32 agreement check over the parameter grid, eager only:
    matrix and levels change scalar constants, not the traced graph."""
    rgb = rgb_frame(_HEIGHT, _WIDTH)
    kw: dict[str, Any] = {"matrix": matrix, "levels": levels}
    out = encode(torch.from_numpy(rgb), **kw, xp=torch)
    np.testing.assert_array_equal(out.to(torch.int64).numpy(), encode(rgb, **kw))


@pytest.mark.parametrize(
    ("subsampling", "width"), [("444", _WIDTH), ("422", _WIDTH), ("422", _WIDTH - 1)]
)
def test_encode_codes_agree_eager_and_compiled(subsampling: str, width: int) -> None:
    rgb = rgb_frame(_HEIGHT, width, seed=1)
    _assert_parity(
        lambda x: encode(x, subsampling=subsampling, xp=torch),
        rgb,
        encode(rgb, subsampling=subsampling),
        torch.uint16,
    )


def test_decode_values_agree_eager_and_compiled() -> None:
    codes = frame("v210", _HEIGHT, _WIDTH - 1)
    _assert_parity(lambda c: decode(c, xp=torch), codes, decode(codes), torch.float32)
