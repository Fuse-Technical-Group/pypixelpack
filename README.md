# pypixelpack

**Pixel layouts and wire encodings for video I/O, device-free.**

The bytes a video frame becomes on a wire — v210, r210, the ST 2110-20
pgroup — and the encoding between a frame's RGB and those bytes: colour
matrix, range, chroma subsampling. Written once against a caller-supplied
array namespace, so the same source packs on numpy for a host frame and
on torch for a frame that never leaves a GPU.

Extracted from [pydecklink](https://github.com/Fuse-Technical-Group/pydecklink)'s
packing module and a GPU render pipeline's colorspace node, which held
the same layout twice. This repository's [SPEC.md](SPEC.md) and
[ROADMAP.md](ROADMAP.md) govern the package.

## Installation

```sh
uv add pypixelpack
```

numpy is the only dependency. torch is a namespace the caller supplies,
never a dependency of this package (`§spec:backend`).

## Usage

```python
import numpy as np
from pypixelpack import decode, encode, pack, unpack

rgb = np.zeros((1080, 1920, 3), dtype=np.float32)  # [R, G, B] in [0, 1]
codes = encode(rgb, subsampling="422")  # (H, W, 3) uint16 [Y, Cb, Cr]
data = pack(codes, "v210", row_bytes=5120)  # 1-D uint8, DMA-ready
back = unpack(data, "v210", width=1920, height=1080, row_bytes=5120)
rgb_again = decode(back, subsampling="422")  # float32 [R, G, B] in [0, 1]
```

On a GPU host, pass the namespace and the arrays stay resident:

```python
import torch

codes = encode(rgb_on_cuda, subsampling="422", xp=torch)
data = pack(codes, "v210", row_bytes=5120, xp=torch)
```

## API

- `pack(pixels, layout, row_bytes, *, xp=numpy)` — `(H, W, 3)` integer
  samples to a 1-D `uint8` buffer of `H × row_bytes` in `layout`, on the
  input's device (`§spec:layouts`). RGB layouts take `[R, G, B]`; `v210`
  and `2vuy` take `[Y, Cb, Cr]` with chroma read from even columns.
- `unpack(data, layout, width, height, row_bytes, *, xp=numpy)` — the
  inverse; `unpack(pack(x)) == x` for every layout. Returns `uint8` for
  8-bit layouts and `uint16` otherwise.
- `row_bytes(layout, width)` — the smallest `row_bytes` that holds a line.
- `LAYOUTS` — the layout table, `name → (pixels per group, bytes per
  group, bit depth)`: `argb`, `bgra`, `r210`, `r10b`, `r10l`, `v210`,
  `2vuy`, `r12b`, `r12l`.

`pack` raises `ValueError` for an unknown layout, a `row_bytes` shorter
than the packed line, or a sample above the layout's bit depth; the
last check is skipped under `torch.compile`, where a compiled caller
trusts its own inputs (`§spec:backend`).

### Encoding

- `encode(rgb, *, matrix="bt709", levels="narrow", layout=None,
  bits=None, subsampling=None, xp=numpy)` — `(H, W, 3)` float RGB in
  `[0, 1]` to `(H, W, 3)` `uint16` `[Y, Cb, Cr]` (`§spec:encoding`).
  `matrix` is `bt709` or `bt2020`; `levels` is `narrow` (luma 16–235,
  chroma 16–240 at 8 bits, shifted up by `bits - 8`) or `full`.
  `layout="v210"` selects the depth and subsampling that wire format
  expects — 10-bit 4:2:2; `2vuy` is 8-bit 4:2:2 — and a `bits` or
  `subsampling` that contradicts it is refused; without a layout they
  default to 10 and `444`. `422` averages chroma over each horizontal
  pair and writes it to both pixels, the shape `pack` reads. Rounding is
  half to even, arithmetic is float32 on every backend, and codes clamp
  to the level range's span.
- `decode(ycbcr, *, matrix, levels, layout=None, bits=None, xp=numpy)`
  — the inverse, per pixel, to within half a code per component;
  returns `float32` RGB clamped to `[0, 1]`.
- `encoding_for(layout)` — the `(bits, subsampling)` a layout carries;
  `ENCODINGS` is the table behind it.
- `legal_codes(*, levels="narrow", bits=10)` — the `(luma, chroma)` code
  spans as `range` objects; `len(legal_codes().luma)` is 877, the levels
  10-bit narrow range can represent.
- `MATRICES` — `name → (KR, KB)`: `bt709`, `bt2020`.

`encode` and `decode` raise `ValueError` for an unknown matrix, levels or
subsampling, a `bits` outside 8–16, or a layout with no component
encoding.

## Development

```sh
uv sync
uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest
```

`uv sync` installs torch (CPU) into the dev group so the parity suite
runs; without it those tests skip. To prove the wheel on numpy alone:

```sh
uv build
uv run --isolated --no-project --with dist/*.whl python tools/check_core_install.py
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
