# pypixelpack — Specification

Solution-space document. Requirements live in
[REQUIREMENTS.md](REQUIREMENTS.md); remaining work in
[ROADMAP.md](ROADMAP.md). A slug in backticks resolves in the named
repository, not here.

## Problem §spec:problem

*Status: complete*

A frame's RGB and the bytes it becomes on a wire are separated by two
steps: an encoding (colour matrix, range, chroma subsampling) and a
layout (how samples sit in words). Both are pure arithmetic, and both
existed inside device tools — the layout twice, once per array backend,
and the encoding once, bound to one matrix (§req:problem-statement).
This library holds each once, written against a caller-supplied array
namespace, so a host binding and a GPU pipeline share one source and a
measurement session declares an encoding it never implements.

## Package shape §spec:package-shape

*Status: complete*

The distribution is `pypixelpack` on PyPI; the import package is
`pypixelpack`. numpy is the only dependency and the reference backend;
importing the package pulls no other array library
(§req:quality-attributes footprint). The package runs on CPython 3.12+
on macOS, Linux and Windows, and opens no device and no socket
(§req:constraints).

**Why one package and no extras.** Layouts and encodings are a few
hundred lines with one dependency between them; an extra would price a
seam nobody pays for.

## Backend §spec:backend

*Status: complete*

Every function takes an `xp` keyword — the array namespace — defaulting
to numpy, and operates on arrays that namespace produced. torch arrays
stay on their device throughout: no function calls `.cpu()`, and the
output is allocated on the input's device. The namespace contract is
the one display-patterns records (`§spec:render-model` there), and
`pypixelpack/_backend.py` lists what this library draws on it. Every
integer intermediate is int64, the widest type both backends shift
without sign, so the bytes agree; the only read of array values, the
range check on `pack`, runs eager and is skipped under a compiler.

**Why a namespace parameter and not a backend registry.** The caller
already knows its backend; a parameter is the whole mechanism. **Why
torch is never a dependency.** The host consumers never import it, and
the device consumer already has it.

**Why words become bytes by memory view.** A 32-bit word is
serialised by reinterpreting its memory, not by shifting four times, so
the library assumes a little-endian host and refuses to import on any
other: every target this organization runs is little-endian, and a
silent byte-swap would be worse than a loud refusal.

**Why fusion is a contract, not a hope.** The device consumer wraps
these functions in `torch.compile`; a rewrite that introduced Python
control flow on array values, or a numpy call on a torch tensor, would
still return correct bytes and silently run eager. The library
therefore asserts, in its own tests, that the torch path compiles and
that its bytes match the eager and numpy paths
(§req:quality-attributes fusion).

## Layouts §spec:layouts

*Status: complete*

A layout names how integer component samples sit in a byte buffer:
its pixel group, bytes per group, and bit depth. `pack(pixels, layout,
row_bytes)` takes `(H, W, 3)` integer samples and returns a 1-D byte
buffer of `H × row_bytes`; `unpack` is its inverse, and `unpack ∘ pack`
is identity for every layout. Layouts are keyed by plain string, and
the table is public.

The catalog is what pydecklink held (`§spec:pixel-packing` there) plus
the one SDK format nothing packed: `argb`, `bgra`, `r210`, `r10b`,
`r10l`, `v210`, `2vuy`, `r12b`, `r12l`. RGB layouts carry `[R, G, B]`;
`v210` and `2vuy` carry `[Y, Cb, Cr]` with chroma sampled from even
columns on pack and replicated on unpack, so identity holds when chroma
agrees within each horizontal pair — inherent to 4:2:2, not a packing
loss. Byte layouts are the format's own reference, and this document
does not restate them.

Provenance, settled against FFmpeg, CoreVideo `CVPixelBuffer.h` and
the DeckLink SDK 15.3 (`DeckLinkAPIModes.h`, manual section 3.4):

| layout | standing | outside the SDK |
| --- | --- | --- |
| `argb`, `bgra` | standard | FFmpeg `AV_PIX_FMT_ARGB` / `BGRA`; CoreVideo `32ARGB` / `32BGRA` |
| `v210` | standard | FFmpeg codec `v210`; CoreVideo `422YpCbCr10` |
| `2vuy` | standard | FFmpeg `AV_PIX_FMT_UYVY422`, tag `'2vuy'`; CoreVideo `422YpCbCr8` |
| `r210` | standard | FFmpeg codec `r210`, tag `'r210'`; `AV_PIX_FMT_X2RGB10BE` |
| `r10b` | SDK spelling of a standard word | byte-identical to FFmpeg codec `r10k` and CoreVideo `30RGB` `'R10k'` |
| `r10l` | SDK spelling | the `r10b` word little-endian; FFmpeg reads it only as `'R10k'` with `DpxE` extradata |
| `r12b`, `r12l` | SDK only | no FFmpeg pixel format, codec or tag; FFmpeg's and GStreamer's DeckLink inputs refuse both |

**Why every layout moves and not v210 alone.** `r210` is uncompressed
10-bit RGB 4:4:4 and is what an RGB 4:4:4 SDI feed needs; hoisting one
layout reopens the question on the next. **Why string keys.** The only
vendor-specific fact in the source module was the map from an SDK enum
to a name; that map stays with the vendor binding, and this library
never sees the enum. **Why `r10b` is not spelled `r10k`.** The key is
the name a DeckLink consumer holds; the identity above tells an FFmpeg
caller which layout reads its bytes.

## Encoding §spec:encoding

*Status: complete*

An encoding maps RGB in [0, 1] to integer component samples and back: a
colour matrix (`bt709`, `bt2020`), a level range (`narrow`, `full`), a
chroma subsampling (`444`, or `422` by pair average) and a bit depth
from 8 to 16 — or a layout name, which selects the depth and subsampling
that wire format carries and refuses a keyword that contradicts them.
`encode` and `decode` are each other's inverse to within half a code per
component, round half to even in float32 on every backend so host and
device agree, and take the same `xp` as the layouts. `legal_codes`
returns the spans a level range represents at a depth.

RGB outside [0, 1] clamps to the span's ends on encode, and decode clamps
its output to [0, 1]: sub-black and super-white codes are representable
on the wire but are not carried through this library's float RGB, which
is a display-referred convention, not a scene-referred one.

Seeded verbatim from a GPU render pipeline's BT.709 narrow-range
conversion, which serves its SDI and ST 2110 outputs from one matrix so
the coefficients cannot drift between them;
a 10-bit narrow-range 4:2:2 encode packed to v210 here reproduces its
bytes exactly.

**Why the matrix is a parameter.** v210 is a container; the colorimetry
is signalled beside it, BT.709 for HD and BT.2020 for UHD. A constant
serves one deployment and forks for the next. **Why narrow range is
explicit.** Narrow range cannot represent every RGB code — 10-bit
luma has 877 levels — so a caller driving exact code values needs to
know which ones survive; the library exposes the representable set
rather than rounding silently. **Why encoding lives with layout.** A
measurement session defines a patch as what the processor receives and
declares the encoding between that and the wire; the declaration is only
meaningful if one implementation answers to it. **Why 4:2:2 keeps the
full shape.** A 4:2:2 encode returns `(H, W, 3)` with chroma equal
within each horizontal pair — what `pack` reads for a layout in
`SUBSAMPLED_422` — rather than a half-width chroma plane, so one array
convention runs from `encode` through `pack` and back. **Why a layout
names its encoding.** `encode(rgb)` at defaults packed to v210 would
silently drop odd-column chroma, and an 8-bit encode would land narrow
levels in a 10-bit container as valid bytes and a wrong picture; the
`ENCODINGS` table makes the wire's expectation the thing a caller names.

## Consumers §spec:consumers

*Status: in progress*

pydecklink imports the layouts and keeps the map from its `PixelFormat`
enum to a layout name; its `pack`/`unpack` surface is unchanged and
importing `pydecklink` still pulls no packing code. A GPU render
pipeline imports layouts and encoding on torch, on the device, and
deletes its copies. display-measure declares an encoding by name and matrix through
bmd-signal-gen and pydecklink and carries no conversion.

**Why adoption is behind one release each.** Extraction is verbatim
first, so a consumer's before-and-after is byte-comparable; reshaping
happens here afterward, against consumers that already pass.

## Scope boundaries §spec:non-goals

*Status: complete*

Not this library: transport (what a buffer is for — a DeckLink frame,
an RTP payload — belongs to pydecklink and pyst2110); device access;
resizing or resampling; colour management beyond the stated matrix and
range; image content (display-patterns); the ST 2110-20 pgroup until a
second consumer exists for it.
