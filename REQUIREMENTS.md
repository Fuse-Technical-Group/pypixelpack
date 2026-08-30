# Requirements

Problem-space document for pypixelpack: pixel layouts and wire encodings
as a device-free library. Extracted from
[pydecklink](https://github.com/Fuse-Technical-Group/pydecklink)
(`§spec:pixel-packing` there) and a GPU render pipeline that outputs to
SDI and ST 2110.

## Problem statement §req:problem-statement

Target users are the maintainers of video I/O tools in this organization
and the display-measurement tools that drive them: a DeckLink binding, a
GPU render pipeline that outputs to SDI and ST 2110, a signal generator,
and a measurement session that puts exact code values on a wire.

The same v210 layout exists twice. pydecklink packs it in numpy on the
host for a DMA buffer; a GPU render pipeline packs it in torch on the
device and DMAs the result, because a round trip through host memory costs a full
uncompressed frame across PCIe every frame. Each is correct for its
caller and neither serves the other, so the next wire format — r210 for
RGB 4:4:4 over SDI — recurs the same way.

The colour encoding between RGB and those samples lives in one of the two
and is hard-bound to one matrix. A measurement session on SDI has to
declare that encoding and cannot implement it without a second copy,
and a wrong matrix or range offsets every code in a way nothing
downstream can detect.

## Success criteria §req:success-criteria

- pydecklink consumes this library for its layouts, keeps only the map
  from its SDK enum to a layout name, and `pack`/`unpack` output is
  byte-identical before and after.
- The GPU render pipeline consumes this library on torch, on the
  device, with no host round trip, and a test proves the compiled path is taken.
- The same source packs a frame on numpy and on torch to identical
  bytes.
- Every layout round-trips: `unpack(pack(x)) == x`.
- display-measure declares an SDI encoding by name and matrix and never
  carries a conversion of its own.

## User stories §req:user-stories

- As the pydecklink maintainer, I delete the layout implementations and
  keep the enum map, so the transport binding holds nothing but the
  vendor's surface.
- As a render-pipeline developer, I pack v210 and r210 on the GPU and
  DMA the bytes, so the frame never crosses the bus uncompressed.
- As a measurement-session author, I declare "10-bit BT.709 narrow-range
  4:2:2 in v210" and drive RGB patches through it, so the encoding on
  the wire is the one the artifact records.
- As a UHD integrator, I select BT.2020, so the encoding follows the
  signal rather than a constant.

## Quality attributes §req:quality-attributes

- **Exactness.** Layouts are bit-exact against the format's own
  reference tables; encodings state their rounding and honor it.
- **Determinism.** Same inputs, same bytes, on every backend.
- **Footprint.** numpy is the only dependency; torch is supplied by the
  caller.
- **Device residency.** No function forces a transfer to host memory.
- **Fusion.** The torch path stays traceable by `torch.compile`; a
  change that silently drops it to eager fails a test.

## Constraints §req:constraints

- Repository under Fuse-Technical-Group; BSD-3-Clause.
- Seeded by extraction: pydecklink's layouts and the GPU pipeline's
  encoding move verbatim before any reshaping; adoption is behind one
  release each.
- No transport and no device: what a byte buffer is *for* — a DeckLink
  frame, an RTP payload — belongs to the consumer.
- No resizing, no colour management beyond the stated matrix and range:
  the library encodes what it is given.
- Renders into a caller-supplied array namespace on the same contract
  display-patterns established.

## Priorities §req:priorities

Essential, in adoption order:

1. The eight layouts pydecklink holds, namespace-generic, with
   reference vectors and numpy/torch byte-identity.
2. pydecklink adoption; first release.
3. BT.709 and BT.2020 encoding at narrow and full range, chroma
   subsampling by pair average.
4. GPU render pipeline adoption on the device, with fusion asserted.

Nice-to-have, after adoption:

- `2vuy`, which no current implementation packs.
- Provenance for `r10b`, `r10l`, `r12b`, `r12l` against FFmpeg's
  format list.
- The ST 2110-20 pgroup, if a second consumer appears.
