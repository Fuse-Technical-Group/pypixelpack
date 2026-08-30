"""Verify the numpy-only footprint of the built wheel (§spec:package-shape).

Run against the wheel in a clean environment (the core-install CI job):
the distribution declares numpy as its only dependency, importing,
encoding and packing pull in no other array library, a reference vector
packs byte-exact, and RGB survives encode, pack, unpack, decode. The
environment is first required to be core-only, so a torch that happened
to be installed cannot mask a stray import.
"""

import importlib.metadata
import importlib.util
import re
import sys

# Array libraries a caller may supply as ``xp`` but the package never
# imports. The probe fails if importing or packing loads any of them.
FORBIDDEN_MODULES = ("torch", "jax", "cupy")


def main() -> None:
    for name in FORBIDDEN_MODULES:
        if importlib.util.find_spec(name) is not None:
            sys.exit(f"{name} is installed; this check requires a core-only env")

    requires = importlib.metadata.requires("pypixelpack") or []
    names = sorted(re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0] for req in requires)
    assert names == ["numpy"], f"wheel depends on more than numpy: {requires}"

    import numpy as np

    from pypixelpack import decode, encode, legal_codes, pack, unpack

    # The v210 reference vector from tests/test_layouts.py (SDK 15.3 §3.4).
    y = [64, 65, 66, 67, 68, 69]
    cb = [100, 0, 101, 0, 102, 0]
    cr = [200, 0, 201, 0, 202, 0]
    px = np.array([[[y[i], cb[i], cr[i]] for i in range(6)]], dtype=np.uint16)
    data = pack(px, "v210", row_bytes=16)
    assert data.tolist() == [
        0x64, 0x00, 0x81, 0x0C, 0x41, 0x94, 0x21, 0x04,
        0xC9, 0x0C, 0x61, 0x06, 0x44, 0x28, 0x53, 0x04,
    ]  # fmt: skip
    back = unpack(data, "v210", width=6, height=1, row_bytes=16)
    assert back[0, :, 0].tolist() == y

    # RGB through the whole path and back: encode at 4:2:2 is what v210
    # packs, and a gray ramp has no chroma detail for 4:2:2 to lose, so
    # only 10-bit narrow-range rounding remains (§spec:encoding).
    assert len(legal_codes().luma) == 877
    ramp = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    rgb = np.broadcast_to(ramp[None, :, None], (1, 6, 3))
    codes = encode(rgb, layout="v210")
    assert codes[0, :, 0].tolist() == [64, 239, 414, 590, 765, 940]
    wire = pack(codes, "v210", row_bytes=16)
    back = decode(unpack(wire, "v210", 6, 1, 16), layout="v210")
    assert float(np.abs(back - rgb).max()) < 1e-3

    loaded = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES)
    assert not loaded, f"packing imported another array library: {loaded}"

    print("pypixelpack encodes and packs on numpy alone")


if __name__ == "__main__":
    main()
