"""The package imports numpy alone (§spec:package-shape)."""

import subprocess
import sys

import pypixelpack


def test_import_pulls_no_other_array_backend() -> None:
    code = "import sys, pypixelpack; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_public_surface() -> None:
    assert set(pypixelpack.__all__) == {
        "ENCODINGS",
        "LAYOUTS",
        "MATRICES",
        "SUBSAMPLED_422",
        "decode",
        "encode",
        "encoding_for",
        "legal_codes",
        "pack",
        "row_bytes",
        "unpack",
    }
