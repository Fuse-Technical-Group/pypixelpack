"""The namespace helpers honour the contract without importing a backend
(§spec:backend)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pypixelpack import _backend


def test_astype_prefers_astype_then_to() -> None:
    assert _backend.astype(np.zeros(2), np.uint8).dtype == np.uint8
    torch_like = SimpleNamespace(to=lambda dtype: ("to", dtype))
    assert _backend.astype(torch_like, "uint8") == ("to", "uint8")


def test_astype_does_not_copy_a_matching_dtype() -> None:
    a = np.zeros(2, dtype=np.uint8)
    assert _backend.astype(a, np.uint8) is a


def test_contiguous_prefers_the_array_method_then_the_namespace() -> None:
    strided = np.zeros((4, 8), dtype=np.uint8)[:, :4]
    assert not strided.flags.c_contiguous
    assert _backend.contiguous(np, strided).flags.c_contiguous
    torch_like = SimpleNamespace(contiguous=lambda: "contiguous")
    assert _backend.contiguous(np, torch_like) == "contiguous"


def test_is_compiling_false_outside_a_compiler() -> None:
    assert _backend.is_compiling(np) is False
    assert _backend.is_compiling(SimpleNamespace()) is False
    stub = SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda: True))
    assert _backend.is_compiling(stub) is True


def test_backend_module_imports_no_array_library() -> None:
    source = Path(_backend.__file__).read_text()
    assert "import numpy" not in source
    assert "import torch" not in source
