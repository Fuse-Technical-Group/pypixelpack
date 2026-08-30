"""Duck-typed helpers over a caller-supplied array namespace (§spec:backend).

The ``xp`` parameter is the backend mechanism: numpy on CPU hosts, torch
on GPU hosts. Nothing here imports a backend. Only the spellings that
differ between the two live here; everything both spell the same is
called raw in ``layouts`` and ``encoding``.

The contract a namespace has to satisfy — recorded here because it is
the answer to "would backend X work?":

- ``asarray``, and ``zeros`` accepting ``dtype`` and ``device`` (numpy 2
  takes ``device="cpu"``, and every array carries ``.device``),
- ``stack``/``concatenate`` taking ``axis``, ``full_like``, and ``flip``
  taking the axes positionally (torch spells the keyword ``dims``; the
  one call site is ``layouts._swap_word_bytes``),
- ``round`` (half to even) and ``clip`` taking scalar bounds, on float
  arrays, and ``*``, ``/``, ``+``, ``-`` between a float array and a
  Python float without widening the array (numpy 2 and torch both keep
  float32),
- dtype attributes ``int32``, ``int64``, ``uint8``, ``uint16`` and
  ``float32``,
- ``astype`` or ``to`` for dtype conversion, and ``ascontiguousarray``
  on the namespace or ``contiguous`` on the array,
- ``view(dtype)`` on the array reinterpreting the last axis,
- ``&``, ``|``, ``<<``, ``>>`` on integer arrays.

**Why every shifted intermediate is int64.** numpy would take uint32,
but torch's unsigned support stops at uint8 for most kernels, and a
signed 32-bit word would sign-extend on ``>>``. int64 holds every 32-bit
word and every shift this library performs; the 8-bit layouts shift
nothing and stay uint8 throughout.

**Why serialisation is a dtype view.** A 32-bit word becomes four bytes
by reinterpreting memory, not by four shift-and-mask passes — that is
what ``view`` is for, and it is the same call on both backends. It
assumes a little-endian host, which ``layouts`` checks once at import.

**Why the range check is skipped under a compiler.** Comparing a
reduction against a Python int is control flow on array values, which
``torch.compile`` cannot trace (§spec:backend fusion). Eager calls keep
the check; a compiled caller has already accepted the cost of trusting
its own inputs.
"""

from typing import Any


def astype(array: Any, dtype: Any) -> Any:
    """``array`` as ``dtype`` — numpy's ``astype`` or torch's ``to``.

    Neither copies when the dtype already matches.
    """
    if hasattr(array, "astype"):
        return array.astype(dtype, copy=False)
    return array.to(dtype)


def contiguous(xp: Any, array: Any) -> Any:
    """``array`` with a contiguous layout, so ``view(dtype)`` is legal.

    A no-op on both backends when the array already is.
    """
    if hasattr(array, "contiguous"):
        return array.contiguous()
    return xp.ascontiguousarray(array)


def is_compiling(xp: Any) -> bool:
    """Whether ``xp`` is tracing the caller for compilation; false for a
    namespace with no compiler (numpy)."""
    compiler = getattr(xp, "compiler", None)
    return compiler is not None and bool(compiler.is_compiling())
