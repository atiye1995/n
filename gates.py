"""
gates.py
Minimal gate set shared by the circuit compiler, both simulators and the
hardware exporter.  Qubit q always corresponds to tensor axis q of an
n-qubit state (qubit 0 = most-significant bit), i.e. the same big-endian
convention as OpenFermion's sparse operators.

Gate set:  x, h, rx(theta), rz(theta), cx(control, target)
  * rz is treated as a *virtual* (error-free, zero-duration) gate, exactly as
    on IBM hardware where it is a frame change.
  * x, h, rx, cx are physical gates and carry noise / are foldable.
"""
from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import numpy as np


class Gate(NamedTuple):
    name: str
    qubits: Tuple[int, ...]
    param: Optional[float] = None


PHYSICAL = frozenset({"x", "h", "rx", "cx"})
SELF_INVERSE = frozenset({"x", "h", "cx"})

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def unitary(g: Gate) -> np.ndarray:
    """Matrix of the gate in the basis ordering given by g.qubits (control first for cx)."""
    n = g.name
    if n == "x":
        return _X
    if n == "h":
        return _H
    if n == "rx":
        c, s = np.cos(g.param / 2), np.sin(g.param / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if n == "rz":
        e = np.exp(-0.5j * g.param)
        return np.diag([e, np.conj(e)])
    if n == "cx":
        return _CX
    raise ValueError(f"unknown gate {n!r}")


def inverse(g: Gate) -> Gate:
    if g.name in SELF_INVERSE:
        return g
    if g.name in ("rx", "rz"):
        return Gate(g.name, g.qubits, -g.param)
    raise ValueError(f"no inverse rule for {g.name!r}")


def is_inverse_pair(a: Gate, b: Gate, tol: float = 1e-12) -> bool:
    if a.qubits != b.qubits or a.name != b.name:
        return False
    if a.name in SELF_INVERSE:
        return True
    if a.name in ("rx", "rz"):
        return abs(a.param + b.param) < tol
    return False
