"""
pauli.py
Pauli-string utilities with no dependency on any quantum SDK.

A Pauli string ("term") uses the OpenFermion convention: a tuple of
(qubit_index, 'X'|'Y'|'Z') sorted by qubit index; the identity is ().
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

PauliString = Tuple[Tuple[int, str], ...]

PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def commute(a: PauliString, b: PauliString) -> bool:
    """Two Pauli strings commute iff they differ on an even number of shared qubits."""
    da = dict(a)
    anti = sum(1 for q, p in b if q in da and da[q] != p)
    return anti % 2 == 0


def dense_matrix(term: PauliString, n_qubits: int) -> np.ndarray:
    """Dense matrix of a Pauli string (small n only; used by tests/diagnostics)."""
    d = dict(term)
    M = np.array([[1.0 + 0j]])
    for q in range(n_qubits):
        M = np.kron(M, PAULI_MATS[d.get(q, "I")])
    return M


def qwc_groups(terms: Sequence[PauliString], coeffs: Sequence[float]) -> List[dict]:
    """Greedy qubit-wise-commuting (QWC) grouping, largest |coefficient| first.
    Every group is measured with ONE circuit (one shared product-basis rotation).
    Returns a list of {"basis": {qubit: letter}, "members": np.ndarray of term indices}.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    order = np.argsort(-np.abs(coeffs), kind="stable")
    groups: List[dict] = []
    for idx in order:
        t = terms[int(idx)]
        for g in groups:
            if all(g["basis"].get(q, p) == p for q, p in t):
                g["basis"].update(dict(t))
                g["members"].append(int(idx))
                break
        else:
            groups.append({"basis": dict(t), "members": [int(idx)]})
    for g in groups:
        g["members"] = np.array(sorted(g["members"]), dtype=int)
    return groups


def parity_matrix(terms: Sequence[PauliString], n_qubits: int) -> np.ndarray:
    """(k, 2**n) matrix P[i, x] = (-1)^{sum_{q in term_i} bit_q(x)}, big-endian bits."""
    x = np.arange(2 ** n_qubits)
    bits = [(x >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    P = np.empty((len(terms), 2 ** n_qubits), dtype=float)
    for i, t in enumerate(terms):
        par = np.zeros_like(x)
        for q, _ in t:
            par ^= bits[q]
        P[i] = 1.0 - 2.0 * par
    return P
