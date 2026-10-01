"""
dm_sim.py
Exact simulators (pure numpy): noiseless statevector and noisy density matrix.

Noisy gates are applied as one fused superoperator per gate (unitary followed by
the gate's noise channel), acting on the affected qubits of the density-matrix
tensor rho[r_0..r_{n-1}, c_0..c_{n-1}].
"""
from __future__ import annotations
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from gates import Gate, unitary
from circuits import Circuit


def zero_rho(n: int) -> np.ndarray:
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def apply_superop(rho: np.ndarray, S4: np.ndarray, qubits: Sequence[int], n: int) -> np.ndarray:
    """Apply a k-qubit superoperator S4 (shape (2,)*4k: out (r',c') axes then in (r,c)
    axes) to qubits `qubits` of the n-qubit density matrix `rho`."""
    k = len(qubits)
    T = rho.reshape((2,) * (2 * n))
    axes = list(qubits) + [n + q for q in qubits]
    out = np.tensordot(S4, T, axes=(list(range(2 * k, 4 * k)), axes))
    out = np.moveaxis(out, list(range(2 * k)), axes)
    return out.reshape(rho.shape)


def unitary_superop(U: np.ndarray) -> np.ndarray:
    d = U.shape[0]
    k = int(np.log2(d))
    return np.kron(U, U.conj()).reshape((2,) * (4 * k))


def _apply_rz(rho: np.ndarray, q: int, theta: float, n: int) -> np.ndarray:
    T = rho.reshape((2,) * (2 * n))
    ph = np.exp(-0.5j * theta * np.array([1.0, -1.0]))
    sr = [1] * (2 * n)
    sc = [1] * (2 * n)
    sr[q] = 2
    sc[n + q] = 2
    T = T * ph.reshape(sr) * ph.conj().reshape(sc)
    return T.reshape(rho.shape)


def run_density_matrix(circuit: Circuit, model=None, rho0: Optional[np.ndarray] = None) -> np.ndarray:
    """Evolve |0..0><0..0| through the circuit.  model=None -> noiseless.
    model is a noise.DeviceNoiseModel providing .superop(gate) -> (2,)*4k tensor."""
    n = circuit.n_qubits
    rho = zero_rho(n) if rho0 is None else rho0
    cache: Dict[Tuple, np.ndarray] = {}
    for g in circuit.all_gates():
        if g.name == "rz":
            rho = _apply_rz(rho, g.qubits[0], g.param, n)
            continue
        key = (g.name, None if g.param is None else round(g.param, 12))
        S4 = cache.get(key)
        if S4 is None:
            S4 = unitary_superop(unitary(g)) if model is None else model.superop(g)
            cache[key] = S4
        rho = apply_superop(rho, S4, g.qubits, n)
    return rho


def statevector(circuit: Circuit) -> np.ndarray:
    n = circuit.n_qubits
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for g in circuit.all_gates():
        U = unitary(g)
        k = len(g.qubits)
        T = psi.reshape((2,) * n)
        out = np.tensordot(U.reshape((2,) * (2 * k)), T,
                           axes=(list(range(k, 2 * k)), list(g.qubits)))
        psi = np.moveaxis(out, list(range(k)), list(g.qubits)).reshape(-1)
    return psi


def energy_from_rho(rho: np.ndarray, H_sparse) -> float:
    """Tr(H rho) for sparse H and dense rho."""
    return float(np.real(H_sparse.multiply(rho.T).sum()))
