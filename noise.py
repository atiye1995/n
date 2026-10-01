"""
noise.py
Device-informed gate-level noise model.

Every physical gate g is followed by
    (1) thermal relaxation over the gate duration (T1, T2), then
    (2) a depolarizing channel whose strength is chosen so that the TOTAL average
        gate infidelity of (relaxation o depolarizing) equals the calibrated gate error.
rz is virtual (noiseless).  Idle-qubit noise and crosstalk are NOT modelled
(all-to-all connectivity is assumed); state the same in the paper.

Readout error is a separate, tensored asymmetric confusion matrix, unaffected by
folding, and is handled in measurement.py (optionally inverted = readout mitigation).

All parameters come from a calibration JSON produced by calibration.py from REAL
backend calibration data.  There are deliberately no built-in default numbers.
"""
from __future__ import annotations
import itertools
import json
from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence
import numpy as np

from gates import Gate, PHYSICAL, unitary

_I2 = np.eye(2, dtype=complex)
_PAULIS = [_I2,
           np.array([[0, 1], [1, 0]], dtype=complex),
           np.array([[0, -1j], [1j, 0]], dtype=complex),
           np.array([[1, 0], [0, -1]], dtype=complex)]


# ----------------------------------------------------------------------
# Kraus / superoperator helpers
# ----------------------------------------------------------------------

def kraus_to_superop(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """Row-major vectorisation: vec(K rho K^dag) = (K kron conj(K)) vec(rho)."""
    return sum(np.kron(K, K.conj()) for K in kraus)


def process_fidelity_of_kraus(kraus: Sequence[np.ndarray]) -> float:
    d = kraus[0].shape[0]
    return float(sum(abs(np.trace(K)) ** 2 for K in kraus) / d ** 2)


def relaxation_kraus_1q(t: float, T1: float, T2: float) -> List[np.ndarray]:
    """Amplitude damping (gamma = 1-e^{-t/T1}) followed by pure dephasing so that the
    coherence decays as e^{-t/T2}.  T2 is clipped to 2*T1 (physical limit)."""
    T2 = min(T2, 2.0 * T1)
    gamma = 1.0 - np.exp(-t / T1)
    lam_phi = 1.0 - np.exp(-2.0 * t / T2 + t / T1)
    lam_phi = min(max(lam_phi, 0.0), 1.0)
    A = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
         np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]
    P = [np.array([[1, 0], [0, np.sqrt(1 - lam_phi)]], dtype=complex),
         np.array([[0, 0], [0, np.sqrt(lam_phi)]], dtype=complex)]
    return [p @ a for a in A for p in P]


def depolarizing_kraus(p: float, n: int) -> List[np.ndarray]:
    """rho -> (1-p) rho + p I/d on n qubits (n = 1 or 2)."""
    d = 2 ** n
    ops = []
    for idx in itertools.product(range(4), repeat=n):
        M = np.array([[1.0 + 0j]])
        for i in idx:
            M = np.kron(M, _PAULIS[i])
        w = 1.0 - p * (d * d - 1) / (d * d) if all(i == 0 for i in idx) else p / (d * d)
        ops.append(np.sqrt(max(w, 0.0)) * M)
    return ops


def _dep_param(e_target: float, F_pro_relax: float, d: int) -> float:
    """Depolarizing parameter p (rho -> (1-p) rho + p I/d) such that the composed channel
    (relaxation then depolarizing) has average gate infidelity exactly e_target.
    Exact identity for trace-preserving relaxation: tr(S_dep S_rel) = (1-p) tr(S_rel) + p,
    i.e. F_pro_total = (1-p) F_pro_relax + p/d^2."""
    F_avg_t = 1.0 - e_target
    F_pro_t = ((d + 1) * F_avg_t - 1.0) / d
    denom = F_pro_relax - 1.0 / d ** 2
    if denom <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (F_pro_relax - F_pro_t) / denom)))


# ----------------------------------------------------------------------
# Device model
# ----------------------------------------------------------------------

@dataclass
class DeviceNoiseModel:
    e1: float          # average error of a physical single-qubit gate
    e2: float          # average error of a CNOT
    T1: float          # seconds
    T2: float          # seconds
    t1q: float         # single-qubit gate duration, seconds
    t2q: float         # two-qubit gate duration, seconds
    p01: float         # P(read 1 | prepared 0)
    p10: float         # P(read 0 | prepared 1)
    provenance: Dict = field(default_factory=dict)
    _cache: Dict = field(default_factory=dict, repr=False, compare=False)

    # ---- construction -------------------------------------------------
    @staticmethod
    def from_json(path: str) -> "DeviceNoiseModel":
        with open(path) as f:
            d = json.load(f)
        return DeviceNoiseModel(e1=d["e1"], e2=d["e2"], T1=d["T1"], T2=d["T2"],
                                t1q=d["t1q"], t2q=d["t2q"], p01=d["p01"], p10=d["p10"],
                                provenance=d.get("provenance", {}))

    def scaled(self, s: float) -> "DeviceNoiseModel":
        """All gate errors x s and coherence times / s (readout unchanged)."""
        return DeviceNoiseModel(self.e1 * s, self.e2 * s, self.T1 / s, self.T2 / s,
                                self.t1q, self.t2q, self.p01, self.p10,
                                dict(self.provenance, noise_scale=s))

    # ---- channels -----------------------------------------------------
    def _noise_superop(self, nq: int) -> np.ndarray:
        d = 2 ** nq
        t = self.t1q if nq == 1 else self.t2q
        e = self.e1 if nq == 1 else self.e2
        K1 = relaxation_kraus_1q(t, self.T1, self.T2)
        Krel = K1 if nq == 1 else [np.kron(a, b) for a in K1 for b in K1]
        F_rel = process_fidelity_of_kraus(Krel)
        p = _dep_param(e, F_rel, d)
        S_rel = kraus_to_superop(Krel)
        S_dep = kraus_to_superop(depolarizing_kraus(p, nq))
        return S_dep @ S_rel

    def superop(self, g: Gate) -> np.ndarray:
        """Fused (gate then noise) superoperator as a (2,)*4k tensor."""
        nq = len(g.qubits)
        key = ("noise", nq)
        if key not in self._cache:
            self._cache[key] = self._noise_superop(nq)
        U = unitary(g)
        S = self._cache[key] @ np.kron(U, U.conj())
        return S.reshape((2,) * (4 * nq))

    def gate_infidelity(self, g: Gate) -> float:
        """Process infidelity of the noise attached to gate g (0 for virtual rz)."""
        if g.name not in PHYSICAL:
            return 0.0
        nq = len(g.qubits)
        key = ("noise", nq)
        if key not in self._cache:
            self._cache[key] = self._noise_superop(nq)
        return float(1.0 - np.real(np.trace(self._cache[key])) / 4 ** nq)

    def readout_confusion(self) -> np.ndarray:
        """C[read, true]."""
        return np.array([[1 - self.p01, self.p10], [self.p01, 1 - self.p10]])


def total_infidelity(gates: Sequence[Gate], model: DeviceNoiseModel) -> float:
    """First-order 'noise budget' of a gate list = sum of gate process infidelities."""
    return float(sum(model.gate_infidelity(g) for g in gates))


def effective_scale(base_gates: Sequence[Gate], folded_gates: Sequence[Gate],
                    model: DeviceNoiseModel) -> float:
    """Actual noise-scale factor of a folded circuit relative to the base circuit.
    USE THIS (not the nominal folding factor) as the extrapolation abscissa."""
    return total_infidelity(folded_gates, model) / total_infidelity(base_gates, model)
