"""
measurement.py
Shot-based measurement of a (noisy) state, exactly as it would be performed on a
device: qubit-wise-commuting (QWC) groups, one basis-rotation circuit per group,
finite shots, asymmetric readout error, optional readout mitigation.

Given the exact noisy density matrix, each group's outcome distribution is
computed once; repeated shot-noise realisations are then drawn with a multinomial
sampler.  Term estimates inside a group share the same shots, so their
correlations (and the correct energy variance) are captured automatically.

Estimator for term i in group g from counts n(s):
    q_i = (1/n_g) sum_s n(s) M_i(s),
    M_i(s) = parity_i(s)                      (no mitigation)
    M_i = (C^{-1})^T parity_i  (tensored)     (readout mitigation, unbiased for
                                                independent readout errors)
Variance estimates are sample variances of M_i(s) (nothing is assumed).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np

from gates import Gate, unitary
from pauli import PauliString, qwc_groups, parity_matrix
from dm_sim import apply_superop, unitary_superop


@dataclass
class MeasurementPlan:
    n_qubits: int
    const: float                       # identity coefficient (measured exactly)
    terms: List[PauliString]           # non-identity terms
    coeffs: np.ndarray                 # real coefficients
    groups: List[dict]                 # {"basis": {q: letter}, "members": ndarray}

    @property
    def weights(self) -> np.ndarray:
        return np.array([len(t) for t in self.terms])

    @property
    def n_terms(self) -> int:
        return len(self.terms)


def build_plan(qubit_operator, n_qubits: int) -> MeasurementPlan:
    const, terms, coeffs = 0.0, [], []
    for term, c in qubit_operator.terms.items():
        if abs(complex(c).imag) > 1e-10:
            raise ValueError("Hamiltonian has a non-real Pauli coefficient")
        if len(term) == 0:
            const += float(complex(c).real)
        else:
            terms.append(tuple(term))
            coeffs.append(float(complex(c).real))
    coeffs = np.array(coeffs)
    return MeasurementPlan(n_qubits, const, terms, coeffs, qwc_groups(terms, coeffs))


def basis_rotation_gates(basis: Dict[int, str]) -> List[Gate]:
    """Gates V with V P V^dagger = Z (measure Z after V == measure P before V)."""
    out = []
    for q, p in sorted(basis.items()):
        if p == "X":
            out.append(Gate("h", (q,)))
        elif p == "Y":
            out.append(Gate("rx", (q,), float(np.pi / 2)))
    return out


def _apply_1q_along_axis(arr: np.ndarray, M: np.ndarray, axis: int) -> np.ndarray:
    out = np.tensordot(M, arr, axes=([1], [axis]))
    return np.moveaxis(out, 0, axis)


def apply_readout(p: np.ndarray, C: np.ndarray, n: int) -> np.ndarray:
    """Forward tensored confusion: p_read = (C x ... x C) p_true."""
    T = p.reshape((2,) * n)
    for q in range(n):
        T = _apply_1q_along_axis(T, C, q)
    return T.reshape(-1)


@dataclass
class GroupDist:
    members: np.ndarray
    p: np.ndarray          # outcome distribution actually sampled (incl. readout error)
    V: np.ndarray          # (k, 2**n) per-outcome term values M_i(s)
    u: np.ndarray          # coeff-weighted per-outcome energy contribution c_g @ V
    exact_terms: np.ndarray  # infinite-shot term expectations V @ p


def prepare_distributions(rho: np.ndarray, plan: MeasurementPlan, readout_C: Optional[np.ndarray],
                          mitigate_readout: bool) -> List[GroupDist]:
    n = plan.n_qubits
    dists: List[GroupDist] = []
    Cinv_T = np.linalg.inv(readout_C).T if (readout_C is not None and mitigate_readout) else None
    for gr in plan.groups:
        r = rho
        for g in basis_rotation_gates(gr["basis"]):
            r = apply_superop(r, unitary_superop(unitary(g)), g.qubits, n)
        p = np.real(np.diag(r)).clip(min=0.0)
        p = p / p.sum()
        if readout_C is not None:
            p = apply_readout(p, readout_C, n)
            p = p.clip(min=0.0)
            p = p / p.sum()
        mem = gr["members"]
        V = parity_matrix([plan.terms[i] for i in mem], n)
        if Cinv_T is not None:
            Vt = V.reshape((len(mem),) + (2,) * n)
            for q in range(n):
                Vt = np.moveaxis(np.tensordot(Cinv_T, Vt, axes=([1], [q + 1])), 0, q + 1)
            V = Vt.reshape(len(mem), -1)
        u = plan.coeffs[mem] @ V
        dists.append(GroupDist(mem, p, V, u, V @ p))
    return dists


def infinite_shot_energy(dists: Sequence[GroupDist], plan: MeasurementPlan) -> float:
    """Energy the estimator would return with infinitely many shots."""
    return float(plan.const + sum(plan.coeffs[g.members] @ g.exact_terms for g in dists))


def unit_group_sigma(dists: Sequence[GroupDist]) -> np.ndarray:
    """Per-shot standard deviation of each group's energy contribution (exact)."""
    return np.array([np.sqrt(max(g.p @ g.u ** 2 - (g.p @ g.u) ** 2, 0.0)) for g in dists])


def allocate_shots(total: int, weights: Sequence[float], min_shots: int = 2) -> np.ndarray:
    """Integer allocation of `total` shots proportional to `weights` (largest remainder),
    at least `min_shots` per group (needed for a variance estimate)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    raw = total * w
    n = np.maximum(min_shots, np.floor(raw)).astype(int)
    deficit = total - n.sum()
    if deficit > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        n[order[:deficit]] += 1
    return n


def group_shot_weights(plan: MeasurementPlan, dists: Sequence[GroupDist], mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(plan.groups))
    if mode == "l1":
        return np.array([np.abs(plan.coeffs[g["members"]]).sum() for g in plan.groups])
    if mode == "oracle":   # Neyman allocation using the exact per-shot variances
        return np.maximum(unit_group_sigma(dists), 1e-12)
    raise ValueError(mode)


def sample_energies(dists: Sequence[GroupDist], plan: MeasurementPlan, shots: np.ndarray,
                    n_reps: int, rng: np.random.Generator, keep_cov_reps: int = 0):
    """Draw n_reps independent shot-noise realisations of one energy estimate.
    Returns dict with
        E        (R,)   energy estimates
        sigma    (R,)   estimated standard error of E (from the samples)
        terms    (R,T)  term estimates
        term_var (R,T)  estimated variance of each term estimate
        cov      list over groups of (keep_cov_reps, k, k) covariance-of-the-mean blocks
    """
    T = plan.n_terms
    terms = np.zeros((n_reps, T))
    term_var = np.zeros((n_reps, T))
    var_E = np.zeros(n_reps)
    cov_blocks = []
    for g, n_g in zip(dists, shots):
        counts = rng.multinomial(int(n_g), g.p, size=n_reps)
        ph = counts / n_g
        m = ph @ g.V.T
        corr = n_g / (n_g - 1.0)
        terms[:, g.members] = m
        term_var[:, g.members] = corr * (ph @ (g.V ** 2).T - m ** 2) / n_g
        e_g = ph @ g.u
        var_E += corr * (ph @ (g.u ** 2) - e_g ** 2) / n_g
        if keep_cov_reps:
            blocks = np.empty((keep_cov_reps, len(g.members), len(g.members)))
            for r in range(keep_cov_reps):
                Vc = g.V - m[r][:, None]
                blocks[r] = corr * (Vc * ph[r]) @ Vc.T / n_g
            cov_blocks.append(blocks)
    E = plan.const + terms @ plan.coeffs
    return {"E": E, "sigma": np.sqrt(np.maximum(var_E, 1e-300)), "terms": terms,
            "term_var": term_var, "cov": cov_blocks}


def counts_to_vector(counts: Dict[str, int], n: int) -> np.ndarray:
    """Qiskit-style counts {'0101': 37, ...} (rightmost char = qubit 0) -> length-2**n
    vector indexed with qubit 0 as the most-significant bit (our convention)."""
    v = np.zeros(2 ** n)
    for s, c in counts.items():
        s = s.replace(" ", "")
        idx = 0
        for q in range(n):
            idx |= int(s[n - 1 - q]) << (n - 1 - q)
        v[idx] += c
    return v


def estimate_from_counts(count_vectors: Sequence[np.ndarray], dists_like: Sequence[GroupDist],
                         plan: MeasurementPlan):
    """Hardware path: same estimator as sample_energies but from observed count vectors
    (one per group).  `dists_like` supplies V and u only (build them with
    prepare_distributions on any state; p is ignored)."""
    T = plan.n_terms
    terms, term_var = np.zeros(T), np.zeros(T)
    var_E = 0.0
    for g, cv in zip(dists_like, count_vectors):
        n_g = cv.sum()
        ph = cv / n_g
        m = g.V @ ph
        corr = n_g / (n_g - 1.0)
        terms[g.members] = m
        term_var[g.members] = corr * (g.V ** 2 @ ph - m ** 2) / n_g
        e_g = g.u @ ph
        var_E += corr * (g.u ** 2 @ ph - e_g ** 2) / n_g
    return {"E": plan.const + terms @ plan.coeffs, "sigma": np.sqrt(var_E),
            "terms": terms, "term_var": term_var}
