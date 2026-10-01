"""
ansatz.py
Noiseless VQE with a spin-conserving fermionic excitation pool (singles + doubles).

  * UCCSD  : all pool operators, ordered [singles, doubles], optimised with L-BFGS-B and
             ANALYTIC gradients (adjoint method) instead of COBYLA.
  * ADAPT  : ADAPT-VQE (Grimsley et al. 2019): grow the ansatz one operator at a time,
             choosing the largest |dE/dtheta| from the pool; re-optimise all parameters.

Exact exponentials.  For a fermionic excitation T (distinct spin orbitals) the generator
G = T - T^dagger satisfies G^3 = -G (checked numerically for every generator), hence
    exp(theta G) = 1 + sin(theta) G + (1 - cos(theta)) G^2
exactly; this is much cheaper than expm_multiply and is what makes analytic gradients cheap.

Every generator is also stored as its Jordan-Wigner Pauli decomposition
G = i * sum_k c_k P_k (commuting P_k, checked) so that circuits.py can compile it exactly.
"""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from pauli import PauliString, commute


@dataclass
class Generator:
    label: str
    terms: List[Tuple[PauliString, float]]     # G = i * sum c_k P_k
    matrix: sp.csr_matrix                       # sparse matrix of G (anti-Hermitian)


# ----------------------------------------------------------------------
# Operator pool
# ----------------------------------------------------------------------

def _make_generator(fermion_T, n_qubits: int, label: str) -> Generator:
    from openfermion import hermitian_conjugated, jordan_wigner
    from openfermion.linalg import get_sparse_operator

    G_f = fermion_T - hermitian_conjugated(fermion_T)
    qop = jordan_wigner(G_f)
    terms = []
    for term, coef in qop.terms.items():
        if len(term) == 0:
            continue
        if abs(complex(coef).real) > 1e-12:
            raise RuntimeError(f"{label}: JW coefficient not purely imaginary")
        terms.append((tuple(term), float(complex(coef).imag)))
    for (a, _), (b, _) in itertools.combinations(terms, 2):
        if not commute(a, b):
            raise RuntimeError(f"{label}: Pauli terms of a single excitation must commute")
    M = get_sparse_operator(qop, n_qubits=n_qubits).tocsr()
    resid = abs(M @ M @ M + M).max()
    if resid > 1e-10:
        raise RuntimeError(f"{label}: G^3 = -G violated ({resid:.2e}); closed-form exp invalid")
    return Generator(label, terms, M)


def build_pool(n_electrons: int, n_qubits: int, kinds: Sequence[str] = ("singles", "doubles")) -> List[Generator]:
    """Spin-orbital 2p = alpha, 2p+1 = beta.  Closed-shell reference with n_electrons//2 doubly
    occupied spatial orbitals.  Singles conserve spin; doubles: alpha-alpha, beta-beta, alpha-beta."""
    from openfermion import FermionOperator

    n_occ, n_tot = n_electrons // 2, n_qubits // 2
    occ, virt = list(range(n_occ)), list(range(n_occ, n_tot))
    pool: List[Generator] = []
    if "singles" in kinds:
        for s in (0, 1):
            for i in occ:
                for a in virt:
                    T = FermionOperator(((2 * a + s, 1), (2 * i + s, 0)))
                    pool.append(_make_generator(T, n_qubits, f"S{'ab'[s]}({i}->{a})"))
    if "doubles" in kinds:
        def dbl(i, j, a, b, lab):
            T = FermionOperator(((b, 1), (a, 1), (j, 0), (i, 0)))
            pool.append(_make_generator(T, n_qubits, lab))
        for s in (0, 1):
            for p1, p2 in itertools.combinations(occ, 2):
                for q1, q2 in itertools.combinations(virt, 2):
                    dbl(2 * p1 + s, 2 * p2 + s, 2 * q1 + s, 2 * q2 + s,
                        f"D{'ab'[s]}{'ab'[s]}({p1},{p2}->{q1},{q2})")
        for p, pp in itertools.product(occ, occ):
            for q, qq in itertools.product(virt, virt):
                dbl(2 * p, 2 * pp + 1, 2 * q, 2 * qq + 1, f"Dab({p},{pp}->{q},{qq})")
    return pool


# ----------------------------------------------------------------------
# State, energy, analytic gradient
# ----------------------------------------------------------------------

def hf_statevector(n_qubits: int, n_electrons: int) -> np.ndarray:
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[int("1" * n_electrons + "0" * (n_qubits - n_electrons), 2)] = 1.0
    return psi


def exp_apply(G, theta: float, psi: np.ndarray) -> np.ndarray:
    Gp = G @ psi
    return psi + np.sin(theta) * Gp + (1.0 - np.cos(theta)) * (G @ Gp)


def prepare_state(thetas: Sequence[float], gens: Sequence[Generator], psi0: np.ndarray) -> np.ndarray:
    psi = psi0
    for th, g in zip(thetas, gens):
        psi = exp_apply(g.matrix, th, psi)
    return psi


def energy(psi: np.ndarray, H) -> float:
    return float(np.real(np.vdot(psi, H @ psi)))


def energy_and_grad(thetas: np.ndarray, gens: Sequence[Generator], psi0: np.ndarray, H):
    """E(theta) and dE/dtheta_k = 2 Re <lambda_k| G_k |phi_k> by the adjoint method
    (phi_k: state after gate k; lambda_k = (U_n..U_{k+1})^dagger H psi)."""
    psi = prepare_state(thetas, gens, psi0)
    Hpsi = H @ psi
    E = float(np.real(np.vdot(psi, Hpsi)))
    grad = np.zeros(len(gens))
    phi, lam = psi, Hpsi
    for k in range(len(gens) - 1, -1, -1):
        G = gens[k].matrix
        grad[k] = 2.0 * np.real(np.vdot(lam, G @ phi))
        phi = exp_apply(G, -thetas[k], phi)
        lam = exp_apply(G, -thetas[k], lam)
    return E, grad


def _optimise(thetas0, gens, psi0, H, maxiter, gtol, log):
    def fun(x):
        E, g = energy_and_grad(x, gens, psi0, H)
        return E, g

    def cb(x):
        log.append(energy_and_grad(x, gens, psi0, H)[0])

    res = minimize(fun, thetas0, jac=True, method="L-BFGS-B", callback=cb,
                   options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15, "maxcor": 30})
    return res.x, float(res.fun), res


def run_uccsd(H, n_qubits: int, n_electrons: int, maxiter: int = 2000, gtol: float = 1e-9,
              seed: int = 0, pool: Optional[List[Generator]] = None) -> dict:
    pool = pool if pool is not None else build_pool(n_electrons, n_qubits)
    psi0 = hf_statevector(n_qubits, n_electrons)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1e-3, len(pool))
    log: List[float] = [energy(psi0, H)]
    th, E, res = _optimise(x0, pool, psi0, H, maxiter, gtol, log)
    psi = prepare_state(th, pool, psi0)
    return {"kind": "uccsd", "generators": pool, "thetas": th, "energy": E, "psi": psi,
            "energy_log": np.array(log), "converged": bool(res.success), "n_iter": int(res.nit),
            "grad_norm": float(np.linalg.norm(energy_and_grad(th, pool, psi0, H)[1])),
            "hf_energy": energy(psi0, H)}


def run_adapt(H, n_qubits: int, n_electrons: int, max_ops: int = 60, grad_tol: float = 1e-4,
              maxiter: int = 500, gtol: float = 1e-9, pool: Optional[List[Generator]] = None) -> dict:
    pool = pool if pool is not None else build_pool(n_electrons, n_qubits)
    psi0 = hf_statevector(n_qubits, n_electrons)
    selected: List[Generator] = []
    th = np.zeros(0)
    log: List[float] = [energy(psi0, H)]
    history = []
    psi = psi0
    while len(selected) < max_ops:
        Hpsi = H @ psi
        grads = np.array([2.0 * np.real(np.vdot(Hpsi, g.matrix @ psi)) for g in pool])
        k = int(np.argmax(np.abs(grads)))
        gmax = float(abs(grads[k]))
        history.append({"iter": len(selected), "max_pool_gradient": gmax, "chosen": pool[k].label})
        if gmax < grad_tol:
            break
        selected.append(pool[k])
        th = np.append(th, 0.0)
        th, E, _ = _optimise(th, selected, psi0, H, maxiter, gtol, log)
        psi = prepare_state(th, selected, psi0)
    E = energy(psi, H)
    return {"kind": "adapt", "generators": selected, "thetas": th, "energy": E, "psi": psi,
            "energy_log": np.array(log), "history": history,
            "converged": bool(history and history[-1]["max_pool_gradient"] < grad_tol),
            "hf_energy": energy(psi0, H)}


def serialise(res: dict) -> dict:
    """JSON-friendly ansatz description (enough to rebuild the circuit without re-running VQE)."""
    return {"kind": res["kind"], "energy": res["energy"], "hf_energy": res["hf_energy"],
            "thetas": [float(t) for t in res["thetas"]],
            "generators": [{"label": g.label, "terms": [[[list(p) for p in t], c] for t, c in g.terms]}
                           for g in res["generators"]]}


def generator_terms_from_json(d: dict):
    return [[(tuple((int(q), str(l)) for q, l in t), float(c)) for t, c in g["terms"]]
            for g in d["generators"]]
