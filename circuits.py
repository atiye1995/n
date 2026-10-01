"""
circuits.py
Gate-level compilation of the VQE ansatz, peephole simplification, and
*local unitary folding* for zero-noise extrapolation.

Compilation of one excitation generator
---------------------------------------
For a fermionic (single or double) excitation T, the anti-Hermitian generator
G = T - T^dagger maps under Jordan-Wigner to G = i * sum_k c_k P_k with real c_k
and MUTUALLY COMMUTING Pauli strings P_k.  Therefore, exactly (no Trotter error)
    exp(theta G) = prod_k exp(i theta c_k P_k) = prod_k R_{P_k}(phi_k),
    R_P(phi) = exp(-i phi P / 2),   phi_k = -2 theta c_k.
Each R_P(phi) = V^dagger exp(-i phi Z..Z / 2) V with V a product of single-qubit
basis changes (H for X, Rx(pi/2) for Y) and a CNOT ladder onto a pivot qubit.

Folding
-------
A physical gate g is replaced by g g^dagger g (identical unitary, 3x the noisy
operations).  Fractional noise-scale factors are obtained by folding an evenly
spaced subset of gates.  The virtual rz gates and the state-preparation block
are never folded.  The *actual* noise-scale factor must be computed from the
noise model (noise.effective_scale) and used as the extrapolation abscissa.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import numpy as np

from gates import Gate, PHYSICAL, inverse, is_inverse_pair
from pauli import PauliString


@dataclass
class Circuit:
    n_qubits: int
    prep: List[Gate] = field(default_factory=list)   # state preparation (never folded)
    body: List[Gate] = field(default_factory=list)   # ansatz block (folded)

    def all_gates(self) -> List[Gate]:
        return list(self.prep) + list(self.body)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for g in self.all_gates():
            c[g.name] += 1
        return dict(c)

    def folded(self, scale: float, weight=None) -> "Circuit":
        return Circuit(self.n_qubits, list(self.prep), fold_local(self.body, scale, weight))


def pauli_rotation(term: PauliString, phi: float) -> List[Gate]:
    """Gates implementing exp(-i phi P / 2) for the Pauli string P."""
    if len(term) == 0 or abs(phi) < 1e-14:
        return []
    qubits = [q for q, _ in term]
    pre, post = [], []
    for q, p in term:
        if p == "X":
            pre.append(Gate("h", (q,)))
            post.append(Gate("h", (q,)))
        elif p == "Y":
            pre.append(Gate("rx", (q,), float(np.pi / 2)))
            post.append(Gate("rx", (q,), float(-np.pi / 2)))
    ladder = [Gate("cx", (qubits[k], qubits[k + 1])) for k in range(len(qubits) - 1)]
    return pre + ladder + [Gate("rz", (qubits[-1],), float(phi))] + ladder[::-1] + post


def compile_generator(gen_terms: Sequence[Tuple[PauliString, float]], theta: float) -> List[Gate]:
    """exp(theta * G) with G = i * sum_k c_k P_k  (gen_terms = [(P_k, c_k), ...])."""
    gates: List[Gate] = []
    for term, c in gen_terms:
        gates += pauli_rotation(term, -2.0 * theta * c)
    return gates


def simplify(gates: Sequence[Gate], tol: float = 1e-12) -> List[Gate]:
    """Peephole pass: cancel adjacent inverse pairs and merge adjacent rz on the same
    qubit.  Applied ONCE to the base circuit before folding (a hardware transpiler at
    optimisation level >= 1 would do the same; folding must act on the circuit that
    is actually executed)."""
    out: List = []
    stacks: Dict[int, List[int]] = defaultdict(list)

    def push(g: Gate):
        idx = len(out)
        out.append(g)
        for q in g.qubits:
            stacks[q].append(idx)

    def pop(idx: int):
        g = out[idx]
        out[idx] = None
        for q in g.qubits:
            assert stacks[q][-1] == idx
            stacks[q].pop()

    for g in gates:
        tops = [stacks[q][-1] if stacks[q] else None for q in g.qubits]
        j = tops[0]
        if j is not None and all(t == j for t in tops) and out[j].qubits == g.qubits:
            prev = out[j]
            if g.name == "rz" and prev.name == "rz":
                ang = prev.param + g.param
                pop(j)
                if abs(ang) > tol:
                    push(Gate("rz", g.qubits, ang))
                continue
            if is_inverse_pair(prev, g, tol):
                pop(j)
                continue
        push(g)
    return [g for g in out if g is not None]


def fold_local(gates: Sequence[Gate], scale: float, weight=None) -> List[Gate]:
    """Fold physical gates so that the noise budget is ~ scale x the original.

    Each fold turns g into g g^dagger g (adds 2 noisy operations).  Whole passes over all
    physical gates are applied first; the fractional remainder folds a well-spread subset
    (golden-ratio ordering).  If `weight` (callable gate -> noise budget, e.g.
    DeviceNoiseModel.gate_infidelity) is given, the budget is counted in those units so
    that scale ~ the true noise-scale factor even when 1q and 2q gates differ a lot;
    otherwise every physical gate counts equally.  scale = 1 -> unchanged.
    Always use noise.effective_scale() for the abscissa of the extrapolation."""
    if scale < 1.0:
        raise ValueError("noise-scale factor must be >= 1 for circuit folding")
    idx = [i for i, g in enumerate(gates) if g.name in PHYSICAL]
    if not idx or scale == 1.0:
        return list(gates)
    w = {i: (1.0 if weight is None else float(weight(gates[i]))) for i in idx}
    W = sum(w.values())
    target = (scale - 1.0) / 2.0 * W               # extra folds' budget, in units of W
    full = int(np.floor(target / W + 1e-12))
    remainder = target - full * W
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    pos = {i: k for k, i in enumerate(idx)}
    order = sorted(idx, key=lambda i: (pos[i] * phi) % 1.0)
    chosen, acc = set(), 0.0
    for i in order:
        if acc + 0.5 * w[i] > remainder + 1e-15:
            continue
        chosen.add(i)
        acc += w[i]
        if acc >= remainder:
            break
    out: List[Gate] = []
    for i, g in enumerate(gates):
        out.append(g)
        if g.name in PHYSICAL:
            reps = full + (1 if i in chosen else 0)
            for _ in range(reps):
                out += [inverse(g), g]
    return out


def build_ansatz_circuit(n_qubits: int, n_electrons: int,
                          generators: Sequence[Sequence[Tuple[PauliString, float]]],
                          thetas: Sequence[float], simplify_gates: bool = True) -> Circuit:
    """HF preparation (X on the first n_electrons qubits) + ordered product of
    exp(theta_k G_k), identical in ordering to the statevector ansatz in ansatz.py."""
    prep = [Gate("x", (q,)) for q in range(n_electrons)]
    body: List[Gate] = []
    for gen_terms, th in zip(generators, thetas):
        if abs(th) < 1e-14:
            continue
        body += compile_generator(gen_terms, th)
    if simplify_gates:
        body = simplify(body)
    return Circuit(n_qubits, prep, body)
