"""
chem.py
Real molecular Hamiltonians: PySCF integrals -> OpenFermion Jordan-Wigner qubit operator.
Geometries / active spaces are those of the original package (STO-3G).

Differences to the old hamiltonian.py that matter for correctness of the paper:
  * the reference for ansatz quality is the EXACT ground energy INSIDE the active space and
    the (Nalpha, Nbeta) sector, computed by exact_sector_energy (not the full-space PySCF FCI,
    which is a different Hamiltonian whenever a core is frozen or orbitals are dropped);
  * full-space FCI is optional (run_fci=True) and only used to report the active-space
    truncation error separately.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh

_ANGLE_H2O = np.radians(104.5)

GEOMETRIES = {
    "H2": {"geometry": lambda d: [("H", (0, 0, 0)), ("H", (0, 0, d))],
           "multiplicity": 1, "charge": 0, "occupied_indices": None, "active_indices": None,
           "eq_bond_length": 0.74},
    "LiH": {"geometry": lambda d: [("Li", (0, 0, 0)), ("H", (0, 0, d))],
            "multiplicity": 1, "charge": 0, "occupied_indices": [0],
            "active_indices": [1, 2, 3, 4, 5], "eq_bond_length": 1.60},
    "H4": {"geometry": lambda d: [("H", (0, 0, 0)), ("H", (d, 0, 0)), ("H", (d, d, 0)), ("H", (0, d, 0))],
           "multiplicity": 1, "charge": 0, "occupied_indices": None, "active_indices": None,
           "eq_bond_length": 1.00},
    "HeH+": {"geometry": lambda d: [("He", (0, 0, 0)), ("H", (0, 0, d))],
             "multiplicity": 1, "charge": 1, "occupied_indices": None, "active_indices": None,
             "eq_bond_length": 0.90},
    "BeH2": {"geometry": lambda d: [("H", (0, 0, -d)), ("Be", (0, 0, 0)), ("H", (0, 0, d))],
             "multiplicity": 1, "charge": 0, "occupied_indices": [0],
             "active_indices": [1, 2, 3, 4, 5, 6], "eq_bond_length": 1.30},
    "H2O": {"geometry": lambda d: [("O", (0, 0, 0)),
                                    ("H", (d * np.sin(_ANGLE_H2O / 2), d * np.cos(_ANGLE_H2O / 2), 0)),
                                    ("H", (-d * np.sin(_ANGLE_H2O / 2), d * np.cos(_ANGLE_H2O / 2), 0))],
            "multiplicity": 1, "charge": 0, "occupied_indices": [0],
            "active_indices": [1, 2, 3, 4, 5], "eq_bond_length": 0.9584},
    "N2": {"geometry": lambda d: [("N", (0, 0, 0)), ("N", (0, 0, d))],
           "multiplicity": 1, "charge": 0, "occupied_indices": [0, 1, 2, 3],
           "active_indices": [4, 5, 6, 7, 8, 9], "eq_bond_length": 1.10},
}


def build_hamiltonian(name: str, bond_length: float, basis: str = "sto-3g", run_fci: bool = False) -> dict:
    from openfermion.chem import MolecularData
    from openfermionpyscf import run_pyscf
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    from openfermion.linalg import get_sparse_operator

    spec = GEOMETRIES[name]
    mol = MolecularData(spec["geometry"](bond_length), basis, spec["multiplicity"], spec["charge"])
    mol = run_pyscf(mol, run_scf=True, run_fci=run_fci)
    mol_ham = mol.get_molecular_hamiltonian(occupied_indices=spec["occupied_indices"],
                                            active_indices=spec["active_indices"])
    qop = jordan_wigner(get_fermion_operator(mol_ham))
    n_qubits = mol.n_qubits if spec["active_indices"] is None else 2 * len(spec["active_indices"])
    n_core = 0 if spec["occupied_indices"] is None else len(spec["occupied_indices"])
    n_elec = mol.n_electrons - 2 * n_core
    return {"qubit_operator": qop, "H_sparse": get_sparse_operator(qop, n_qubits=n_qubits).tocsr(),
            "n_qubits": n_qubits, "n_electrons_active": n_elec,
            "hf_energy_pyscf": float(mol.hf_energy),
            "fci_energy_full_space": float(mol.fci_energy) if run_fci else None,
            "basis": basis, "bond_length": float(bond_length), "name": name}


def sector_indices(n_qubits: int, n_alpha: int, n_beta: int) -> np.ndarray:
    """Basis-state indices with the given alpha/beta electron numbers.  Spin orbital
    2p is alpha and 2p+1 beta (OpenFermion convention); qubit 0 is the most-significant bit."""
    x = np.arange(2 ** n_qubits)
    bits = np.stack([(x >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)], axis=1)
    mask = (bits[:, 0::2].sum(1) == n_alpha) & (bits[:, 1::2].sum(1) == n_beta)
    return np.nonzero(mask)[0]


def exact_sector_energy(H_sparse, n_qubits: int, n_electrons: int) -> float:
    """Exact ground energy of H restricted to the closed-shell (N/2, N/2) sector."""
    idx = sector_indices(n_qubits, n_electrons // 2, n_electrons // 2)
    Hs = H_sparse[idx][:, idx]
    if Hs.shape[0] <= 4000:
        return float(eigh(Hs.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])
    return float(spla.eigsh(Hs, k=1, which="SA")[0][0])
