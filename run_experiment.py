"""
run_experiment.py
Generate the raw data of one experiment (one molecule / bond length / ansatz / noise level / shot budget):

  1. real Hamiltonian (PySCF)                       -> chem.build_hamiltonian
  2. noiseless VQE (UCCSD or ADAPT, analytic grads) -> ansatz.py   => E_ref = noiseless VQE energy
  3. gate-level circuit, simplified once            -> circuits.py
  4. device-informed noise model from REAL calibration data -> noise.py / data/calibration.json
  5. for every noise scale: fold the circuit, exact noisy density matrix, effective scale
  6. QWC-grouped shot sampling with readout error (+ optional mitigation), R independent repetitions
  7. save everything needed by analyze.py (no estimator is applied here)

    python run_experiment.py --molecule H2 --ansatz uccsd --shots 1000000 --reps 500

The reference for ALL error statistics is E_ref (the noiseless VQE energy of the same ansatz):
zero-noise extrapolation can only recover that, not the FCI energy.  The ansatz error
(E_ref - E_exact_active) is reported separately.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import platform
import time

import numpy as np

import ansatz as A
import chem
import circuits as C
import dm_sim
import estimators as EST
import measurement as M
import noise as N


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True, choices=list(chem.GEOMETRIES))
    ap.add_argument("--bond", type=float, default=None, help="bond length (Angstrom); default equilibrium")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--ansatz", choices=["uccsd", "adapt"], default="uccsd")
    ap.add_argument("--adapt-max-ops", type=int, default=60)
    ap.add_argument("--adapt-grad-tol", type=float, default=1e-4)
    ap.add_argument("--calibration", default="data/calibration.json")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="multiply all gate errors (and divide T1,T2)")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--shots", type=int, default=1_000_000, help="total shots per energy estimate at each noise level")
    ap.add_argument("--reps", type=int, default=500, help="independent shot-noise repetitions")
    ap.add_argument("--hier-reps", type=int, default=0, help="repetitions for which covariance blocks are stored (hierarchical model)")
    ap.add_argument("--group-alloc", choices=["uniform", "l1", "oracle"], default="l1")
    ap.add_argument("--lambda-alloc", choices=["uniform", "neyman"], default="uniform")
    ap.add_argument("--neyman-order", type=int, default=1)
    ap.add_argument("--no-readout-mitigation", action="store_true")
    ap.add_argument("--max-qubits", type=int, default=10, help="density-matrix memory guard")
    ap.add_argument("--with-fci", action="store_true", help="also compute full-space FCI (truncation error)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def main():
    a = parse()
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    bond = a.bond if a.bond is not None else chem.GEOMETRIES[a.molecule]["eq_bond_length"]
    tag = f"{a.molecule}_r{bond:.3f}_{a.ansatz}_S{a.shots}_x{a.noise_scale:g}_{a.group_alloc}_{a.lambda_alloc}"

    # 1. Hamiltonian ------------------------------------------------------
    ham = chem.build_hamiltonian(a.molecule, bond, a.basis, run_fci=a.with_fci)
    n, ne, H = ham["n_qubits"], ham["n_electrons_active"], ham["H_sparse"]
    if n > a.max_qubits:
        raise SystemExit(f"{n} qubits > --max-qubits {a.max_qubits}: the exact noisy density-matrix "
                         f"simulation would need {16 * 4 ** n / 2 ** 30:.1f} GiB per state. "
                         f"Use a smaller molecule or raise --max-qubits deliberately.")
    E_exact = chem.exact_sector_energy(H, n, ne)

    # 2. noiseless VQE (cached) -------------------------------------------
    cache = os.path.join(a.out_dir, f"ansatz_{a.molecule}_r{bond:.3f}_{a.ansatz}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            ans = json.load(f)
    else:
        if a.ansatz == "uccsd":
            res = A.run_uccsd(H, n, ne)
        else:
            res = A.run_adapt(H, n, ne, max_ops=a.adapt_max_ops, grad_tol=a.adapt_grad_tol)
        ans = A.serialise(res)
        ans.update({"converged": res["converged"], "n_params": len(res["thetas"]),
                    "energy_log": [float(x) for x in res["energy_log"]]})
        if "history" in res:
            ans["history"] = res["history"]
        with open(cache, "w") as f:
            json.dump(ans, f)
    E_ref = ans["energy"]
    print(f"[{a.molecule}] HF={ans['hf_energy']:.8f}  VQE({a.ansatz})={E_ref:.8f}  "
          f"exact(active)={E_exact:.8f}  ansatz error={1e3 * (E_ref - E_exact):.4f} mHa  "
          f"params={ans['n_params']}  converged={ans['converged']}")

    # 3. circuit -----------------------------------------------------------
    gens = A.generator_terms_from_json(ans)
    circ = C.build_ansatz_circuit(n, ne, gens, ans["thetas"])
    psi_c = dm_sim.statevector(circ)
    E_circ = float(np.real(np.vdot(psi_c, H @ psi_c)))
    if abs(E_circ - E_ref) > 1e-7:
        raise SystemExit(f"compiled circuit energy {E_circ} != VQE energy {E_ref}: compilation bug")
    print(f"circuit gate counts (base): {circ.counts()}  (verified against statevector VQE, |dE|={abs(E_circ - E_ref):.1e})")

    # 4. noise model from real calibration ---------------------------------
    model = N.DeviceNoiseModel.from_json(a.calibration).scaled(a.noise_scale)

    # 5. folded circuits, exact noisy states -------------------------------
    lam_nom = np.array(a.lambdas, float)
    plan = M.build_plan(ham["qubit_operator"], n)
    ro = model.readout_confusion()
    mitigate = not a.no_readout_mitigation
    dists_all, lam_eff, E_noisy_rho = [], [], []
    for lam in lam_nom:
        f = circ.folded(lam, model.gate_infidelity)
        lam_eff.append(N.effective_scale(circ.all_gates(), f.all_gates(), model))
        rho = dm_sim.run_density_matrix(f, model)
        E_noisy_rho.append(dm_sim.energy_from_rho(rho, H))
        dists_all.append(M.prepare_distributions(rho, plan, ro, mitigate))
        print(f"  lambda_nom={lam:g} -> lambda_eff={lam_eff[-1]:.4f}  gates={len(f.all_gates())}  "
              f"E(exact noisy)={E_noisy_rho[-1]:.8f}  ({time.time() - t0:.0f}s)")
    lam_eff = np.array(lam_eff)
    E_inf = np.array([M.infinite_shot_energy(d, plan) for d in dists_all])

    # 6. shot allocation & sampling -----------------------------------------
    m, G = len(lam_nom), len(plan.groups)
    gw = [M.group_shot_weights(plan, d, a.group_alloc) for d in dists_all]
    if a.lambda_alloc == "neyman":
        sig_unit = []
        for d, w in zip(dists_all, gw):
            f_g = w / w.sum()
            sig_unit.append(np.sqrt(np.sum(M.unit_group_sigma(d) ** 2 / f_g)))
        w_ext = EST.linear_extrapolation_weights(lam_eff, order=min(a.neyman_order, m - 1))
        S_lam = EST.neyman_lambda_allocation(w_ext, sig_unit, m * a.shots, min_shots=max(100, 2 * G))
    else:
        S_lam = np.full(m, a.shots, dtype=int)
    shots_group = np.array([M.allocate_shots(int(S), w) for S, w in zip(S_lam, gw)])
    children = np.random.SeedSequence(a.seed).spawn(m)
    E_hat = np.zeros((a.reps, m))
    sig_hat = np.zeros((a.reps, m))
    terms = np.zeros((a.reps, m, plan.n_terms))
    term_var = np.zeros((a.reps, m, plan.n_terms))
    cov_store = []
    for j in range(m):
        out = M.sample_energies(dists_all[j], plan, shots_group[j], a.reps,
                                np.random.default_rng(children[j]), keep_cov_reps=a.hier_reps)
        E_hat[:, j], sig_hat[:, j] = out["E"], out["sigma"]
        terms[:, j], term_var[:, j] = out["terms"], out["term_var"]
        cov_store.append(out["cov"])
    print(f"sampled {a.reps} repetitions; median sigma(E) per level [mHa]: {np.round(1e3 * np.median(sig_hat, 0), 4)}")

    # 7. save ---------------------------------------------------------------
    np.savez_compressed(os.path.join(a.out_dir, f"raw_{tag}.npz"),
                        lam_nom=lam_nom, lam_eff=lam_eff, E_hat=E_hat, sig_hat=sig_hat, terms=terms,
                        term_var=term_var, E_inf=E_inf, E_noisy_rho=np.array(E_noisy_rho), E_ref=E_ref,
                        E_exact_active=E_exact, E_hf=ans["hf_energy"],
                        E_fci_full=np.nan if ham["fci_energy_full_space"] is None else ham["fci_energy_full_space"],
                        shots_lambda=S_lam, shots_group=shots_group, coeffs=plan.coeffs, const=plan.const,
                        weights=plan.weights)
    with open(os.path.join(a.out_dir, f"plan_{tag}.pkl"), "wb") as f:
        pickle.dump({"plan": plan, "cov": cov_store if a.hier_reps else None}, f)
    import scipy
    meta = {"tag": tag, "args": vars(a), "bond": bond, "n_qubits": n, "n_electrons_active": ne,
            "n_params": ans["n_params"], "ansatz_converged": ans["converged"],
            "gate_counts_base": circ.counts(), "n_pauli_terms": plan.n_terms, "n_groups": G,
            "noise_model": {k: getattr(model, k) for k in ("e1", "e2", "T1", "T2", "t1q", "t2q", "p01", "p10")},
            "noise_provenance": model.provenance, "readout_mitigated": mitigate,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version()},
            "runtime_s": time.time() - t0}
    with open(os.path.join(a.out_dir, f"meta_{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("saved", tag)


if __name__ == "__main__":
    main()
