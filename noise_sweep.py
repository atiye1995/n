"""
noise_sweep.py
Infinite-shot (no statistical noise) bias of every estimator as a function of the noise strength.
This is the quantitative test of the extrapolation theory: for a low-order estimator the bias
should scale as a power of the noise strength, and the sweep shows where each method breaks down.

    python noise_sweep.py --molecule H2 --ansatz uccsd --scales 0.1 0.3 1 3 10

Uses the same real Hamiltonian, ansatz cache and calibration-derived noise model as run_experiment.py;
the noise strength is scaled by --scales (gate errors x s, T1/T2 / s).  Output: results/sweep_*.csv
"""
import argparse, json, os
import numpy as np
import pandas as pd

import ansatz as A, chem, circuits as C, dm_sim, estimators as EST, measurement as M, noise as N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True, choices=list(chem.GEOMETRIES))
    ap.add_argument("--bond", type=float, default=None)
    ap.add_argument("--ansatz", choices=["uccsd", "adapt"], default="uccsd")
    ap.add_argument("--calibration", default="data/calibration.json")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    bond = a.bond if a.bond is not None else chem.GEOMETRIES[a.molecule]["eq_bond_length"]
    cache = os.path.join(a.out_dir, f"ansatz_{a.molecule}_r{bond:.3f}_{a.ansatz}.json")
    if not os.path.exists(cache):
        raise SystemExit(f"{cache} not found: run run_experiment.py for this molecule/ansatz first")
    ans = json.load(open(cache))
    ham = chem.build_hamiltonian(a.molecule, bond)
    n, ne, H = ham["n_qubits"], ham["n_electrons_active"], ham["H_sparse"]
    circ = C.build_ansatz_circuit(n, ne, A.generator_terms_from_json(ans), ans["thetas"])
    plan = M.build_plan(ham["qubit_operator"], n)
    base = N.DeviceNoiseModel.from_json(a.calibration)
    E_ref, rows = ans["energy"], []
    for s in a.scales:
        model = base.scaled(s)
        lam_eff, E = [], []
        for lam in a.lambdas:
            f = circ.folded(lam, model.gate_infidelity)
            lam_eff.append(N.effective_scale(circ.all_gates(), f.all_gates(), model))
            rho = dm_sim.run_density_matrix(f, model)
            d = M.prepare_distributions(rho, plan, model.readout_confusion(), True)
            E.append(M.infinite_shot_energy(d, plan))
        lam_eff, E = np.array(lam_eff), np.array(E)
        res = EST.run_all(lam_eff, E, np.full(len(E), 1e-6))
        row = {"noise_scale": s, "raw_error": E[0] - E_ref}
        row.update({f"err_{k}": (v.F - E_ref) if v.ok else np.nan for k, v in res.items()})
        rows.append(row)
        print(f"scale {s:g}: raw error {1e3 * row['raw_error']:.4f} mHa")
    df = pd.DataFrame(rows)
    out = os.path.join(a.out_dir, f"sweep_{a.molecule}_r{bond:.3f}_{a.ansatz}.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    # power-law exponents of |error| vs noise strength (log-log slope)
    for c in [c for c in df.columns if c.startswith("err_") or c == "raw_error"]:
        v = np.abs(df[c].values)
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 3:
            print(f"slope d log|{c}| / d log(noise) = {np.polyfit(np.log(df['noise_scale'][ok]), np.log(v[ok]), 1)[0]:.2f}")
    print("saved", out)


if __name__ == "__main__":
    main()
