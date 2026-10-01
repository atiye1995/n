"""
hardware_run.py
Run the SAME folded circuits, grouping, estimators and reference on a Qiskit backend
(a real IBM device via qiskit-ibm-runtime SamplerV2, or a fake backend with real calibration
snapshots for a dry run).  The result is a genuine (or device-calibrated) experiment for the paper.

    pip install qiskit qiskit-ibm-runtime qiskit-aer
    # 1. dry run on a fake backend (needs the ansatz cache from run_experiment.py)
    python hardware_run.py --molecule H2 --ansatz uccsd --fake FakeSherbrooke --shots 20000 --reps 5
    # 2. real device (saved QiskitRuntimeService account)
    python hardware_run.py --molecule H2 --ansatz uccsd --backend ibm_brisbane --qubits 0 1 2 3 --shots 20000 --reps 5

Important details already handled:
  * transpile(optimization_level=0): higher levels would cancel the folded g g^dagger g and destroy
    the noise scaling;  routing SWAPs (added when the layout is not all-to-all) DO add noise that is
    not folded -- pick a --qubits layout with the couplings the circuit needs, or use a small molecule.
  * readout error: measured from two calibration circuits (|0..0>, |1..1>) and inverted
    (tensored, qubit-averaged) exactly as in the simulation.
  * lambda_eff: computed from the calibration-derived noise model (data/calibration.json) -- an
    APPROXIMATION on hardware because real routing/compilation changes the gate mix.  Report it as such,
    or replace it by the nominal folding factor and say so.
NOT executed by the author of this file: first run the fake-backend dry run and inspect the output.
"""
import argparse, json, os
import numpy as np

import ansatz as A, chem, circuits as C, estimators as EST, measurement as M, noise as N


def to_qiskit(gates, n, basis=None, measure=True):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for g in gates:
        if g.name == "x": qc.x(g.qubits[0])
        elif g.name == "h": qc.h(g.qubits[0])
        elif g.name == "rx": qc.rx(g.param, g.qubits[0])
        elif g.name == "rz": qc.rz(g.param, g.qubits[0])
        elif g.name == "cx": qc.cx(*g.qubits)
        else: raise ValueError(g.name)
    if basis is not None:
        for g in M.basis_rotation_gates(basis):
            qc.h(g.qubits[0]) if g.name == "h" else qc.rx(g.param, g.qubits[0])
    if measure:
        qc.measure_all()
    return qc


def execute(circuits, shots_list, backend, fake):
    """Returns list of counts dicts (bitstring -> count), one per circuit."""
    from qiskit import transpile
    if fake:
        out = []
        for qc, s in zip(circuits, shots_list):      # backend.run needs one shots value per call
            res = backend.run(qc, shots=int(s)).result()
            out.append(res.get_counts())
        return out
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    out = []
    for qc, s in zip(circuits, shots_list):
        job = sampler.run([qc], shots=int(s))
        out.append(job.result()[0].data.meas.get_counts())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--bond", type=float, default=None)
    ap.add_argument("--ansatz", default="uccsd")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fake"); g.add_argument("--backend")
    ap.add_argument("--qubits", type=int, nargs="*", default=None, help="physical qubits (initial layout)")
    ap.add_argument("--calibration", default="data/calibration.json")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--shots", type=int, default=20000, help="shots per circuit (per group)")
    ap.add_argument("--reps", type=int, default=5, help="independent repetitions of the whole experiment")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    bond = a.bond if a.bond is not None else chem.GEOMETRIES[a.molecule]["eq_bond_length"]
    ans = json.load(open(os.path.join(a.out_dir, f"ansatz_{a.molecule}_r{bond:.3f}_{a.ansatz}.json")))
    ham = chem.build_hamiltonian(a.molecule, bond)
    n, ne = ham["n_qubits"], ham["n_electrons_active"]
    circ = C.build_ansatz_circuit(n, ne, A.generator_terms_from_json(ans), ans["thetas"])
    plan = M.build_plan(ham["qubit_operator"], n)
    model = N.DeviceNoiseModel.from_json(a.calibration)

    from qiskit import transpile
    if a.fake:
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, a.fake)()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(a.backend)
    layout = a.qubits if a.qubits is not None else list(range(n))

    def run_batch(qcs, shots):
        t = transpile(qcs, backend=backend, optimization_level=0, initial_layout=layout)
        return execute(t, shots, backend, bool(a.fake))

    # readout calibration (tensored, qubit-averaged) ---------------------------------
    from qiskit import QuantumCircuit
    z0, z1 = QuantumCircuit(n), QuantumCircuit(n)
    z1.x(range(n)); z0.measure_all(); z1.measure_all()
    c0, c1 = run_batch([z0, z1], [a.shots, a.shots])
    v0, v1 = M.counts_to_vector(c0, n), M.counts_to_vector(c1, n)
    bits = lambda q: (np.arange(2 ** n) >> (n - 1 - q)) & 1
    p01 = np.mean([(v0 / v0.sum() * bits(q)).sum() for q in range(n)])
    p10 = np.mean([(v1 / v1.sum() * (1 - bits(q))).sum() for q in range(n)])
    C_ro = np.array([[1 - p01, p10], [p01, 1 - p10]])
    print(f"measured readout: p01={p01:.4f} p10={p10:.4f}")
    dists = M.prepare_distributions(np.eye(2 ** n) / 2 ** n, plan, C_ro, True)   # V,u only are used

    lam_eff, folded = [], []
    for lam in a.lambdas:
        f = circ.folded(lam, model.gate_infidelity)
        folded.append(f)
        lam_eff.append(N.effective_scale(circ.all_gates(), f.all_gates(), model))
    lam_eff = np.array(lam_eff)
    print("lambda_eff (model-based, approximate on hardware):", np.round(lam_eff, 3))

    E = np.zeros((a.reps, len(a.lambdas)))
    S = np.zeros_like(E)
    for r in range(a.reps):
        for j, f in enumerate(folded):
            qcs = [to_qiskit(f.all_gates(), n, gr["basis"]) for gr in plan.groups]
            counts = run_batch(qcs, [a.shots] * len(qcs))
            cv = [M.counts_to_vector(c, n) for c in counts]
            out = M.estimate_from_counts(cv, dists, plan)
            E[r, j], S[r, j] = out["E"], out["sigma"]
        print(f"repetition {r + 1}/{a.reps}: E(lambda) = {np.round(E[r], 5)}")

    tag = f"HW_{a.molecule}_{a.fake or a.backend}_{a.ansatz}"
    np.savez(os.path.join(a.out_dir, f"hw_{tag}.npz"), lam_eff=lam_eff, lam_nom=np.array(a.lambdas),
             E_hat=E, sig_hat=S, E_ref=ans["energy"], shots_per_circuit=a.shots, n_groups=len(plan.groups))
    y = ans["energy"]
    for r in range(a.reps):
        res = EST.run_all(lam_eff, E[r], S[r])
        print(f"rep {r}: " + "  ".join(f"{k}={1e3 * (v.F - y):+.2f}mHa" for k, v in res.items() if v.ok))
    print("saved", tag, "(errors are relative to the noiseless VQE energy E_ref)")


if __name__ == "__main__":
    main()
