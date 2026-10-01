"""
calibration.py
Extract REAL calibration data from an IBM backend and write data/calibration.json,
the only input from which noise.DeviceNoiseModel takes its numbers.

Two sources (both give real calibration data, recorded with provenance):
  --fake  FakeSherbrooke      snapshot of a real device shipped inside qiskit-ibm-runtime
  --backend ibm_xxx           the live calibration of a device you can access
                              (needs a saved QiskitRuntimeService account)

    pip install qiskit qiskit-ibm-runtime
    python calibration.py --fake FakeSherbrooke --out data/calibration.json
    python calibration.py --backend ibm_brisbane --out data/calibration.json

Aggregation: median over the selected qubits / couplings (default: all).
Use --qubits 0 1 2 3 to restrict to the physical qubits you would actually use.
Mapping used for the numerical model (state this in the paper):
    e1  = median error of 'sx'          t1q = its duration
    e2  = median error of the native two-qubit gate (ecr/cx/cz), t2q = its duration
    T1, T2 = median qubit coherence times
    p01, p10 = median prob_meas1_prep0 / prob_meas0_prep1 when available,
               else the symmetric assignment error of 'measure'.
NOTE: written against the qiskit-ibm-runtime BackendV2 / Target API; run once and
inspect the printed values before using them.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import numpy as np


def _median(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        raise ValueError("no finite calibration values found")
    return float(np.median(x))


def extract_from_backend(backend, qubits=None) -> dict:
    target = backend.target
    nq = target.num_qubits
    qubits = list(range(nq)) if qubits is None else list(qubits)
    qset = set(qubits)

    def op_stats(name, arity):
        errs, durs = [], []
        for qargs, props in target[name].items():
            if qargs is None or len(qargs) != arity or not set(qargs) <= qset:
                continue
            if props is None:
                continue
            errs.append(props.error)
            durs.append(props.duration)
        return _median(errs), _median(durs)

    e1, t1q = op_stats("sx", 1)
    two_q = next((n for n in ("ecr", "cz", "cx") if n in target.operation_names), None)
    if two_q is None:
        raise RuntimeError("no native two-qubit gate (ecr/cz/cx) found in target")
    e2, t2q = op_stats(two_q, 2)

    T1s, T2s = [], []
    for q in qubits:
        qp = backend.qubit_properties(q)
        T1s.append(qp.t1)
        T2s.append(qp.t2)

    p01s, p10s, sym = [], [], []
    props_obj = None
    try:
        props_obj = backend.properties()
    except Exception:
        props_obj = None
    for q in qubits:
        if props_obj is not None:
            try:
                p01s.append(props_obj.qubit_property(q, "prob_meas1_prep0")[0])
                p10s.append(props_obj.qubit_property(q, "prob_meas0_prep1")[0])
                continue
            except Exception:
                pass
        try:
            sym.append(target["measure"][(q,)].error)
        except Exception:
            pass
    if p01s and p10s:
        p01, p10, ro_source = _median(p01s), _median(p10s), "prob_meas1_prep0/prob_meas0_prep1"
    else:
        p01 = p10 = _median(sym)
        ro_source = "symmetric 'measure' error"

    try:
        import qiskit
        qv = qiskit.__version__
    except Exception:
        qv = "unknown"
    return {
        "e1": e1, "e2": e2, "T1": _median(T1s), "T2": _median(T2s),
        "t1q": t1q, "t2q": t2q, "p01": p01, "p10": p10,
        "provenance": {
            "backend": getattr(backend, "name", str(backend)),
            "native_two_qubit_gate": two_q, "readout_source": ro_source,
            "qubits": qubits, "aggregation": "median",
            "extracted_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "qiskit_version": qv,
            "calibration_last_update": str(getattr(props_obj, "last_update_date", "n/a")),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fake", help="e.g. FakeSherbrooke (from qiskit_ibm_runtime.fake_provider)")
    g.add_argument("--backend", help="live IBM backend name")
    ap.add_argument("--qubits", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="data/calibration.json")
    a = ap.parse_args()
    if a.fake:
        import qiskit_ibm_runtime.fake_provider as fp
        backend = getattr(fp, a.fake)()
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(a.backend)
    cal = extract_from_backend(backend, a.qubits)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(cal, f, indent=2)
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
