"""
analyze.py
Apply every ZNE estimator to every shot-noise repetition of a raw_*.npz and compute, against the
CORRECT reference E_ref (noiseless VQE energy of the same ansatz):

    bias, sd, RMSE, mean reported sd, 95 % interval coverage (with Wilson CI), mean interval score,
    mean CRPS, failure fraction, fraction within chemical accuracy (1.6 mHa),
    and the *infinite-shot bias* (estimator applied to the exact expectation curve) which isolates
    model-form error from statistical error.

    python analyze.py results/raw_H2_r0.740_uccsd_S1000000_x1_l1_uniform.npz
    python analyze.py <raw.npz> --hier-reps 30        # add the hierarchical Bayesian model (needs jax, numpyro)

Coverage is computed over ALL repetitions in which the estimator returned a result and failures are
reported separately (no selection bias: an estimator that fails is not silently excluded).
"""
from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import beta

import estimators as EST
import curvature as CURV

CHEM_ACC = 1.6e-3


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def summarise(name, rows, n_total, y):
    ok = [r for r in rows if r["ok"]]
    out = {"method": name, "n_reps": n_total, "fail_frac": 1 - len(ok) / n_total}
    if not ok:
        return out
    err = np.array([r["err"] for r in ok])
    cov = np.array([r["covered"] for r in ok])
    lo, hi = wilson(int(cov.sum()), len(cov))
    out.update({"bias_mHa": 1e3 * err.mean(), "sd_mHa": 1e3 * err.std(ddof=1) if len(err) > 1 else np.nan,
                "rmse_mHa": 1e3 * np.sqrt(np.mean(err ** 2)),
                "mean_reported_width_mHa": 1e3 * np.mean([r["width"] for r in ok]),
                "coverage95": cov.mean(), "coverage_lo": lo, "coverage_hi": hi,
                "interval_score_mHa": 1e3 * np.mean([r["interval_score"] for r in ok]),
                "crps_mHa": 1e3 * np.mean([r["crps"] for r in ok]),
                "frac_chem_acc": float(np.mean(np.abs(err) < CHEM_ACC))})
    return out


def analyse_file(path, hier_reps=0, hier_kwargs=None):
    d = np.load(path)
    R, m = d["E_hat"].shape
    lam, y = d["lam_eff"], float(d["E_ref"])
    per_method = {}
    for r in range(R):
        res = EST.run_all(lam, d["E_hat"][r], d["sig_hat"][r])
        for k, v in res.items():
            per_method.setdefault(k, []).append(EST.score_result(v, y))
    table = [summarise(k, rows, R, y) for k, rows in per_method.items()]

    if hier_reps:
        import hier_model as HM
        with open(path.replace("raw_", "plan_").replace(".npz", ".pkl"), "rb") as f:
            blob = pickle.load(f)
        plan, cov = blob["plan"], blob["cov"]
        if cov is None:
            raise SystemExit("raw file was generated without --hier-reps; rerun run_experiment with --hier-reps N")
        H = min(hier_reps, cov[0][0].shape[0])
        rows = []
        for r in range(H):
            blocks = [[cov[j][g][r] for g in range(len(plan.groups))] for j in range(m)]
            data = HM.prepare_data(lam, d["terms"][r], blocks, plan, d["shots_group"], )
            res = HM.fit_hierarchical(data, seed=r, **(hier_kwargs or {}))
            rows.append(dict(EST.score_result(res, y), rhat=res.info.get("rhat"),
                             div=res.info.get("divergences"), phi=res.info.get("phi_mean")))
        row = summarise("hier", rows, H, y)
        row["mean_rhat"] = float(np.nanmean([r.get("rhat", np.nan) for r in rows if r["ok"]] or [np.nan]))
        table.append(row)

    # infinite-shot (exact curve) bias: statistical noise removed
    sig_flat = np.full(m, float(np.median(d["sig_hat"])))
    inf = EST.run_all(lam, d["E_inf"], sig_flat)
    df = pd.DataFrame(table)
    df["inf_shot_bias_mHa"] = df["method"].map({k: 1e3 * (v.F - y) if v.ok else np.nan for k, v in inf.items()})
    meta = {"E_ref": y, "E_exact_active": float(d["E_exact_active"]), "E_hf": float(d["E_hf"]),
            "ansatz_error_mHa": 1e3 * (y - float(d["E_exact_active"])),
            "raw_bias_mHa": 1e3 * (float(d["E_inf"][0]) - y), "lam_eff": lam.tolist(),
            "median_sigma_mHa": 1e3 * float(np.median(d["sig_hat"]))}
    # curvature test on the first three noise levels (first repetition = one 'experiment')
    c = CURV.curvature_summary(lam[:3], d["terms"][0][:3].T, d["term_var"][0][:3].T)
    meta["curvature_frac_sig_theory"], meta["curvature_frac_sig_empirical"] = c["frac_theory"], c["frac_empirical"]
    meta["curvature_null_sigma"] = c["null_sigma"]
    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--hier-reps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    a = ap.parse_args()
    df, meta = analyse_file(a.raw, a.hier_reps, {"num_warmup": a.warmup, "num_samples": a.samples})
    out = a.raw.replace("raw_", "metrics_").replace(".npz", ".csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in meta.items()})
    print(df.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
