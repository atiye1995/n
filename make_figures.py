"""
make_figures.py
Publication figures from REAL outputs of run_experiment.py / analyze.py / noise_sweep.py.

    python make_figures.py --results results --molecule H2 --ansatz uccsd --noise-scale 1

Figures (PDF + PNG in figures/):
  fig_coverage_vs_shots   empirical coverage of the nominal 95 % intervals vs shot budget (Wilson bars)
  fig_rmse_vs_shots       RMSE vs shot budget, chemical-accuracy line
  fig_score_vs_shots      mean interval score (proper scoring rule; lower is better)
  fig_example_fit         one repetition: measured energies, fits and 95 % bands, E_ref line
  fig_noise_sweep         infinite-shot |error| vs noise strength (needs noise_sweep.py output)
Only files that exist are used; nothing is simulated here.
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import estimators as EST

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150})
METHODS = ["raw", "richardson_linear", "poly2", "exp3_exact", "exp", "stacked", "hier"]
STYLE = {"raw": ("k", "x"), "richardson_linear": ("tab:gray", "s"), "poly2": ("tab:orange", "^"),
         "exp3_exact": ("tab:red", "v"), "exp": ("tab:purple", "D"), "stacked": ("tab:blue", "o"),
         "hier": ("tab:green", "*")}


def load(results, mol, ans, xs):
    rows = []
    for p in glob.glob(os.path.join(results, f"metrics_{mol}_r*_{ans}_S*_x{xs}_*.csv")):
        S = int(re.search(r"_S(\d+)_", p).group(1))
        df = pd.read_csv(p)
        df["shots"] = S
        df["file"] = p
        rows.append(df)
    return pd.concat(rows) if rows else None


def save(fig, name, out):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--ansatz", default="uccsd")
    ap.add_argument("--noise-scale", default="1")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.results, a.molecule, a.ansatz, a.noise_scale)
    if df is None:
        raise SystemExit("no metrics_*.csv found; run analyze.py first")

    for col, ylabel, name, logy in (("coverage95", "empirical coverage of nominal 95 % interval", "fig_coverage_vs_shots", False),
                                    ("rmse_mHa", "RMSE vs noiseless VQE (mHa)", "fig_rmse_vs_shots", True),
                                    ("interval_score_mHa", "mean interval score (mHa)", "fig_score_vs_shots", True)):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for mth in METHODS:
            s = df[df.method == mth].sort_values("shots")
            if s.empty or col not in s or s[col].isna().all():
                continue
            c, mk = STYLE[mth]
            if col == "coverage95":
                ax.errorbar(s.shots, s[col], yerr=[s[col] - s.coverage_lo, s.coverage_hi - s[col]], color=c,
                            marker=mk, label=mth, capsize=2, lw=1)
            else:
                ax.plot(s.shots, s[col], color=c, marker=mk, label=mth, lw=1)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if col == "coverage95":
            ax.axhline(0.95, color="k", ls="--", lw=0.8)
            ax.set_ylim(0, 1.02)
        if col == "rmse_mHa":
            ax.axhline(1.6, color="k", ls=":", lw=0.8)
            ax.text(s.shots.min(), 1.7, "chemical accuracy", fontsize=7)
        ax.set_xlabel("shots per noise level")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=2)
        save(fig, name, a.out)

    # example fit from the file with the largest shot budget
    big = df.loc[df.shots.idxmax(), "file"].replace("metrics_", "raw_").replace(".csv", ".npz")
    if os.path.exists(big):
        d = np.load(big)
        r = 0
        lam, E, sg, y = d["lam_eff"], d["E_hat"][r], d["sig_hat"][r], float(d["E_ref"])
        res = EST.run_all(lam, E, sg)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.errorbar(lam, 1e3 * (E - y), yerr=1e3 * 1.96 * sg, fmt="ko", capsize=2, label="measured (95 %)")
        ax.axhline(0, color="k", lw=0.8)
        xs = np.linspace(0, lam.max(), 100)
        th, _, _, _ = EST.fit_exp(lam, E, sg)
        ax.plot(xs, 1e3 * (th[0] + th[1] * np.exp(-th[2] * xs) - y), color="tab:purple", lw=1, label="exp fit")
        p2, _, _ = EST.gls_poly(lam, E, sg, 2)
        ax.plot(xs, 1e3 * (np.vander(xs, 3, increasing=True) @ p2 - y), color="tab:orange", lw=1, label="poly2 fit")
        for mth in ("stacked",):
            v = res[mth]
            ax.errorbar([0.0], [1e3 * (v.F - y)], yerr=[[1e3 * (v.F - v.lo)], [1e3 * (v.hi - v.F)]], fmt="s",
                        color=STYLE[mth][0], label=f"{mth} at lambda=0", capsize=3)
        ax.set_xlabel("effective noise scale")
        ax.set_ylabel("E - E_ref (mHa)")
        ax.legend(fontsize=6)
        save(fig, "fig_example_fit", a.out)

    sw = glob.glob(os.path.join(a.results, f"sweep_{a.molecule}_r*_{a.ansatz}.csv"))
    if sw:
        s = pd.read_csv(sw[0])
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for col in [c for c in s.columns if c.startswith("err_") or c == "raw_error"]:
            mth = col.replace("err_", "").replace("raw_error", "raw")
            c, mk = STYLE.get(mth, ("tab:cyan", "."))
            ax.loglog(s.noise_scale, np.abs(s[col]) * 1e3, marker=mk, color=c, label=mth, lw=1)
        ax.axhline(1.6, color="k", ls=":", lw=0.8)
        ax.set_xlabel("noise strength (x calibrated)")
        ax.set_ylabel("|infinite-shot error| (mHa)")
        ax.legend(fontsize=6, ncol=2)
        save(fig, "fig_noise_sweep", a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
