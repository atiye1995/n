"""
curvature.py  (replaces spike_slab.py)

Per-Pauli-term curvature test on REAL sampled data.  For a term measured at three noise levels
lam1 < lam2 < lam3 the second divided difference
    c2 = [ (y3-y2)/(lam3-lam2) - (y2-y1)/(lam2-lam1) ] / (lam3-lam1)
vanishes for a term that is linear in lambda; z = c2 / se(c2) with the term variances estimated
from the shots.  (Works for unequal spacing, i.e. for effective noise scales.)

Why the old spike-and-slab code was retired:
  * it used np.trapz (removed in NumPy 2.4) and a Laplace evidence whose prior volume V_prior was
    dimensionally inconsistent between models;
  * its z-statistic was computed from exact expectation values divided by an ASSUMED shot variance,
    so it was deterministic, not a statistic.
Here z uses shot-estimated variances, its null is calibrated empirically (Efron's central matching)
and multiplicity is controlled with Benjamini-Hochberg.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def curvature_z(lam3, y3, var3):
    l1, l2, l3 = lam3
    y3 = np.asarray(y3, float)
    var3 = np.asarray(var3, float)
    w1 = 1.0 / ((l2 - l1) * (l3 - l1))
    w3 = 1.0 / ((l3 - l2) * (l3 - l1))
    w2 = -(w1 + w3)
    c2 = w1 * y3[..., 0] + w2 * y3[..., 1] + w3 * y3[..., 2]
    se = np.sqrt(w1 ** 2 * var3[..., 0] + w2 ** 2 * var3[..., 1] + w3 ** 2 * var3[..., 2])
    return c2, c2 / np.maximum(se, 1e-300)


def empirical_null(z):
    """Efron central matching: N(mu0, sigma0^2) from the bulk of the z distribution."""
    z = np.asarray(z, float)
    mu0 = float(np.median(z))
    sigma0 = float(1.4826 * np.median(np.abs(z - mu0)))
    return mu0, max(sigma0, 1e-6)


def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    ok = p[order] <= thresh
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(m, dtype=bool)
    sig[order[:k]] = True
    return sig


def curvature_summary(lam3, y3, var3, q=0.05):
    """Returns dict with z-scores, empirical null, BH-significant mask and the fraction of
    significantly curved terms (uses both the theoretical N(0,1) and the empirical null)."""
    c2, z = curvature_z(lam3, y3, var3)
    mu0, s0 = empirical_null(z)
    p_theory = 2 * norm.sf(np.abs(z))
    p_emp = 2 * norm.sf(np.abs((z - mu0) / s0))
    return {"c2": c2, "z": z, "null_mu": mu0, "null_sigma": s0,
            "sig_theory": benjamini_hochberg(p_theory, q), "sig_empirical": benjamini_hochberg(p_emp, q),
            "frac_theory": float(np.mean(benjamini_hochberg(p_theory, q))),
            "frac_empirical": float(np.mean(benjamini_hochberg(p_emp, q)))}
