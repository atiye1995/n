"""
estimators.py
Zero-noise extrapolation estimators that ALL take
    lam   (m,)  actual noise-scale factors (effective, see noise.effective_scale)
    E     (m,)  measured energies
    sig   (m,)  standard errors of E (estimated from the shots)
and return a ZNEResult with a point estimate, standard deviation and 95 % interval.

Estimators
  raw                 energy at the smallest noise scale (no mitigation)
  richardson_linear   two-point linear Richardson (the old baseline)
  poly{1,2,3}         weighted polynomial regression on all points (exact Richardson if m = order+1)
  exp                 A + B exp(-c lam), weighted NLS by variable projection, 3 parameters
  exp3_exact          same model through the first 3 points (equivalent of the closed form in the
                      original paper, generalised to unequal spacing; flagged if the fit hits a bound)
  stacked             cross-validated model averaging of {poly1, poly2, exp}: weights come from
                      *extrapolation* cross-validation (fit on points k+1.., predict point k), NOT from
                      a Laplace evidence with an arbitrary prior volume.

No estimator requires r in (0,1); c is only bounded to [C_MIN, C_MAX] and hitting the bound
is reported in `info["boundary"]`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

Z95 = 1.959964
C_MIN, C_MAX = 1e-3, 20.0


@dataclass
class ZNEResult:
    name: str
    F: float
    sd: float
    lo: float
    hi: float
    ok: bool = True
    draws: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)

    @staticmethod
    def gaussian(name, F, sd, **info):
        return ZNEResult(name, float(F), float(sd), float(F - Z95 * sd), float(F + Z95 * sd), True, None, info)

    @staticmethod
    def from_draws(name, draws, **info):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return ZNEResult(name, float(np.mean(draws)), float(np.std(draws)), float(lo), float(hi), True, draws, info)

    @staticmethod
    def failed(name, reason):
        return ZNEResult(name, np.nan, np.nan, np.nan, np.nan, False, None, {"reason": reason})


# ----------------------------------------------------------------------
# Linear-in-parameters models
# ----------------------------------------------------------------------

def gls_poly(lam, E, sig, order):
    lam, E, sig = map(lambda a: np.asarray(a, float), (lam, E, sig))
    if len(lam) < order + 1:
        raise ValueError("not enough points")
    X = np.vander(lam, order + 1, increasing=True)
    XtW = X.T / sig ** 2
    cov = np.linalg.inv(XtW @ X)
    theta = cov @ (XtW @ E)
    chi2 = float(np.sum(((E - X @ theta) / sig) ** 2))
    return theta, cov, chi2


def linear_extrapolation_weights(lam, order=1):
    """Weights w with F = w @ E for an (equal-variance) polynomial fit -- used for optimal shot allocation."""
    lam = np.asarray(lam, float)
    X = np.vander(lam, order + 1, increasing=True)
    return (np.linalg.inv(X.T @ X) @ X.T)[0]


def neyman_lambda_allocation(weights, sigma_unit, total_shots, min_shots=100):
    """Shots per noise level minimising Var(F) = sum_j w_j^2 sigma_unit_j^2 / S_j at fixed sum S_j:
    S_j proportional to |w_j| sigma_unit_j."""
    a = np.abs(np.asarray(weights)) * np.asarray(sigma_unit)
    a = a / a.sum()
    S = np.maximum(min_shots, np.floor(total_shots * a)).astype(int)
    return S


# ----------------------------------------------------------------------
# Exponential model  E = A + B exp(-c lam)   (variable projection)
# ----------------------------------------------------------------------

def _vp_linear(lam, E, sig, c):
    X = np.column_stack([np.ones_like(lam), np.exp(-c * lam)])
    XtW = X.T / sig ** 2
    M = XtW @ X
    try:
        ab = np.linalg.solve(M, XtW @ E)
    except np.linalg.LinAlgError:
        return None, np.inf
    r = (E - X @ ab) / sig
    return ab, float(r @ r)


def fit_exp(lam, E, sig):
    """Returns (theta=(A,B,c), cov(3x3), chi2, boundary_flag) or raises ValueError."""
    lam, E, sig = map(lambda a: np.asarray(a, float), (lam, E, sig))
    if len(lam) < 3:
        raise ValueError("exp needs >= 3 points")
    grid = np.exp(np.linspace(np.log(C_MIN), np.log(C_MAX), 400))
    chi = np.array([_vp_linear(lam, E, sig, c)[1] for c in grid])
    j = int(np.argmin(chi))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    res = minimize_scalar(lambda lc: _vp_linear(lam, E, sig, np.exp(lc))[1],
                          bounds=(np.log(lo), np.log(hi)), method="bounded",
                          options={"xatol": 1e-10})
    c = float(np.exp(res.x))
    ab, chi2 = _vp_linear(lam, E, sig, c)
    if ab is None or not np.isfinite(chi2):
        raise ValueError("degenerate exponential fit")
    A, B = ab
    ex = np.exp(-c * lam)
    J = np.column_stack([np.ones_like(lam), ex, -B * lam * ex])
    JtW = J.T / sig ** 2
    cov = np.linalg.pinv(JtW @ J)
    boundary = bool(c <= C_MIN * 1.01 or c >= C_MAX / 1.01)
    return np.array([A, B, c]), cov, chi2, boundary


def _exp_F_sd(theta, cov):
    g = np.array([1.0, 1.0, 0.0])            # F = A + B (lam = 0)
    return float(theta[0] + theta[1]), float(np.sqrt(max(g @ cov @ g, 0.0)))


# ----------------------------------------------------------------------
# Individual estimators
# ----------------------------------------------------------------------

def est_raw(lam, E, sig):
    return ZNEResult.gaussian("raw", E[0], sig[0])


def est_richardson_linear(lam, E, sig):
    w = linear_extrapolation_weights(lam[:2], 1)
    F = w @ E[:2]
    return ZNEResult.gaussian("richardson_linear", F, np.sqrt(np.sum(w ** 2 * sig[:2] ** 2)))


def est_poly(order):
    def f(lam, E, sig):
        name = f"poly{order}"
        if len(lam) < order + 1:
            return ZNEResult.failed(name, "too few points")
        th, cov, chi2 = gls_poly(lam, E, sig, order)
        return ZNEResult.gaussian(name, th[0], np.sqrt(cov[0, 0]), chi2=chi2, dof=len(lam) - order - 1)
    return f


def est_exp(lam, E, sig):
    try:
        th, cov, chi2, bd = fit_exp(lam, E, sig)
    except ValueError as e:
        return ZNEResult.failed("exp", str(e))
    F, sd = _exp_F_sd(th, cov)
    return ZNEResult.gaussian("exp", F, sd, c=th[2], chi2=chi2, boundary=bd)


def est_exp3(lam, E, sig):
    try:
        th, cov, chi2, bd = fit_exp(lam[:3], E[:3], sig[:3])
    except ValueError as e:
        return ZNEResult.failed("exp3_exact", str(e))
    F, sd = _exp_F_sd(th, cov)
    r = (E[2] - E[1]) / (E[1] - E[0]) if E[1] != E[0] else np.nan
    return ZNEResult.gaussian("exp3_exact", F, sd, c=th[2], boundary=bd, r=r)


# ----------------------------------------------------------------------
# Cross-validated model averaging
# ----------------------------------------------------------------------

_MODELS = ("poly1", "poly2", "exp")
_NPAR = {"poly1": 2, "poly2": 3, "exp": 3}


def _predict(model, lam_fit, E_fit, sig_fit, lam_new):
    """Predictive mean and (parameter) variance at lam_new."""
    if model.startswith("poly"):
        order = int(model[-1])
        th, cov, _ = gls_poly(lam_fit, E_fit, sig_fit, order)
        x = np.vander(np.array([lam_new]), order + 1, increasing=True)[0]
        return float(x @ th), float(x @ cov @ x)
    th, cov, _, _ = fit_exp(lam_fit, E_fit, sig_fit)
    ex = np.exp(-th[2] * lam_new)
    g = np.array([1.0, ex, -th[1] * lam_new * ex])
    return float(th[0] + th[1] * ex), float(g @ cov @ g)


def extrapolation_cv_scores(lam, E, sig, models=_MODELS):
    """Sum over folds of the predictive log density when the lowest-lambda point k is held out and
    predicted from the points above it (a genuine extrapolation to smaller lambda)."""
    m = len(lam)
    K = m - max(_NPAR[x] for x in models)
    scores = {x: 0.0 for x in models}
    if K < 1:
        return scores, 0
    for k in range(K):
        for mod in models:
            try:
                mu, var = _predict(mod, lam[k + 1:], E[k + 1:], sig[k + 1:], lam[k])
                v = var + sig[k] ** 2
                scores[mod] += -0.5 * np.log(2 * np.pi * v) - 0.5 * (E[k] - mu) ** 2 / v
            except (ValueError, np.linalg.LinAlgError):
                scores[mod] += -1e6
    return scores, K


def est_stacked(lam, E, sig, rng=None, n_draws=4000, temperature=1.0):
    """Pseudo-BMA weights  w_j proportional to exp(score_j / temperature)  from extrapolation CV,
    mixture of the Gaussian posteriors of the member models."""
    rng = rng or np.random.default_rng(0)
    fits = {}
    for mod, f in (("poly1", est_poly(1)), ("poly2", est_poly(2)), ("exp", est_exp)):
        r = f(lam, E, sig)
        if r.ok and np.isfinite(r.F) and np.isfinite(r.sd) and r.sd > 0:
            fits[mod] = r
    if not fits:
        return ZNEResult.failed("stacked", "no model could be fitted")
    scores, K = extrapolation_cv_scores(lam, E, sig, tuple(fits))
    s = np.array([scores[k] for k in fits]) / temperature
    w = np.exp(s - s.max())
    w /= w.sum()
    if K < 1:
        w = np.full(len(fits), 1.0 / len(fits))
    counts = rng.multinomial(n_draws, w)
    draws = np.concatenate([rng.normal(fits[k].F, fits[k].sd, c) for k, c in zip(fits, counts) if c > 0])
    return ZNEResult.from_draws("stacked", draws, weights=dict(zip(fits, w.tolist())), cv_folds=K)


def all_estimators(m: int) -> Dict[str, Callable]:
    d = {"raw": est_raw, "richardson_linear": est_richardson_linear, "poly1": est_poly(1),
         "poly2": est_poly(2), "exp": est_exp, "exp3_exact": est_exp3,
         "stacked": lambda lam, E, sig: est_stacked(lam, E, sig)}
    if m >= 5:
        d["poly3"] = est_poly(3)
    return d


def run_all(lam, E, sig) -> Dict[str, ZNEResult]:
    lam, E, sig = (np.asarray(a, float) for a in (lam, E, sig))
    out = {}
    for name, f in all_estimators(len(lam)).items():
        try:
            out[name] = f(lam, E, sig)
        except Exception as e:                       # never let one estimator kill a study
            out[name] = ZNEResult.failed(name, repr(e))
    return out


# ----------------------------------------------------------------------
# Proper scoring rules
# ----------------------------------------------------------------------

def interval_score(y, lo, hi, alpha=0.05):
    """Gneiting & Raftery (2007) interval score (smaller is better)."""
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return sd * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def crps_samples(y, draws):
    draws = np.asarray(draws)
    a = np.mean(np.abs(draws - y))
    d = np.sort(draws)
    n = len(d)
    gini = np.sum((2 * np.arange(1, n + 1) - n - 1) * d) / (n * n)      # E|X-X'| = 2*gini_mean_difference/2
    return a - gini


def score_result(res: ZNEResult, y: float) -> dict:
    if not res.ok or not np.isfinite(res.F):
        return {"ok": False}
    crps = crps_samples(y, res.draws) if res.draws is not None else crps_gaussian(y, res.F, max(res.sd, 1e-300))
    return {"ok": True, "err": res.F - y, "covered": bool(res.lo <= y <= res.hi),
            "width": res.hi - res.lo, "interval_score": float(interval_score(y, res.lo, res.hi)),
            "crps": float(crps)}
