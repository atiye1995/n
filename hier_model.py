"""
hier_model.py
Structured hierarchical Bayesian zero-noise extrapolation (the proposed method).

Idea.  Instead of extrapolating the scalar energy E(lambda) with a generic curve (which needs
model-form assumptions and has ~as many parameters as data points), extrapolate EVERY Pauli
expectation value separately with a physically motivated decay and let the terms *share
strength*:

    q_i(lambda) = b_i + (a_i - b_i) * exp(-kappa_i * lambda),      F = const + sum_i c_i a_i

  a_i = q_i(0) in [-1, 1]            noiseless expectation value (what we want)
  kappa_i                            decay rate; log kappa_i = mu_{w(i)} + tau_{w(i)} z_i, z_i ~ N(0,1)
                                     (hierarchical over Pauli-weight classes w(i))
  b_i ~ N(0, s_b^2)                  small asymptote (non-unital noise, e.g. amplitude damping)

Likelihood.  Terms in one QWC group share shots, so their estimates are correlated.  For every
(group, lambda) the sample covariance C_gj of the group's term means is Cholesky-whitened and the
residual r = L^{-1}(y - mean) enters a Gaussian likelihood with a global variance-inflation
factor phi ~ LogNormal(0, 0.3^2) that absorbs residual misspecification:
    log p = -0.5 * sum ||r||^2 / phi - 0.5 * N * log(phi).
Posterior of F is obtained by NUTS -> genuine credible interval; its frequentist calibration is
checked by analyze.py (coverage over many independent shot realisations).

Requires: pip install jax numpyro     (x64 is enabled below)
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import split_gelman_rubin, effective_sample_size

from estimators import ZNEResult


def pooled_kappa_anchor(lam: np.ndarray, y: np.ndarray) -> float:
    """Data-driven centre for the kappa prior: median log-linear decay rate of the terms whose
    estimates are large at every noise level.  Used only as the centre of a wide (sd = 1.5 on log
    scale) prior."""
    slopes = []
    for row in y:
        if np.all(np.abs(row) > 0.05) and np.all(np.sign(row) == np.sign(row[0])):
            s = -np.polyfit(lam, np.log(np.abs(row)), 1)[0]
            if np.isfinite(s) and s > 0:
                slopes.append(s)
    return float(np.clip(np.median(slopes), 0.02, 2.0)) if slopes else 0.1


def prepare_data(lam, term_means, cov_blocks, plan, shots_per_group, ridge: float = 1e-3):
    """Build padded/whitened arrays.
    term_means : (m, T)   term estimates for one shot realisation
    cov_blocks : list over lambda of list over groups of (k_g, k_g) covariance-of-the-mean
    shots_per_group : (m, G)
    """
    lam = np.asarray(lam, float)
    m, G = len(lam), len(plan.groups)
    kmax = max(len(g["members"]) for g in plan.groups)
    idx = np.zeros((G, kmax), dtype=int)
    mask = np.zeros((G, kmax))
    W = np.zeros((G, m, kmax, kmax))
    for g, gr in enumerate(plan.groups):
        mem = gr["members"]
        k = len(mem)
        idx[g, :k] = mem
        mask[g, :k] = 1.0
        for j in range(m):
            C = 0.5 * (cov_blocks[j][g] + cov_blocks[j][g].T)
            n_g = shots_per_group[j, g]
            d = np.diag(C)
            C = C + np.diag(np.maximum(ridge * d, 1.0 / n_g ** 2))
            L = np.linalg.cholesky(C)
            W[g, j, :k, :k] = np.linalg.inv(L)
    wclass = np.minimum(plan.weights, 5) - 1
    uniq = np.unique(wclass)
    remap = {u: i for i, u in enumerate(uniq)}
    cls = np.array([remap[c] for c in wclass])
    return {"lam": lam, "y": np.asarray(term_means).T, "idx": idx, "mask": mask, "W": W,
            "cls": cls, "n_cls": len(uniq), "coeffs": plan.coeffs, "const": plan.const,
            "anchor": pooled_kappa_anchor(lam, np.asarray(term_means).T)}


def _model(d):
    lam = jnp.asarray(d["lam"])
    y = jnp.asarray(d["y"])                     # (T, m)
    T, m = y.shape
    idx, mask, W = jnp.asarray(d["idx"]), jnp.asarray(d["mask"]), jnp.asarray(d["W"])
    cls, n_cls = jnp.asarray(d["cls"]), d["n_cls"]
    N_obs = float(np.sum(d["mask"]) * m)

    def model():
        mu = numpyro.sample("mu", dist.Normal(jnp.log(d["anchor"]), 1.5).expand([n_cls]))
        tau = numpyro.sample("tau", dist.HalfNormal(0.7).expand([n_cls]))
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([T]))
        kappa = jnp.exp(mu[cls] + tau[cls] * z)
        s_b = numpyro.sample("s_b", dist.HalfNormal(0.05))
        b_raw = numpyro.sample("b_raw", dist.Normal(0.0, 1.0).expand([T]))
        b = s_b * b_raw
        a = numpyro.sample("a", dist.Uniform(-jnp.ones(T), jnp.ones(T)))
        phi = numpyro.sample("phi", dist.LogNormal(0.0, 0.3))
        mean = b[:, None] + (a - b)[:, None] * jnp.exp(-kappa[:, None] * lam[None, :])   # (T, m)
        # group-padded residuals: (G, m, kmax)
        r = jnp.transpose(y[idx] - mean[idx], (0, 2, 1))            # (G, m, kmax)
        r = r * mask[:, None, :]
        wr = jnp.einsum("gjkl,gjl->gjk", W, r)
        numpyro.factor("loglik", -0.5 * jnp.sum(wr ** 2) / phi - 0.5 * N_obs * jnp.log(phi))
        numpyro.deterministic("F", d["const"] + jnp.dot(jnp.asarray(d["coeffs"]), a))

    return model


def fit_hierarchical(d: dict, num_warmup: int = 500, num_samples: int = 500, num_chains: int = 2,
                     seed: int = 0, target_accept: float = 0.9) -> ZNEResult:
    kernel = NUTS(_model(d), target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    by_chain = mcmc.get_samples(group_by_chain=True)
    F_chain = np.asarray(by_chain["F"])                      # (chains, draws)
    div = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
    rhat = float(split_gelman_rubin(F_chain)) if num_chains > 1 else np.nan
    ess = float(effective_sample_size(F_chain))
    res = ZNEResult.from_draws("hier", F_chain.reshape(-1), rhat=rhat, ess=ess, divergences=div,
                               phi_mean=float(np.mean(np.asarray(by_chain["phi"]))))
    res.ok = bool(div == 0 or div < 0.01 * F_chain.size)
    return res
