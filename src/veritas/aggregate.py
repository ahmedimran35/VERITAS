"""VERITAS aggregation — the three defense layers combined into one server step."""
from __future__ import annotations
import numpy as np
from .robust import (
    consistency_flags, robust_mu, trimmed_mean, krum, cap_weights, Reputation,
)

DEFAULT_CFG = {
    "tau": 3.0,              # consistency flag threshold (MAD z-score)
    "mode": "ratio",        # sigma-consistency mode: "ratio" | "theilsen"
    "mu_method": "median",  # robust mu: "median" | "trimmed" | "mean"
    "w_cap": 0.4,           # max aggregation weight any single client may hold
    "update_agg": "trimmed",   # "weighted" | "trimmed" | "krum"
    "trim": 1,
    "n_byz": 1,
    "use_reputation": True,
}


def plain_aggregate(reports):
    """Baseline Fed-GRPO aggregation: weight by reported sigma, mean the mu.
    No verification — this is the vulnerable system we attack.
    """
    sigmas = np.asarray([r["sigma"] for r in reports], float)
    w = sigmas / (sigmas.sum() + 1e-9)
    updates = np.stack([np.asarray(r["update"], float) for r in reports], 0)
    g_update = (w[:, None] * updates).sum(0)
    g_mu = float(np.mean([r["mu"] for r in reports]))
    return {"update": g_update, "mu": g_mu, "weights": w,
            "flags": np.zeros(len(reports), bool)}


def veritas_aggregate(reports, cfg=None, reputation: Reputation | None = None):
    """Robust aggregation with the three VERITAS layers."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    sigmas = np.asarray([r["sigma"] for r in reports], float)
    mus = [r["mu"] for r in reports]
    updates = [np.asarray(r["update"], float) for r in reports]
    norms = np.asarray([np.linalg.norm(u) for u in updates], float)

    # Layer 1 — sigma/update consistency
    flags, z = consistency_flags(sigmas, norms, tau=cfg["tau"], mode=cfg["mode"])

    # Layer 2 — robust global mu
    g_mu = robust_mu(mus, method=cfg["mu_method"], trim=cfg["trim"])

    # Layer 3 — weights: sigma, zero the flagged, apply reputation, cap influence
    raw = np.where(flags, 0.0, sigmas)
    if cfg["use_reputation"] and reputation is not None:
        raw = raw * reputation.trust
    w = cap_weights(raw, cfg["w_cap"])

    kept_idx = [i for i in range(len(reports)) if not flags[i]] or list(range(len(reports)))
    kept = [updates[i] for i in kept_idx]

    if cfg["update_agg"] == "trimmed" and len(kept) > 2:
        g_update = trimmed_mean(kept, trim=cfg["trim"])
    elif cfg["update_agg"] == "krum" and len(kept) > cfg["n_byz"] + 2:
        j, _ = krum(kept, n_byz=cfg["n_byz"])
        g_update = np.asarray(kept[j], float)
    else:  # weighted average over all, using w (flagged already 0)
        U = np.stack(updates, 0)
        g_update = (w[:, None] * U).sum(0)

    if cfg["use_reputation"] and reputation is not None:
        reputation.update(flags)

    return {"update": g_update, "mu": g_mu, "flags": flags, "z": z, "weights": w}
