"""VERITAS defense primitives.

All server-side, numpy-only. These are the real implementations used both by the
synthetic simulator and (later) by the GRPO integration.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# robust statistics helpers
# ---------------------------------------------------------------------------
def mad(a, eps: float = 1e-9):
    """Median Absolute Deviation (scaled to be ~std for normal data)."""
    a = np.asarray(a, float)
    med = np.median(a)
    scale = 1.4826 * np.median(np.abs(a - med))
    return float(scale + eps), float(med)


def theil_sen(x, y):
    """Robust line y ~ slope*x + intercept via median of pairwise slopes."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(n) for j in range(i + 1, n)
        if x[j] != x[i]
    ]
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


# ---------------------------------------------------------------------------
# Layer 1 — sigma vs update consistency  (catches the weight-hijack, A1)
# ---------------------------------------------------------------------------
def consistency_flags(sigmas, update_norms, tau: float = 3.0, mode: str = "ratio",
                      fold: float = 2.5):
    """Flag clients whose reported sigma is inconsistent with their update size.

    Fed-GRPO Observation 3: a truthful sigma tracks the magnitude of the update.
    So sigma / ||update|| should be roughly constant across honest clients.
    A client inflating sigma (A1) blows that ratio up and becomes an outlier.

    We require a client to be BOTH a statistical outlier (|z| > tau) AND off the
    median ratio by at least `fold`x. The fold guard stops honest clients from
    being flagged when all updates shrink near convergence (tiny-norm noise).

    Returns (flags: bool array, z: standardized deviation).
    """
    s = np.asarray(sigmas, float)
    g = np.asarray(update_norms, float)

    if mode == "ratio":
        denom = g + 0.05 * np.median(g) + 1e-9    # regularise tiny norms
        signal = s / denom
    elif mode == "theilsen":
        slope, intercept = theil_sen(g, s)        # expected sigma from update
        signal = s - (slope * g + intercept)      # residual
    else:
        raise ValueError(f"unknown mode {mode!r}")

    scale, med = mad(signal)
    z = (signal - med) / scale

    if mode == "ratio":
        foldr = signal / (med + 1e-9)
        flags = (np.abs(z) > tau) & ((foldr > fold) | (foldr < 1.0 / fold))
    else:
        flags = np.abs(z) > tau
    return flags, z


# ---------------------------------------------------------------------------
# Layer 2 — robust calibration  (neutralises the mu-poison, A2)
# ---------------------------------------------------------------------------
def robust_mu(mus, method: str = "median", trim: int = 1) -> float:
    """Aggregate reported mu robustly so one fake mu cannot move the global mu."""
    m = np.asarray(mus, float)
    if method == "median":
        return float(np.median(m))
    if method == "trimmed":
        s = np.sort(m)
        if 2 * trim < len(s):
            s = s[trim:len(s) - trim]
        return float(s.mean())
    return float(np.mean(m))  # non-robust fallback


# ---------------------------------------------------------------------------
# Layer 3 — Byzantine-robust update aggregation + bounded influence (A3)
# ---------------------------------------------------------------------------
def _flat(updates):
    return np.stack([np.asarray(u, float).ravel() for u in updates], axis=0)


def krum(updates, n_byz: int = 1):
    """Classic Krum: index of the update closest to its nearest neighbours."""
    U = _flat(updates)
    n = U.shape[0]
    k = max(1, n - n_byz - 2)
    scores = []
    for i in range(n):
        d = np.sort([np.sum((U[i] - U[j]) ** 2) for j in range(n) if j != i])
        scores.append(float(np.sum(d[:k])))
    return int(np.argmin(scores)), scores


def trimmed_mean(updates, trim: int = 1):
    """Coordinate-wise trimmed mean of the updates."""
    U = _flat(updates)
    s = np.sort(U, axis=0)
    if 2 * trim < U.shape[0]:
        s = s[trim:U.shape[0] - trim]
    return s.mean(axis=0)


def cap_weights(weights, cap: float):
    """Normalise weights and cap any single client's share at `cap`.

    Uses iterative water-filling: cap the over-limit clients and redistribute
    their excess to the rest, repeating until the cap holds (a single cap +
    renormalise would re-inflate the capped client above the cap).
    """
    w = np.clip(np.asarray(weights, float), 0, None)
    n = len(w)
    cap = max(cap, 1.0 / n)                 # cap below 1/n is infeasible
    if w.sum() <= 0:
        return np.ones(n) / n
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        pool = float(w[under].sum())
        if not under.any() or pool <= 0:
            w[under] += excess / max(1, int(under.sum()))
            break
        w[under] += excess * (w[under] / pool)
    return w / w.sum()


class Reputation:
    """Per-client trust that decays on repeated flags; dampens false positives."""
    def __init__(self, n: int, decay: float = 0.85, recover: float = 1.02, floor: float = 0.05):
        self.trust = np.ones(n, float)
        self.decay, self.recover, self.floor = decay, recover, floor

    def update(self, flags):
        for i, f in enumerate(flags):
            if f:
                self.trust[i] = max(self.floor, self.trust[i] * self.decay)
            else:
                self.trust[i] = min(1.0, self.trust[i] * self.recover)
        return self.trust
