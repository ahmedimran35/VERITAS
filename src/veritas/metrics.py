"""Evaluation metrics: detection quality and model recovery."""
from __future__ import annotations
import numpy as np


def detection_metrics(flags, liar_idx):
    """TPR / FPR of the liar-detection flags against the ground-truth liar set."""
    flags = np.asarray(flags, bool)
    n = len(flags)
    truth = np.zeros(n, bool)
    truth[list(liar_idx)] = True
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    tn = int(np.sum(~flags & ~truth))
    return {
        "tpr": tp / max(1, tp + fn),
        "fpr": fp / max(1, fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def accuracy_recovery(err_baseline, err_attack, err_defended):
    """Fraction of the attack's damage recovered by the defense (1.0 = full)."""
    damage = err_attack - err_baseline
    if abs(damage) < 1e-12:
        return 1.0
    return float((err_attack - err_defended) / damage)


def roc_sweep(sigmas, update_norms, liar_idx, taus=None, mode="ratio"):
    """Detection TPR/FPR across a sweep of the flag threshold tau."""
    from .robust import consistency_flags
    if taus is None:
        taus = np.linspace(0.5, 8.0, 25)
    out = []
    for t in taus:
        flags, _ = consistency_flags(sigmas, update_norms, tau=float(t), mode=mode)
        m = detection_metrics(flags, liar_idx)
        out.append({"tau": float(t), "tpr": m["tpr"], "fpr": m["fpr"]})
    return out
