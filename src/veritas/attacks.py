"""Attacks against Federated GRPO reward statistics.

A 'report' is a dict: {"mu": float, "sigma": float, "update": np.ndarray}.
Each attack takes an honest report and returns the tampered one.
"""
from __future__ import annotations
import numpy as np


def honest(report: dict) -> dict:
    return report


def sigma_inflation(report: dict, factor: float = 1000.0) -> dict:
    """A1 — weight hijack: claim a huge spread to grab aggregation weight."""
    r = dict(report)
    r["sigma"] = report["sigma"] * factor
    return r


def mu_poison(report: dict, shift: float = 5.0) -> dict:
    """A2 — calibration poison: report a shifted mean reward."""
    r = dict(report)
    r["mu"] = report["mu"] + shift
    return r


def adaptive(report: dict, factor: float = 5.0) -> dict:
    """A3 — adaptive: inflate sigma AND scale the update so sigma/||update||
    stays near the honest ratio, trying to pass the consistency check.
    The scaled update becomes a magnitude outlier for the Byzantine filter.
    """
    r = dict(report)
    r["sigma"] = report["sigma"] * factor
    r["update"] = np.asarray(report["update"], float) * factor
    return r


ATTACKS = {
    "honest": honest,
    "A1_sigma_inflation": sigma_inflation,
    "A2_mu_poison": mu_poison,
    "A3_adaptive": adaptive,
}
