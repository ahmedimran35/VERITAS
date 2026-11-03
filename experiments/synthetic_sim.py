"""Synthetic proof-of-mechanism for VERITAS (CPU only, numpy).

A toy horizontal-federated task stands in for Fed-GRPO:
  * honest clients push the global model toward theta_star;
  * one liar pushes toward theta_bad and/or lies about (mu, sigma);
  * honest sigma tracks the update magnitude (mimics Fed-GRPO Observation 3).

We show, for each attack, that the plain (sigma-weighted) server is corrupted
and that VERITAS recovers the honest outcome and flags the liar.

Run:  python experiments/synthetic_sim.py
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from veritas.attacks import ATTACKS                      # noqa: E402
from veritas.aggregate import plain_aggregate, veritas_aggregate  # noqa: E402
from veritas.robust import Reputation                    # noqa: E402
from veritas.metrics import detection_metrics, accuracy_recovery  # noqa: E402

RNG = np.random.default_rng(0)
D = 10
N = 6
LIAR = 5
ROUNDS = 40
LR = 0.3
SIGMA_C = 2.0          # honest relation: sigma ~ SIGMA_C * ||update||
TRUE_MU = 1.0

THETA_STAR = np.ones(D)
THETA_BAD = -np.ones(D)
# per-client local targets: all honest, with mild non-IID heterogeneity
TARGETS = [THETA_STAR + 0.05 * RNG.standard_normal(D) for _ in range(N)]


def client_report(k, theta, targets):
    """Honest report for client k given the current global model."""
    update = LR * (targets[k] - theta)
    sigma = SIGMA_C * np.linalg.norm(update) + abs(0.02 * RNG.standard_normal())
    mu = TRUE_MU + 0.05 * RNG.standard_normal()
    return {"mu": float(mu), "sigma": float(sigma), "update": update}


def run(attack="honest", defense="plain", **akw):
    theta = np.zeros(D)
    rep = Reputation(N) if defense == "veritas" else None
    last_flags = np.zeros(N, bool)
    targets = [t.copy() for t in TARGETS]
    if attack != "honest":
        targets[LIAR] = THETA_BAD.copy()      # the liar also has divergent intent
    for _ in range(ROUNDS):
        reports = [client_report(k, theta, targets) for k in range(N)]
        reports[LIAR] = ATTACKS[attack](reports[LIAR], **akw)  # liar tampers
        out = (veritas_aggregate(reports, reputation=rep) if defense == "veritas"
               else plain_aggregate(reports))
        theta = theta + out["update"]
        last_flags = out["flags"]
    model_err = float(np.linalg.norm(theta - THETA_STAR))
    calib_err = abs(out["mu"] - TRUE_MU)
    det = detection_metrics(last_flags, [LIAR])
    return {"model_err": model_err, "calib_err": calib_err, **det}


def main():
    base = run("honest", "plain")
    print(f"\nVERITAS — synthetic proof of mechanism  (d={D}, clients={N}, rounds={ROUNDS})")
    print("=" * 78)
    print(f"{'scenario':<26}{'model_err':>11}{'calib_err':>11}{'TPR':>7}{'FPR':>7}{'recovery':>10}")
    print("-" * 78)
    print(f"{'honest / baseline':<26}{base['model_err']:>11.3f}{base['calib_err']:>11.3f}"
          f"{'-':>7}{'-':>7}{'-':>10}")

    specs = [
        ("A1_sigma_inflation", {"factor": 1000.0}),
        ("A2_mu_poison", {"shift": 5.0}),
        ("A3_adaptive", {"factor": 6.0}),
    ]
    for atk, kw in specs:
        pl = run(atk, "plain", **kw)
        ve = run(atk, "veritas", **kw)
        # recovery is only meaningful for attacks that damage the model (A1/A3);
        # A2 damages calibration (calib_err), not the model.
        model_damage = pl["model_err"] - base["model_err"]
        if abs(model_damage) > 0.3:
            rec = f"{accuracy_recovery(base['model_err'], pl['model_err'], ve['model_err']):.2f}"
        else:
            rec = "n/a"
        print(f"{atk + ' / plain':<26}{pl['model_err']:>11.3f}{pl['calib_err']:>11.3f}"
              f"{'-':>7}{'-':>7}{'-':>10}")
        print(f"{atk + ' / VERITAS':<26}{ve['model_err']:>11.3f}{ve['calib_err']:>11.3f}"
              f"{ve['tpr']:>7.2f}{ve['fpr']:>7.2f}{rec:>10}")
        print("-" * 78)

    print("\nRead: lower model_err / calib_err is better. Under attack the plain server")
    print("is corrupted; VERITAS pulls it back near the honest baseline, flags the liar")
    print("(TPR high), and leaves honest clients alone (FPR low).\n")


if __name__ == "__main__":
    main()
