# VERITAS

**VE**rifying **R**eward sta**T**istics **I**n feder**A**ted **S**ystems

> Detecting lying clients in Federated GRPO — a reward-statistics poisoning attack and a server-side consistency defense.

**Status:** pilot complete ✅. On a real federated GRPO model (Qwen2.5-0.5B + LoRA, GSM8K, 4 non-IID clients), the lying-client attack collapses global accuracy and the VERITAS defense detects the liar and recovers it:

| scenario | accuracy | detection |
|---|---|---|
| honest | 0.200 | — |
| **attack** (lying client) | **0.013** (−0.188) | — |
| **defense** (VERITAS) | **0.225** (recovered) | liar flagged, 0 false positives |

See [`results/`](results/) for figures and [`results/RESULTS.md`](results/RESULTS.md). Next: the full NuminaMath-CoT run.

---

## 1. One-line idea

In Federated GRPO, every client self-reports two reward numbers — **μ** (the mean reward) and **σ** (its spread) — and the server trusts them without verification. A single malicious client can lie about μ and σ to hijack the aggregation or poison calibration, corrupting the shared model for everyone. VERITAS catches the liar by checking the self-reported σ against the client's *actual* update, which the server already receives.

## 2. 30-second primer (μ and σ)

- **μ (mu)** = the **average** reward of a client's group of answers.
- **σ (sigma)** = the **spread** of those rewards (how much they vary).

GRPO scores each answer by its *advantage* = `(reward − μ) / σ`, i.e. relative to its own group. In the federated setting a client sends the server its model update plus its μ and σ. The server weights clients by σ and broadcasts a global μ — both taken on trust.

## 3. Threat model

- **Setting:** horizontal Federated GRPO (clients hold their own prompts; same task).
- **Adversary:** one (or a few) malicious / Byzantine clients that follow the message format but send falsified reward statistics. They may also adapt their model update.
- **Goal:** integrity, not privacy — corrupt the global model, not steal data.
- **Server:** honest-but-blind; sees updates and reported (μ, σ); cannot re-run client data.

## 4. Attacks (what we inject to test)

| ID | Name | Lie | Effect |
|----|------|-----|--------|
| A1 | Weight hijack | report huge fake **σ** | captures the largest aggregation weight → dominates the model |
| A2 | Calibration poison | report fake **μ** | broadcast global μ carries the poison to *honest* clients |
| A3 | Adaptive | fake σ **and** a crafted update to keep σ↔update consistent | tries to evade the consistency check |

## 5. The defense (VERITAS)

Three layers, all server-side, all cheap:

1. **σ–update consistency check** (catches A1). The server has each client's update, so it checks whether the reported σ is consistent with the update's magnitude. Grounded in Fed-GRPO's own Observation 3 (σ tracks update size). Technique: robust ratio / Theil–Sen relation + **MAD z-score** outlier flag.
2. **Robust calibration** (neutralises A2). Build the global μ with a **median / trimmed-mean** instead of a plain average, so a single fake μ is an outlier and is ignored.
3. **Byzantine-robust aggregation + bounded influence** (backstops A3). Aggregate updates with **Krum / trimmed-mean**, **cap** any single client's weight, and track a per-client **reputation** that decays on repeated flags (keeps false positives from hurting honest clients).

## 6. How we prove it works

Run matrix: `{no-attack, A1, A2, A3} × {defense off, on}` on NuminaMath-CoT (non-IID), then report:

- **Attack success** — accuracy drop under attack vs honest baseline.
- **Detection rate (TPR)** — fraction of rounds the liar is flagged.
- **False-positive rate (FPR)** — honest clients wrongly flagged (must stay low).
- **Accuracy recovery** — accuracy with defense on vs the honest baseline.
- **ROC** — sweep the flag threshold τ.

Plus ablations: number of liars, lie strength, non-IID degree, seeds (mean ± std).

## 7. Roadmap

- [x] Defense primitives (`src/veritas/`) — real, tested.
- [x] Synthetic proof-of-mechanism (`experiments/synthetic_sim.py`) — CPU, no GPU.
- [x] GPU pilot: Qwen2.5-0.5B + GSM8K via TRL `GRPOTrainer`, 4 clients — attack + defense + detection demonstrated (`experiments/step34_controlled.py`, `step5_sweeps.py`, `results/`).
- [ ] Full run on NuminaMath-CoT: multi-round, more clients/steps, seeds, μ-poison (A2) + adaptive (A3), full metric matrix + ablations (~12 figures, 4 tables).
- [ ] Write-up.

## 8. Repo layout

```
veritas-fedgrpo/
├── src/veritas/        # defense + attack primitives (runnable, numpy)
│   ├── robust.py       # Theil–Sen, MAD, consistency flags, Krum, trimmed-mean, reputation
│   ├── attacks.py      # A1 sigma-inflation, A2 mu-poison, A3 adaptive
│   ├── aggregate.py    # VERITAS aggregation (the 3 layers combined)
│   └── metrics.py      # TPR / FPR / accuracy recovery / ROC sweep
├── experiments/
│   └── synthetic_sim.py  # CPU demo: attack breaks it, VERITAS recovers it
├── fedgrpo/            # GRPO-on-LLM integration (GPU phase, stubs)
│   ├── client.py
│   └── server.py
├── configs/default.yaml
├── tests/test_robust.py
└── requirements.txt
```

## 9. Quickstart

```bash
pip install -r requirements.txt
python experiments/synthetic_sim.py      # runs the attack/defense demo on CPU
pytest -q                                 # sanity tests for the defense primitives
```

## 10. References

- Fed-GRPO — arXiv:2610.11502 (the target system; Observation 3 grounds the defense)
- Weight Attack on Federated Learning — arXiv:2112.14468
- Byzantine-Robust FL with Learnable Aggregation Weights — arXiv:2511.03529 (ICLR 2026)
- Hail to the Thief — Attacks on Decentralised GRPO — arXiv:2511.09780
- FedMOA — arXiv:2602.00453 · FGRPO — arXiv:2606.03094
- Dataset: NuminaMath-CoT — huggingface.co/datasets/AI-MO/NuminaMath-CoT

---

Maintainer: MD Imran Ahmed ([@ahmedimran35](https://github.com/ahmedimran35)). Private research repo — please do not share without permission.
