# VERITAS — Pilot Results (2026-11-04)

First end-to-end demonstration of the attack and the defense on a **real**
Federated GRPO model. Pilot scale (proof-of-mechanism), not the final run.

## Setup
- Model: Qwen2.5-0.5B-Instruct + LoRA · Dataset: GSM8K (non-IID, 4 clients)
- GRPO local training, σ-weighted server aggregation (vulnerable Fed-GRPO)
- Single-round, controlled: 4 honest client updates trained once and cached;
  the three scenarios differ **only in aggregation**, isolating our contribution.

## Headline result (`step34_controlled.py`)
| scenario | global accuracy |
|----------|-----------------|
| honest   | 0.200 |
| **attack** (lying client: sign-flip ×100 + fake σ ×30) | **0.013**  (−0.188) |
| **defense** (VERITAS) | **0.225**  (recovered +0.212) |
| liar detected | **True** (client 3), false positives **0** |

The lying client collapses the model (94% of accuracy destroyed); VERITAS
detects it (0 false positives) and restores accuracy to the honest level.

## Figures (`step5_sweeps.py`, from the cached updates)
- **fig1_attack_strength.png** — accuracy vs attack BOOST. Weak attacks are
  harmless; past a threshold the model collapses; the defense line stays flat.
- **fig2_attacker_dilemma.png** — as the liar inflates σ it gains aggregation
  weight but becomes easier to flag. Note: at σ-factor ≈ update-boost the ratio
  check is **evaded** (adaptive attack) → motivates the norm-based defense layer.
- **fig3_threshold.png** — TPR=1, FPR=0 across all thresholds τ for the strong
  attack: detection is robust to tuning.

## Honest caveats
- Pilot scale: GSM8K, 0.5B, single round. The honest baseline does not yet
  improve over init (too few steps) — a learning baseline needs the larger run.
- BOOST=100 is a strong Byzantine attack; the sweep shows the full range.
- fig2 exposes a ratio-check blind spot under an adaptive attacker → the full
  defense must keep Layer 3 (Krum / trimmed-mean norm filter) active.

## Next (final run)
NuminaMath-CoT (~860k), multi-round, more clients/steps, seeds, μ-poison (A2),
adaptive attacker (A3), full metric matrix + ablations (~12 figures, 4 tables).

## Reproduce
```
python experiments/step34_controlled.py   # trains once, caches, prints results
python experiments/step5_sweeps.py          # figures from the cache
```
