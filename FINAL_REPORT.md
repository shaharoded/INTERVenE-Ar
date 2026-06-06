# EMR Event-Prediction Transformer — Benchmarking Final Report

Dataset: `data/source/` — 57,078 patients / 16.8M temporal events.
Split (seeded, fixed): 39,954 train / 8,562 val / 8,562 held-out test.
Evaluation: per-patient peak-detector on the held-out test set (immutable `evaluation.py`).
6 evaluated outcomes (5 complications + DEATH; KETOACIDOSIS & ACIDOSIS auto-filtered
below 1% prevalence; RELEASE excluded from AUC, reported as length-of-stay).

---

## Headline model

**M-128 + QA data** — 4-layer transformer, embed_dim=128, 2 heads, **1.75M params**,
seed 42, patience 15, `USE_QA_DATA=True`. Reproduce with `python api.py` (current config).

| Metric | Value | 95% CI (bootstrap, 2000× patient resample) |
|---|---|---|
| **AUROC** (support-weighted) | **0.885** | [0.881, 0.889] |
| AUPRC (support-weighted) | 0.784 | [0.776, 0.792] |
| AUROC (simple mean) | 0.884 | — |
| max-F1 (weighted) | 0.724 | — |

**Per-outcome AUROC [95% CI] · peak-MAE:**
| Outcome | AUROC [CI] | peak-MAE |
|---|---|---|
| CARDIO-VASCULAR | 0.957 [0.948, 0.966] | 34h |
| DISGLYCEMIA-Hyperglycemia | 0.911 [0.904, 0.917] | 23h |
| HYPEROSMOLALITY | 0.884 [0.876, 0.891] | 25h |
| KIDNEY | 0.882 [0.874, 0.890] | 25h |
| DISGLYCEMIA-Hypoglycemia | 0.873 [0.859, 0.888] | 42h |
| DEATH | 0.796 [0.782, 0.809] | 186h |

Length-of-stay MAE 84h. (NB the QA model under-generates at default inference,
gen_to_gt 0.24 — a weaker ttt-gate is the natural inference-side counter, untested.)

---

## What we tried, and what moved the needle

| Lever | Effect on AUROC_w | Verdict |
|---|---|---|
| **QA data** (`%_PATTERN%` events + ComplianceScore context) | **+0.038** | **largest single win** |
| Training patience 5→15 (at fixed seed) | +0.023 | helps (over-trains generation) |
| ttt-gate inference strengthening (non-QA) | +0.016 | free inference-time win |
| Architecture M-128 → M-256 → M-384 (±QA) | within noise | **capacity does not help** |
| Temperature-schedule decoding (F2) | −0.150 | greedy is optimal |
| Beam search (F1) | not run | deprioritized (over-gen model; greedy wins) |
| Seed context length k>2 days | declines | k=2 optimal; more context does not help |

## P6 architecture sweep (single-seed, full data)

| Variant | Params | AUROC_w | honesty (gen_to_gt) |
|---|---|---|---|
| M-128 | 1.75M | 0.883 | 0.61 |
| M-256 | 6.7M | 0.891 | 0.41 |
| M-384 | 14.9M | 0.876 | 0.32 |
| M-512/M-768 | — | not run (curve had turned down) | — |

---

## Methodological findings (the paper's contributions)

1. **Initialization variance dominates architecture choice.** The *same* M-128 config
   gave AUROC 0.824–0.883 across inits (std ≈ 0.024, range ≈ 0.06) — ~4× the entire
   architecture-sweep spread (0.015) and ~6× the test-set bootstrap CI half-width
   (±0.004). **Single-seed architecture comparisons at this scale are within noise**;
   the M-128↔M-256↔M-384 ranking is not statistically meaningful. (We added reproducible
   seeding mid-study to establish this cleanly.)

2. **AUROC ↔ calibration Pareto, along every axis.** More capacity, more training epochs,
   and QA all *lower validation BCE* (better calibration/likelihood) while not improving —
   sometimes hurting — *ranking AUROC* and trajectory honesty. M-384 had the best val-loss
   of the sweep but the worst AUROC; M-256+QA had the lowest loss of any run, not the best
   AUROC. Reported as a primary observation.

3. **Greedy decoding is optimal for the peak-detector eval.** Stochastic temperature
   decoding cost −0.15 AUROC; the eval rewards the model's confident MAP trajectory.

4. **Trajectory length is highly sensitive and controllable post-hoc.** Models swing
   between under- and over-generation with training duration / QA; the inference-time
   ttt-gate cleanly re-calibrates length (gen_to_gt 1.91→1.05) for +0.016 AUROC, no retraining.

5. **More patient-history context does not help** under this generative peak-detector eval
   (k=2 days optimal; monotone decline to k=7) — partly because a longer seed leaves less
   future to predict.

## Confidence

- **Test-set uncertainty:** bootstrap 95% CIs (2000 patient resamples; point estimates
  validated against the reported headlines) — winner AUROC_w 0.885 [0.881, 0.889].
- **Training/init uncertainty:** std ≈ 0.024 (empirical, from 4 existing full-data M-128
  runs). This dominates, and renders the M-128↔M-256 gap (0.007) non-significant.
- A full 3-seed retrain study was deemed unnecessary given the above (init variance is
  already characterized; test-set CI is tight).

## Reproducibility

- Branch `autoresearch-trajectory`. Ledger: `results/results-trajectory-fix.tsv`.
- Winner config is the repo default (`python api.py` → M-128+QA, seed 42, patience 15).
- Backups: `checkpoints.bak_M128_QA_s42` (winner), `checkpoints.bak_seed42_p15_nonQA`.
- Bootstrap CIs: `python bootstrap_ci.py <checkpoint_dir>`.
- `evaluation.py` / `api.py` never modified (except the user-scoped, reverted
  `EVAL_INPUT_DAYS` edit for the k-ablation).
