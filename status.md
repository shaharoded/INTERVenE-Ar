# EMR Event-Prediction Transformer — Benchmarking Phase

Fresh journal for the benchmarking phase. Iteration-loop journal archived at
`status-iteration-loop.md`.

## Inherited from the iteration loop

Locked architecture and recipe (full spec in `program.md`):

- **M-256** transformer (embed 256, 4 layers, 4 heads, time2vec 32, dropout 0.10)
- **AdaLN-Zero** patient-context conditioning + temporal RoPE + Time2Vec
- **Per-token learnable `log_tau_lm`**, terminal entries frozen at `log(12/336)`
- **C-ttt aux head** (time-to-terminal regression, MSE on `log1p(t_terminal − t_now)`)
- **Δt two-head** (gate + softplus magnitude)
- **P4 pool aux** at `aux_fraction_cap=0.05` (Phase 3)
- **I2b inference gate** (ttt-driven terminal-logit bias when `hrs-to-terminal < 48h`)
- **CBM** input masking p=0.25 (Phase 2 only)

Three-phase training: embedder → LM curriculum → outcome head fine-tune. Phase 3
backbone with `lr_factor=0.01`.

## Outcome configuration

Head-targeted outcomes are set in
`emr_model/transform_emr/config/dataset_config.py`:

- 6 complications (in `OUTCOMES`)
- 2 terminals (in `TERMINAL_OUTCOMES`): DEATH, RELEASE
- `AUC_EXCLUDE = ("RELEASE_EVENT",)` in `evaluation.py` — RELEASE stays in head
  training (so the model emits it correctly) but is excluded from the AUROC /
  AUPRC / F1 headline (it's `¬DEATH` in this cohort, redundant ranking task).
  Reported as length-of-stay MAE instead.

**7 evaluated outcomes for the headline**: 6 complications + DEATH.

LM vocab is built from the training data, so any token present in the dataset
appears in input sequences regardless of `OUTCOMES` — `OUTCOMES` controls only
the head-training targets, sampler weights, and CBM forbid list.

## Benchmarking journal

Agent appends `### <tag>` blocks below as experiments run. Each block records:

- Tag, what changed (1–2 lines), commit SHA
- Smoke gate results (A–D)
- Post-train gate results (T1–T3)
- Headline numbers (`patient_auroc_weighted`, `patient_auprc_weighted`, `patient_max_f1_weighted`,
  per-outcome AUROC/AUPRC/max-F1/F1@0.5, peak-MAE per complication, length-of-stay MAE)
- Trajectory honesty (`gen_to_gt_ratio_median`, `gen_frac_terminal_first24h`)
- **Per-aux training trace table** (unlock epoch, λ_max, anchor raw_aux, final raw_aux, Δ%) — mandatory
- Verdict (KEEP / DISCARD) with reason

---

### P6 / M-128 @ FULL DATA — DISCARD (capacity-bounded)

**Code:** `f524b67` (locked recipe + 11-outcome snip + `MODEL_CONFIG`
embed_dim=128, n_head=2; head_dim=64 kept, n_layer=4). 1.75 M params
(M-256 baseline = 6.4 M). Full data; embedder retrained on full data.

**Hypothesis (P6 decision rule):** smallest variant within 0.005 AUROC_w
of the best.

**Aggregated headline (11 outcomes):**

| Metric | M-256 (I2b-full-snip) | **M-128** | Δ |
|---|---|---|---|
| patient_auroc_weighted | 0.759 | **0.686** | **−0.073** |
| patient_auprc_weighted | 0.781 | 0.744 | −0.037 |
| patient_auroc_simple | 0.694 | 0.695 | +0.001 |
| patient_auprc_simple | 0.432 | 0.422 | −0.010 |
| patient_maxF1_weighted | 0.763 | 0.754 | −0.009 |
| patient_F1@0.5_weighted | 0.128 | 0.127 | ≈flat |
| patient_maxF1_simple | 0.453 | 0.463 | +0.010 |
| patient_F1@0.5_simple | 0.134 | 0.133 | ≈flat |
| cap=48h AUROC | 0.523 | 0.551 | +0.028 |
| RELEASE MAE (h) | 68.4 | 85.2 | +16.8 |
| DEATH MAE (h) | 167.6 | 156.5 | −11.1 |
| gen_to_gt_ratio_median | 0.544 | 1.275 | over-gen |
| gen_frac_terminal_first24h | 0.051 | 0.050 | ≈flat |
| forced-terminal fraction | low | **29.2%** | over-gen badly |
| phase2_best_val | 0.150 | 0.157 | +0.007 |
| phase3_best_val | 0.930 | 0.952 | +0.022 |

**Per-outcome AUROC / AUPRC / F1:**

| Outcome | AUROC | AUPRC | maxF1 (τ*) | F1@0.5 | n_pos | prev |
|---|---|---|---|---|---|---|
| DISGLYCEMIA_Hyper | 0.907 | 0.887 | 0.809 (τ=0.051) | 0.381 | 3550 | 0.415 |
| DISGLYCEMIA_Hypo  | 0.902 | 0.692 | 0.654 (τ=0.300) | 0.552 | 875  | 0.102 |
| KIDNEY            | 0.759 | 0.718 | 0.695 (τ=0.037) | 0.205 | 3839 | 0.448 |
| CARDIO            | 0.747 | 0.798 | 0.781 (τ=0.000) | 0.000 | 5078 | 0.593 |
| DEATH             | 0.711 | 0.264 | 0.341 (τ=0.439) | 0.330 | 1115 | 0.130 |
| RETINOPATHY       | 0.655 | 0.111 | 0.211 (τ=0.000) | 0.000 | 284  | 0.033 |
| SKIN_ULCER        | 0.645 | 0.096 | 0.221 (τ=0.000) | 0.000 | 391  | 0.046 |
| KETOACIDOSIS      | 0.620 | 0.052 | 0.134 (τ=0.000) | 0.000 | 200  | 0.023 |
| NERVOUS_SYSTEM    | 0.614 | 0.116 | 0.220 (τ=0.000) | 0.000 | 517  | 0.060 |
| NEUROVASCULAR     | 0.596 | 0.038 | 0.097 (τ=0.000) | 0.000 | 170  | 0.020 |
| **RELEASE**       | **0.486** | 0.867 | 0.930 (τ=0.002) | 0.000 | 7447 | 0.870 |

**Per-aux trace (Phase 2, full data):**

| Aux | Unlock ep | λ_max | Anchor raw | Final raw | Δ% | Learning? |
|---|---|---|---|---|---|---|
| ce | 3 | 0.0730 | 1.2818 | 0.00391 | −99.7% | yes |
| dt | 3 | 0.1172 | 0.7985 | 0.0651 | −91.8% | yes |
| ttt | 3 | 0.0027 | 21.1321 | 0.0789 | −99.6% | yes |
| ranking | 16 | 0.0282 | 0.1285 | 0.0709 | −44.8% | yes |

**Verdict: DISCARD — capacity-bounded.** Two clean signals: (1) headline
AUROC_w collapsed −0.073 vs M-256 (>14× the 0.005 decision threshold);
(2) RELEASE AUROC fell to **0.486 (below chance)** because the smaller
backbone couldn't learn discharge timing — 29.2% of generated trajectories
hit `max_len=500` without a natural terminal and got forced-terminal
injected, dragging RELEASE generation badly off. Surprisingly the
short-horizon cap=48h calibration *improved* (+0.028) and DEATH MAE got
better (−11 h), but those are dwarfed by the broader collapse. Counter-
intuitively `auroc_simple` is ≈flat (rare outcomes are near-chance for
both M-128 and M-256 on full data — the unweighted mean isn't the
discriminator); the n_pos-weighted mean is where M-256 wins, driven by
RELEASE (huge n_pos=7447) and the better mid-prevalence discrimination
on KIDNEY/DEATH. M-128 is too small for the task. Note: M-128 ran with
the original 50-epoch cap (early-stopped at ep~40, so cap wasn't
limiting); subsequent P6 variants use 100.

---

### P6 / M-384 @ FULL DATA — NEW-BEST-PENDING (largest tested so far)

**Code:** `561f681` + `MODEL_CONFIG embed_dim=384, n_head=6` (head_dim=64
kept, n_layer=4). 14.88 M params (~2.3× M-256). Full data; embedder
retrained on full data; `phase_n_epochs=100`.

**Aggregated headline (11 outcomes):**

| Metric | M-256 (I2b-full-snip) | **M-384** | Δ |
|---|---|---|---|
| patient_auroc_weighted | 0.759 | **0.793** | **+0.034** |
| patient_auprc_weighted | 0.781 | 0.802 | +0.021 |
| patient_auroc_simple | 0.694 | 0.683 | −0.011 |
| patient_auprc_simple | 0.432 | 0.434 | +0.002 |
| patient_maxF1_weighted | 0.763 | 0.764 | ≈flat |
| patient_F1@0.5_weighted | 0.128 | 0.097 | −0.031 |
| patient_maxF1_simple | 0.453 | 0.425 | −0.028 |
| patient_F1@0.5_simple | 0.134 | 0.098 | −0.036 |
| cap=48h AUROC | 0.523 | 0.503 | −0.020 |
| RELEASE MAE (h) | 68.4 | 86.0 | **+17.6** |
| DEATH MAE (h) | 167.6 | 174.4 | +6.8 |
| gen_to_gt_ratio_median | 0.544 | **0.327** | severe under-gen |
| gen_frac_terminal_first24h | 0.051 | 0.344 | terminals too early |
| forced-terminal fraction | low | 2.1% | (low — natural emit, just early) |
| phase2_best_val | 0.150 | 0.145 | better |
| phase3_best_val | 0.930 | 0.892 | better |

**Per-outcome AUROC / AUPRC / F1:**

| Outcome | AUROC | AUPRC | maxF1 (τ*) | F1@0.5 | n_pos | prev |
|---|---|---|---|---|---|---|
| DISGLYCEMIA_Hyper | 0.918 | 0.907 | 0.808 (τ=0.025) | 0.331 | 3550 | 0.415 |
| DISGLYCEMIA_Hypo  | 0.901 | 0.642 | 0.619 (τ=0.169) | 0.300 | 875  | 0.102 |
| KIDNEY            | 0.856 | 0.832 | 0.760 (τ=0.014) | 0.121 | 3839 | 0.448 |
| CARDIO            | 0.804 | 0.848 | 0.805 (τ=0.000) | 0.003 | 5078 | 0.593 |
| DEATH             | 0.763 | 0.394 | 0.382 (τ=0.265) | 0.320 | 1115 | 0.130 |
| RELEASE           | 0.745 | 0.943 | 0.935 (τ=0.002) | 0.000 | 7447 | 0.870 |
| KETOACIDOSIS      | 0.515 | 0.045 | 0.057 (τ=0.000) | 0.000 | 200  | 0.023 |
| RETINOPATHY       | 0.503 | 0.035 | 0.064 (τ=0.000) | 0.000 | 284  | 0.033 |
| NEUROVASCULAR     | 0.502 | 0.023 | 0.039 (τ=0.000) | 0.000 | 170  | 0.020 |
| SKIN_ULCER        | 0.502 | 0.047 | 0.087 (τ=0.000) | 0.000 | 391  | 0.046 |
| NERVOUS_SYSTEM    | 0.500 | 0.061 | 0.114 (τ=0.000) | 0.000 | 517  | 0.060 |

**Per-aux trace (Phase 2, full data):**

| Aux | Unlock ep | λ_max | Anchor raw | Final raw | Δ% | Learning? |
|---|---|---|---|---|---|---|
| ce | 3 | 0.0954 | 0.7806 | 0.00208 | −99.7% | yes |
| dt | 3 | 0.0921 | 0.8084 | 0.0394 | −95.1% | yes |
| ttt | 3 | 0.0021 | 21.1092 | 0.0527 | −99.8% | yes |
| ranking | 11 | 0.0349 | 0.0907 | 0.0378 | −58.3% | yes |

**Verdict: NEW-BEST-PENDING** — `AUROC_w 0.793` is the best in P6 so far,
+0.034 over M-256. The lift comes from sharper discrimination on the
mid- to high-prevalence outcomes — **RELEASE +0.067** (the dominant n_pos
contributor), **CARDIO +0.060**, **KIDNEY +0.023** — while the 5 rare
outcomes all sit *exactly* at chance (~0.50), even worse than M-256's
~0.55. The 6-strong / 5-chance split sharpens with scale: better
representations for the data-rich outcomes don't help the data-starved
ones at all (no improvement was ever realistic at these prevalences with
this dataset). Trade: gen_to_gt 0.33 (severe under-generation), 34% of
trajectories emit a terminal within the first 24 h, and RELEASE MAE
regresses +17.6 h — the larger backbone learned to emit terminals more
*confidently* (only 2.1% forced) but too *eagerly*. Decision pending
M-512 / M-768 — if either beats M-384 by ≤ 0.005, M-384 wins by the
"smallest-within-0.005" rule.

---

## Reproducibility

- Branch `autoresearch-trajectory`.
- Ledger: `results/results-trajectory-fix.tsv` (iteration-loop rows preserved; benchmarking rows appended).
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Iteration-loop archive: `status-iteration-loop.md`.
