# EMR Event-Prediction Transformer — Patient-Level Eval Loop

## Inherited from prior session

Decisions carried forward from the architecture sweep + ablations. The
specific AUC numbers from that session were computed under the
**per-window** eval framing and **are not comparable** to the new
patient-level peak-detector headline — don't anchor to them.

- **Architecture**: M-256 — `embed_dim=256`, `n_layer=4`, `n_head=4`,
  `time2vec_dim=32`, `dropout=0.10`. Params ~6.4 M. Peak VRAM at training ~5 GB.
- **Optimiser**: AdamW. `phase{1,2}_lr=3e-4`, `phase3_lr=1e-4`,
  `phase3_backbone_lr_factor=0.01`. Aux caps `{ce: 0.5, dt: 0.5, ranking: 0.2}`.
- **Training**: three-phase. Phase 1 embedder; Phase 2 GPT pretrain with
  curriculum (BCE → CE + Δt → pairwise ranking); Phase 3 outcome-head fine-tune.
- **Evaluation seed**: 2-day input → 14-day generation horizon. (Prior
  k-day-seed scan ruled k=1 below operational floor; AUROC plateaued
  from k=2 onward; k=2 chosen for the operational use case.)
- **QA data**: `USE_QA_DATA=False`. The QA-augmented variant added new
  context features + tokens; in the prior eval framing it didn't move
  the headline. The new loop will revisit this in **P7** as the final
  step on the running-best model.
- **Running best on HEAD**: Z (direction E — narrow + frozen terminal
  `log_tau_lm`). No checkpoint on disk; pod is fresh and Phase 1
  retrains.

---

## Patient-level eval loop

Per `program.md`. New eval framing (per-patient peak detector). The
agent appends `### <tag>` blocks here as experiments run.

Each block records: tag, what changed (1–2 lines), smoke gate results,
post-train gate results, headline numbers (`patient_auroc_weighted`,
per-outcome AUROC for DEATH/RELEASE/each complication, peak-MAE),
trajectory honesty (`gen_to_gt_ratio_median`,
`gen_frac_terminal_first24h`), verdict (KEEP / DISCARD) with reason.

### B0-Z @ 10k (SHA 8d3cf18)

P0 baseline. Z (direction E — narrow + frozen terminal `log_tau_lm`,
init `log(12/336)`) on HEAD. No code change — first run on new
patient-level peak-detector eval.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gate A pass — no NaN, train=8.5680, val=7.8539 at Phase-3 epoch 1.
- Gate B pass — raw_out=8.568, raw_rank=0.691 (~12×, within 1–2 OOM).
- Gate C pass — λ_ranking calibrated 2.479 ∈ [1e-3, 10].
- Gate D pass — summary block + all headline keys emit.

Post-train (10k):
- T1 pass — Phase-3 raw_out 2.20→1.05, raw_rank 0.66→0.41 across 29 epochs.
- T2 pass — Phase-2 early stop at epoch 46 (ranking ramped from epoch 32,
  fully active by 35, ran active 11+ epochs before stop). Phase-3 early
  stop at epoch 29 with best val at epoch 9 (1.0105).
- T3 pass — patient AUROC shows real discrimination on the headline
  outcomes (see below).

Headline:
- `patient_auroc_weighted`: **0.6671**
- `patient_auprc_weighted`: 0.6205
- `patient_auroc_simple`:   0.6932
- `patient_auprc_simple`:   0.3032
- `n_outcomes_used`:        16

Per-outcome AUROC (top):
- DISGLYCEMIA_Hyperglycemia 0.904 (AUPRC 0.871, n_pos=619)
- DISGLYCEMIA_Hypoglycemia  0.797
- KETOACIDOSIS              0.791
- NERVOUS_SYSTEM_DISORDER   0.788
- RETINOPATHY               0.776
- NEUROVASCULAR             0.749
- KIDNEY_COMPLICATION       0.702 (AUPRC 0.634, n_pos=685)
- CARDIO-VASCULAR_DISORDER  0.701 (AUPRC 0.744, n_pos=860)
- **DEATH**                 0.693 (AUPRC 0.228, n_pos=192)
- SKIN_ULCER                0.663
- HYPEROSMOLALITY           0.644
- ATHEROSCLEROSIS           0.608
- ACUTE_RESPIRATORY         0.605
- ACIDOSIS                  0.585
- INFECTION                 0.566
- **RELEASE**               0.521 (AUPRC 0.881, n_pos=1308)

Peak MAE (hours, positives only):
- DEATH:    158.84  (n=191)
- RELEASE:   85.97  (n=1308)
- DISGLYCEMIA_Hyper: 43.98
- DISGLYCEMIA_Hypo:  66.15
- KIDNEY:           106.36
- CARDIO:           107.99
- (others 145–234 h)

Trajectory honesty:
- `gen_median_hours`:         114.48
- `gen_to_gt_ratio_median`:     1.116 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`: 0.148
- `gen_length_mae_hrs`:       101.48

Phase stats: phase2_best_val 0.184 / 46 epochs (early stopped),
phase3_best_val 1.157 / 29 epochs (early stopped).

Verdict: **BASELINE-KEEP** — first patient-level eval reference.
Running best until B0-C-ttt result is in. Checkpoints backed up to
`emr_model/checkpoints.bak_keep_B0-Z/`.

---

### B0-C-ttt @ 10k (SHA ea65988)

P0 baseline #2. Cherry-pick of dd3fc1b "C-ttt-head" (time-to-terminal
regression aux) on top of B0-Z. Adds an MSE head predicting
`log1p(t_terminal − t_now)` at every non-terminal, non-pad position,
sharing the backbone. Joins Phase-2 stage 0 alongside ce/dt with
fraction_cap=0.30. Goal: force the backbone to encode distance-to-
terminal explicitly so the LM head can decide WHEN to emit terminal
tokens.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gate A pass — no NaN; RawTrain ce=1.31, dt=0.81, ranking=0.69,
  ttt=19.19, all finite.
- Gate B pass — ttt within ~25× of ce/dt (within 1–2 OOM).
- Gate C pass — λ_ranking calibrated 2.497 ∈ [1e-3, 10].
- Gate D pass — summary block + all headline keys present.

Post-train (10k):
- T1 pass — Phase-3 raw_out 2.11→1.01, raw_rank 0.66→0.38; ttt λ_max
  calibrated at Phase-2 epoch 3 (λ_max=0.0040, raw_aux=20.86 — head
  starts well above ce/dt then decays).
- T2 pass — Phase-2 ranking calibrated epoch 31, ramp 31→35, full
  active by 35; Phase-2 early stop at epoch 40 (5 epochs of full
  stage-1 activity before stop). Phase-3 best val at epoch 15 (0.996),
  early stop at epoch 23.
- T3 pass — DEATH AUROC 0.710 (+0.017 vs B0-Z), KIDNEY 0.715,
  CARDIO 0.709, KETOACIDOSIS 0.915 (+0.124 — biggest single per-outcome
  swing).

Headline (Δ vs B0-Z @ 10k):
- `patient_auroc_weighted`: **0.6831** (+0.0160 ✓)
- `patient_auprc_weighted`: 0.6336 (+0.0131 ✓)
- `patient_auroc_simple`:   0.6959 (+0.0027)
- `patient_auprc_simple`:   0.3239 (+0.0207)
- `n_outcomes_used`:        16

Per-outcome AUROC vs B0-Z:
- KETOACIDOSIS              0.915  (+0.124, n_pos=37)
- DISGLYCEMIA_Hyperglycemia 0.896  (−0.008)
- NERVOUS_SYSTEM            0.796  (+0.008)
- RETINOPATHY               0.785  (+0.009)
- DISGLYCEMIA_Hypoglycemia  0.771  (−0.026)
- KIDNEY                    0.715  (+0.013)
- **DEATH**                 0.710  (+0.017) ✓
- CARDIO                    0.709  (+0.008)
- NEUROVASCULAR             0.686  (−0.063)  ← biggest regression
- SKIN_ULCER                0.679  (+0.016)
- ATHEROSCLEROSIS           0.595  (−0.013)
- ACUTE_RESPIRATORY         0.591  (−0.014)
- HYPEROSMOLALITY           0.585  (−0.059)
- **RELEASE**               0.581  (+0.060) ✓
- ACIDOSIS                  0.570  (−0.015)
- INFECTION                 0.551  (−0.015)

Peak MAE vs B0-Z (hours):
- DEATH:    168.97  (+10.13  — REGRESSION ≥ 5h threshold)
- RELEASE:   71.29  (−14.68 ✓)
- DISGLYCEMIA_Hyper:  36.07 (−7.91)
- KIDNEY:            79.11  (−27.25)
- CARDIO:            79.08  (−28.91)

Trajectory honesty:
- `gen_median_hours`:           75.05  (vs B0-Z 114.48 — generates shorter)
- `gen_to_gt_ratio_median`:      0.720  (vs B0-Z 1.116 — still ≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  0.165  (vs B0-Z 0.148 — slight bump)

Phase stats: phase2_best_val 0.187 / 41 epochs (early stopped);
phase3_best_val 1.144 / 23 epochs (early stopped). Both terminate
earlier than B0-Z (46/29) — Phase-3 best val is also lower (1.144 vs
1.157), so faster convergence on a better minimum.

Verdict: **BASELINE-KEEP, RUNNING BEST** — between the two P0
baselines, B0-C-ttt clearly wins on the primary headline
(`patient_auroc_weighted` 0.683 > 0.667) and lifts both DEATH and
RELEASE AUROC simultaneously, which is the precise pattern program.md
predicted under the new framing. The DEATH-MAE regression (+10 h) and
the NEUROVASCULAR / HYPEROSMOLALITY AUROC dips are real costs, but
n_pos is small (29, 83) so per-outcome variance is high, and the model
is generating 35 % shorter sequences (75 h vs 114 h) which mechanically
explains the slight DEATH-MAE drift toward the rare-DEATH median.
P0 KEEP rule (better of two baselines) applies — KEEP/DISCARD threshold
test is for subsequent experiments vs this running best.

Checkpoints backed up to `emr_model/checkpoints.bak_keep_B0-C-ttt/`.
This is the running best for P1 (MIL max-BCE).

---

## Reproducibility

| Artefact | Location |
|---|---|
| Branch | `autoresearch-trajectory` |
| Canonical baseline checkpoints (read-only) | `emr_model/checkpoints.bak_originals/` |
| Running-best backups | `emr_model/checkpoints.bak_keep_<tag>/` |
| Ledger | `results/results-trajectory-fix.tsv` |
| Source data (not in repo) | `emr_model/data/source/temporal_data.csv` + `context_data.csv` |
| Train / val / test split | `PatientId`-stratified 70 / 15 / 15, `random_state=42` (in `api.py`) |

To reproduce from a fresh clone: place source CSVs under
`emr_model/data/source/`, then `python api.py`. The pipeline builds a
tokenizer + scaler from the train split, caches the processed dataset,
runs the three phases (training in one subprocess, eval in another),
and prints the summary block.
