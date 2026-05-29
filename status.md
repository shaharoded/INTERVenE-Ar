# EMR Event-Prediction Transformer — Benchmarking Phase

Fresh journal for the benchmarking phase. The iteration-loop journal is in
prior git commits if anyone needs the history.

## Inherited from the iteration loop

Locked architecture and recipe (full spec in `program.md`):

- **4-layer transformer** with `time2vec_dim=32, dropout=0.10, head_dim=64` (heads scale with `embed_dim`). Size selected by P6 sweep — start small (M-128) and grow.
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

- 7 complications (in `OUTCOMES`) (but in data only 5 will pass the min. support threshold)
- 2 terminals (in `TERMINAL_OUTCOMES`): DEATH, RELEASE
- `AUC_EXCLUDE = ("RELEASE_EVENT",)` in `evaluation.py` — RELEASE stays in head
  training (so the model emits it correctly) but is excluded from the AUROC /
  AUPRC / F1 headline (it's `¬DEATH` in this cohort, redundant ranking task).
  Reported as length-of-stay MAE instead.

**6 evaluated outcomes for the headline**: 5 complications + DEATH.

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

### P6-M-128-10k — RECIPE-TRANSFER SMOKE (Step 1)

**What:** Validate the locked I2b recipe transfers to the FRESH dataset. M-128
(embed_dim=128, n_head=2, n_layer=4; 1.75M params), `sample=10000` (7000 train /
1500 val / 1500 test patients), full epochs (100 cap each phase, early-stopped).
Tokenizer/scaler/embedder rebuilt from this sample (smoke-built tiny checkpoints
wiped first). Config commit `d05589b`.

**Smoke gates (sample=50, epochs=1 — commit ee4ce76→d05589b path):**
- A (no NaN/inf): PASS — all phase losses finite.
- B (aux raw within 1–2 OOM of BCE): PASS — ttt highest at ~1.6 OOM, within bound.
- C (calibrated λ in [1e-3,10]): PASS — only P3 λ calibrate in 1 epoch (ranking 1.49,
  pool 0.29); P2 auxes pending (bce_only_epochs=4 > 1) as expected.
- D (summary + all headline keys): PASS — `n_outcomes_used=6` (5 complications +
  DEATH; KETOACIDOSIS+ACIDOSIS auto-filtered <1%, RELEASE AUC-excluded).

**Post-train gates (10k full run):**
- T1 (every aux descends across active phase): PASS — see trace table.
- T2 (early-stop after auxes ramped): PASS — P2 ranking unlocked ep54, warmup ep57,
  P2 early-stopped ep70 (57 < 70); P3 λ calibrated ep1, ran 48 ep.
- T3 (real discrimination): PASS — all 6 outcomes AUROC 0.704–0.931.

**Headline (held-out test, 1500 patients):**
- `patient_auroc_weighted` **0.813**, `patient_auprc_weighted` 0.683,
  `patient_max_f1_weighted` 0.639, `patient_f1_at_0_5_weighted` 0.410,
  simple AUROC 0.812. n_outcomes=6.
- Per-outcome AUROC / AUPRC / maxF1 / F1@0.5 / peak-MAE(h):
  - CARDIO-VASCULAR 0.931 / 0.667 / 0.637 / 0.609 / 39.0
  - DISGLYCEMIA_Hyper 0.848 / 0.793 / 0.729 / 0.636 / 25.0
  - KIDNEY 0.817 / 0.743 / 0.650 / 0.592 / 27.5
  - HYPEROSMOLALITY 0.808 / 0.743 / 0.704 / 0.205 / 32.0
  - DISGLYCEMIA_Hypo 0.765 / 0.282 / 0.378 / 0.076 / 47.4
  - DEATH 0.704 / 0.329 / 0.343 / 0.116 / 153.9
- Length-of-stay MAE 63.8h (median 53.1, p90 134.7, n=1307).
- Multi-horizon AUROC: cap48 0.642, cap168 0.657, cap336 0.605.

**Trajectory honesty (near-perfect):** `gen_to_gt_ratio_median` 1.023,
`gen_frac_terminal_first24h` 0.075, gen_median 106.3h vs gt_median 103.9h,
1499/1500 natural terminals (no forced-terminal over-generation).

**Per-aux training trace table (mandatory):**

| Phase | Aux | Unlock/calib ep | λ_max | anchor raw | final raw | Δ% |
|---|---|---|---|---|---|---|
| 1 | dt | calib ep3 (active ep4) | 0.0415 | 1.918 (ep1) | 0.752 (ep40) | −60.8% |
| 2 | ce | calib ep3 (active ep4) | 0.1167 | 1.476 | 0.0041 | −99.7% |
| 2 | dt | calib ep3 (active ep4) | 0.2151 | 0.801 | 0.103 | −87.2% |
| 2 | ttt | calib ep3 (active ep4) | 0.0048 | 21.430 | 0.125 | −99.4% |
| 2 | ranking | unlock ep54 (calib ep53) | 0.0259 | 0.245 (ep53) | 0.107 (ep69) | −56.4% |
| 3 | outcome BCE | ep1 | — | 2.829 | 1.867 (ep48) | −34.0% |
| 3 | ranking | calib ep1 | 0.9643 | 0.587 | 0.347 (ep48) | −40.9% |
| 3 | pool | calib ep1 | 0.1504 | 0.941 | 0.060 (ep48) | −93.7% |

All |Δ| ≥ 34% — every aux is learning (none flagged <5%).

**Verdict: RECIPE-TRANSFER CONFIRMED.** The locked recipe transfers cleanly to the
fresh dataset — all smoke gates A–D and post-train T1–T3 pass, all auxes descend
strongly, no degenerate outputs (honest trajectories), AUROC headline 0.813 with
all 6 outcomes well above chance. This is a verification probe, not a KEEP/DISCARD.
phase2_val 0.191 (70 ep), phase3_val 2.117 (48 ep), 1.75M params, peak_vram 196MB.
**Proceed to Step 2 — P6 full-data architecture sweep (M-128 → M-256 → M-384 → M-512 → M-768).**

## Reproducibility

- Branch `autoresearch-trajectory`.
- Ledger: `results/results-trajectory-fix.tsv` (iteration-loop rows preserved; benchmarking rows appended).
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Iteration-loop history: prior git commits (not on disk).
