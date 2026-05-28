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

## Reproducibility

- Branch `autoresearch-trajectory`.
- Ledger: `results/results-trajectory-fix.tsv` (iteration-loop rows preserved; benchmarking rows appended).
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Iteration-loop history: prior git commits (not on disk).
