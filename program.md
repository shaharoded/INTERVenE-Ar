# autoresearch — EMR Event Prediction: Trajectory-Generation Fix (v2)

Fix the **generation-collapse failure mode** on the deployed M-256 model:
median generation length 3 tokens, 100 % of patients emit a terminal token
(DEATH / RELEASE) within ~1 hour of the 2-day seed end. Under the original
truncated eval this looked great (AUROC 0.918 / AUPRC 0.630). Under the honest
**horizon-extended eval** (windows extend to each patient's true horizon, score
= 0 for windows past generation end) AUROC drops to ~0.452, AUPRC to ~0.107.

The model is excellent at next-48 h risk scoring; it does not produce
multi-day trajectories. This branch is about closing that gap *without*
sacrificing the near-term signal.

`status.md` Sections 1 / 1b carry the full diagnosis. M-256 architecture is
locked.

---

## What is on disk at branch start

- `emr_model/checkpoints/` — the deployed M-256 checkpoints (P1 / P2 / P3 best,
  tokenizer, scaler). **This is the canonical baseline.** Every training
  experiment starts from these.
- `emr_model/checkpoints.bak_originals/` — read-only backup of the canonical
  baseline. **Treat as immutable.** Restore from here before every retrain.
- `emr_model/data/source/` — MIMIC-IV processed data (CSVs, gitignored).
- `results/` — published TSV ledgers, plus per-experiment artefacts the agent
  appends to.

---

## Goal

Improve the **horizon-extended** eval headlines without trading the
truncated-eval capability away:

| Metric (current baseline)     | Direction | Why |
|-------------------------------|-----------|-----|
| `outcome_auroc` 0.452         | ↑         | Multi-day ranking. |
| `outcome_auprc` 0.107         | ↑         | Lift over prevalence. |
| `mae_release_hrs`, `mae_death_hrs` | ↓ | Direct test that generation timing is calibrated. |
| `gen_median_hours` ~1         | ↑         | Trajectory length vs median patient horizon (~152 h). |
| `gen_frac_terminal_first24h` 1.0 | ↓     | Premature terminal emission. |
| Truncated `outcome_auroc` 0.918 | report  | Must not drop more than 0.02 — that capability already exists. |

---

## Hard constraints

- **No hand-coded inference rules** — no terminal token masking, no min-length
  floor, no fixed "must generate N steps". The fix must be learned.
- **`api.py`, `evaluation.py`, `emr_model/data/`, M-256 architecture, VRAM ≤ 24 GB** — locked.
- **`checkpoints.bak_originals/` is read-only.** Never overwrite it. It is the
  source from which `emr_model/checkpoints/` is restored at the start of every
  experiment.

---

## Validation discipline (READ THIS BEFORE EVERY EXPERIMENT)

The previous session burned hours on experiments that looked healthy but
weren't. Three gates must pass before a full training experiment runs:

### Gate 1 — smoke test runs end-to-end

`sample=50, phase{1,2,3}_n_epochs=1` → `python api.py > smoke.log 2>&1` → confirm
the summary block prints including `gen_*` lines and the `multi_horizon`
block.

### Gate 2 — every loss term is honestly active

In the smoke run's per-epoch print lines (`tr_bce`, `tr_ce`, `tr_dt`,
`tr_ranking`, plus any new aux you added):

1. **All raw loss magnitudes within 1–2 orders of magnitude of BCE.** If your
   new aux's raw value is 30 000 while BCE is 0.4, the scheduler's lambda
   calibration will compute `λ ≈ fraction × tr_bce / tr_aux ≈ 1e-6` — the
   gradient on your aux head is then ~zero. **A loss "descending" at λ ≈ 1e-6
   is not being trained**; the descent is driven by the other heads.
   Fix the loss scale (e.g. normalise by sequence length, take log1p of
   hours, clamp targets, use MAE instead of MSE) until raw values land near
   the same magnitude.
2. **All loss terms descend** across the smoke's epoch budget. No flat aux.
3. **The intended target signal of the new mechanism moves on the smoke
   checkpoint**, even faintly. E.g. for a trajectory-length loss: smoke-eval
   `gen_length_mae_hrs` should drop vs the canonical baseline. If the metric
   the new loss targets doesn't budge on a 50-patient 1-epoch smoke, the
   formulation is broken — don't pay for a full run.

### Gate 3 — every change starts from the canonical baseline

Before each training experiment:

```bash
rm -rf emr_model/checkpoints
cp -r emr_model/checkpoints.bak_originals emr_model/checkpoints
```

Never chain on top of a previous experiment's "fixed" backbone. Each
experiment's headline must be measured **against the canonical baseline**,
not against the most recent intermediate.

Only when all three gates pass → run `python api.py > run.log 2>&1`.

---

## Multi-horizon evaluation (automatic, every run)

`evaluation.py::evaluate_on_test_set` now emits a **multi-horizon AUC table**
on every full eval. Same generated `risk_df` scored against three
per-patient horizon caps: 48 h, 168 h, 336 h. The eval block prints lines:

```
multi_horizon<TAB>horizon_cap_hrs<TAB>outcome<TAB>auroc<TAB>auprc<TAB>n_pos<TAB>n_neg
multi_horizon	48	DEATH_EVENT	...	...	...	...
multi_horizon	168	DEATH_EVENT	...	...	...	...
multi_horizon	336	DEATH_EVENT	...	...	...	...
...
multi_horizon_csv: results/multi_horizon_<commit>.tsv
```

Use this to read the **horizon curve** for an experiment:

- A healthy fix improves AUROC at the longer caps (168 h, 336 h) without
  losing the 48 h signal.
- A model that's still collapsed will look strong at 48 h and crash to chance
  at 168 h / 336 h — that's the "before" picture.
- A model that's only learned long-horizon at the cost of near-term will
  improve 336 h but regress 48 h — also worth knowing.

Report all three horizon caps (per-outcome and mean) in the journal row for
every experiment.

Cheap pre-screen recipe: if you suspect a training change might break the
near-term capability, you can call `pooled_episode_auc` with
`patient_horizons=None` directly on the smoke `risk_df` for a fast read on
truncated AUROC before paying for the full multi-horizon pass.

---

## Workflow per experiment

1. **Re-read this file.** Check `git status`, last few rows of
   `results/results-trajectory-fix.tsv`, recent log lines from the last run.
2. **Restore the canonical baseline** (Gate 3 command above).
3. Propose ONE change with a falsifiable hypothesis.
4. **Smoke test** → Gate 1 + Gate 2 checks. Do not skip.
5. **Code commit** (just the code files) with a 3-part message: change /
   diagnostic / expectation. Push. Note the commit hash `<CODE_SHA>`.
6. **Full run**: `python api.py > run.log 2>&1`. For pure inference-side
   experiments use `python api.py --eval-only`; no training cost.
7. Pull `outcome_*`, `gen_*`, `mae_*`, `multi_horizon` lines from `run.log`
   into a single new row in `results/results-trajectory-fix.tsv`.
8. Write an experiment block in `status.md` ending with `Verdict: <KEEP|DISCARD> — <one-sentence reason>`.
9. **Journal commit** (only `status.md` + `results/`) with message
   `journal: <tag> <VERDICT> — <summary>`. Push.
10. On **DISCARD**: `git revert --no-edit <CODE_SHA>` then push. The journal
    commit stays — the failure record is visible.
11. On **KEEP**: do nothing more; next experiment starts here.
12. **Never `git reset --hard`** for a DISCARD. That erases the failure
    record from the journal, which is the user's communication channel.

### KEEP iff all of:

- All three Validation gates passed during smoke.
- Peak VRAM ≤ 24 GB.
- At least one headline horizon-extended metric improves past the noise
  floor (AUROC ≥ +0.005, AUPRC ≥ +0.005, MAE ≥ −5 h).
- No headline metric regresses past the noise floor.
- Truncated `outcome_auroc` (the 48 h row of `multi_horizon`) does not drop
  more than 0.02 below the 0.918 baseline.
- `gen_median_hours` strictly above previous best, or already ≥ 50 % of the
  median patient horizon.
- `gen_frac_terminal_first24h` strictly below previous best, or already
  below 0.10.

Otherwise DISCARD → revert the code commit.

---

## Research directions

Inference-first directions cost almost no compute (use `--eval-only` against
the deployed checkpoints — no retraining). Training-side directions cost a
full Phase-1/2/3 pipeline. Combine, replace, or invent alternatives — every
change needs a falsifiable hypothesis about why it should extend generations
without hard rules.

**Inference-side (cheap — `python api.py --eval-only`):**

- **F1. Beam search with length-normalised scoring.** Multi-candidate decoding,
  score / length^α. Falsifiable: if generation extends without any constraint,
  the single-trajectory sampler was the bottleneck.
- **F2. Sampling-temperature schedule.** Higher temperature in the first N
  steps to escape the immediate-terminal local minimum, anneal as generation
  proceeds. Falsifiable: `gen_median_hours` rises monotonically with starting
  temperature.
- **F3. Hazard-driven terminal sampling at inference.** Use the existing
  outcome head's terminal logits to draw the terminal time from a smoothed
  distribution instead of emitting on first peak. No retraining needed.

**Training-side (requires retraining; uses Phase-1 cache when embed_dim
unchanged):**

- **A. Scheduled sampling.** Gradually replace teacher-forced inputs with
  the model's own predictions during Phase 2. Anneal `p` from 0 → ~0.3.
  Falsifiable: median generation length rises as `p` increases on an
  in-training probe.
- **B. Trajectory-length loss.** Phase-2 sequence-level loss on cumulative
  Δt mismatch. **Watch the scale**: with both `pred_abs` and `true_abs` in
  normalised units ([0, 1] = hours / 336), `MSE(sum_pred_Δt, sum_true_Δt)`
  must land in O(1), not O(10⁴). If raw values blow up, the formulation is
  wrong — fix it before paying for a full run. Falsifiable:
  `gen_length_mae_hrs` drops below ~48 h.
- **C. Time-to-terminal regression head.** Auxiliary regression on
  `log1p(t_terminal − t_now)` at every non-terminal position. Falsifiable:
  head R² > 0.3 on the regression target; terminal MAE drops.
- **D. Discrete-time hazard for terminals.** Replace BCE on DEATH/RELEASE
  with hazard bins (1, 6, 24, 72, 168 h); inference samples terminal time
  from the hazard distribution. Falsifiable: terminal MAE drops, no 0 h
  collapse.
- **E. Narrow terminal `tau_lm`.** The 168 h soft-kernel window for terminals
  teaches the model that "predict terminal soon" minimises BCE almost
  everywhere. Narrow to 12–24 h and/or down-weight terminal in `pos_weight`.
  Falsifiable: `gen_frac_terminal_first24h` drops without hurting
  complication-class AUROC.
- **G. Curriculum: short-horizon → long-horizon supervision.** Train Phase 2
  with a weighted mix of next-48 h BCE (the existing capability) and a
  multi-day cumulative term, with the multi-day weight ramping up across
  epochs. Anchors training on what the model already does well, then asks
  it to extend. Falsifiable: at the end of training the 48 h `multi_horizon`
  AUROC ≥ 0.91 and the 336 h AUROC strictly above 0.45.

**Encouraged ordering**: try F1–F3 first on the canonical checkpoints —
no retraining, results in seconds. If those don't fix the collapse, move
to training-side directions one at a time, restoring the canonical baseline
between each.

---

## Stop criterion

Stop when you have a **publishable multi-day event-prediction result** under
the horizon-extended eval:

- 336 h `multi_horizon` AUROC meaningfully above the 0.452 baseline.
- 336 h AUPRC meaningfully above the prevalence baselines (lifts ≥ 2× for
  most outcomes).
- 48 h `multi_horizon` AUROC essentially preserved (within 0.02 of 0.918).
- `gen_median_hours` a meaningful fraction of the median patient horizon
  (~150 h).
- Terminal MAE small enough to be clinically informative.

No hard threshold on any single metric — use judgement on whether the
combined picture is defensible. If after honest attempts across multiple
directions the gap can't be closed, document the trade-off honestly. The
deployed M-256 remains a publishable result under the "next-48 h event-
window risk scorer" framing — that's a real, useful model.

---

## Reproducibility

- Branch: `autoresearch-trajectory`. Code commits here only; no force-push.
- Ledger: `results/results-trajectory-fix.tsv` — one row per experiment.
- Multi-horizon raw TSV per run: `results/multi_horizon_<commit>.tsv`.
- Per-outcome (horizon-extended) raw TSV per run: `results/per_outcome_<commit>.tsv`.
- Checkpoints: `emr_model/checkpoints/` gitignored; `checkpoints.bak_originals/`
  read-only baseline. Restore the latter into the former between experiments.
- Journal: `status.md` at repo root. Sections 1 and 1b from
  `autoresearch-optimization` stay intact at the top.
