# autoresearch — Trajectory-Generation Fix

## The problem

Deployed M-256 collapses generation: median trajectory = 3 tokens / ~1 hour;
100 % of patients emit a terminal (DEATH/RELEASE) within 24 h of seed end.
Truncated eval looks great (AUROC 0.918 / AUPRC 0.630). The honest
**horizon-extended eval** (windows extend to each patient's true admission
horizon, score = 0 past generation end) collapses to AUROC 0.452 / AUPRC
0.107. Model is a strong **next-48 h scorer**; it does **not** produce
multi-day trajectories.

Fix that without sacrificing the near-term capability. M-256 architecture
locked. VRAM ≤ 24 GB. `api.py`, `evaluation.py`, data, and
`emr_model/checkpoints.bak_originals/` are read-only.

## Tools at your disposal

Every `python api.py > run.log` or `python api.py --eval-only > run.log`
emits a grep-friendly summary block:

| Key prefix              | What |
|-------------------------|------|
| `outcome_auroc:` / `outcome_auprc:` | Horizon-extended headlines |
| `onset_mae_hrs:` / `mae_release_hrs:` / `mae_death_hrs:` | Onset & terminal timing |
| `gen_median_hours:`, `gen_p90_hours:`, `gen_max_hours:` | Generated trajectory length (time) |
| `gen_median_steps:`, `gen_n_with_terminal:`, `gen_frac_terminal_first24h:` | Generation behaviour |
| `gen_length_mae_hrs:`, `gt_median_hours:`, `gen_to_gt_ratio_median:` | Trajectory length vs GT (1.0 = full coverage) |
| `multi_horizon\t<cap_h>\t<outcome>\t...` | Per-outcome AUROC/AUPRC at every 24 h cap, 24…336 h |
| `per_outcome\t<outcome>\t...`              | Per-outcome AUROC/AUPRC at the horizon-extended cap |
| `phase{2,3}_best_val:` | Training loss curves' best val per phase |

Plus persisted CSVs in `results/`: `per_outcome_<commit>.tsv`,
`multi_horizon_<commit>.tsv`. Use `--eval-only` for inference-side
experiments (no training cost; reads the deployed checkpoints).

`api.py` already does the right thing: Phase 2/3 are wiped and retrained
from scratch every run (with current code); Phase 1 cache-hits on
`(embed_dim, time2vec_dim, ctx_dim)`; `processed_datasets.pt` is invariant
across model experiments. **No mid-training resume ever happens.** Don't
use `phase2_warm_start_path` / `phase3_warm_start_path` — that pattern is
banned (the previous session's failure mode).

## Goal — current → target

| Metric (current)                     | Direction |
|--------------------------------------|-----------|
| `outcome_auroc` 0.452                | ↑ |
| `outcome_auprc` 0.107                | ↑ |
| `gen_length_mae_hrs` ~140            | ↓ |
| `gen_to_gt_ratio_median` ~0          | → 1.0 |
| `gen_frac_terminal_first24h` 1.0     | ↓ |
| Truncated AUROC (cap=48 h) 0.918     | preserve (don't drop > 0.07) |

## The loop

```
1. Read `program.md`. Check `git log --oneline -5`, last rows of
   `results/results-trajectory-fix.tsv`.
2. Propose ONE change with a falsifiable hypothesis.
3. SMOKE TEST  (sample=50, phase{1,2,3}_n_epochs=1):
      python api.py > smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 orders of magnitude of
           BCE — otherwise `λ_aux ≈ 1e-6` and the loss isn't actually
           being trained (looks "calibrated", but gradient is dead).
   Gate-C: calibrated λ from LambdaScheduleController in [1e-3, 10].
   Gate-D: summary block + `multi_horizon` block + `per_outcome` block
           all print. No silent exceptions.
   If any gate fails → fix formulation, do NOT pay for a full run.
4. Commit code only:
      git add <code files> && git commit -m "<tag>: change / why / expected"
      git push
   Note <CODE_SHA>.
5. FULL RUN:  python api.py > run.log 2>&1
   (or --eval-only for inference-side experiments)

   POST-TRAIN VALIDATION (mandatory before logging a verdict):
   Gate-T1: Read run.log epoch-by-epoch. Every aux's RAW loss term must
            show a real decrease across its active phase (no flat aux,
            no NaNs, no oscillation indicating exploded gradients).
   Gate-T2: Early stop / warmup gate did not trigger before all aux
            losses finished ramping (scheduler prints `warmup ends at
            epoch N` — early stop must not have fired before N).
   Gate-T3: Run diagnose.py and read it. The model must show real
            discrimination — not trivial predictions:
              cd emr_model && python -m transform_emr.diagnose > ../diag.log 2>&1
            Check Report 1 (per-outcome AUROC: bottom outcomes not
            collapsed to ~0.5), Report 2 (sigmoid separation ≥ 0.05),
            Report 4 (calibrated λ ≥ 1e-3 per aux), Report 5 (outcome
            tokens ranked in top-half by grad/occ), Δt probe (R² > 0.05).
            If any of these regress materially vs the running best, the
            training is not honest — treat as DISCARD candidate even if
            headlines look OK.
   Only when Gates T1–T3 all pass: write the verdict.
6. Append one row to results/results-trajectory-fix.tsv with the
   summary-block headlines (incl. multi_horizon caps 24/48/168/336 and
   gen_* / mae_* / outcome_*).
7. Write a `### <tag>` block in status.md ending with
   `Verdict: KEEP|DISCARD — <reason>`.
8. Journal commit:
      git add status.md results/ && git commit -m "journal: <tag> <VERDICT> — <summary>"
      git push
9. If DISCARD:  git revert --no-edit <CODE_SHA>  &&  git push
   (Never `git reset --hard` — that erases the journal entry.)
10. If KEEP: this is the new running best. Back up checkpoints:
      cp -r emr_model/checkpoints emr_model/checkpoints.bak_keep_<tag>
    The KEEP'd code stays in HEAD; the next experiment builds on top
    of it. Compare future experiments against THIS state, not against
    bak_originals.
```

## Research directions

Ordered by leverage. **Loss / training first** — teach the model *when*
to stop. Inference-side tweaks only after the backbone has been honestly
improved.

### Primary — train the model to know when to finish

Loss-scale matters more than novelty: any aux whose raw magnitude differs
from BCE by 4+ orders gets `λ ≈ 1e-6` after calibration and isn't trained.

- **B. Trajectory-length loss.** Penalise `|sum_pred_Δt − sum_true_Δt|`
  per patient. `pred_abs`, `true_abs` are normalised to [0, 1]; keep the
  loss in O(1) — use log1p-hours or MAE, not raw squared normalised
  units. *Falsifiable*: `gen_length_mae_hrs` drops; `gen_to_gt_ratio` rises.
- **C. Time-to-terminal regression head.** Auxiliary regression on
  `log1p(t_terminal − t_now)` at every non-terminal position. The
  backbone learns *distance* to discharge/death, not just *imminence*.
  *Falsifiable*: head R² > 0.3; terminal MAE drops.
- **A. Scheduled sampling.** Anneal teacher-forcing replacement
  `p: 0 → ~0.3` across Phase 2. Closes the train/inference gap.
  *Falsifiable*: median generation length rises monotonically with `p`.
- **G. Short → long horizon curriculum.** Phase 2 loss = weighted mix of
  next-48 h BCE (preserves 0.918) + multi-day cumulative term, multi-day
  weight ramping up. *Falsifiable*: end of training, 48 h
  `multi_horizon` AUROC ≥ 0.91 *and* 336 h clearly above 0.45.

### Secondary

- **D. Discrete-time hazard for terminals** (replace BCE on DEATH/RELEASE
  with hazard bins, sample terminal time from the distribution). Larger
  structural change.
- **E. Narrow terminal `tau_lm`** (current 168 h kernel teaches "terminal
  is always near"; try 12–24 h, or down-weight terminal in `pos_weight`).
  Small surface; may not be enough on its own.

### Inference-side — only after the backbone is honest

(`python api.py --eval-only` — no retraining cost.)

- **F1. Beam search** with length-normalised score (score / length^α).
- **F2. Temperature schedule** — higher temperature in first N steps to
  escape the immediate-terminal local minimum, anneal back.
- **F3. Hazard-driven terminal sampling** at inference using the existing
  outcome head.

## KEEP vs DISCARD

Comparison reference = **the most recent KEEP** (= running best). First
experiment compares to `bak_originals` (canonical M-256, AUROC 0.452 /
AUPRC 0.107).

**KEEP** iff *all*:
- Smoke gates A–D all passed.
- Peak VRAM ≤ 24 GB.
- ≥ 1 horizon-extended headline improves past noise floor
  (AUROC ≥ +0.005, AUPRC ≥ +0.005, MAE ≥ −5 h).
- No headline regresses past noise floor.
- Truncated AUROC (`multi_horizon` cap=48 h) drops by < 0.07.
- `gen_median_hours` strictly above running best (or already ≥ 50 % of
  median patient horizon).
- `gen_frac_terminal_first24h` strictly below running best (or already
  < 0.10).

**Otherwise DISCARD** → `git revert <CODE_SHA>`.

## Stop criterion

When the horizon-extended `outcome_auroc` is clearly above the 0.452
baseline, AUPRC clearly above prevalence (lifts ≥ 2× for most outcomes),
`gen_median_hours` a meaningful fraction of the median patient horizon,
and terminal MAE clinically informative — that is a publishable
multi-day predictor. Stop and write a final session report in `status.md`.

If after honest attempts the gap can't be closed, document the trade-off
honestly. The deployed M-256 remains publishable under the "next-48 h
event-window scorer" framing.

## Reproducibility

- Branch `autoresearch-trajectory`; no force-push to `main`.
- Ledger: `results/results-trajectory-fix.tsv`.
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Journal: `status.md` (Sections 1 / 1b from prior branch stay intact).
