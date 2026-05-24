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

## Goal — three coupled objectives, not one

Closing the gap is **simultaneously** about:

1. **Trajectory collapse** — generation length and termination timing
   (`gen_to_gt_ratio_median`, `gen_frac_terminal_first24h`, `gen_length_mae_hrs`).
2. **Full-trajectory AUC** — horizon-extended ranking & precision-recall
   (`outcome_auroc`, `outcome_auprc`, multi_horizon cap=336h).
3. **Short-term AUC** — preserve the strong next-48h capability
   (`multi_horizon` cap=48h ≥ canonical −0.07).

A fix that solves (1) at the cost of (2) or (3) isn't a fix — it's a
trade. The expectation is that **several different strategies will be
needed**, each addressing one or two of the three. No single experiment
has to solve everything; the goal is for the running-best stack to
collectively close all three.

| Metric (current)                     | Direction | Tied to |
|--------------------------------------|-----------|---------|
| `outcome_auroc` 0.452                | ↑         | (2) full AUC |
| `outcome_auprc` 0.107                | ↑         | (2) full AUC |
| `gen_length_mae_hrs` ~140            | ↓         | (1) collapse |
| `gen_to_gt_ratio_median` ~0          | → 1.0     | (1) collapse |
| `gen_frac_terminal_first24h` 1.0     | ↓         | (1) collapse |
| multi_horizon cap=48h AUROC mean     | preserve  | (3) short-term |

## The loop

**10k-sample loop is the primary workspace.** Trajectory collapse, full-
horizon AUC, and short-horizon (cap=48h) AUC are three coupled
objectives — closing the gap likely needs many different architectural
attempts, not one silver bullet. Burning a full-data run on every idea
wastes hours per experiment. Instead:

- **All experiments run at `sample=10000`** (~25 min on GPU per training
  experiment) — this is the running iteration loop. KEEP/DISCARD
  verdicts and the running best are tracked at 10k scale.
- **Full-data confirm runs are reserved for the end of an architectural
  block** — when the agent has run through the primary directions, the
  running best has settled, and further 10k experiments are not yielding
  meaningful gains. Only then do we re-run the running best (and
  optionally the second/third candidates) at `sample=None` to produce
  publishable numbers.
- The full-data run is a **validation step**, not a verdict step. If
  the 10k running best fails to replicate at full scale, that's a
  signal we have a sample-size artifact and need more 10k probing —
  not an automatic DISCARD.

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
   If any gate fails → fix formulation, do NOT pay for any longer run.
4. Commit code only:
      git add <code files> && git commit -m "<tag>: change / why / expected"
      git push
   Note <CODE_SHA>.
5. EXPERIMENT RUN  (sample=10000, full epochs):
      # set sample=10000 in model_config.py
      python api.py > run.log 2>&1
   ~25-30 min on GPU. Gates T1-T3 apply.
   (Inference-side experiments: --eval-only against current
   checkpoints — already cheap, no sample knob needed.)

   POST-TRAIN VALIDATION (mandatory before logging a verdict):
   Gate-T1: Read run.log epoch-by-epoch. Every aux's RAW loss term
            must show a real decrease across its active phase (no
            flat aux, no NaNs, no oscillation).
   Gate-T2: Early stop / warmup gate did not trigger before all aux
            losses finished ramping (scheduler prints `warmup ends at
            epoch N` — early stop must not have fired before N).
   Gate-T3: Run diagnose.py and read it. The model must show real
            discrimination:
              cd emr_model && python -m transform_emr.diagnose > ../diag.log 2>&1
            Reports 1/2/4/5 + Δt probe per program.md rules. If any
            regresses materially vs the running best, treat as DISCARD
            candidate even if headlines look OK.
   Only when Gates T1–T3 all pass: write the verdict.

   KEEP iff (at 10k scale, comparing against the running best
   re-evaluated at sample=10000 — see step 11):
     - All T1-T3 gates passed.
     - At least one headline metric clearly moves in the right
       direction beyond the 10k noise floor:
         AUROC ≥ +0.010, AUPRC ≥ +0.010, MAE ≤ −10 h, OR
         gen_to_gt_ratio_median up by ≥ 0.05, OR
         gen_frac_terminal_first24h down by ≥ 0.10.
     - No headline metric regresses by the same threshold.
     - Truncated AUROC (multi_horizon cap=48h mean) doesn't drop
       more than 0.07 vs running best.

6. Append one row to results/results-trajectory-fix.tsv with the
   summary-block headlines (multi_horizon caps 24/48/168/336 and
   gen_* / mae_* / outcome_*). Mark sample size in description
   (e.g. "10k" or "FULL").
7. Write a `### <tag>` block in status.md ending with
   `Verdict: KEEP|DISCARD — <reason>`.
8. Journal commit:
      git add status.md results/ && git commit -m "journal: <tag> <VERDICT> — <summary>"
      git push
9. If DISCARD:  git revert --no-edit <CODE_SHA>  &&  git push
   (Never `git reset --hard` — that erases the journal entry.)
10. If KEEP: new running best. Back up checkpoints:
      cp -r emr_model/checkpoints emr_model/checkpoints.bak_keep_<tag>
    KEEP'd code stays in HEAD; next experiment builds on top.

11. AFTER each new KEEP, re-eval the running best at sample=10000 once
    to refresh the baseline for the next screen. Inference-only
    re-evals (`python api.py --eval-only` on the new checkpoints)
    are fine here — no need to retrain.

12. FULL-DATA CONFIRM (sample=None) is NOT part of the per-experiment
    loop. Trigger it only when:
      a) the running best has been stable across at least 2–3
         consecutive DISCARD experiments (i.e. you've exhausted the
         most promising primary directions and aren't finding more
         10k gains), OR
      b) all primary directions (C, B-rollout, G, D) have been
         honestly attempted, OR
      c) the user explicitly asks.
    Then re-run the running best (and optionally the runner-up) at
    `sample=None` for publishable numbers. Mark these rows "FULL" in
    the ledger. If full-scale numbers don't replicate the 10k story,
    that's a sample-size artifact to investigate — not an automatic
    DISCARD.
```

## Research directions

Ordered by **whether the gradient actually reaches the failure mode**.
Generation length at inference is set by **when the LM head emits a
terminal token** (DEATH/RELEASE). A direction is high-priority iff its
loss / mechanism has a credible gradient path to that terminal-token
decision (directly or via the shared backbone). Loss-scale matters
too: any aux whose raw magnitude differs from BCE by 4+ orders gets
`λ ≈ 1e-6` after calibration and isn't trained.

### Primary — credible gradient path to terminal-token decision

- **C. Time-to-terminal regression head** (cheapest of the primaries —
  try first). Add an auxiliary head predicting
  `log1p(t_terminal − t_now)` at every non-terminal position; MSE
  against the GT time. The head shares the backbone with the LM head,
  so the gradient forces the backbone representation to encode
  distance-to-terminal — which the LM head can then use to decide *when*
  to emit DEATH/RELEASE. Direct attack on the failure mode; one extra
  head + one MSE term — minimal engineering surface.
  *Falsifiable*: head R² > 0.3; `mae_release_hrs` / `mae_death_hrs` drop;
  `gen_median_hours` rises; LM head's terminal logits shift across
  positions accordingly.

- **B-rollout. Scheduled multi-step rollout with sequence-level length
  loss** (reformulated B; the dead TF-only B is ruled out below). Try
  if C doesn't move the needle — it's the more expensive but more
  direct attack. Phase 2 starts with pure teacher forcing (existing
  BCE / CE / dt losses). After `bce_only_epochs`, anneal a rollout
  depth `k` from 1 upward across epochs — at the last `k` positions of
  each sequence the model emits its own token (Gumbel-softmax with
  annealed temperature τ: 2.0 → 0.5) and uses its own Δt prediction.
  Per-position BCE / CE / dt losses still apply at rolled-out positions,
  but they're now measured on the model's *own* output distribution.
  Additionally, a **sequence-level length loss** on the rollout:
  `|log1p(Σ_rollout pred_Δt_hrs) − log1p(target_rollout_horizon_hrs)|`.
  This is the gradient path X lacked: emitting terminal early in rollout
  → tiny Σ → high length loss → backprop pushes the LM-head terminal
  logit down. Naturally combines A (scheduled sampling) + B (length
  loss); as `k` grows the model's effective training distribution
  shifts smoothly from TF to autoregressive.
  *Falsifiable*: at end of training (k_max ~ 16–32), rolled-out
  `gen_median_hours` matches GT within ±25 % on the smoke checkpoint;
  `gen_frac_terminal_first24h` drops; 48 h `multi_horizon` AUROC
  doesn't drop more than 0.07 below baseline.
  *Engineering risks*: memory (k extra forward passes retained for
  backward — likely halve batch + double grad-accum); gradient noise
  (anneal both k AND τ slowly); Gumbel-softmax stability (start τ=2,
  anneal to 0.5 not 0). Watch raw length-loss magnitude vs BCE for the
  Gate-B check — keep both within 1–2 orders of magnitude.

- **G. Short → long horizon curriculum.** Phase-2 loss = weighted mix of
  next-48 h BCE (preserves 0.918) + multi-day cumulative signal,
  multi-day weight ramped up across epochs. Anchored on what the model
  already does well; the curriculum forces the LM head to extend its
  prediction horizon without abandoning near-term.
  *Falsifiable*: end of training, 48 h `multi_horizon` AUROC ≥ 0.91 AND
  336 h clearly above 0.45.

- **D. Discrete-time hazard for terminals.** Replace BCE on DEATH/RELEASE
  with a hazard head predicting `P(terminal in [t, t+Δ])` over log-spaced
  Δ bins; structured time-to-terminal supervision. Direct attack on the
  LM head's terminal output.
  *Falsifiable*: terminal MAE drops; calibrated hazard CDF matches GT
  terminal-time distribution within ~24 h.

### Secondary — addresses adjacent issues, not the terminal decision directly

- **A. Scheduled sampling.** Anneal teacher-forcing replacement
  `p: 0 → ~0.3` across Phase 2. Closes the train/inference distribution
  gap — important if the failure mode partly reflects exposure bias.
  Doesn't directly target the terminal-token decision; pairs naturally
  with C or G.
  *Falsifiable*: median generation length rises monotonically with `p`
  on an in-training probe.

### Symptom-attacking — last resort, not root cause

- **E. Narrow terminal `tau_lm`** (frozen). Tries to make the LM head's
  terminal soft-kernel narrower so terminal BCE targets aren't positive
  at distant positions. Attacks the kernel-widening symptom, not the
  upstream "model doesn't know when terminal should come" cause. Y
  (current run) is the freeze-via-gradient-hook version. A prior
  initialise-only attempt landed flat; if Y is flat too, the direction
  is exhausted.

### Ruled out

- **B (original — TF-only trajectory-length loss).** X DISCARDed;
  post-hoc analysis showed the loss formulation has **no gradient path
  to the LM head's terminal-token decision** — it only constrains the
  dt head's per-step Δt predictions during teacher forcing (which the
  per-step dt MSE already covers), and the dt head is decoupled from
  token choice at inference. The `log_tau_lm` drift in X is a downstream
  symptom, not the cause. The legitimate reformulation is **B-rollout**
  above — same length-loss intuition, but applied to autoregressive
  rollout output where the gradient actually reaches the LM head.

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
