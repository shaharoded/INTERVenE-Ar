# autoresearch — Patient-Level Eval Reframing

## The reframing

Prior loop used per-(patient, window) AUC. It was dragged by rare
outcomes and double-punished partial wins. New eval is per-patient
peak-detector: for each (patient, outcome), score = `max P_outcome` over
the generated portion, label = outcome occurred in GT. MAE = distance
between `argmax_t P` and the **nearest** GT occurrence.

### Headline keys (in `api.py` summary block)

| Key | Meaning |
|---|---|
| `patient_auroc_weighted:` | Support-weighted mean per-outcome AUROC. **Primary**. |
| `patient_auprc_weighted:` | Support-weighted mean AUPRC. |
| `patient_auroc_simple:`   | Unweighted mean (sanity). |
| `n_outcomes_used:`        | Outcomes passing the 1 % prevalence threshold. |
| `patient_per_outcome\t…`  | Per-outcome AUROC/AUPRC/n_pos/n_neg/prevalence. |
| `peak_mae_hrs\t…`         | Per-outcome MAE to nearest GT occurrence. |

Legacy keys (`outcome_auroc`, `multi_horizon\t…`, etc.) still emit for
back-compat. Outcomes below 1 % test-set prevalence get `auroc=nan` and
are excluded from the weighted mean.

VRAM ≤ 24 GB. Data and `bak_originals` are read-only.

## What's already on this branch

You inherit a clean project. **Do not touch `api.py` or `evaluation.py`.**
Edit only `emr_model/transform_emr/**` and its `config/**`.

- `evaluation.py` already has `per_patient_max_auc`,
  `weighted_mean_auc`, `time_accuracy_nearest`, and the legacy
  per-window functions.
- `api.py` summary block emits all keys above.
- Z architecture (narrow + frozen `log_tau_lm[terminal]`) is the
  starting point — code on HEAD, no checkpoint on disk (pod is fresh).
- Ledger: `results/results-trajectory-fix.tsv`.

## The loop

10k-sample is the primary workspace (~25–30 min per training run).
Full-data confirm only at end of a block or when running best is stable.

```
1. Read program.md. Check git log + last rows of
   results/results-trajectory-fix.tsv.
2. Propose ONE change with a falsifiable hypothesis.
3. SMOKE (sample=50, phase{1,2,3}_n_epochs=1):
      python api.py > smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 OOM of BCE.
   Gate-C: calibrated λ in [1e-3, 10].
   Gate-D: summary block prints, all headline keys present.
   (P3-specific gates listed in P3 section.)
4. git add <files> && git commit -m "<tag>: change / why / expected" && git push
5. EXPERIMENT (sample=10000):
      python api.py > run.log 2>&1
   POST-TRAIN:
   T1: every aux's raw loss decreases across its active phase.
   T2: early stop didn't fire before auxes finished ramping.
   T3: diagnose.py shows real discrimination on key probes.
6. Append row to results/results-trajectory-fix.tsv (new headline keys).
7. Write `### <tag>` block in status.md → `Verdict: KEEP|DISCARD — …`.
   **Mandatory in every block — per-aux training trace** (one table per
   experiment, for every aux loss term active in any phase, both new ones
   AND inherited ones):
   ```
   | Aux       | Unlock epoch | λ_max  | Anchor raw_aux | Final raw_aux | Δ      |
   |-----------|--------------|--------|----------------|---------------|--------|
   | ce        | 4 (Ph-2)     | 0.0779 | 1.504          | 0.234         | -84%   |
   | dt        | 4 (Ph-2)     | 0.1440 | 0.814          | 0.412         | -49%   |
   | ttt       | 4 (Ph-2)     | 0.0034 | 20.86          | <FILL>        | <FILL> |
   | ranking   | 32 (Ph-2)    | 0.0337 | 0.150          | <FILL>        | <FILL> |
   ```
   The "final raw_aux" is the value at the last epoch the phase ran
   (= the last `RawTrain <aux>=…` line in run.log for that phase before
   early-stop fires). Δ shows whether the aux actually descended. If
   |Δ| < 5 %, the aux is not learning — flag it explicitly in the
   verdict reasoning. T1 gate is about checking descent; this is how we
   make it auditable.
8. Journal commit + push.
9. DISCARD → git revert --no-edit <CODE_SHA> && git push.
10. KEEP → cp -r emr_model/checkpoints emr_model/checkpoints.bak_keep_<tag>.
    Then run an ABLATION (next section) before proceeding.
11. After each KEEP, re-eval running best at 10k (--eval-only) to refresh.
12. FULL-DATA CONFIRM (sample=None) when running best stable across
    2–3 DISCARDs, OR a block ends, OR user asks.
```

### KEEP rule (vs running best at 10k)

- All smoke gates A–D + post-train T1–T3 passed.
- ≥ 1 headline lifts past noise: AUROC ≥ +0.010, AUPRC ≥ +0.010, MAE ≤ −5 h.
- No headline regresses by the same threshold.
- `gen_to_gt_ratio_median` doesn't drop below 0.4.

Otherwise DISCARD → revert.

### Ablation discipline (mandatory after each KEEP)

A KEEP that stacks on a prior KEEP creates **attribution debt** — you don't
know whether the gain came from the new change or from the prior one
still doing the work. Before declaring a new KEEP final and moving to
the next direction, run **one ablation that isolates the new change** at
10k: strip the prior intervention and re-test with only the new change
on the bare baseline.

Example: after `B0-C-ttt` KEEPs vs `B0-Z`, run `C-ttt-on-baseline` —
the C-ttt aux head applied to the bare M-256 (without Z's frozen-tau).
If AUROC is comparable, the simpler recipe wins; demote Z and use bare
M-256 + C-ttt as the running best. If the stacked version is meaningfully
better, the prior intervention is doing real work — keep the stack.

Ablation outcomes get a journal block (`### <tag>-ablation`) and a
ledger row tagged `ABLATION`. They never count as new KEEPs; they
either confirm the running best or simplify it.

This step is **not optional** — skipping it causes the loop to silently
stack interventions that may be redundant.

## Research directions (in order)

### P0 — Baselines under the new headline

Two baselines must exist before any new direction is judged. Both are
straight 10k runs; the better one becomes the running best for P1.

**B0-Z** — Z is already on HEAD. No code change required, just run.
Z = direction E (narrow + frozen terminal `log_tau_lm`). Marker block
is in `emr_model/transform_emr/transformer.py` ~line 463 — search for
the comment `# Direction E: freeze the terminal entries of log_tau_lm`.
Terminal-init value is set in
`emr_model/transform_emr/config/model_config.py` (`_log_tau_terminal`,
currently `math.log(12.0 / 336.0)`).

**B0-C-ttt** — re-apply commit **`dd3fc1b`** ("C-ttt-head: time-to-
terminal regression aux (direction C) on Z") on top of HEAD. Adds an
auxiliary head predicting `log1p(t_terminal − t_now)` per non-terminal
position with MSE loss; shares the backbone. Touches four files:
```
emr_model/transform_emr/config/model_config.py
emr_model/transform_emr/diagnose.py
emr_model/transform_emr/inference.py
emr_model/transform_emr/transformer.py
```
Recipe:
```
git show dd3fc1b --stat                 # inspect scope
git cherry-pick --no-commit dd3fc1b     # apply diff, keep staged
# resolve conflicts if any (HEAD is post-Z, dd3fc1b was on top of Z)
# then commit as your own B0-C-ttt tag
```
Was DISCARDed under the old per-window eval (rare-7 flipped), but
produced DEATH window-AUC 0.79 — expected to dominate on patient-level
DEATH AUC.

### P1 — MIL patient-level max-BCE aux loss (Phase 3)

```
score_patient = softmax_t(logit_outcome(t) / T) · logit_outcome(t)
loss_mil = BCE(σ(score_patient), patient_binary_label)
```

Soft max (temperature ~1.0, learnable per-outcome). Schedule via
`LambdaScheduleController` with `aux_fraction_cap` ~ 0.20. Existing
per-position BCE stays as the 48-h calibration anchor.

**Falsifiable**: patient AUROC ≥ +0.03 vs best of {B0-Z, B0-C-ttt};
per-window AUROC drop < 0.10.

### P2 — Soft-argmax time loss, positives only (Phase 3)

```
weights = softmax(logit_outcome(t) / T_time)
predicted_t = sum_t(weights · t)
loss_time = smooth_l1(predicted_t, nearest_t_in_gt(outcome, patient))
```

Per-outcome learnable `T_time` (~13 scalars). Only for patients with the
outcome.

**Falsifiable**: `peak_mae_hrs` for {DEATH, RELEASE} drops ≥ 5 h;
patient AUROC doesn't regress.

### P3 — Risk-aware LM head (architectural coupling)

Currently the LM and outcome heads only share a backbone. P3 makes the
outcome-head's prediction influence which tokens the LM emits.

**The change.** Linear projection `bias_proj: n_outcomes → vocab_size`.
Per position:
```python
# B=batch, T=seq, D=hidden, V=vocab, K=n_outcomes
h           # (B,T,D)
lm_logits = LM_head(h)                  # (B,T,V)
o_logits  = outcome_head(h)             # (B,T,K)  ← must be 3D
P         = torch.sigmoid(o_logits)     # (B,T,K)

assert P.shape == (B,T,K)
assert lm_logits.shape == (B,T,V)

bias = bias_proj(P)                     # (B,T,V)
assert bias.shape == lm_logits.shape

combined_logits = lm_logits + bias      # (B,T,V)
```
No shift — `P[t]` and `lm_logits[t]` both predict from `h_t`. (Prior
attempts died on: silent broadcast from missing T axis on P; off-by-one
shift; bias_proj weight not zero-initialised so step-0 disrupted CE.)

**Init must be no-op**: `nn.init.zeros_(bias_proj.weight)`. Step 0
combined_logits == lm_logits exactly.

**Phase 3 unfreezing**: LM head must have `requires_grad=True`, LR
multiplier ~0.1×base. `bias_proj` and outcome head get full base LR.

**Smoke gates** (additions on top of A–D):
- **P3a**: zero-init no-op — CE on first batch matches non-coupled
  baseline to 1e-6.
- **P3b**: after first backward, all three grads non-zero:
  `bias_proj.weight.grad.norm()`, `outcome_head[-1].weight.grad.norm()`,
  `LM_head.weight.grad.norm()`, each > 1e-8.
- **P3c**: shape asserts pass.

**Per-epoch print** (Phase 3 and `diagnose.py`):
```
P3 coupling stats epoch <e>:
  ||bias|| / ||lm_logits||  mean: <r>  max: <r>
  bias_proj.weight row norms (per outcome): [DEATH=.., RELEASE=.., ...]
  per-outcome contribution to terminal logits: [DEATH→TERM=<>, ...]
```
Healthy ratio band: **0.05–0.3**. Below → no coupling formed.
Above → bias dominates, LM atrophies.

**Behavioural probe** (`diagnose.py`, held-out batch):
```
Pearson(P_DEATH[t], terminal_token_logit[t]):  <ρ>  (> 0.3 healthy)
gen_to_terminal_hrs on positives vs negatives, Δ in hours (>0 healthy)
```
Δ ≤ 0 → coupling didn't form behaviourally → DISCARD even if AUC moved.

**Falsifiable**: patient DEATH/RELEASE AUROC ≥ +0.03 vs P1+P2;
behavioural Δ > 12 h; coupling ratio in [0.05, 0.3].

The `bias_proj` row weights — which outcomes bias which tokens — are
publication-worthy figure material.

### P4 — Patient-level pooling head

Learned attention pool over generated hidden states, queried by
per-outcome embeddings → per-patient score replaces "max P_outcome" in
eval. Per-position outcome head stays for 48-h calibration. ~150 LOC.

Defer unless P1+P2+P3 plateau.

**Falsifiable**: patient AUROC ≥ +0.05 vs P1+P2+P3.

### P5 — M0 ablation: per-position outcome BCE redundancy

After P1–P4 honestly attempted, down-weight / disable per-position BCE
(`aux_fraction_cap` → 0.02 or off). Structural diagnostic at 10k, not
a KEEP/DISCARD candidate:
- Patient AUROC holds + cap=48h doesn't collapse → per-position BCE
  was redundant for ranking; keep small for calibration only.
- cap=48h collapses → per-position BCE is the calibration anchor; keep.

## Post-P5 iteration (before P6 lock-in)

P0–P5 done. Running best `B0-C-ttt`. Three retries of close misses + three
new directions + one agent-discretion slot before the recipe is locked
for P6 full-data scale-up. Run in this exact order at 10k; same loop
discipline (smoke gates A–D, post-train T1–T3, two-commit pattern,
per-aux trace in the journal, revert on DISCARD, ablation discipline on
KEEPs).

### I1 — P3-v2: risk-aware LM head with LM head FROZEN in Phase 3

Same `bias_proj: K → V` and shape-contract design as original P3, but
in Phase 3 set `phase3_backbone_lr_factor=0.0` AND freeze the LM head
explicitly (`requires_grad=False` on `lm_head.*`). Only `bias_proj` +
outcome head train. Original P3 died because the LM head atrophied as
`bias_proj` learned; freezing it removes that failure path. All P3-prefix
smoke gates (P3a zero-init no-op, P3b gradient probe on bias_proj +
outcome head, P3c shape asserts) and per-epoch coupling-ratio /
behavioural-probe diagnostics still apply.

**Falsifiable**: patient-level AUROC ≥ +0.010 vs running best; bias_proj
row weights show interpretable outcome→token routing
(e.g. DEATH→TERMINAL row norm large and positive).

### I2 — P4-tight: pooling head with cap=0.05

Same attention pool design as original P4, lower `aux_fraction_cap`
(0.20 → 0.05). Single config change. Original P4 lifted AUROC +0.018 but
tripped RELEASE MAE +7.4 h and per-outcome regressions; lower cap should
preserve the AUROC lift while killing the calibration disruption.

**Falsifiable**: patient-level AUROC ≥ +0.010 vs running best; RELEASE
MAE doesn't regress past 5 h; no per-outcome AUROC drops past 0.020.

### I3 — P-CTTT-bounds: structural bounds on the ttt head

Currently `ttt_head = Linear → ReLU → Linear` with unconstrained scalar
output and pure MSE loss against `log1p(t_terminal − t_now)`. Nothing
enforces output ≥ 0, monotonicity across positions, or consistency
with the GT hospitalization duration. Two bounds, recommended together
as one experiment:

**(a) Positivity** — final layer activation = `softplus`, mirroring the
dt magnitude head. ~5 LOC.

**(b) Consistency loss** — at each position with absolute time `T(t)`:
```
loss_ttt_consistency = | expm1(ttt_pred[t]) + T(t) − GT_duration |
```
averaged over valid positions. Pins the two time-heads (dt and ttt) to
agree on patient duration. Added as an auxiliary loss with its own
scheduler entry; calibrated to ~0.1 fraction of main BCE. ~15 LOC.

If results are ambiguous, agent may split into two experiments
(I3a positivity-only, I3b consistency-only). Default: both in one run.

**Falsifiable**: patient-level AUROC doesn't regress; raw_ttt
**descends** (now properly auditable per the per-aux trace requirement);
ttt-vs-dt consistency error < 10 h on held-out batch.

### I4 — Sub-trajectory augmentation in Phase 2

Build multiple views per training patient, each a coherent sub-trajectory:
- View A: full sequence (current)
- View B: drop a random 12 h gap in the middle
- View C: keep only labs + outcomes (drop interventions/meals)
- View D: keep only clinical events + outcomes (drop labs)

**Hard constraint: outcome tokens AND terminal tokens are never removed**
from any view (they are events with clinical reality). Pass a forbid_ids
list to the augmenter exactly like CBM uses.

K=2–3 views per patient per epoch. Implemented in dataloader, not in
the loss. **Targets Phase 2 backbone training** — gives the model
multiple consistent "views" of the same patient. Coexists with CBM
(both at Phase 2; CBM is token-level noise, augmentation is structural
view variety).

**Falsifiable**: patient-level AUROC ≥ +0.010; auxes still descend
cleanly under the larger effective training set; trajectory honesty
preserved.

### I5 — P-AR-FT: AR-generated data + frozen Phase-3 backbone

Closes the train/eval distribution gap. Pre-generate K=1–2 trajectories
per train patient using the current running-best model (cached to disk,
`torch.no_grad`). In Phase 3:
- `phase3_backbone_lr_factor = 0.0` (backbone frozen)
- Mix dataloader: GT sequences + cached generated sequences (50/50 or
  similar; ramp generated fraction across epochs)
- BCE labels at every position computed from GT outcome timestamps
  using the existing soft-kernel mechanism — labels are GT-derived,
  inputs are model-derived. **The outcome head reads backbone features,
  not emitted tokens**, so the LM not emitting outcomes is fine — the
  head learns to predict from context.

~190 LOC: generation+caching script, dataloader mix mode, Phase-3 freeze
flag. Strongest single bet for moving RELEASE because nothing else
attacks the train/eval distribution mismatch.

**Falsifiable**: RELEASE AUROC ≥ +0.030; patient-level AUROC ≥ +0.010;
no per-outcome regressions past 0.020; gen_to_gt_ratio_median preserved.

### I6 — CBM in Phase 3 with aggressive masking

CBM currently runs in Phase 2 (input token masking, p=0.25) with a
forbid list that excludes intervals/meals to preserve LM-head temporal
coherence. Phase 3 doesn't need that coherence — outcome head reads
backbone features, can tolerate masked input. Move/duplicate CBM to
Phase 3 with:
- p starts at 0.25, can probe up to 0.40
- Same outcome-preserving forbid list (outcomes + terminals never masked)
- Other tokens (interventions, labs, meals, context) all eligible

Also consider raising Phase-2 CBM ratio if I4 doesn't already saturate
the input-noise regime.

**Falsifiable**: patient-level AUROC ≥ +0.005; outcome head shows
robustness gain measurable as smaller eval-time AUROC variance across
seeds.

### I7 — Agent-discretion slot

ONE experiment of agent latitude IF (and only if) a specific observable
problem has surfaced in earlier experiments' logs/journals. Strict rules:

- **Trigger required**: cite the specific run.log line / journal entry /
  diagnose probe motivating the proposed change. "I felt like trying X"
  is NOT a trigger. "Aux Y stayed within 5% of anchor across the full
  phase" IS.
- **Single modification, falsifiable hypothesis, normal loop discipline**:
  smoke gates A–D, post-train T1–T3, two-commit pattern, per-aux trace,
  ablation discipline if KEEP.
- **Scope limits**: stays within loss / aux / data-side modifications.
  NO architecture/size changes (that's P6). NO QA toggle (that's P7).
  NO scope creep into "tried a paper I read".
- **Hard rules to respect**: every active aux must contribute and visibly
  descend during its phase; no degenerate outputs (trajectory collapse,
  terminal starvation, majority-class collapse); all token types emit
  with reasonable frequency.

If no clear trigger has surfaced by this point, skip the slot — go
straight to P6.

### Recipe lock

After I1–I7 complete, the running best's loss / aux / schedule recipe is
**locked**. No further training-side changes until P6/P7 finish. The
recipe at recipe-lock time is what gets scaled in P6.

### P6 — Architecture scale-up (FULL DATA ONLY, second-to-last)

**This is NOT a 10k probe.** Architecture scale-up is reserved for the
end of the loop, on the locked recipe, at full data. Running it at 10k
mid-loop is **explicitly out of scope** — it burns hours on an
architecture decision before the recipe is even finalised.

**Strict trigger** (ALL must hold):
- P0 through P5 (incl. ablations) have all been honestly attempted.
- A clear running best exists with `patient_auroc_weighted` ≥ +0.03 vs
  the B0-Z baseline, trajectory honesty preserved
  (`gen_to_gt_ratio_median` ≥ 0.5).
- The running best's loss recipe is **locked** — no further aux / coupling /
  schedule changes will be made.
- Last 2–3 10k experiments DISCARDed (running best stable).

Lift M-256 lock. Scan grid with the locked recipe, **each variant a full-
data run (sample=None, ~hours per variant)**:

| Tag | embed_dim | n_layer | n_head | Approx params |
|---|---|---|---|---|
| M-128 | 128 | 4 | 4 | ~2 M |
| M-256 | 256 | 4 | 4 | ~6 M (baseline) |
| M-384 | 384 | 6 | 6 | ~15 M |
| M-512 | 512 | 6 | 8 | ~25 M |
| M-768 | 768 | 8 | 12 | ~55 M |

OOM → halve batch + double grad-accum; if still OOM, that's the size
ceiling.

**Decision**: smallest variant within ~0.005 of best at full data
(prefer smaller). The winning architecture becomes the substrate for P7.

### P7 — Final: toggle `USE_QA_DATA` on the very best model

**Strict trigger** (ALL must hold):
- P5 has settled the loss recipe.
- P6 has chosen the winning architecture at full data.
- The running best is the genuine end-of-loop candidate.

Toggle `USE_QA_DATA = True` in
`emr_model/transform_emr/config/dataset_config.py`. This adds context
features AND new tokens — the cached vocab/scaler/datasets are stale.

**Pre-flight (mandatory before `python api.py`)**:
```bash
rm -f emr_model/checkpoints/tokenizer.pt
rm -f emr_model/checkpoints/scaler.pkl
rm -f emr_model/checkpoints/processed_datasets.pt
rm -rf emr_model/checkpoints/phase1
```

Phase 1 retrains from scratch (only experiment in the whole loop that
does). Verify after rebuild: `len(tokenizer.token2id)` must be strictly
greater than the non-QA value — if equal, QA tokens weren't picked up.

Smoke (sample=50) before full data; this final run is at sample=None
since the result is publishable, not a 10k probe.

**Falsifiable**: full-data `patient_auroc_weighted` lifts ≥ +0.005 OR
QA-introduced tokens visibly emitted in generated trajectories. If
neither, the non-QA running best stands as the final result.

## Inference-side directions (no retraining) — between P6 and P7

`python api.py --eval-only`:
- **F1**. Beam search with length-normalised scoring.
- **F2**. Temperature schedule to escape immediate-terminal local minimum.

Run after P6's winning architecture is locked, before P7's QA toggle.
Cheap (no training), so they're sequenced after architecture choice but
before the final QA experiment.

## Stop criterion

No quality target — push as high as the model honestly allows. Stop when:
- All directions in order — P0–P5, I1–I7, P6, F1/F2, P7 — honestly attempted,
- Last 2–3 10k experiments DISCARDed at recipe-lock time,
- Full-data confirm of running best done.

Write final report in `status.md`: running-best numbers (weighted
AUROC, per-outcome AUROC for DEATH/RELEASE/each complication, peak-MAE,
trajectory honesty stats), and what was tried.

## Reproducibility

- Branch `autoresearch-trajectory`; no force-push to `main`.
- Ledger: `results/results-trajectory-fix.tsv`.
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Journal: `status.md` (Sections 1 / 1b stay intact).

