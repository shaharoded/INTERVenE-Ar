# autoresearch — Benchmarking Phase

## Where we are

The architecture has been discovered, iterated, and locked through a long
iteration loop (see prior commits for the full journal). The
recipe transfers to any dataset with the temporal-event token structure this
codebase ingests — only the outcome head's K changes when the OUTCOMES list
changes.

Entering the benchmarking phase: validate the locked recipe on the dataset,
run the architecture size sweep at full data, polish with inference-side and
QA-data ablations, report.

## Locked architecture

- **4-layer transformer** with `time2vec_dim=32, dropout=0.10, head_dim=64` (heads scale with `embed_dim`). Size is selected in P6 (sweep below); start M-128 and grow.
- **AdaLN-Zero** patient-context conditioning
- **Temporal RoPE** + Time2Vec for absolute timestamps
- **Per-token learnable `log_tau_lm`** soft-kernel BCE, **terminal entries frozen** at
  `log(12/336)` via a gradient hook (direction E / Z marker, `transformer.py:463`)
- **C-ttt aux head** — `Linear→ReLU→Linear` predicting `log1p(t_terminal − t_now)` at
  every non-terminal position, MSE loss. Shares the backbone. Calibrated via the
  `LambdaScheduleController` in Phase 2 stage 0.
- **Δt two-head** — gate (`P(Δt>0)`) + magnitude (softplus). Phase 2 stage 0.
- **P4 pool aux** at `aux_fraction_cap=0.05` — patient-level attention pool added
  in Phase 3.
- **I2b inference gate** — `inference.py` ramps a bias onto terminal logits when
  predicted `hrs-to-terminal < 48 h` (bias 3.0). Caps over-generation cleanly.
- **CBM** (Curriculum by Masking) input-token noise at p=0.25, Phase 2 only.

## Locked training recipe

| Phase | Trainable | Objectives |
|---|---|---|
| 1 — embedder | All embedder params | Per-window outcome BCE + Δt aux |
| 2 — LM (transformer) | All backbone + LM head | `MaskedFocalBCE(γ=0.5)` next-token + curriculum: stage 0 = ce + dt + ttt; stage 1 = ranking (plateau-gated). CBM at p=0.25. |
| 3 — outcome head | Outcome head + ranking head; backbone with `phase3_backbone_lr_factor=0.01` (mostly frozen) | Per-position outcome BCE (soft-kernel, learnable `outcome_log_tau`) + ranking + I2 pool aux (cap=0.05) |

Auxiliary losses calibrated by `LambdaScheduleController` against tr_main:
`{ce: 0.5, dt: 0.5, ranking: 0.2, ttt: 0.3}` fraction caps. Plateau-gated stage transition.

## Eval framework (locked)

Per-patient peak-detector. For each (patient, outcome):
- **AUROC / AUPRC / max-F1 / F1@0.5** from per-patient max P_outcome over generated positions vs patient-level binary GT label
- **Peak-MAE** = distance from argmax-P_outcome time to nearest GT occurrence
- **Length-of-stay MAE** = `|max(TimePoint) − GT_RELEASE_time|` (cleaner length-of-stay regression than RELEASE peak-MAE)
- **Trajectory honesty**: `gen_to_gt_ratio_median`, `gen_frac_terminal_first24h`

**RELEASE excluded from AUC/AUPRC/F1** via `AUC_EXCLUDE = ("RELEASE_EVENT",)` —
`RELEASE ≈ ¬DEATH` in this cohort; including both double-counts the same terminal
ranking task. RELEASE stays in the LM vocab + outcome head training (so the
model emits it correctly and learns its timing), reported as length-of-stay MAE.

Headline keys in `api.py` summary block:
- `patient_auroc_weighted`, `patient_auprc_weighted`, `patient_max_f1_weighted`, `patient_f1_at_0_5_weighted`
- `patient_auroc_simple` (unweighted sanity)
- `length_of_stay_mae_hours`
- `patient_per_outcome\t…` rows with AUROC/AUPRC/max_f1/threshold/F1@0.5/n_pos/n_neg/prevalence
- `peak_mae_hrs\t…` rows
- Trajectory honesty: `gen_to_gt_ratio_median`, `gen_frac_terminal_first24h`, etc.

**Do not touch `api.py` or `evaluation.py`.** Agent edits `emr_model/transform_emr/**` and config only.

## Roadmap — what's left

```
1. RECIPE-TRANSFER SMOKE  (10k, full pipeline, M-128 to start)
   - Verify auxes descend cleanly on the dataset under the locked recipe.
   - Smoke gates A–D + post-train T1–T3 + per-aux trace in journal.
   - If clean → proceed. If anything regresses, investigate before scaling.

2. P6 ARCHITECTURE SWEEP  (full data, locked recipe; produces the headline)
   - Grid in order: M-128, M-256, M-384, M-512, M-768
     (each: embed_dim → head_dim=64; n_head = embed_dim/64; n_layer=4 fixed).
   - Each variant a full-data run (sample=None), ~hours per run.
   - OOM → halve batch + double grad-accum; if still OOM, that's the size ceiling.
   - Decision: smallest variant within ~0.005 weighted AUROC of best (prefer smaller).
   - The winning variant is the publishable headline #1.

3. F1 + F2 INFERENCE-SIDE  (eval-only on the P6 winner)
   - F1: beam search with length-normalised scoring.
   - F2: temperature schedule to escape immediate-terminal local minimum.

4. P7 USE_QA_DATA TOGGLE  (full data, on P6 winner)
   - Pre-flight: delete tokenizer.pt, scaler.pkl, processed_datasets.pt, phase1/.
   - Phase 1 retrains. Smoke first; verify len(tokenizer.token2id) > non-QA value.

5. Confidence:
   - At this point we'll have the most final architecture.
   - We need to rerun it with 3 different seeds, to report confident results KDD style.

6. FINAL REPORT
   - P6 winner full-data headline numbers (per-outcome AUROC/AUPRC/max-F1/F1@0.5/peak-MAE + LoS MAE + trajectory honesty)
   - P6 sweep table (all sizes tested)
   - F1/F2 and P7 deltas
   - The AUROC↔calibration Pareto observation as a methodological finding
```

## Loop discipline

```
1. Read program.md. Check git log + last rows of results/results-trajectory-fix.tsv.
2. Propose ONE change with a falsifiable hypothesis.
3. SMOKE (sample=50, phase{1,2,3}_n_epochs=1):
     python api.py > smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 OOM of BCE.
   Gate-C: calibrated λ in [1e-3, 10].
   Gate-D: summary block prints, all headline keys present.
4. git add <files> && git commit -m "<tag>: change / why / expected" && git push.
5. EXPERIMENT (sample=10000 OR sample=None for full-data):
     python api.py > run.log 2>&1
   POST-TRAIN:
   T1: every aux's raw loss decreases across its active phase.
   T2: early stop didn't fire before auxes finished ramping.
   T3: diagnose.py shows real discrimination on key probes.
6. Append row to results/results-trajectory-fix.tsv (new headline keys).
7. Write `### <tag>` block in status.md → `Verdict: KEEP|DISCARD — …`.
   **Mandatory per-aux training trace table** (unlock epoch, λ_max, anchor
   raw_aux, final raw_aux, Δ%). Flag |Δ| < 5 % as "not learning."
8. Journal commit + push.
9. DISCARD → git revert --no-edit <CODE_SHA> && git push.
10. KEEP → cp -r emr_model/checkpoints emr_model/checkpoints.bak_keep_<tag>.
    Run ablation that strips the new change → confirms gain attribution.
11. After each KEEP, re-eval running best at 10k (`--eval-only`) to refresh.
12. FULL-DATA CONFIRM (sample=None) once a milestone (smoke→headline→P6 winner) is reached.
```

### KEEP rule

- All smoke gates A–D + post-train T1–T3 passed.
- ≥ 1 headline lifts past noise: AUROC ≥ +0.010, AUPRC ≥ +0.010, MAE ≤ −5 h.
- No headline regresses by the same threshold.
- `gen_to_gt_ratio_median` doesn't drop below 0.4.

Benchmarking phase expects mostly verification rather than KEEPs — the recipe
is locked. Unexpected regressions are signal that the new dataset has
structural differences worth investigating before scaling.

## Cheap optimisation: reuse Phase-1 / Phase-2 cached checkpoints

If the new dataset's temporal-event structure is identical to a prior cached
run (same vocab, same scaler):
- Phase 1 may cache-hit on `(embed_dim, time2vec_dim, ctx_dim)`.
- Phase 2 LM head is independent of OUTCOMES (predicts the same vocabulary).
- Phase 3 outcome head's K is the only thing that genuinely changes.

Verify the cache key matches, then optionally only retrain Phase 3.

## Lessons learned (carry forward)

From the iteration loop, abridged (full details in prior commits):

- **Trajectory honesty fixable via narrow + frozen terminal `log_tau_lm` + C-ttt aux.** Z's freeze hook + C-ttt aux head together produced multi-day trajectories with `gen_to_gt_ratio_median ≈ 0.6` and `gen_frac_terminal_first24h ≈ 0.05`.
- **Patient-level pool aux (cap=0.05) is the largest single win.** Single config knob, +0.04 AUROC on the iteration-loop dataset.
- **I2b inference ttt-gate** — reusing the ttt head at inference (originally trained as a regression aux) caps over-generation cleanly without retraining. Adds another +0.005 AUROC.
- **AUROC↔calibration Pareto frontier**: every direction that improved trajectory honesty / MAE / calibration tended to hurt rare-outcome AUROC. Capacity-bounded trade-off. The lift came from amplifying ranking, not from fixing calibration. **Worth one paragraph in the methods section.**
- **Eval-time validation of OUTCOMES**: always check per-outcome GT timestamp distribution against the prediction-anchor before training. Outcomes that occur only at admission time aren't predictable from a post-admission seed regardless of recipe — this is dataset-side validation that should happen on day one.

## Reproducibility

- Branch `autoresearch-trajectory`; no force-push to `main`.
- Ledger: `results/results-trajectory-fix.tsv`. Benchmarking rows append; iteration-loop rows preserved.
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Iteration-loop history lives in prior git commits (not on disk).
- Benchmarking journal: `status.md` (fresh, agent appends `### <tag>` blocks here).
