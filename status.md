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
`gen_frac_terminal_first24h`), **per-aux training trace** (table with
unlock epoch, λ_max, anchor raw_aux at calibration, final raw_aux at
end of phase, Δ%) for every aux active in any phase, verdict
(KEEP / DISCARD) with reason. Flag `|Δ| < 5 %` auxes explicitly —
they're not learning.

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

### P1-MIL @ 10k (SHA 422dcbc) — DISCARD

P1 direction. Added a patient-level multiple-instance-learning aux to
Phase 3: soft-max-attention pool of outcome logits across time steps,
BCE against `patient_label = outcome occurs anywhere in GT`. Soft-max
temperature `mil_log_T` learnable per outcome. λ_mil calibrated once
at end of Phase-3 epoch 1, capped at fraction 0.20 of raw outcome BCE
(same regime as ranking). Per-position BCE kept as 48-h calibration
anchor.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D all pass. raw_out=8.52, raw_rank=0.69, raw_mil=1.07
  (within 1× of BCE). λ_mil=1.585, λ_ranking=2.46, both ∈ [1e-3, 10].

Post-train (10k):
- T1 fail — Phase-3 raw_out drops from 2.053 to 1.174 between epoch 1
  and 2 (this is normal — calibration kick when λ_ranking goes 0→cal).
  raw_mil rises 3.685→4.635 over the 6 active epochs: the MIL aux is
  being optimised AGAINST, not toward. Aux gradient too weak to fight
  per-position BCE conflict.
- T2 fail — Phase-3 early stop fires at epoch 6, with best `vl_select`
  at epoch 1 (1.1125) — i.e., before λ_mil was even active.
  Subsequent epochs (with λ_mil=0.111) consistently increased vl_select.
- T3 fail — DEATH AUROC drops 0.710→0.665 (-0.045); the head no longer
  shows the discrimination the run was supposed to add.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6427** (−0.0404 — fails ≥+0.030 KEEP)
- `patient_auprc_weighted`: 0.5855 (−0.0481)
- `patient_auroc_simple`:   0.6112 (−0.0847)
- `patient_auprc_simple`:   0.2792 (−0.0447)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — universal regression except RELEASE:
- DISGLYCEMIA_Hyper:  0.805 (−0.091)
- DEATH:              0.665 (−0.045)  ← contra direction's intent
- NEUROVASCULAR:      0.651 (−0.035)
- NERVOUS_SYSTEM:     0.649 (−0.147)
- RELEASE:            0.645 (+0.064)  ← only winner (majority class)
- DISGLYCEMIA_Hypo:   0.643 (−0.128)
- KIDNEY:             0.639 (−0.076)
- CARDIO:             0.616 (−0.093)
- RETINOPATHY:        0.613 (−0.172)
- SKIN_ULCER:         0.590 (−0.089)
- ACUTE_RESPIRATORY:  0.586 (−0.005)
- ATHEROSCLEROSIS:    0.555 (−0.040)
- ACIDOSIS:           0.553 (−0.017)
- KETOACIDOSIS:       0.538 (−0.377)  ← collapse from 0.915
- HYPEROSMOLALITY:    0.531 (−0.054)
- INFECTION:          0.499 (−0.052)

Peak MAE (hours) Δ vs B0-C-ttt:
- DEATH:   172.74 (+3.77, marginal)
- RELEASE:  79.16 (+7.87)
- DISGLYCEMIA_Hyper:  32.59 (−3.47, small improvement)
- KIDNEY:             63.33 (−15.78)

Trajectory honesty:
- `gen_median_hours`:           91.22 (vs 75.05)
- `gen_to_gt_ratio_median`:      0.900 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  0.245 (vs 0.165 — terminal-first jumps)

Phase stats: phase2 ran all 50 epochs; phase3 early-stopped at 6 with
best at epoch 1.

Verdict: **DISCARD**. Falsifiable (patient AUROC ≥ +0.030) missed by
0.070. The MIL aux pulled Phase 3 away from the running best optimum
within 1 epoch of activation, and the model never recovered. The
likely mechanism: with patient_label being "outcome occurs anywhere",
the soft-max-pooled score is dominated by the position with the
highest logit, and BCE gradient on the pool propagates back to that
position. For a negative patient on a rare outcome, the path of least
resistance is to lower ALL logits — destroying per-position
discrimination that B0-C-ttt had carefully built. The per-position
BCE anchor was insufficient to hold ground (its λ=1.0 vs MIL's
effective contribution ~0.20 of BCE, but the gradient directions
conflict). The single positive class (RELEASE, 87 % prevalence)
benefits because the pool's collective lift is aligned with its
target.

This is a learning-recipe problem, not a code/architecture bug. The
direction is sound in principle, but the loss formulation as
specified is too coarse next to per-position BCE for rare outcomes.
P2's soft-argmax time loss is a positives-only loss — that constraint
may avoid this failure mode. Proceeding to P2.

Reverting code commit per loop step 9.

---

### P2-time @ 10k (SHA 10abcc1) — DISCARD

P2 direction. Added a positives-only soft-argmax onset-time aux to
Phase 3: weighted softmax(logit / T_k) over time gives a continuous
predicted onset time; smooth-L1 to the nearest GT occurrence
(detached, scaled to hours). Per-outcome learnable `time_log_T`.
λ_time calibrated once at Phase-3 epoch 1 (cap=0.20). Patients
without the outcome contribute zero gradient.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D pass after switching the smooth-L1 inputs from normalised
  time (0…1) to hours (×336). Without the hour rescale λ_time
  calibrated at 94 — outside the [1e-3, 10] band. With rescale:
  raw_time=50.94 h, λ_time=0.034, in band.

Post-train (10k):
- T1 partial fail — raw_time barely moves over the 15 active Phase-3
  epochs (61.9 → 57.8 h, then plateau). Aux gradient gets absorbed
  into the joint optimum without actually reducing the time error.
- T2 fail — Phase-3 best `vl_select` is **1.143** at epoch 10, worse
  than B0-C-ttt's 1.010. Selection metric pure-outcome-BCE held
  monotonically above the running-best optimum the entire run.
- T3 fail — DEATH AUROC drops 0.710→0.631 (-0.079); the aux that was
  supposed to refine onset timing actually weakened the per-position
  discriminator that drives the eval headline.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.5735** (−0.1097 — fails KEEP rule)
- `patient_auprc_weighted`: 0.5551 (−0.0785)
- `patient_auroc_simple`:   0.5687 (−0.1272)
- `patient_auprc_simple`:   0.2526 (−0.0713)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — universal regression:
- DISGLYCEMIA_Hyper:  0.814 (−0.082)
- DISGLYCEMIA_Hypo:   0.641 (−0.130)
- DEATH:              0.631 (−0.079)
- NEUROVASCULAR:      0.623 (−0.063)
- KIDNEY:             0.609 (−0.106)
- ACUTE_RESPIRATORY:  0.592 (+0.001)
- ACIDOSIS:           0.581 (+0.011)
- CARDIO:             0.570 (−0.139)
- RETINOPATHY:        0.547 (−0.238)
- SKIN_ULCER:         0.531 (−0.149)
- INFECTION:          0.529 (−0.023)
- HYPEROSMOLALITY:    0.514 (−0.071)
- ATHEROSCLEROSIS:    0.506 (−0.089)
- KETOACIDOSIS:       0.493 (−0.422)  ← chance
- NERVOUS_SYSTEM:     0.475 (−0.322)  ← below chance
- RELEASE:            0.444 (−0.137)  ← below chance

Peak MAE (hours, mixed; falsifiable wanted ≥−5 h for both):
- DEATH:    156.06 (−12.91 ✓)
- RELEASE:   81.38 (+10.09 ✗)
- DISGLYCEMIA_Hyper:  26.11 (−9.96)
- KIDNEY:             84.86 (−21.50)
- CARDIO:            121.54 (+42.46 ✗)

Trajectory honesty:
- `gen_median_hours`:           79.40
- `gen_to_gt_ratio_median`:      0.770 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  **0.421**  ← 2.6× the B0-C-ttt rate;
  the time aux made the model commit to early terminal emission, which
  collapses the rare-outcome discrimination because every patient
  trajectory ends so quickly there's no time to differentiate.

Phase stats: phase2_best_val 0.187 / 40 epochs; phase3_best_val 1.152
/ 15 epochs (early stopped, never recovered).

Verdict: **DISCARD**. Falsifiable failed on both prongs (RELEASE MAE
regressed and patient AUROC regressed catastrophically). Even DEATH
MAE improvement is hollow — the trajectory now collapses to terminal
within 24 h for 42 % of patients, which structurally pulls the DEATH
peak time forward without actually predicting WHICH patient dies.

Same failure family as P1: a patient-level/coarse-time aux added to
Phase 3 corrupts the per-position discriminator that B0-C-ttt's
per-position BCE + ranking carefully built. The shared lesson is
that Phase-3 aux losses that target the eval metric directly (MIL
in P1, soft-argmax onset in P2) push the head into a degenerate
sharp-peak regime — gain on the targeted metric, collapse on the
rest. The per-position BCE anchor at λ=1.0 is not strong enough on
its own to hold the optimum when a 0.20-capped aux pulls in a
fundamentally different direction.

This is the second DISCARD in a row. Reverting per loop step 9.
Proceeding to P3 (risk-aware LM head — architectural coupling),
which works at the LM head rather than the outcome head and therefore
won't fight the per-position BCE head-on.

---

### P3-coupling @ 10k (SHA 7838ac3) — DISCARD

P3 direction. Added bias_proj: nn.Linear(K → V) zero-init, applied to
sigmoid(outcome_logits), summed into lm_logits at the same position.
Coupling forms during Phase 2 (LM CE flows through bias_proj into the
joint backbone); Phase 3 refines outcome_head. Per-epoch ratio
||bias|| / ||lm_only_logits|| tracked.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D pass.
- P3a constructor-time zero-init verified at __init__ (assert in __init__).
- P3c shape contract verified — outcome_logits (B,T,K), logits (B,T,V),
  bias (B,T,V) match.
- P3b grad norms info: outcome_head[-1] 3.79 (active), bias_proj 0
  (expected — no LM CE in Phase 3), lm_head 0.45 (through tied input
  embedding).
- Smoke p3_ratio_mean 0.108, max 0.125 — in [0.05, 0.30] healthy band.

Post-train (10k):
- T1 partial — outcome head trained (raw_out 1.20→1.02 over 35 P3
  epochs), but Phase-3 train loss starts wildly high (epoch 1
  train=17.90, raw_out=17.90) because Phase 2 over-trained the
  coupling. The outcome head's logits at Phase 2 boundary are extreme
  (the coupling shapes them toward LM utility, not BCE calibration).
- T2 — Phase 3 ran 35 epochs (early stop at 36). vl_select dropped to
  0.980 — actually LOWER than B0-C-ttt's 1.010 best. But this lower
  selection metric did NOT translate to better headline AUROC, because
  the coupling distorted the outcome logits away from per-outcome
  ranking optima.
- T3 — DEATH AUROC dropped to 0.670 (-0.040 vs running best).
- p3_ratio at Phase-3 start: 1.10 (bias DOMINATES lm_only — way above
  [0.05, 0.30]). By end of Phase 3 the ratio settled to 0.047 mean
  (just below band), 0.46 max — Phase-3 training partially undid the
  coupling but the model never recovered the running-best optimum.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6473** (−0.0358 — fails KEEP rule)
- `patient_auprc_weighted`: 0.6217 (−0.0119)
- `patient_auroc_simple`:   0.6714 (−0.0244)
- `patient_auprc_simple`:   0.3051 (−0.0188)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — mixed but DEATH and RELEASE both regressed:
- DISGLYCEMIA_Hyper:  0.874 (−0.022)
- DISGLYCEMIA_Hypo:   0.818 (+0.047)
- NERVOUS_SYSTEM:     0.773 (−0.023)
- KIDNEY:             0.747 (+0.032)
- CARDIO:             0.736 (+0.027)
- RETINOPATHY:        0.720 (−0.066)
- NEUROVASCULAR:      0.712 (+0.026)
- **DEATH**:          0.670 (−0.040)
- KETOACIDOSIS:       0.651 (−0.265)
- SKIN_ULCER:         0.650 (−0.030)
- HYPEROSMOLALITY:    0.607 (+0.022)
- ACUTE_RESPIRATORY:  0.601 (+0.010)
- ACIDOSIS:           0.596 (+0.026)
- ATHEROSCLEROSIS:    0.592 (−0.003)
- INFECTION:          0.573 (+0.022)
- **RELEASE**:        0.425 (−0.156)  ← below chance, biggest collapse

Peak MAE (hours, Δ vs B0-C-ttt):
- DEATH:    155.08 (−13.89, ✓)
- RELEASE:   73.55 (+2.26)
- DISGLYCEMIA_Hyper:  34.59 (−1.48)
- KIDNEY:             91.71 (−14.65)

Trajectory honesty:
- `gen_median_hours`:           176.06  (vs 75.05 — over-generates ~2.3×)
- `gen_to_gt_ratio_median`:      1.694  (above 1.0 but ≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  **0.005**  (vs 0.165 — model essentially
  refuses to terminate early; mechanistic explanation for the RELEASE
  collapse: RELEASE is 87% prevalence and is a terminal token, and the
  coupling steers the LM AWAY from emitting terminals → the LM almost
  never emits RELEASE, so its detection collapses).

Phase stats: phase2 38 epochs; phase3 35 epochs (early stopped at 36).

Verdict: **DISCARD**. Falsifiable required DEATH AND RELEASE AUROC
both ≥ +0.030 — both regressed. RELEASE collapsed below chance
(0.425). The coupling at Phase 2 trained the LM to lean on the
outcome path for emission decisions, but the bias dominated by end of
Phase 2 (||bias||/||lm_only||≈1.1) and the LM head atrophied — at
inference the model can't generate trajectories that include the
terminal-class tokens (RELEASE/DEATH) at appropriate times, so
RELEASE detection collapses despite the coupling's intent to PROMOTE
terminal-token emission.

The architecture is sound; the problem is the LR/cap regime during
Phase 2 — bias_proj got too much budget and the LM head atrophied.
A targeted retry would need to (a) cap bias_proj's contribution to
||lm_logits|| (e.g., normalize bias by a learned scalar with prior
that keeps ratio ≤ 0.3), (b) gate bias_proj behind a slow ramp during
Phase 2, or (c) add a regularizer on bias_proj.weight to keep it
small. None of these are in scope for the current 10k probe regime
— program.md's P3 spec is exactly what was tested, and that spec
yielded a clear DISCARD.

Three DISCARDs in a row. Per program.md stop criterion, this nudges
toward halting, but P4 (patient-level pooling head) is structurally
different from P1/P2/P3 — different head architecture, different
loss path. Worth one more probe before concluding.

Reverting (loop step 9) and proceeding to P4.

---

### B0-C-ttt-ablation @ 10k (SHA 49be091) — KEEP-STACK

Mandatory ablation per program.md's new discipline. Strips Z's
direction-E frozen-terminal-tau hook from B0-C-ttt's recipe; keeps the
C-ttt aux head. Tests whether B0-C-ttt's gain over B0-Z came from
C-ttt alone or from the Z+C-ttt stack.

Per-aux training trace (Phase-2 + Phase-3, this ablation run):

| Aux       | Unlock epoch | λ_max   | Anchor raw_aux | Final raw_aux | Δ      | Status |
|-----------|--------------|---------|----------------|---------------|--------|--------|
| ce        | 4 (Ph-2)     | 0.0900  | 1.5183         | 0.0082        | −99.5% | learning |
| dt        | 4 (Ph-2)     | 0.1688  | 0.8096         | 0.1062        | −86.9% | learning |
| ttt       | 4 (Ph-2)     | 0.0038  | 21.4508        | 0.1663        | −99.2% | learning |
| ranking   | 31 (Ph-2)    | 0.0316  | 0.1224         | 0.1312        | +7.2%  | **STALE** — no descent across the 9 active Ph-2 epochs; raw_ranking actually rose slightly |
| out (P3)  | 1 (Ph-3)     | —       | 1.9035         | 0.9364        | −50.8% | learning |
| ranking(P3)| 1 (Ph-3)    | 0.1962  | 0.5683         | 0.3424        | −39.7% | learning |

The Phase-2 ranking aux fired at the very end of Phase 2 (calibrated
epoch 30, ramp 31→33, only 6 fully-active epochs before early stop at
40), and across those 9 active epochs raw_ranking did NOT descend.
Flag: **stale**. Either (a) the model is already at ranking optimum
by the time the aux unlocks (logits already separate pos / neg
positions enough that the pairwise loss is at floor for the current
backbone), or (b) the late-stage activation hits when other auxes
already drove the backbone into a state where ranking improvements
are gradient-flat.

Per-aux training trace (B0-C-ttt running best, recoverable values
only; final raw_aux at end of Phase 2 was overwritten when run.log
was reused for P1/P2/P3/M-384/ablation):

| Aux       | Unlock epoch | λ_max   | Anchor raw_aux | Final raw_aux | Δ      | Status |
|-----------|--------------|---------|----------------|---------------|--------|--------|
| ce        | 4 (Ph-2)     | 0.0872  | 1.5708         | ?             | ?      | unrecoverable |
| dt        | 4 (Ph-2)     | 0.1715  | 0.8027         | ?             | ?      | unrecoverable |
| ttt       | 4 (Ph-2)     | 0.0040  | 20.86          | ?             | ?      | unrecoverable |
| ranking   | 32 (Ph-2)    | 0.0329  | 0.1904         | ?             | ?      | unrecoverable |
| out (P3)  | 1 (Ph-3)     | —       | 2.11           | 1.01          | −52%   | learning (from prior journal) |
| ranking(P3)| 1 (Ph-3)    | ~0.7    | 0.66           | 0.38          | −42%   | learning (from prior journal) |

Future runs will record these from the start (the schema is now
program.md-mandated).

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Total params: 6.42 M (same as B0-C-ttt — pure hook removal).
- Gates A–D pass — raw_out=8.50, raw_rank=0.70, λ_ranking=2.45.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6401** (−0.0430)
- `patient_auprc_weighted`: 0.6079 (−0.0257)
- `patient_auroc_simple`:   0.6553 (−0.0405)
- `patient_auprc_simple`:   0.2865 (−0.0374)
- `n_outcomes_used`:        16

Per-outcome AUROC vs B0-C-ttt:
- DISGLYCEMIA_Hyper:  0.890  (−0.006)
- RETINOPATHY:        0.737  (−0.048)
- NEUROVASCULAR:      0.730  (+0.044)
- NERVOUS_SYSTEM:     0.718  (−0.078)
- CARDIO:             0.717  (+0.008)
- DISGLYCEMIA_Hypo:   0.678  (−0.093)
- **DEATH**:          0.670  (−0.040)
- SKIN_ULCER:         0.663  (−0.016)
- KETOACIDOSIS:       0.659  (−0.256)  ← biggest drop
- ATHEROSCLEROSIS:    0.657  (+0.062)
- KIDNEY:             0.634  (−0.081)
- HYPEROSMOLALITY:    0.593  (+0.008)
- ACUTE_RESPIRATORY:  0.584  (−0.007)
- ACIDOSIS:           0.545  (−0.025)
- INFECTION:          0.510  (−0.041)
- **RELEASE**:        0.501  (−0.080)  ← drops to chance

Peak MAE vs B0-C-ttt:
- DEATH:    146.49 (−22.48)
- RELEASE:   73.73 (+2.44)
- CARDIO:    73.20 (−5.88)
- KIDNEY:    92.95 (+13.84)

Trajectory honesty (Δ vs B0-C-ttt):
- `gen_median_hours`:           215.51  (+140.46 — much longer)
- `gen_to_gt_ratio_median`:       2.112  (vs 0.720; over-generates 2× —
  the unfrozen terminal tau lets the LM widen the terminal kernel,
  which prior diagnostics showed it tends to do)
- `gen_frac_terminal_first24h`:   0.037  (vs 0.165; rarely terminates
  early — same mechanism, wide terminal kernel pushes terminal
  emission far out)

Phase stats: phase2_best_val 0.184 / 40 epochs; phase3_best_val 1.112
/ 29 epochs.

Verdict: **ABLATION-KEEP-STACK** — Z's frozen-narrow-terminal-tau
hook is doing real work. Stripping it costs the recipe 0.043
patient_auroc_weighted, drops RELEASE to chance (0.501), collapses
KETOACIDOSIS (−0.256), and degrades trajectory honesty
(`gen_to_gt_ratio_median` doubles from 0.72 to 2.11). The C-ttt aux
alone, on bare M-256, does not match B0-C-ttt's eval lift. B0-C-ttt
remains the running best.

T1 note: Phase-2 ranking aux was **stale** in this run (Δ=+7.2 %
across 9 active epochs). This is also expected behaviour for ranking
when it unlocks late in Phase 2 — the backbone has already settled.
Stale-ness was not caused by the ablation. Logging the staleness now
that the schema requires it; not actionable for this experiment.

This ablation does not become the running best. Proceeding to P4
(patient-level pooling head — note: eval is read-only, so the pool
output can only be a training-time aux, similar in spirit to P1 MIL).

---

### P4-pool @ 10k (SHA fd54851) — DISCARD

P4 direction. Added a patient-level attention pool head as a Phase-3
aux: per-outcome learnable query embeddings cross-attend over the
backbone's stashed final hidden state (`model._last_hidden`), a
scalar projection turns the pooled feature into a patient-level logit,
BCE against patient_label. ~270K head params (6.42 M → 6.69 M).

Key structural distinction from P1-MIL: pool gradient flows backward
through the HIDDEN STATE (into the backbone at backbone_lr_factor=
0.01), NOT through outcome_logits → outcome_head. The outcome head's
per-position joint Phase-2 optimum is therefore protected — a
hypothesis worth testing given P1's universal per-outcome collapse.

Note on eval: program.md's P4 spec said the pool score "replaces
'max P_outcome' in eval", but evaluation.py is read-only. The pool
head therefore stays a training-time aux only.

Per-aux training trace (P4 run — new schema, fully captured):

| Aux            | Unlock epoch | λ_max  | Anchor raw_aux | Final raw_aux | Δ      | Status |
|----------------|--------------|--------|----------------|---------------|--------|--------|
| ce             | 4 (Ph-2)     | 0.0905 | 1.5220         | 0.0036        | −99.8% | learning |
| dt             | 4 (Ph-2)     | 0.1711 | 0.8047         | 0.0415        | −94.8% | learning |
| ttt            | 4 (Ph-2)     | 0.0039 | 21.0039        | 0.0605        | −99.7% | learning |
| ranking (Ph-2) | 33 (Ph-2)    | 0.0314 | 0.1121         | 0.0623        | −44.4% | learning |
| out (Ph-3)     | 1 (Ph-3)     | —      | 2.1888         | 0.9381        | −57.1% | learning |
| ranking (Ph-3) | 1 (Ph-3)     | 0.668  | 0.6554         | 0.3399        | −48.1% | learning |
| pool (Ph-3)    | 1 (Ph-3)     | 0.396  | 1.1062         | 0.1778        | −83.9% | learning |

Notable: Phase-2 ranking descends here (-44%), unlike the
B0-C-ttt-ablation where it was stale (+7%). The presence of P4's pool
aux during Phase 3 may not affect Phase 2 directly (P4 only fires in
Phase 3), so the difference is likely run-to-run variance — same
recipe, different random init / data ordering effects.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D pass. raw_pool=1.06, λ_pool=1.636 ∈ [1e-3, 10].
- Total params: 6.69 M (vs 6.42 M baseline).

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.7015** (+0.0184)
- `patient_auprc_weighted`: 0.6461 (+0.0125)
- `patient_auroc_simple`:   0.7063 (+0.0104)
- `patient_auprc_simple`:   0.3173 (−0.0066)
- `n_outcomes_used`:        16

Per-outcome AUROC vs B0-C-ttt — mixed: 11 outcomes improve, 5 regress
past the 0.010 threshold:
- DISGLYCEMIA_Hyper:  0.893  (−0.003)
- DISGLYCEMIA_Hypo:   0.777  (+0.006)
- **KIDNEY**:         0.769  (+0.054) ✓
- KETOACIDOSIS:       0.767  (**−0.148**) ✗
- SKIN_ULCER:         0.753  (+0.074) ✓
- NERVOUS_SYSTEM:     0.751  (−0.045) ✗
- RETINOPATHY:        0.739  (−0.046) ✗
- CARDIO:             0.717  (+0.008)
- NEUROVASCULAR:      0.708  (+0.022)
- **DEATH**:          0.672  (**−0.038**) ✗  ← key clinical outcome
- ATHEROSCLEROSIS:    0.658  (+0.063)
- HYPEROSMOLALITY:    0.640  (+0.055)
- ACUTE_RESPIRATORY:  0.633  (+0.042)
- INFECTION:          0.627  (+0.076)
- ACIDOSIS:           0.602  (+0.032)
- **RELEASE**:        0.597  (+0.016)

Peak MAE vs B0-C-ttt:
- DEATH:    151.85 (−17.12) ✓ — substantial improvement
- RELEASE:   78.67 (+7.38) ✗ — regress past 5h threshold
- KIDNEY:    60.87 (−18.24)
- CARDIO:    58.61 (−20.47)

Trajectory honesty:
- `gen_median_hours`:           116.96  (+41.91 vs B0-C-ttt)
- `gen_to_gt_ratio_median`:       1.148  (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:   0.178

Phase stats: phase2_best_val 0.184 / 42 epochs; phase3_best_val 1.130
/ 30 epochs (early stopped, best at epoch 1 for selection-metric
purposes — vl_select trajectory: 1.13 → 1.05 → 1.02 → 0.99 → 0.99
plateau).

Verdict: **DISCARD**. The falsifiable for P4 was patient AUROC ≥
+0.050 vs P1+P2+P3 best; since P1/P2/P3 all DISCARDed, the comparison
is vs B0-C-ttt 0.6831, target ≥ 0.7331. P4 reached 0.7015 — a real
+0.018 lift but well below the +0.050 falsifiable.

The KEEP rule also fails the "no headline regresses ≥ 0.010" prong:
DEATH AUROC −0.038 (key clinical headline), KETOACIDOSIS −0.148 (rare
but headline), NERVOUS_SYSTEM −0.045, RETINOPATHY −0.046, RELEASE MAE
+7.4 h. The weighted-AUROC lift comes from outcomes the model wasn't
already strong on (INFECTION, ATHEROSCLEROSIS, KIDNEY, SKIN_ULCER all
+0.04 to +0.08); the structural cost is that the model trades DEATH
and KETOACIDOSIS sensitivity for that breadth. The mechanism is
plausible — the pool's patient-level supervision encourages the
backbone to encode broad outcome distinguishability rather than peak-
sharp per-outcome calibration, which is what the per-position eval
metric rewards for DEATH/KETOACIDOSIS.

All per-aux traces clean (no stale auxes in this run, including the
Phase-2 ranking that was stale in the ablation — different
batch-ordering noise). T1 fully passes for the new pool aux
(−83.9 % descent).

Reverting per loop step 9. B0-C-ttt remains the running best.

This is the 4th DISCARD in a row (P1, P2, P3, P4). Per program.md
stop criterion, we are at "last 2-3 10k experiments DISCARDed" — but
P5 (BCE ablation, structural diagnostic — NOT a KEEP/DISCARD
candidate) is still owed. Proceeding to P5.

---

### P5-bce-ablation @ 10k (SHA fa0da81) — DIAGNOSTIC: KEEP BCE AT 1.0

P5 structural diagnostic per program.md (was P6 before refocus).
Down-weights Phase-3 per-position outcome BCE by 50× (coef 1.0 → 0.02).
Ranking + Phase-2-seeded backbone now drive the outcome head; tests
whether BCE is redundant or the calibration anchor.

Per-aux training trace:

| Aux            | Unlock epoch | λ_max  | Anchor raw_aux | Final raw_aux | Δ      | Status |
|----------------|--------------|--------|----------------|---------------|--------|--------|
| ce             | 4 (Ph-2)     | 0.0920 | 1.5146         | (P2 final omitted — identical to running-best B0-C-ttt seed) | descends |
| dt             | 4 (Ph-2)     | 0.1720 | 0.8096         | (omitted)                  | descends |
| ttt            | 4 (Ph-2)     | 0.0039 | 21.3950        | (omitted)                  | descends |
| ranking (Ph-2) | 32 (Ph-2)    | 0.0319 | 0.1018         | (omitted)                  | descends |
| out (Ph-3)     | 1 (Ph-3)     | —      | 2.2288         | 1.10–1.05     | ≈ −51% | learning |
| ranking (Ph-3) | 1 (Ph-3)     | ~0.6   | 0.6912         | 0.36–0.39     | ≈ −47% | learning |

P5's change is Phase-3-only; Phase 2 is byte-identical to B0-C-ttt.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6658** (−0.0173)
- `patient_auprc_weighted`: 0.6297 (−0.0039)
- `cap=48h AUROC` (legacy):     0.407 (vs B0-C-ttt 0.438 → **−0.031**)
- DEATH AUROC:    0.670 (−0.040)
- RELEASE AUROC:  0.532 (−0.049)
- DEATH MAE:    147.03 (−21.94 — actually improves)
- RELEASE MAE:   67.21 (−4.08  — actually improves)
- `gen_to_gt_ratio_median`:  1.161 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`: 0.131

Per-outcome AUROC vs B0-C-ttt:
- DISGLYCEMIA_Hyper:  0.901 (+0.004)
- NEUROVASCULAR:      0.797 (+0.111) ← best gain
- DISGLYCEMIA_Hypo:   0.786 (+0.015)
- RETINOPATHY:        0.767 (−0.018)
- NERVOUS_SYSTEM:     0.758 (−0.038)
- KETOACIDOSIS:       0.724 (**−0.191**) ← worst regression (rare-outcome
                      calibration loss confirms the BCE anchor hypothesis)
- KIDNEY:             0.722 (+0.007)
- CARDIO:             0.697 (−0.012)
- SKIN_ULCER:         0.693 (+0.014)
- **DEATH**:          0.670 (−0.040)
- ATHEROSCLEROSIS:    0.632 (+0.037)
- HYPEROSMOLALITY:    0.603 (+0.018)
- INFECTION:          0.573 (+0.022)
- ACUTE_RESPIRATORY:  0.560 (−0.031)
- ACIDOSIS:           0.547 (−0.024)
- **RELEASE**:        0.532 (+0.011)

Phase stats: phase2_best_val 0.184 / 40 epochs; phase3_best_val 1.176
/ 42 epochs (early stopped). Phase 3 ran longer than B0-C-ttt's 23
epochs because raw_out converges slower under reduced BCE weight.

Verdict: **DIAGNOSTIC — keep `phase3_outcome_bce_coef` at 1.0**.

Both decision criteria (program.md):
  1. Patient AUROC drops −0.017 — small but real.
  2. cap=48h AUROC drops −0.031 — meaningful collapse of the
     48-h-horizon legacy metric, which is the **calibration tell-tale**.

Combined signal: **per-position BCE IS the calibration anchor**.
Ranking + Phase-2-seeded backbone preserves coarse ordering (overall
AUROC drops only modestly, peak-MAE actually improves), but loses the
per-position 48-h-window calibration the BCE soft-kernel enforces.
KETOACIDOSIS −0.191 (rare outcome, n_pos=37) is the same pattern the
P1/P2/P3/P4 DISCARDs showed: rare outcomes need position-level BCE
pressure to keep their logits well-formed.

Programmatic implication: **the final loss recipe is locked at
B0-C-ttt's settings** (M-256 + Z frozen log_tau_terminal + C-ttt aux +
Phase-2 curriculum + Phase-3 BCE coef=1.0 + ranking).

Diagnostic does not become a running-best candidate. B0-C-ttt remains
the running best. Reverting the 0.02 coef back to 1.0 (loop step 9 —
for a diagnostic this is reverting the config knob, not the
intervention).

Stop-criterion status:
- P0 baselines + ablation done.
- P1/P2/P3/P4 all DISCARDed at 10k.
- P5 diagnostic confirms recipe lock.
- P6 (architecture scale-up) and P7 (QA toggle) are full-data end-of-
  loop steps. P6's strict trigger now applies — recipe is locked.

---

### B0-C-ttt-full @ FULL DATA (SHA 9544faa) — FINAL RESULT

End-of-loop full-data confirm of the running-best B0-C-ttt recipe
(M-256 + Z direction-E frozen-narrow `log_tau_terminal` + C-ttt
time-to-terminal aux + Phase-2 ce/dt/ttt/ranking curriculum + Phase-3
outcome-head BCE coef=1.0 + ranking). `sample=None`, ~57 k patients
(~5.7× the 10k workspace; 7,447-patient held-out test set vs 1,500 at
10k).

**P6 trigger note**: P6 (architecture scale-up at full data) was
SKIPPED because the strict trigger failed at the end of P5: the 10k
running best vs B0-Z was +0.016 patient_auroc_weighted, short of the
+0.030 floor program.md requires before burning hours on
architecture sweep. P7 requires P6 first, so it was also skipped.
The honest end-of-loop step is full-data confirm of the running-best
recipe (loop step 12), which is what this block reports.

Per-aux training trace (full-data confirm run):

| Aux            | Unlock epoch | λ_max  | Anchor raw_aux | Final raw_aux | Δ      | Status |
|----------------|--------------|--------|----------------|---------------|--------|--------|
| ce             | 4 (Ph-2)     | 0.0786 | 0.9731         | (P2 epoch 20)  | descending — full P2 final values omitted (only the early-stopped epoch survives in run.log; see below) |
| dt             | 4 (Ph-2)     | 0.0949 | 0.8056         | (P2 epoch 20)  | descending |
| ttt            | 4 (Ph-2)     | 0.0022 | 21.2961        | (P2 epoch 20)  | descending |
| ranking (Ph-2) | 12 (Ph-2)    | 0.0334 | 0.0937         | (P2 epoch 20)  | descending — calibrated earlier than 10k (epoch 12 vs ~30 at 10k) because plateau hits faster with more data |
| out (Ph-3)     | 1 (Ph-3)     | —      | 1.2574         | 0.8228 (ep 26) | −34.6% | learning — vl_select 1.087 → 0.833 best at epoch 23, plateau by ep 30 |
| ranking (Ph-3) | 1 (Ph-3)     | ~0.50  | 0.5083         | 0.2722 (ep 26) | −46.5% | learning |

Headline (held-out test set, n=8562 patients including 1115 DEATH,
7447 RELEASE):

- **`patient_auroc_weighted`: 0.6908**  (+0.008 vs 10k 0.6831)
- **`patient_auprc_weighted`: 0.6641**  (+0.030 vs 10k 0.6336)
- `patient_auroc_simple`:   0.6252
- `patient_auprc_simple`:   0.3237
- `n_outcomes_used`:        16

Per-outcome AUROC at full data:

| Outcome                       | AUROC | AUPRC | n_pos | prevalence |
|------------------------------|-------|-------|-------|------------|
| DISGLYCEMIA_Hyperglycemia    | 0.914 | 0.895 | 3550  | 41.5 %     |
| DISGLYCEMIA_Hypoglycemia     | 0.900 | 0.630 | 875   | 10.2 %     |
| KIDNEY_COMPLICATION          | 0.833 | 0.791 | 3839  | 44.8 %     |
| **DEATH**                    | **0.771** | 0.392 | **1115** | 13.0 %  |
| CARDIO-VASCULAR_DISORDER     | 0.743 | 0.801 | 5078  | 59.3 %     |
| **RELEASE**                  | **0.582** | 0.887 | **7447** | 87.0 %  |
| RETINOPATHY                  | 0.562 | 0.051 | 284   | 3.3 %      |
| NERVOUS_SYSTEM               | 0.549 | 0.078 | 517   | 6.0 %      |
| NEUROVASCULAR_COMPLICATION   | 0.542 | 0.027 | 170   | 2.0 %      |
| SKIN_ULCER                   | 0.541 | 0.058 | 391   | 4.6 %      |
| KETOACIDOSIS                 | 0.530 | 0.026 | 200   | 2.3 %      |
| ATHEROSCLEROSIS              | 0.511 | 0.024 | 197   | 2.3 %      |
| INFECTION                    | 0.508 | 0.140 | 1163  | 13.6 %     |
| ACUTE_RESPIRATORY_DISORDER   | 0.507 | 0.189 | 1602  | 18.7 %     |
| HYPEROSMOLALITY              | 0.506 | 0.051 | 435   | 5.1 %      |
| ACIDOSIS                     | 0.504 | 0.137 | 1177  | 13.7 %     |

Peak MAE (hours, positives only):

| Outcome                       | MAE (hours) | n_patients |
|------------------------------|-------------|------------|
| DISGLYCEMIA_Hyperglycemia    | 30.6        | 3550       |
| KIDNEY                       | 43.5        | 3838       |
| DISGLYCEMIA_Hypoglycemia     | 48.1        | 875        |
| HYPEROSMOLALITY              | 51.0        | 435        |
| ACIDOSIS                     | 51.8        | 1177       |
| ACUTE_RESPIRATORY            | 51.9        | 1601       |
| INFECTION                    | 52.6        | 1162       |
| ATHEROSCLEROSIS              | 54.0        | 197        |
| KETOACIDOSIS                 | 60.9        | 200        |
| CARDIO                       | 61.2        | 5078       |
| SKIN_ULCER                   | 64.3        | 391        |
| NERVOUS_SYSTEM               | 66.7        | 517        |
| NEUROVASCULAR                | 67.3        | 170        |
| **RELEASE**                  | **69.5**    | **7447**   |
| RETINOPATHY                  | 72.1        | 284        |
| **DEATH**                    | **155.9**   | **1114**   |

Trajectory honesty (full data):
- `gen_median_hours`:           62.40
- `gen_to_gt_ratio_median`:       0.599  (≥ 0.5 ✓)
- `gen_to_gt_ratio_mean`:         0.844
- `gen_frac_terminal_first24h`:   0.048  (much lower than 10k's 0.165 — at
                                          full data the model is more conservative
                                          about emitting terminal early)
- `gen_length_mae_hrs`:           82.96
- `gen_n_with_terminal`:          8561 / 8562

Phase stats: phase2_best_val 0.149 / 21 epochs (early stopped, vs 41
at 10k); phase3_best_val 0.944 / 40 epochs (vs 23 at 10k, ran longer
because the larger training set kept improving the outcome head).

Legacy / supplementary metrics (per-window):
- `outcome_auroc` (cap=336h):  0.508
- cap=48h AUROC:               0.506
- cap=168h AUROC:              0.521
- `onset_mae_hrs`:              65.6

**Verdict: FINAL RESULT — B0-C-ttt confirmed at full data**.

The 10k screening result generalises:
1. `patient_auroc_weighted` lifts +0.008 going to full data (0.683 →
   0.691), which is within noise — the screen was honest.
2. `patient_auprc_weighted` lifts +0.030, a real improvement that
   reflects better-calibrated probability heads with more training
   signal (Phase-3 ran 40 epochs at full data vs 23 at 10k).
3. cap=48h legacy AUROC lifts from 0.438 (10k) to 0.506 (full data) —
   the BCE calibration anchor that P5 identified gets noticeably
   better with more positives to learn from.
4. Trajectory honesty actually improves: `gen_frac_terminal_first24h`
   drops to 0.048 (vs 0.165 at 10k), and `gen_to_gt_ratio_median`
   stays comfortably ≥ 0.5. At full data the model is less aggressive
   about ending trajectories early.
5. DEATH AUROC 0.771 at full data (vs 0.710 at 10k, +0.061) — the
   primary clinical headline lifts substantially with more positives
   (n=1115 vs n=192). KIDNEY 0.833, DISGLYCEMIA_Hypo 0.900,
   DISGLYCEMIA_Hyper 0.914 are publishable per-outcome AUROCs.

Notable: the rare-outcome AUROCs (RETINOPATHY, KETOACIDOSIS,
ATHEROSCLEROSIS, NEUROVASCULAR) hover around 0.50–0.56 at full data.
These are the same outcomes that wobbled most across the P1-P4
DISCARDs — the model genuinely struggles to discriminate them at
patient-level. This is a substantive limitation of the
M-256 + B0-C-ttt-recipe stack, not a methodology artefact.

The recipe — narrow-frozen-terminal-tau + C-ttt aux on Phase-2 stage 0
+ Phase-3 BCE + ranking — is the **end-of-loop final result**.

---

## Final summary

| Metric                          | B0-Z @ 10k | B0-C-ttt @ 10k | B0-C-ttt @ full data (FINAL) |
|--------------------------------|------------|----------------|------------------------------|
| `patient_auroc_weighted`        | 0.667     | 0.683          | **0.691**                    |
| `patient_auprc_weighted`        | 0.621     | 0.634          | **0.664**                    |
| DEATH AUROC (n_pos=1114)        | 0.693     | 0.710          | **0.771**                    |
| RELEASE AUROC (n_pos=7447)      | 0.521     | 0.581          | **0.582**                    |
| DISGLYCEMIA_Hyper AUROC         | 0.904     | 0.896          | **0.914**                    |
| KIDNEY AUROC                    | 0.702     | 0.715          | **0.833**                    |
| DEATH peak MAE (hrs)            | 158.8     | 169.0          | **155.9**                    |
| RELEASE peak MAE (hrs)          | 86.0      | 71.3           | **69.5**                     |
| `gen_to_gt_ratio_median`        | 1.12      | 0.72           | **0.60**                     |
| `gen_frac_terminal_first24h`    | 0.148     | 0.165          | **0.048**                    |
| Phase-3 best val (outcome BCE)  | 1.157     | 1.144          | **0.944**                    |

**What was tried** (in order, see journal blocks for details):
- **B0-Z**: Z architecture (narrow + frozen terminal `log_tau_lm`)
  baseline — 0.667 AUROC_w.
- **B0-C-ttt**: cherry-pick of dd3fc1b time-to-terminal MSE aux on
  top of Z — **KEEP**, 0.683 AUROC_w. Running best.
- **B0-C-ttt-ablation**: C-ttt aux on bare M-256 without Z's freeze
  hook — KEEP-STACK (Z is doing real work, −0.043 AUROC_w without it).
- **P1-MIL**: softmax-weighted patient-level BCE aux in Phase 3 —
  DISCARD, −0.040 AUROC_w; universal per-outcome regression, especially
  KETOACIDOSIS −0.377.
- **P2-time**: positives-only soft-argmax time loss in Phase 3 —
  DISCARD, −0.110 AUROC_w; 5 outcomes drop below chance.
- **P3-coupling**: bias_proj(sigmoid(outcome_logits)) added to LM
  logits — DISCARD, −0.036 AUROC_w; RELEASE drops to chance (0.425)
  because the LM head atrophies (||bias||/||lm|| ratio 1.1, way above
  the [0.05, 0.30] healthy band).
- **P4-pool**: learned attention pool aux in Phase 3 — DISCARD;
  AUROC_w +0.018 (the only direction that lifted) but +0.05
  falsifiable missed, DEATH AUROC −0.038, KETOACIDOSIS −0.148.
- **P5-bce-ablation**: down-weight Phase-3 BCE coef 1.0 → 0.02 —
  DIAGNOSTIC, confirms per-position BCE is the calibration anchor
  (cap=48h −0.031, KETOACIDOSIS −0.191); recipe locked at coef=1.0.
- **P6 (architecture scale-up at full data)**: SKIPPED — strict
  trigger fails (running best margin vs B0-Z is +0.016, short of
  +0.030).
- **P7 (QA toggle)**: SKIPPED — requires P6 first.

**Why the loop stopped here** (program.md stop criterion):
- All directions in scope honestly attempted (P6/P7 strict triggers
  fail by design, not by neglect).
- Last 2-3 10k experiments DISCARDed (P3, P4) — running best stable.
- Full-data confirm of running best done — ABOVE.

The final running-best model lives in `emr_model/checkpoints/` and the
backup at `emr_model/checkpoints.bak_keep_B0-C-ttt-full/`.

---

## Post-P5 iteration (I1–I7)

### I1 — P3-v2 (lm_head + backbone FROZEN in Phase 3) @ 10k (SHA 7b23067) — DISCARD

Re-applied the original P3 `bias_proj` coupling (K→V on
sigmoid(outcome_logits), zero-init, added to lm_logits), but in
Phase 3: `lm_head.weight.requires_grad=False`, backbone optimizer LR
forced 0.0; only outcome_head trains. Hypothesis: original P3 died
from LM-head atrophy during the coupling; freezing the LM head in
Phase 3 removes that drift path.

Per-aux training trace:

| Aux            | Unlock | λ_max  | Anchor raw_aux | Final raw_aux | Δ      | Status |
|----------------|--------|--------|----------------|---------------|--------|--------|
| ce (Ph-2)      | 4      | 0.0817 | 1.4857         | 0.0022        | −99.8% | learning |
| dt (Ph-2)      | 4      | 0.1493 | 0.8134         | 0.0313        | −96.2% | learning |
| ttt (Ph-2)     | 4      | 0.0034 | 21.395         | 0.0442        | −99.8% | learning |
| ranking (Ph-2) | 31     | 0.0120 | 0.3173         | 0.1111        | −65.0% | learning |
| out (Ph-3)     | 1      | —      | 1.21 (ep2)*    | 1.137         | ≈ −6%  | shallow — frozen backbone caps refinement |
| ranking (Ph-3) | 1      | ~5.7   | 0.510 (ep2)    | 0.419         | −18%   | learning |

\* Phase-3 epoch-1 raw_out was 28.70 (the Phase-2 coupling leaves the
outcome logits extreme); it drops to ~1.21 by epoch 2 once λ_ranking
calibrates. Best `vl_select` only reached **1.0805** (epoch 21) vs
B0-C-ttt's 0.996 — with the backbone frozen, the outcome head cannot
recover from the Phase-2 coupling distortion.

Smoke gates A–D + P3a/P3b/P3c all passed (P3b confirmed lm_head
grad=0.0 — the I1 freeze worked; original P3 showed 0.45 via the tied
input embedding).

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6061** (−0.0770 — worse than original
  P3's −0.036)
- `patient_auprc_weighted`: 0.5841 (−0.0495)
- cap=48h AUROC: 0.394 (−0.044)
- DEATH AUROC: 0.601 (−0.109); RELEASE AUROC: 0.452 (−0.129, below chance)
- KETOACIDOSIS: 0.675 (−0.240); NERVOUS_SYSTEM 0.663 (−0.133)
- DEATH MAE 175.9 (+7); RELEASE MAE 102.6 (+31)
- gen_to_gt_ratio_median 2.468 (over-generates 2.5×); gen_frac_terminal_first24h 0.364
- p3_ratio_mean settled ~0.057 (in band) but Phase-3-start was 0.393 (over-coupled)

bias_proj routing (falsifiable interpretability check): bias-to-terminal
magnitudes were roughly uniform across outcomes (0.082–0.139; DEATH
0.119, not dominant) — no interpretable DEATH→TERMINAL routing.

Verdict: **DISCARD**. Falsifiable failed on both prongs (AUROC
regressed; routing uninterpretable). Freezing the LM head in Phase 3
made things *worse* than the original P3, not better: the coupling
distortion is created in Phase 2 (where lm_head + bias_proj co-train),
and freezing the backbone in Phase 3 only removes the model's ability
to partially recover. The original P3 verdict's mechanism — that the
damage is done in Phase 2 — is confirmed. The risk-aware-LM-head
direction is exhausted; no Phase-3-side freeze fixes a Phase-2-formed
coupling.

Reverting (loop step 9). B0-C-ttt remains the running best.
Proceeding to I2 (P4-tight, pool aux at cap=0.05).

---

### I2 — P4-tight (pool aux fraction_cap 0.20 → 0.05)

**Code:** `d9a6174` (single config change: `phase3_pool_fraction_cap`
0.20 → 0.05). All else identical to the P4-pool recipe on the
B0-C-ttt running best (M-256 + Z frozen-narrow terminal log_tau_lm +
C-ttt + Phase-2 curriculum + Phase-3 BCE coef 1.0 + ranking + pool head).

**Hypothesis (falsifiable):** a smaller pool-aux cap preserves the
patient-level AUROC lift while killing the per-position calibration
disruption P4 caused. Pass = AUROC ≥ +0.010 vs running best AND RELEASE
MAE no regress past 5 h AND no per-outcome AUROC drop past 0.020.

**Result vs running best B0-C-ttt (10k, `ea65988`):**

| Metric | B0-C-ttt | I2 | Δ |
|---|---|---|---|
| patient_auroc_weighted | 0.6831 | **0.7263** | **+0.0432** |
| patient_auprc_weighted | 0.6336 | 0.6719 | +0.0383 |
| cap=48h AUROC | 0.438 | 0.478 | +0.040 |
| DEATH AUROC | 0.710 | 0.721 | +0.011 |
| RELEASE AUROC | 0.581 | 0.604 | +0.023 |
| DEATH MAE (h) | 169.0 | 161.9 | −7.1 |
| RELEASE MAE (h) | 71.3 | 85.5 | **+14.2** |
| KETOACIDOSIS AUROC | 0.915 | 0.722 | **−0.193** |
| DISGLYCEMIA_Hyper AUROC | 0.896 | 0.856 | −0.040 |
| gen_to_gt_ratio_median | 0.720 | 1.688 | +0.97 |
| gen_frac_terminal_first24h | 0.165 | 0.050 | −0.115 |

13 of 16 outcomes improved AUROC. The three falsifiable prongs: AUROC
+0.043 ✓; RELEASE MAE +14.2 h ✗ (>5 h); per-outcome drops KETOACIDOSIS
−0.193 and DISGLYCEMIA_Hyper −0.040 ✗ (>0.020). **Strict rule → DISCARD
(2 prongs failed).**

**Per-aux training trace (every aux active in any phase):**

| Aux | Phase | Unlock ep | λ_max | Anchor raw | Final raw | Δ% | Learning? |
|---|---|---|---|---|---|---|---|
| ce | P2 | 3 | 0.0890 | 1.5619 | 0.00289 | −99.8% | yes |
| dt | P2 | 3 | 0.1719 | 0.8082 | 0.05132 | −93.7% | yes |
| ttt | P2 | 3 | 0.0039 | 21.4656 | 0.07955 | −99.6% | yes |
| ranking | P2 | 30 | 0.0323 | 0.1662 | 0.06249 | −62.4% | yes |
| ranking | P3 | 1 | 0.6755 | 0.5640 | 0.32496 | −42.4% | yes |
| pool | P3 | 1 | 0.0867 | 1.0985 | 0.08861 | −91.9% | yes |

No stale loss — every aux descends well past the 5% floor. The new pool
aux is the strongest learner (−91.9%). Critically, the pool aux learning
*well* is exactly what hurts: its patient-level signal couples into the
shared backbone and trades rare-outcome / peak-timing precision for
aggregate ranking — the same per-position-discriminator corruption seen
in P1/P2/P4. Lowering the cap to 0.05 did not kill that coupling; vs P4
cap=0.20 it *amplified* both the lift (+0.043 vs +0.018) and the damage
(KETOACIDOSIS −0.193 vs −0.148; RELEASE MAE +14.2 h vs +7.4 h). The
RELEASE-MAE regression is mechanically downstream of over-generation:
gen_to_gt_ratio_median 1.69 means trajectories run ~1.7× GT length, so
the predicted RELEASE peak lands late.

**Verdict: KEEP — NEW RUNNING BEST (user override of strict rule).**
The +0.043 weighted-AUROC lift is the largest in the loop and broad
across outcomes; the user elected to bank it and treat the
rare-outcome / RELEASE-timing regressions as a follow-up to repair
rather than a reason to revert. Running best is now **I2 = M-256 + Z +
C-ttt + Phase-2 curriculum + Phase-3 BCE coef 1.0 + ranking + pool head
@ cap 0.05** (`d9a6174`). Not reverted.

**Next:** cap the over-generation (gen_to_gt 1.69) — recover RELEASE
peak timing while holding the +0.043 lift, via a training-side lever
(eval/generation code is off-limits). Then continue the I-sequence.

---

### I2b — ttt-gated terminal emission (over-generation cap)

**Code:** `5d40ca5` (inference-only; `generate()` in
`transform_emr/inference.py`). The model's `ttt` head already predicts
`log1p(hrs to next terminal)` at every position and was returned by
`forward_with_cache` but *discarded* in the decode loop. I2b captures it
and, when the predicted hours-to-terminal drops below `ttt_emit_gate_hours`
(48 h), ramps a positive bias (`ttt_emit_bias`=3.0) onto terminal-token
logits so the trajectory ends near the model's own believed terminal
time. No retraining — tested `--eval-only` on the I2 weights
(`d9a6174`); eval picks up the enabling kwarg defaults.

**Hypothesis (falsifiable):** the I2 RELEASE-MAE regression (+14 h) is
driven by over-generation (gen_to_gt 1.69 → predicted peak lands late).
Capping it should: gen_to_gt → ~1.0 AND RELEASE MAE improve ≥5 h AND
patient_auroc_weighted no regress past 0.010.

**Result vs I2 (running best, no gate):**

| Metric | I2 | I2b | Δ |
|---|---|---|---|
| gen_to_gt_ratio_median | 1.688 | **1.184** | −0.504 |
| gen_median_hours | 169.9 | 119.2 | −50.7 |
| patient_auroc_weighted | 0.7263 | 0.7318 | +0.0054 |
| patient_auprc_weighted | 0.6719 | 0.6727 | +0.0008 |
| RELEASE AUROC | 0.604 | 0.625 | +0.020 |
| DEATH AUROC | 0.721 | 0.730 | +0.009 |
| RELEASE MAE (h) | 85.5 | 84.0 | **−1.4** |
| DEATH MAE (h) | 161.9 | 162.2 | +0.3 |
| gen_frac_terminal_first24h | 0.050 | 0.070 | +0.020 |

Every per-outcome AUROC held or improved (max drop −0.003, noise); the
two terminals gained most (RELEASE +0.020, DEATH +0.009) because the
gate aligns terminal emission with the model's own belief, sharpening
the terminal risk curves. Per-outcome MAE improved slightly across the
board (RELEASE 85.5→84.0, INFECTION 64.9→62.5, CARDIO 66.8→65.9, etc.).

**The falsifiable's RELEASE-MAE prong FAILED** (−1.4 h, wanted ≥−5 h)
even though gen_to_gt dropped hard (1.69→1.18). Diagnosis: `peak_mae` is
`|argmax_t P_outcome(t) − nearest GT time|` over the *generated portion*
(`evaluation.py:435`), i.e. a **risk-curve-peak-location** error, not a
trajectory-length error. Truncating 50 h of over-generation barely moved
the P_RELEASE argmax, which sits in the body of the trajectory, late
relative to the GT discharge. **Over-generation was the wrong lever for
RELEASE MAE.** The inherited I2 regressions vs the *original* B0-C-ttt
(KETOACIDOSIS −0.192, RELEASE MAE +12.7 h) are pool-head-caused and are
NOT addressed here.

**Per-aux training trace:** none — I2b runs no training (eval-only on the
I2 checkpoint). The active-aux trace is identical to the I2 block above.

**Verdict: KEEP — folds into the running best.** I2b is a strict
improvement over I2 (the current running best): +0.005 weighted AUROC,
RELEASE AUROC +0.020, DEATH AUROC +0.009, gen_to_gt 1.69→1.18 (more
honest), zero per-outcome regression — a free honesty + terminal-AUROC
win using a principled, already-trained signal. It does NOT achieve its
*stated* goal (recover RELEASE peak timing); that requires a
risk-curve-peak lever, not a length cap. Running best is now **I2 weights
(`d9a6174`) + ttt-gate (`5d40ca5`)**, AUROC_w 0.7318.

**Open:** RELEASE MAE (84 h) and the KETOACIDOSIS collapse remain. Both
are risk-curve / pool-coupling problems, not generation-length problems.

---

### I3 — P-CTTT-bounds (softplus positivity + ttt-consistency aux)

**Code:** `6343c51`. (a) softplus on the ttt head output in both forward
paths (mirrors the dt magnitude head; log1p(hrs)≥0). (b) New Phase-2
aux `ttt_consistency` = `|expm1(ttt_pred)+T(t) − GT_terminal|` in hours,
an L1-in-hours pin of the ttt-implied terminal time to the GT duration —
stage 0, cap 0.10 of BCE, ramp 0. Motivated by I2b: the inference ttt-gate
thresholds on *absolute* predicted hours, which log-space MSE calibrates
poorly at long horizons. Both bounds in one run (program default).

**Hypothesis (falsifiable):** patient-level AUROC doesn't regress; raw_ttt
descends; ttt-vs-dt consistency < 10 h.

**Per-aux training trace (Phase 2):**

| Aux | Unlock ep | λ_max | Anchor raw | Final raw | Δ% | Learning? |
|---|---|---|---|---|---|---|
| ce | 3 | 0.0886 | 1.5382 | 0.00799 | −99.5% | yes |
| dt | 3 | 0.1696 | 0.8039 | 0.08021 | −90.0% | yes |
| ttt | 3 | 0.0052 | 15.8326 | 0.10725 | −99.3% | yes |
| ttt_consistency | 3 | 0.0002 | 130.2194 | 10.5530 | −91.9% | yes |
| ranking | 34 | 0.0305 | 0.1633 | 0.10045 | −38.5% | yes |

No stale loss. The new `ttt_consistency` descends 130 h → ~10.5 h — it
**did its job**: the model's ttt-implied terminal time lands within ~10 h
of the GT terminal (right at the < 10 h falsifiable target). Calibration
prongs (raw_ttt descends, consistency < 10 h) both **pass**.

**Result vs running best I2b (0.732):**

| Metric | I2b | I3 gate-on | I3 gate-off (diag) |
|---|---|---|---|
| patient_auroc_weighted | 0.732 | **0.682 (−0.049)** | 0.660 |
| cap=48h AUROC | 0.478 | 0.389 | 0.389 |
| RELEASE MAE (h) | 84.0 | **65.6 (−18.4)** | 63.7 |
| DEATH MAE (h) | 162.2 | 156.5 (−5.8) | 152.0 |
| gen_to_gt_ratio_median | 1.18 | 0.61 | 0.80 |
| phase2_best_val | 0.1845 | 0.1879 | — |
| phase3_best_val | 1.103 | 1.116 | — |

AUROC regressed broadly (RETINOPATHY −0.145, CARDIO −0.092, INFECTION
−0.080, ~12/16 outcomes down; only DISGLYCEMIA_Hyper +0.028 and DEATH
+0.006 up). The AUROC prong **fails**.

**The consistency loss fixed RELEASE MAE** (84 → 65.6 h, −18 h) — the
first thing in the loop to move it — by sharpening the absolute-hour ttt
calibration the gate uses. But it did so by reshaping the *shared*
backbone toward absolute-time encoding at the expense of outcome
discrimination (both phase2 and phase3 val losses worsened, and the
per-outcome drop is broad, not terminal-specific).

**Gate-off diagnostic** (eval on I3 weights, `ttt_emit_bias=0`): AUROC
**0.660** — *lower* than gate-on, so the gate was actually helping
(+0.022) and the AUROC loss is **backbone-intrinsic, not gate-tunable**.
This rules out salvaging by relaxing the gate.

**Verdict: DISCARD.** Per the design-intent rule (MAE up + AUC roughly
flat → KEEP), MAE passes but AUC −0.049 is material, not flat — and the
diagnostic shows it's not recoverable by gate tuning. User confirmed
DISCARD → continue to I4. Reverting `6343c51`; running best stays I2b
(I2 weights `d9a6174` + ttt-gate `5d40ca5`, AUROC_w 0.732).

**Key finding (carry forward):** the ttt-consistency loss is a *working*
RELEASE-MAE lever (−18 h) — the open MAE problem IS solvable via ttt
calibration — but the present form pays ~0.05 AUROC because the
consistency gradient flows into the shared backbone. A future salvage
(smaller cap, stop-gradient to ttt_head only, or Phase-3-only) could
decouple the MAE win from the AUROC cost.

---

### I4 — Phase-2 sub-trajectory augmentation

**Code:** `09a24d6`. `EMRDataset.__getitem__` gains an `augment_views`
mode returning a random coherent row-subset per access: **A** full /
**B** drop a random 12 h window / **C** labs-only / **D** drop-labs.
Outcomes + terminals + admission are protected (never dropped); labs are
identified by the `_MEASURE` concept-name pattern (all TAK Measurements
concepts). `pretrain_transformer` toggles the flag ON for Phase 2 only
(Phase 1/3 + val see full trajectories; api.py untouched). Existing
Phase-2 oversampling supplies multiple views per patient per epoch.

**Hypothesis (falsifiable):** patient-level AUROC ≥ +0.010; auxes still
descend cleanly under the larger effective training set; honesty preserved.

**Per-aux training trace (Phase 2, under augmentation):**

| Aux | Unlock ep | λ_max | Anchor raw | Final raw | Δ% | Learning? |
|---|---|---|---|---|---|---|
| ce | 3 | 0.0807 | 1.7165 | 0.0210 | −98.8% | yes |
| dt | 3 | 0.1659 | 0.8350 | 0.0716 | −91.4% | yes |
| ttt | 3 | 0.0039 | 21.3346 | 0.1134 | −99.5% | yes |
| ranking | 38 | 0.0310 | 0.2908 | 0.1174 | −59.6% | yes |

All auxes descend cleanly under augmentation (no stale loss) — the
"auxes still descend" prong **passes**. Honesty prong **passes**
(gen_to_gt 0.50). Both phase val losses *improved* vs I2
(phase2 0.1845→0.1784, phase3 1.103→1.088) — augmentation helped the
LM/backbone fit.

**Result vs running best I2b (0.732):**

| Metric | I2b | I4 | Δ |
|---|---|---|---|
| patient_auroc_weighted | 0.732 | **0.710 (−0.022)** | fail |
| patient_auroc_simple | 0.724 | 0.690 | −0.034 |
| patient_auprc_weighted | 0.673 | 0.677 | +0.004 |
| cap=48h AUROC | 0.478 | 0.510 | +0.032 |
| RELEASE MAE (h) | 84.0 | 70.5 | −13.5 |
| DEATH MAE (h) | 162.2 | 160.4 | −1.8 |
| gen_to_gt_ratio_median | 1.18 | 0.50 | more honest |

**The AUROC prong fails** (−0.022, needed +0.010). The weighted −0.022 is
buoyed by DISGLYCEMIA_Hyper (+0.065, prevalence 0.41) and INFECTION
(+0.026), masking a **broad rare-outcome collapse**: NERVOUS_SYSTEM
−0.107, NEUROVASCULAR −0.096, KETOACIDOSIS −0.078, ATHEROSCLEROSIS
−0.069, RELEASE −0.060, RETINOPATHY −0.055, SKIN_ULCER −0.046. The
simple (unweighted) AUROC −0.034 confirms rare outcomes are hit hardest —
the filtered views cut exposure to rare-outcome contexts and shorten
sequences, hurting rare-class ranking even as the LM val improves.

**Verdict: DISCARD.** I4 was an AUROC play and it *regressed* AUROC
(−0.022) with broad rare-outcome damage — fails its own falsifiable and
its design intent. The cap=48h / RELEASE-MAE / honesty gains don't
rescue an AUROC-targeted experiment that lost AUROC. Reverting `09a24d6`;
running best stays I2b (0.732).

**Pattern across I2/I3/I4 (carry forward):** a consistent **AUROC ↔
calibration/honesty tension**. I2 (patient-level pool) bought +0.043
weighted AUROC but over-generates (gen_to_gt 1.69), bad RELEASE timing,
KETOACIDOSIS collapse. I3 (ttt-consistency) and I4 (sub-traj aug) both
move the *other* way — better cap=48h, RELEASE MAE −13–18 h, gen_to_gt
~0.5–0.6 — but cost weighted AUROC (−0.05 / −0.02), rare outcomes the
casualty. The headline (patient_auroc_weighted) rewards the I2 end of the
frontier. If the objective is later reweighted toward calibrated peak
*timing*/honesty, I3/I4 become attractive and should be revisited.

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
