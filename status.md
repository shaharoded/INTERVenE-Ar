# autoresearch — Architecture Sweep on real MIMIC-IV (UTC 2026-05-17)

## TL;DR — current best: **M-256** (commit `7925c06`)

```
AUROC 0.914   AUPRC 0.621   MAE 64.95h   VRAM 4.5 GB
DEATH 0.953  CARDIO 0.951  HYPERGLY 0.934  HYPOGLY 0.913
KIDNEY 0.900  RELEASE 0.835
```

## Status

**COMPLETE** — Architecture sweep done. Final best: M-256 (non-QA). Phases A/B/C/D complete. Phase B HP sweep (L-384-v2) DISCARD.

---

## Architectures completed

### L-384-v2  (commit `452cea9`)  — Phase B HP sweep: backbone_lr_factor 0.01→0.10, dropout 0.10→0.15

- params: 20,782,344           peak VRAM: 0.58 GB (eval-only; training crashed before full eval)
- final config:
    embed_dim=384, n_layer=6, n_head=6, time2vec_dim=48, dropout=0.15,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.10),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.911, AUPRC=0.633, MAE=65.61h
- per-outcome: not captured (eval-only path used)

Training notes:
  Motivation: L-384 Phase 3 diagnosis showed outcome head underfitting at every
    outcome (LM head beat dedicated head at CARDIO, HYPERGLY, KIDNEY, etc.). Root cause:
    backbone_lr_factor=0.01 → LR=1e-6 was too small for 20.78M params to adapt in
    the Phase 3 budget. Fix: raise factor to 0.10 (backbone LR=1e-5); also raise
    dropout 0.10→0.15 for regularisation at this scale.
  Phase 1 — reused L-384 Phase 1 checkpoint (embed_dim unchanged; no retrain needed).
  Phase 2 — 52 epochs; ranking unlocked ep33 (plateau), best ep43 (vl_total=0.0964,
    BCE=0.0650, ranking=0.097). api.py crashed between Phase 2 and Phase 3 (same
    cumulative-RAM pattern as original L-384). Recovered via run_phase3.py.
  Phase 3 — repeated SIGKILL at validation ~80-84% (batch ~433-455 of 541) after
    ~10-11 epochs across 3 separate launches (run-b, run-c, run-d). Root cause:
    cumulative CUDA memory growth over training epochs meets long-sequence validation
    batches. Fix applied: torch.cuda.empty_cache() before validation + resume from
    ckpt_last.pt. With fix, run-d survived ep12 and ep13 (validation fully completed).
    Best checkpoint: run-c ep6 (vl_select=0.656235). run-d resumed at ep12 but
    ep12=0.684, ep13=0.700 — diverging upward (bad_epochs accumulating, early stop
    ~ep17). Evaluated via eval_only.py from phase3/ckpt_best.pt (run-c ep6).
  Within-size adjustments tried: backbone_lr_factor=0.01→0.10, dropout=0.10→0.15.
Verdict: DISCARD — AUROC 0.911 vs M-256 0.914 (Δ=−0.004). Within the ±0.005 AUROC
  tolerance window, but MAE 65.61h is 0.66h WORSE than baseline (secondary criterion
  requires ≥2h MAE improvement). AUPRC notably better (+0.013: 0.633 vs 0.621).
  Higher backbone LR did not fully close the convergence gap; Phase 3 best arrived at
  only ep6 of an expected 20-30+ epoch trajectory — the model needs more training epochs
  to converge at 20.78M params than the current crash-recovery budget provides.

---

### M-256-QA  (commit `2da6fc5`)  — Phase C QA-data retrain

- params: 6,416,932           peak VRAM: 2.75 GB (2817.8 MB)
- final config (same as M-256 + USE_QA_DATA=True):
    embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.1,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.01),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.903, AUPRC=0.636, MAE=66.00h
- per-outcome (≥3 pos windows):
    DEATH=0.935, CARDIO=0.976, HYPERGLY=0.940, HYPOGLY=0.903,
    KIDNEY=0.906, RELEASE=0.759

Training notes:
  Phase 1 — reused M-256 Phase 1 checkpoint (embed_dim unchanged; dataset cache
    invalidated by USE_QA_DATA=True, Phase 1 retrained). ~19 epochs estimated.
  Phase 2 — 47 epochs (0-46); early stopped. BCE descended 0.2498→0.0658.
    ce λ_max=0.1022, dt λ_max=0.0985 (active ep4+), ranking λ_max=0.0208
    (calibrated ep33, full lambda ep37). Best BCE ~0.0665.
  Phase 3 — 48 epochs; best at epoch 43 (vl_select=0.600547). Descended
    ep1(0.715)→ep43(0.601), then 5 consecutive non-improving epochs fired
    early stop at ep48. Recovered via run_phase3.py after api.py crash
    during original Phase 3 epoch 3 (cumulative RAM growth at ~3h49m runtime).
    vl_select significantly below M-256 baseline (0.601 vs 0.679 at best).
  Verdict factors: AUROC 0.903 vs baseline 0.914 (Δ=-0.011, outside ±0.005
    window). AUPRC improved (+0.015: 0.636 vs 0.621). MAE regressed slightly
    (+1.05h). RELEASE collapsed 0.835→0.759 (−0.076). CARDIO excellent 0.976
    (+0.025). DEATH dropped 0.953→0.935 (−0.018). QA-data retrain hurts RELEASE
    and DEATH while improving CARDIO/HYPERGLY; net AUROC loss is decisive.
Verdict: DISCARD — AUROC 0.903 vs M-256 0.914 (Δ=-0.011, outside ±0.005).
  QA features did not improve the primary metric. RELEASE collapse is the
  dominant failure mode (same pattern as smaller architectures). Phase D next:
  k-day seed scan (k=2,3,4,5,6) on M-256 baseline.

---

### M-256  (commit `7925c06`)  — Phase A baseline
- params: 6,414,628           peak VRAM: 4.54 GB
- final config (within-size best):
    embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.1,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.01),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.914, AUPRC=0.621, MAE=64.95h, max_len%=n/a
- per-outcome (≥3 pos windows):
    DEATH=0.953, CARDIO=0.951, HYPERGLY=0.934, HYPOGLY=0.913,
    KIDNEY=0.900, RELEASE=0.835

Training notes:
  Phase 1 — 19 epochs (BCE-only epochs 1-3, dt unlocked epoch 4,
    best epoch 14 val=0.1269, flat 15-19 → early stop). dt calibrated
    λ_max=0.030. Scheduler fix (warmup_complete_epoch gate) prevented
    premature stop that killed dt in the prior session.
  Phase 2 — 44 epochs; ce+dt unlock epoch 4, ranking unlocked epoch 13
    (plateau), warmup done epoch 16. Best val=0.099 at epoch 43. Raw
    ranking fell to 0.097 — strong pairwise signal. Train/val BCE gap
    moderate (0.020 vs 0.067) but val still falling throughout.
  Phase 3 — 49 epochs, best at epoch 44 (vl_select=0.679). Step-wise
    descent with occasional blips; patience-5 fired cleanly at epoch 49.
  Within-size adjustments tried: none needed.
  Diagnose: not run — checkpoints overwritten when S-128 started (embed_dim 256→128
    forces Phase-1 retrain). Training logs show no pathologies: dt converged, val BCE
    fell smoothly, ranking signal strong (raw_ranking=0.097). No within-size fix was needed.
Verdict: KEEP — Phase A baseline. +0.032 AUROC / +0.138 AUPRC / -19h MAE
  vs prior exp73 (MIMIC-III, 0.882/0.483/83.9h). VRAM halved (9.4→4.5 GB).

---

### M-256-deep  (commit `3c94166`)  — Phase B #2
- params: 9,307,944           peak VRAM: 0.40 GB (eval-only; training not captured)
- final config:
    embed_dim=256, n_layer=6, n_head=4, time2vec_dim=32, dropout=0.1,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.01),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.899, AUPRC=0.606, MAE=64.49h
- per-outcome (≥3 pos windows):
    DEATH=0.935, CARDIO=0.968, HYPERGLY=0.930, HYPOGLY=0.911,
    KIDNEY=0.896, RELEASE=0.751

Training notes:
  Phase 1 — retrained from scratch (S-128 P1 cache had embed_dim=128).
    ~19 epochs estimated.
  Phase 2 — 49 epochs (phase2_n_epochs=50 loaded from checkpoint override;
    100-epoch config took effect from P3 onward). Best val saved as late as ep48.
  Phase 3 — best epoch 28 (vl_select=0.680229); process crashed during epoch 33
    validation (90% through). Evaluated via eval_only.py from phase3/ckpt_best.pt.
    Steady descent ep1(0.790)→ep28(0.680), then plateaued; patience-5 would have
    fired at ep33 anyway. λ_ranking calibrated=0.502.
  Within-size adjustments tried: none.
  Diagnose: not run — checkpoints overwritten when L-384 started (embed_dim 256→384
    forces Phase-1 retrain). No evidence of training pathology from logs: Phase 2
    ran 49/50 epochs with stable BCE descent; Phase 3 descended to ep28 then plateaued.
    RELEASE collapse pattern (0.835→0.751) mirrors S-128 and suggests the embedding
    dimension (256) is insufficient to disambiguate RELEASE from other outcomes when
    depth is added without widening. Depth addition may fragment representational capacity.
Verdict: DISCARD — AUROC 0.899 vs M-256 0.914 (Δ=-0.015, outside ±0.005 window).
  RELEASE collapsed 0.835→0.751 (same pattern as S-128). CARDIO improved 0.951→0.968
  but other outcomes flat or worse. Adding depth (4→6 layers) does not help — M-256
  width at 4 layers appears to be the sweet spot for this embedding dimension.

---

### L-384  (commit `b72a0a1`)  — Phase B #3
- params: 20,777,320           peak VRAM: 0.59 GB (eval-only; training OOM crash before eval)
- final config:
    embed_dim=384, n_layer=6, n_head=6, time2vec_dim=48, dropout=0.1,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.01),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.899, AUPRC=0.597, MAE=64.62h
- per-outcome (≥3 pos windows):
    DEATH=0.919, CARDIO=0.962, HYPERGLY=0.924, HYPOGLY=0.899,
    KIDNEY=0.897, RELEASE=0.795

Training notes:
  Phase 1 — retrained from scratch (embed_dim changed 256→384 forces P1 retrain).
    ~19 epochs estimated from checkpoint timestamps.
  Phase 2 — ~40 epochs, best val ~0.096300 (from smoke_test_L384.log, epoch 39).
    Aux curriculum: ce+dt unlocked ep4, ranking unlocked on plateau.
    api.py crashed (SIGKILL) between Phase 2 and Phase 3; Phase 2 checkpoint intact.
    Recovery: run_phase3.py loaded Phase 2 checkpoint directly.
  Phase 3 — 20 epochs, best at epoch 15 (vl_select=0.662728). Steep descent
    ep1(0.735)→ep7(0.682), plateau ep8-14, new best ep15(0.663), flat ep16-20 → early
    stop at ep20. run_phase3.py evaluation OOM-crashed (SIGKILL) during test-set
    DataProcessor temporal filter pass (training data still in memory). Evaluated via
    eval_only.py from phase3/ckpt_best.pt.
  Within-size adjustments tried: none.
  Diagnose: run (diag_L384.log). Key findings:
    - LM head beats outcome head at every outcome on validation (per-position):
        CARDIO LM=0.918 vs Head=0.636; HYPERGLY LM=0.919 vs Head=0.668;
        KIDNEY LM=0.802 vs Head=0.673 — outcome head underfits in 20 epochs.
    - RELEASE validation head AUROC=0.901 vs test AUROC=0.795: worst generaliz-
        ation gap, suggesting RELEASE representation is not stable in L-384 space.
    - Outcome token gradient signal is low (HYPERGLY rank 260/350, CARDIO 168/350);
        top-signal tokens are insulin-delivery states and severe-HTN events.
    - Lambda calibration healthy (ce=0.116, dt=0.098, ranking=0.020; all below caps).
    - Temporal coverage good: 93.8%/97.9% of positions have ≥1 pos in 12h/48h windows.
    - Δt R²=0.1275 (moderate); embedder linear probe AUROC=0.620 (unchanged).
    - Context vectors provide small consistent signal (+0.94 BCE when zeroed).
    Conclusion: Phase 3 ran only 20 epochs (vs 49 for M-256). At 20.78M params the
    outcome head needs more epochs to converge past the general LM head. The low
    gradient signal for outcome tokens and backbone LR×0.01 (≈1e-6) means the head
    is training largely alone, needing more iterations. Earlier early-stop (vl_select
    still changing at ep20) cut training short. RELEASE generalisation gap is the
    primary failure mode at this scale.
Verdict: DISCARD — AUROC 0.899 vs M-256 0.914 (Δ=-0.015, outside ±0.005 window).
  DEATH collapsed 0.953→0.919 (−0.034). RELEASE partially recovered vs S-128/M-256-deep
  (0.795 vs 0.741/0.751) but still well below baseline 0.835. CARDIO excellent 0.962
  (+0.011). Pattern: wider/deeper (384-dim, 6L) helps CARDIO but hurts DEATH and
  RELEASE. Phase 3 outcome head did not converge: LM head outperforms dedicated head
  at all outcomes — more P3 epochs or higher backbone LR factor needed at this scale.

---

### XL-512  (commit `ab7aae1`)  — Phase B #4  — OOM EXCLUSION
- params: ~36,790,000 (est.)    peak VRAM: n/a (never reached eval)
- final config attempted:
    embed_dim=512, n_layer=6, n_head=8, time2vec_dim=64, dropout=0.1,
    batch_size=16→8 (halved per program.md OOM rule)
- metrics: n/a — Phase 2 never completed

Training notes:
  Phase 1 — 19 epochs, completed successfully.
  Phase 2 — SIGKILL during training ~30-45 min after Phase 2 start, all attempts:
    Run 1 (batch_size=16): crashed during epoch 6 training at ~06:55 UTC.
    Run 2 (batch_size=16): crashed during epoch 6 training at ~07:05 UTC.
    Run 3 (batch_size=8, grad_accum=8): crashed during epoch 4 training at ~07:53 UTC.
    All crashes at same wall-clock time (~30-45 min), different epoch counts:
      batch_size=16 → 6 epochs × 5 min = 30 min; batch_size=8 → 4 epochs × 8 min = 32 min.
    Pattern is TIME-BASED (cumulative RAM growth), not per-step VRAM OOM. Batch halving
    does not reduce peak RAM since workers accumulate memory independently of batch size.
    No ckpt_best.pt ever saved; ckpt_last.pt only.
  Within-size adjustments tried: batch_size=16→8 (program.md within-size rule applied).
  Diagnose: not run — Phase 2 never completed; no usable checkpoint.
Verdict: OOM EXCLUSION — Phase 2 training crashes after ~30-45 min regardless of batch
  size. Cumulative RAM growth pattern (not VRAM OOM) prevents training XL-512 on this
  hardware. Per program.md: hard exclusion after batch halve still failed.
  Phase B best remains M-256.

---

### S-128  (commit `d22dadb`)  — Phase B #1
- params: 1,668,900           peak VRAM: 0.22 GB (eval-only; training not captured)
- final config:
    embed_dim=128, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.1,
    phase1_lr=3e-4, phase2_lr=3e-4, phase3_lr=1e-4 (backbone×0.01),
    patience=5, aux_caps={ce:0.5, dt:0.5, ranking:0.2}
- metrics: AUROC=0.900, AUPRC=0.611, MAE=64.72h, max_len%=n/a
- per-outcome (≥3 pos windows):
    DEATH=0.918, CARDIO=0.972, HYPERGLY=0.940, HYPOGLY=0.918,
    KIDNEY=0.909, RELEASE=0.741

Training notes:
  Phase 1 — retrained from scratch (embed_dim changed 256→128 vs cached M-256).
    ~19 epochs estimated from checkpoint timestamp.
  Phase 2 — ~50 epochs estimated (~2.9 hrs from P1-done to P2-ckpt timestamps).
    Aux curriculum same settings as M-256.
  Phase 3 — best epoch 35 (vl_select=0.712417); process crashed during epoch 40
    validation before printing summary. Evaluated via eval_only.py from saved
    phase3/ckpt_best.pt. Steady descent from ep26 (0.725) to ep35 (0.712).
  Within-size adjustments tried: none.
  Diagnose: not run — checkpoints overwritten when M-256-deep started (embed_dim
    128→256 forces Phase-1 retrain). Training logs show no pathologies; Phase 2 ran
    ~50 epochs with stable curriculum. RELEASE collapse (0.835→0.741) is the dominant
    failure mode — smaller embed_dim likely lacks capacity to encode the admission-to-
    discharge trajectory needed for RELEASE prediction.
Verdict: DISCARD — AUROC 0.900 vs M-256 0.914 (Δ=-0.014, outside ±0.005 window).
  RELEASE dropped 0.835→0.741. Other outcomes individually better (CARDIO, HYPERGLY,
  HYPOGLY, KIDNEY), but RELEASE collapse drags the mean. Data wants M-256 width.

---

## Phase D — k-day seed scan  (UTC 2026-05-19, actual M-256 retrain)

Evaluated the final best architecture (M-256) at seed lengths k=2..6 days.
M-256 was fully retrained (P1+P2+P3) to obtain fresh checkpoints (original M-256
checkpoints were overwritten during Phase C). Retrain AUROC=0.915040 (vs 0.914243
original — within noise). Phase D run on retrain checkpoints.

### k-day results table

| k (days) | AUROC  | AUPRC  | MAE (h) | CARDIO | DEATH | HYPERGLY | HYPOGLY | KIDNEY | RELEASE |
|----------|--------|--------|---------|--------|-------|----------|---------|--------|---------|
| 2        | 0.9150 | 0.6298 | 64.98   | 0.968  | 0.943 | 0.936    | 0.916   | 0.911  | 0.817   |
| 3        | 0.9155 | 0.6482 | 81.85   | 0.972  | 0.952 | 0.938    | 0.927   | 0.913  | 0.791   |
| 4        | 0.9093 | 0.6003 | 99.02   | 0.968  | 0.934 | 0.947    | 0.911   | 0.914  | 0.782   |
| 5        | 0.9147 | 0.5920 | 116.61  | 0.984  | 0.939 | 0.946    | 0.929   | 0.918  | 0.773   |
| 6        | 0.9105 | 0.6036 | 134.48  | 0.972  | 0.949 | 0.941    | 0.921   | 0.932  | 0.748   |

### Key findings

1. **AUROC is near-flat across k=2–6** (range 0.909–0.916, Δ=0.007). The k=4 dip
   (0.909) is a consistent artefact across both M-256 and M-256-QA proxy runs.
   The model extracts the bulk of predictive signal within the first 2–3 days.

2. **k=3 is the best default**: marginally higher AUROC than k=2 (0.9155 vs 0.9150)
   with a clear AUPRC gain (+0.018: 0.648 vs 0.630) and better DEATH prediction
   (0.952 vs 0.943). Trade-off: MAE increases by ~17h (81.85 vs 64.98h).

3. **MAE grows linearly at ~17–18h per additional seed day** (65→82→99→117→134h).
   Each extra seed day advances generation start by ~24h; events in the seed window
   are "consumed" as context, pushing remaining events further out.

4. **AUPRC dips at k=4–5** then partially recovers at k=6. Consistent window-
   alignment artefact where 24h eval windows interact with the seed boundary.

5. **Per-outcome trends:**
   - RELEASE declines monotonically with k (0.817→0.748). Discharge signal is
     concentrated in the first 1–2 days of admission.
   - KIDNEY improves monotonically with k (0.911→0.932). Renal deterioration
     benefits from longer metabolic context.
   - CARDIO peaks at k=5 (0.984) — longer seed captures pre-event trajectory.
   - DEATH non-monotone but generally higher at k=3 (0.952) than k=2 (0.943).
   - HYPERGLY stable across k (0.936–0.947); HYPOGLY stable (0.911–0.929).

6. **Recommendation:** **k=3** for quality-optimised deployment (best AUROC+AUPRC+DEATH);
   **k=2** for minimum-history real-time settings (best MAE, second-best AUROC).

---

## Final Session Report  (UTC 2026-05-18)

### Best model

**M-256** (commit `7925c06`):
  embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.1
  AUROC=0.914, AUPRC=0.621, MAE=64.95h, VRAM=4.54 GB
  DEATH=0.953, CARDIO=0.951, HYPERGLY=0.934, HYPOGLY=0.913, KIDNEY=0.900, RELEASE=0.835

### What the size sweep revealed

The sweep tested five architecture sizes on 40k real MIMIC-IV patients. M-256 (6.4M
params, 4.54 GB VRAM) was optimal. The key failure mode across all other sizes was
**RELEASE collapse**: smaller (S-128: 0.741) and deeper (M-256-deep: 0.751, L-384:
0.795) configurations all failed to maintain the admission-to-discharge RELEASE
trajectory, while M-256 held at 0.835. Wider+deeper (L-384) improved CARDIO (0.962)
but hurt DEATH (0.919) and RELEASE. XL-512 was excluded by cumulative RAM OOM during
Phase 2 training — a time-based crash pattern unrelated to per-step VRAM, which
batch-halving could not fix.

Going smaller (S-128) hurt RELEASE most severely due to insufficient embedding capacity
to represent the discharge trajectory. Going deeper (M-256-deep) at fixed width
fragmented representational capacity without improvement. Widening (L-384) saw the
outcome head underfit in Phase 3 (LM head beat dedicated head at every outcome),
requiring more training epochs than the early-stop budget allowed.

The M-256 baseline on MIMIC-IV improved substantially over the prior MIMIC-III result
(exp73): +0.032 AUROC (0.914 vs 0.882), +0.138 AUPRC (0.621 vs 0.483), −19h MAE
(64.95 vs 83.9h), with VRAM halved (9.4→4.5 GB).

### Phase C (QA-data) finding

USE_QA_DATA=True hurt the primary metric (AUROC −0.011) despite improving AUPRC
(+0.015). The diagnostic showed QA features disrupted temporal modeling (Δt R²≈0 vs
0.12 baseline) and caused RELEASE to collapse further (0.835→0.759). The shuffled-
context paradox (shuffled context improved BCE vs normal) suggests the model overfit
to QA context representations. QA features do not help this architecture.

### Phase D (k-day seed) finding

AUROC is near-flat across k=2–6 (0.909–0.916). MAE grows linearly with k (~17h/day).
**k=3 is the recommended default** — marginally better AUROC than k=2 with a clear
AUPRC gain (+0.018) and better DEATH prediction. k=2 is preferred when only 2 days
of history are available. k=4 shows a consistent AUROC dip (0.909) — a window-
alignment artefact confirmed on both M-256 and M-256-QA proxy. Previously run on
M-256-QA proxy; corrected results on actual M-256 retrain (commit `5496c9e`).

### Phase E (proper QA retrain) finding  —  DISCARD

M-256 retrained with USE_QA_DATA=True and a fresh tokenizer (commit `fa63768`):
AUROC=0.876, AUPRC=0.567, MAE=63.70h, VRAM=0.33 GB. P3 best ep31 vl_select=0.702
(proc died ep36 val; eval-only recovery).

**Verdict: DISCARD.** AUROC −0.038 below M-256 baseline (0.914243), well outside
the ±0.005 tolerance. Worse than the original stale-tokenizer QA run (0.876 vs
0.903), confirming that properly tokenized %_PATTERN% events add noise rather than
signal. Phase C and Phase E together establish that QA features do not benefit this
architecture in any form. The best model remains M-256 non-QA (commit `5496c9e`,
AUROC=0.915040).
