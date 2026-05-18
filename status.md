# autoresearch — Architecture Sweep on real MIMIC-IV (UTC 2026-05-17)

## TL;DR — current best: **M-256** (commit `7925c06`)

```
AUROC 0.914   AUPRC 0.621   MAE 64.95h   VRAM 4.5 GB
DEATH 0.953  CARDIO 0.951  HYPERGLY 0.934  HYPOGLY 0.913
KIDNEY 0.900  RELEASE 0.835
```

## Status

Phase B in progress — S-128 DISCARD, M-256-deep DISCARD, L-384 DISCARD. Next: XL-512 or L-384 variation per program.md.

---

## Architectures completed

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
