# autoresearch — Architecture Sweep on real MIMIC-IV (UTC 2026-05-17)

## TL;DR — current best: **M-256** (commit `7925c06`)

```
AUROC 0.914   AUPRC 0.621   MAE 64.95h   VRAM 4.5 GB
DEATH 0.953  CARDIO 0.951  HYPERGLY 0.934  HYPOGLY 0.913
KIDNEY 0.900  RELEASE 0.835
```

## Status

Phase B in progress — S-128 DISCARD, M-256-deep DISCARD. Running L-384 next.

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
Verdict: DISCARD — AUROC 0.899 vs M-256 0.914 (Δ=-0.015, outside ±0.005 window).
  RELEASE collapsed 0.835→0.751 (same pattern as S-128). CARDIO improved 0.951→0.968
  but other outcomes flat or worse. Adding depth (4→6 layers) does not help — M-256
  width at 4 layers appears to be the sweet spot for this embedding dimension.

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
Verdict: DISCARD — AUROC 0.900 vs M-256 0.914 (Δ=-0.014, outside ±0.005 window).
  RELEASE dropped 0.835→0.741. Other outcomes individually better (CARDIO, HYPERGLY,
  HYPOGLY, KIDNEY), but RELEASE collapse drags the mean. Data wants M-256 width.
