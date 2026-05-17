# autoresearch — Architecture Sweep on real MIMIC-IV (UTC 2026-05-17)

## TL;DR — current best: **M-256** (commit `7925c06`)

```
AUROC 0.914   AUPRC 0.621   MAE 64.95h   VRAM 4.5 GB
DEATH 0.953  CARDIO 0.951  HYPERGLY 0.934  HYPOGLY 0.913
KIDNEY 0.900  RELEASE 0.835
```

## Status

Phase B in progress — running S-128 next.

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
