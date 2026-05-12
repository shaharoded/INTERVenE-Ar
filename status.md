# autoresearch — loop status (UTC 2026-05-12 ~22:30)

## TL;DR

Current best — **exp63** (`033e019`)
```
AUROC 0.833    AUPRC 0.434    MAE 81.6h    max_len 8.5%
DEATH 0.988   CARDIO 0.863   HYPERGLY 0.843   HYPOGLY 0.805
KIDNEY 0.802  RELEASE 0.694                  peak VRAM 8.3 GB
```

## Status — in flight

**exp66 — P3 ranking loss, methodology-fixed retry** (`82387ca`).

This is exp65 with the val-selection bug fixed. exp65 added the
pairwise ranking loss to Phase 3 but watched `val_total` for
early-stop. Because λ_ranking calibrates at the end of epoch 1
(λ goes 0 → λ_cal), val_total jumps at the epoch-1/2 boundary and the
selector locks onto epoch 1 — so the saved checkpoint never saw a
ranking gradient. Despite that, RELEASE moved +0.038 just from
P3-shortening to 1 epoch.

exp66 fix: track `val_outcome_raw` separately, use it for early-stop.
It is stable across the λ-transition and is the metric the
generation-based eval ultimately measures (outcome head calibration on
natural distribution). The ranking term still affects training; it
just doesn't pollute the selector.

Sub-questions exp66 answers:
- AUROC ≥ exp63 + AUPRC ≥ 0.434 + HYPOGLY recovers → KEEP.
- AUROC drops > 0.005 → ranking isn't the right P3 fix; move to
  sub-2 (replace BCE with ranking-only) or Direction G (project-wide).

## Last completed

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| audit_0.2a | `083bfdb` | 0.828 | 0.401 | 0.698 | 14.9 | KEEP (hazard removed) |
| audit_0.2c | `b0cabac` | 0.819 | 0.428 | 0.651 | 12.7 | AUDIT (outcome BCE borderline) |
| exp62 | `c56108c` | 0.842 | 0.435 | 0.813 | 13.9 | DISCARD (P3 NaN'd → fluke) |
| exp62b | `aa267eb` | 0.831 | 0.419 | 0.674 | 13.9 | DISCARD (P3 destroyed RELEASE) |
| **exp63** | **`033e019`** | **0.833** | **0.434** | **0.694** | **8.5** | **KEEP — current** |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3: P3 IS net-positive +0.036 AUROC) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (methodology bug — ckpt never saw ranking grad) |

## What we now know

- exp64 (skip P3) cost AUROC −0.036, AUPRC −0.070. P3 is net-positive
  on average; HYPOGLY (−0.121) and HYPERGLY (−0.050) are the
  outcomes that lean on P3. Only CARDIO benefited from skipping.
- exp65 (P3 + ranking, buggy selector) hit RELEASE 0.732 even though
  the ranking gradient never trained the saved checkpoint. The
  +0.038 RELEASE gain came from running P3 for just 1 epoch — that
  alone is a useful diagnostic about P3 over-fitting RELEASE.
- The "exp62 RELEASE = 0.813" story is officially retired (exp64
  ran cleanly without P3 and RELEASE stayed at 0.688 — that number
  was a NaN-fallback fluke).

## Open directions (post-exp65)

- **A sub-1 (in flight as exp66)** — add ranking loss to P3 with
  stable val selection.
- **A sub-2** — replace P3's BCE entirely with ranking-only. Run if
  exp66 KEEPs but AUPRC doesn't move; sharper test.
- **A sub-3** — use P2's oversampled DataLoader in P3 (natural→
  oversampled). Different angle if loss-shape doesn't fix it.
- **A sub-4** — explicitly limit P3 epochs (the exp65 accidental
  evidence). Smaller change than skipping P3, tests "less is more".
- **G** — remove outcome soft-BCE project-wide, keep ranking-only.
  Bigger surgery; informed by A sub-1 / sub-2 outcomes.
- **C** (refreshed) — soft-kernel BCE at LM head, learnable per-class
  tau. Comes after the P3-loss work lands.
- **B** — patient-trajectory contrastive aux for RELEASE. Defer.
- **E** — inference-side hazard boost (un-trained hazard head in
  model). Cheap; opportunistic.

## Process discipline

- `results.tsv`: 72 data rows + header. Untracked; survives reset.
- DISCARDed exp64 and exp65 with `git reset --hard`. inference.py
  hazard-removal fix preserved as standalone bug-fix commit
  (`c0a8ed0`); transformer.py reverted along with exp65.
- 7-decimal raw aux logging now also in Phase 3 (every epoch logs
  `raw_out` and `raw_rank`). exp65's "best-val locked at epoch 1"
  was visible in the run.log; making this loud helps future audits.
- Committing locally only.
