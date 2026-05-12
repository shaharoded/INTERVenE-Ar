# autoresearch — loop status (UTC 2026-05-12 ~23:30)

## TL;DR — **NEW BEST exp66** (`82387ca`)

```
AUROC 0.850   (+0.017 vs exp63)    AUPRC 0.452   (+0.018)
MAE   81.5h   flat                 max_len 11.6%  (+3.1pp)
DEATH    0.983  (-0.005)
CARDIO   0.899  (+0.036)  KIDNEY 0.819 (+0.017)
HYPERGLY 0.836  (-0.007)  HYPOGLY 0.835 (+0.030)
RELEASE  0.727  (+0.033)               peak VRAM 8.4 GB
```

The fix that unlocked this was *methodology*, not just architecture:
add the P2 pairwise ranking loss to P3 (`λ_ranking` calibrated once at
end of epoch 1, cap inherited from `phase2_scheduler.aux_fraction_caps["ranking"]`)
**and** early-stop on `val_outcome_raw` (stable across the λ=0 → λ_cal
transition) rather than `val_total` (which jumps when λ activates and
locks the selector onto epoch 1 — the exp65 bug).

P3 ran 27 epochs; raw_ranking dropped 18.5%, raw_outcome 7.7%; best
saved at epoch 17. Rule 6 still met (ranking + outcome both active).

## Status — picking next

In flight: nothing. Next: **exp67 — Direction A sub-2 (replace P3 BCE
with ranking-only)**. exp66 proved both losses contribute in P3; sub-2
tests whether BCE is strictly net-positive once ranking is there.
audit_0.2c showed BCE in *P2* was borderline (cost −0.009 AUROC,
gave +0.027 AUPRC when removed). Same test for P3 is the cleanest
follow-up.

## Last completed

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| audit_0.2a | `083bfdb` | 0.828 | 0.401 | 0.698 | 14.9 | KEEP (hazard removed) |
| audit_0.2c | `b0cabac` | 0.819 | 0.428 | 0.651 | 12.7 | AUDIT (outcome BCE borderline) |
| exp62 | `c56108c` | 0.842 | 0.435 | 0.813 | 13.9 | DISCARD (P3 NaN'd → fluke) |
| exp62b | `aa267eb` | 0.831 | 0.419 | 0.674 | 13.9 | DISCARD (P3 destroyed RELEASE) |
| exp63 | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | KEEP (was best before exp66) |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3 cost +0.036 AUROC) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug — ckpt never saw ranking grad) |
| **exp66** | **`82387ca`** | **0.850** | **0.452** | **0.727** | **11.6** | **KEEP — current** |

## What we now know

- P3 IS net-positive on average (exp64 lost AUROC −0.036 with P3 off).
- The ranking loss is the *dominant* outcome-direction signal in both
  P2 (audit_0.2b: ablation cost −0.044 AUROC) and now P3 (exp65→exp66
  delta is +0.021 just from the selector fix → meaningful ranking-
  gradient training time).
- P3's outcome head was being pulled out of the P2 joint optimum by
  the BCE-only loss surface; restoring ranking closes that seam.
- RELEASE 0.694 → 0.727 (+0.033) — biggest single-experiment RELEASE
  jump since the data-shape work. HYPOGLY +0.030 confirms the P3
  beneficiaries are stronger, not just RELEASE.

## Open directions

- **A sub-2 (next as exp67)** — replace P3 BCE with ranking-only.
  Smallest follow-up that isolates whether BCE adds value in P3
  once ranking is there. One-line change to `loss = …` in
  `finetune_transformer`.
- **A sub-3** — use P2's oversampled DataLoader in P3 (currently
  natural distribution). Different lever.
- **G** — remove outcome soft-BCE project-wide (P2 AND P3). Bigger
  surgery; informed by exp67 outcome.
- **C** (refreshed) — soft-kernel BCE at LM head, learnable per-class
  tau. Comes after the P3-loss work fully lands.
- **B** — patient-trajectory contrastive aux for RELEASE. Defer.
- **E** — inference-side hazard boost (un-trained hazard head still
  in model). Cheap; opportunistic.

## Process discipline

- `results.tsv`: 73 data rows + header. Untracked.
- exp66 codebase = exp63 + ranking-in-P3 + stable-val selector +
  hazard-removal inference.py fix (from `c0a8ed0`).
- 7-decimal raw aux logging now in both P2 and P3 — visible per epoch.
- Committing locally only.
