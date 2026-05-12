# autoresearch — FINAL SESSION REPORT — 2026-05-12 16:30 UTC

> Session paused at user request. No new runs queued. Pickup-ready below.

## TL;DR

Starting point: exp52 baseline AUROC **0.788** / AUPRC **0.239** / MAE **87.6** / max_len **~83 %** / RELEASE **0.601**. After 13 experiments + 5 audit runs, current best is **exp63** (commit `033e019`):

```
AUROC    0.833    (+0.045 vs exp52, +0.029 vs exp49 program-best 0.804)
AUPRC    0.434    (+0.195)
MAE      81.59    (-6.0)
max_len  8.5 %    (-74.5 pp — inference termination effectively fixed)
RELEASE  0.694    (+0.093)
CARDIO   0.863, HYPERGLY 0.843, HYPOGLY 0.805, KIDNEY 0.802, DEATH 0.988
```

The headline driver was the **data-shape** wide-BCE-window for terminals (exp59 / 0.4b), confirmed by audit Task 0.4. The auxiliary contributions (ranking, outcome BCE) were quantified honestly via audit Task 0.2; hazard was removed (Rule 2(b) decorative). Per-outcome learnable tau (exp62 / 63) was the last principled "learn the window" extension — landed as a borderline KEEP.

## Status vs program.md tasks

| Task in program.md | Status | Evidence |
|---|---|---|
| Task A (Phase-1 Δt) | LOCKED before this session (exp52) | Raw audit 71 % drop, Pearson r 0.50 |
| Task B (Phase-1 MLM) | LOCKED removed before this session (exp54) | three honest attempts failed |
| Task C (pairwise ranking) | **LOCKED** | Audit 0.2b: ablation cost −0.044 AUROC |
| Task D (efficiency) | LOCKED earlier in session | 25 → 8.3 GB peak VRAM |
| Task 0 (honest audit) | **COMPLETE** | All 5 sub-tasks resolved; 6-decimal raw logging committed |
| Open-ended: RELEASE | partially addressed | 0.601 → 0.694. exp62 P2-only suggests 0.81+ is reachable but P3 destroys it |
| Open-ended: inference termination | **fixed** | 83 % → 8.5 % max_len % |
| Open-ended: learnable aux | one tried (exp62/63 outcome_log_tau) — borderline KEEP |

## Final aux task health (Rule 2(a) raw-drop + Rule 2(b) ablation)

| Aux | Raw drop | Ablation cost | Verdict | In codebase? |
|---|---|---|---|---|
| BCE primary | 8× ↓ | (mandatory) | LOCKED | yes |
| Δt (P1) | 71 % ↓ | (mandatory) | LOCKED | yes |
| ce (P2) | 84 % ↓ | (mandatory) | LOCKED | yes |
| dt (P2 gate+mag) | 77 % ↓ | (mandatory) | LOCKED | yes |
| outcome soft-BCE | 94 % ↓ | −0.009 AUROC | KEPT | yes |
| ranking | strong ↓ | **−0.044 AUROC** | LOCKED | yes |
| hazard | 97.6 % ↓ | +0.003 AUROC | decorative | **removed** |
| outcome_log_tau (exp63) | param trained in P2, frozen in P3 | learnable | KEPT | yes |

All mandatory aux pass the bar honestly. The outcome head is pushed by **two** losses (soft-BCE + ranking) — the audit confirmed both are real, ranking dominant.

## Experiment-by-experiment trace (this session)

| Exp / Audit | Commit | AUROC | AUPRC | RELEASE | max_len % | Status |
|---|---|---|---|---|---|---|
| exp52 baseline (start) | `d4a94ec` | 0.788 | 0.239 | 0.601 | — | — |
| Task D1+D2 | `6534aa8` | 0.810 | 0.281 | 0.616 | — | KEEP |
| exp54 redo (MLM remove) | `488d8b5` | 0.793 | 0.268 | 0.608 | — | KEEP |
| exp55 (Task C ranking bug) | `6fd38d9` | 0.794 | 0.271 | — | — | DISCARD |
| exp56 (Task C fix) | `9b9ac73` | 0.807 | 0.294 | 0.597 | 83.3 | KEEP |
| exp57 (inference hazard boost) | not committed | — | — | — | — | discarded (no effect) |
| exp58 (terminal-imminent aux) | `3ff2490` | 0.778 | 0.235 | 0.610 | 87.8 | DISCARD |
| exp59 (terminal wide BCE) | `8361146` | 0.810 | 0.386 | 0.654 | 12.4 | KEEP |
| exp60 (two-tier wide) | `c2f3856` | 0.833 | 0.396 | 0.756 | 8.7 | KEEP (later flagged) |
| exp61 (P1 wide too) | `c093e86` | 0.829 | 0.392 | 0.701 | 14.1 | DISCARD |
| audit_0.4 (no tiers) | `18a3caa` | 0.794 | 0.242 | 0.614 | 82.4 | AUDIT — terminals are the driver |
| **audit_0.4b (terminals only)** | `71ddbe9` | 0.825 | 0.427 | 0.681 | 8.9 | KEEP (principled, per caveat) |
| audit_0.2a (+ hazard removed) | `083bfdb` | 0.828 | 0.401 | 0.698 | 14.9 | KEEP |
| audit_0.2b (− ranking) | `27b8809` | 0.784 | 0.350 | 0.610 | 14.1 | AUDIT — ranking is real |
| audit_0.2c (− outcome BCE) | `b0cabac` | 0.819 | 0.428 | 0.651 | 12.7 | AUDIT — outcome real (borderline) |
| exp62 (learnable tau, P3 NaN) | `c56108c` | 0.842 | 0.435 | **0.813** | 13.9 | DISCARD (NaN bug; P2 only) |
| exp62b (NaN fix) | `aa267eb` | 0.831 | 0.419 | 0.674 | 13.9 | DISCARD (RELEASE collapsed in P3) |
| **exp63 (freeze tau in P3)** | `033e019` | **0.833** | **0.434** | 0.694 | **8.5** | KEEP (current) |

## Key learnings

1. **Data-shape changes can dwarf architecture work.** The single wide-window-for-terminals fix gave +0.031 AUROC / +0.185 AUPRC / −74 pp max_len. No aux head came close.

2. **The 4-decimal weighted log was hiding learning.** The 6-decimal raw audit showed every aux drops 70–97 %. "Flat at 4 decimals" was a logging artefact, not a learning failure.

3. **Hazard learned its sub-task but the gradient didn't reach AUROC.** A clean case of Rule-2(b) decorative. Removing it cost +0.003 AUROC and helped the weakest outcomes (RELEASE +0.017, HYPOGLY +0.020).

4. **Phase-3 can hurt.** exp62 vs exp62b: when P3 NaN'd and the eval fell back to P2's checkpoint, RELEASE was 0.813. With P3 working normally, RELEASE collapsed to 0.674. Freezing tau in P3 (exp63) only partly recovered things (0.694). Phase 3 is doing damage to RELEASE that we haven't isolated yet.

5. **Per-family BCE windows are dangerous overfitting** (caveat `f770850`). The complications-48h tier (exp60) gave AUROC +0.008 / RELEASE +0.075 but flagged retroactively as hand-picked. Adopted terminals-only (audit_0.4b) as the principled split.

## Suggested next experiments (NOT launched)

Ordered by my read of impact × principle. Pick whichever fits the next session's direction:

### Direction A — recover RELEASE via different P3 mechanics
exp62 P2-only had RELEASE 0.813. exp63 with P3-frozen-tau had 0.694. The P2-only number is achievable; P3 keeps destroying it. Two sub-experiments:

- **exp64a — P3 with much smaller LR.** `phase3_learning_rate: 1e-4 → 1e-5`. Tests if P3 overfits the outcome head.
- **exp64b — Save P2 best as the eval checkpoint when P3 doesn't improve val_loss.** Currently P3 best is preferred over P2 best regardless. This is an `api.py` / `finetune_transformer` selection change.

### Direction B — Patient-trajectory contrastive aux for RELEASE
New head + loss in Phase-2 that pulls healthy-discharge trajectories together in embedding space and pushes them apart from complication-trajectories. Per Rule 6 it must pass the learning bar. Risk: backbone capacity diversion (exp58 style failure).

- **exp65** — SimCLR-style contrastive head on mean-pooled patient trajectory representation. Targets RELEASE specifically.

### Direction C — Learned BCE window at the LM-head level
Apply the exp62 "learn the window" idea to the lm-head BCE (where the big exp59 data-shape gain lives). Replace the hard 168h/12h two-tier with per-token-class learnable log-tau in a soft-target formulation. Risky — touches the locked-in data-shape gain.

- **exp66** — `get_temporal_soft_hot_targets` with per-token-class learnable tau. Keep terminals' hard 168h as an upper bound (regulariser).

### Direction D — Audit Phase-3 entirely
Phase 3's role is unclear after the audits. Multiple experiments hint Phase 3 is over-fitting:
- audit_0.2c (no outcome): AUPRC +0.027 when outcome removed in P3.
- exp62 (P3 NaN'd, fell back to P2): RELEASE +0.115.
- exp63 (P3 with frozen tau): RELEASE only partly recovered.

- **exp67** — Run with `phase3_n_epochs: 0` (skip P3 entirely). Compare to exp63. If P3 is net-negative, that's the headline.

### Direction E — Inference-side fixes
- **exp68** — Re-try inference-side hazard boost from exp57 now that the hazard head no longer trains (audit_0.2a). The hazard head still exists in the model (untrained), and its predictions are different now. Different signal might help.

## Stop conditions (from program.md)

Not yet at the stop point. Multiple structural directions remain. Specifically:
- Phase-3 dynamics not yet investigated honestly (Direction A, D).
- Contrastive aux for RELEASE never tried (Direction B).
- Learned window at lm-head level never tried (Direction C).
- Inference-side hazard boost not re-tested under current setup (Direction E).

## Resume snippet

```bash
cd /workspace/autoresearch

# Quick state check
git log --oneline -3
tail -n 3 results.tsv | awk -F'\t' '{print $1, $5, $6}'
ps aux | grep "python api" | grep -v grep   # should be empty

# Re-read program.md before launching anything
less program.md

# Smoke test always before the full run
# (set sample=50 + 1/1/1 in config, run api.py, check summary, revert config)
```

## Process discipline still in effect
- `results.tsv` is **untracked**. Survives `git reset --hard`.
- 6-decimal raw aux logging is committed (`39c3896`). Any future "flat aux" claim must check raw values first.
- DISCARD = `git reset --hard HEAD~1` (results.tsv survives).
- Rule 5: any +<0.005 AUROC "win" needs fresh-P1 re-run before logging KEEP.

---

**Session stopped at user request. No experiments queued. Resume by selecting one of the 5 directions above and re-engaging.**
