# autoresearch — EMR Event Prediction: Architecture Sweep on real MIMIC-IV

Autonomous architecture sweep on the EMR complication-prediction model now that
real processed MIMIC-IV data is available at scale.

The structural research phase is finished. Every previously-locked architectural
fix (soft-kernel LM head, three-tier `log_tau_lm` init, P3 ranking loss, AdaLN-Zero,
temporal RoPE, Time2Vec log-spaced init, gradient checkpointing, BF16 AMP, etc.)
stays in the codebase. **No new structural ideas.** This is about finding the
right *size* of the existing architecture for the real data, then a final
training round with treatment-quality QA data included.

---

## Background

Three-phase training of an event-stream transformer:

- **Phase 1 — EMREmbedding**: hierarchical token embeddings + Time2Vec + static
  patient context. Loss: teacher-forced BCE + Δt MSE.
- **Phase 2 — GPT (AdaLN-Zero + temporal RoPE)**: causal LM over Phase-1
  embeddings. Soft-kernel BCE on the LM head with learnable per-class `tau`
  (three-tier init: terminals 168h / outcome-class 48h / default 12h) + next-token
  CE + Δt regression + P2 ranking aux on the outcome head.
- **Phase 3 — Outcome head fine-tune**: backbone with differential LR
  (`backbone 1e-6, head 1e-4`); outcome BCE on time-decayed soft labels + P3
  ranking loss. `val_outcome_raw` is the checkpoint selector.

Data is split 70/15/15 train/val/test **by `PatientId`** (seed=42, two-stage
`train_test_split`). Train fits the scaler and tokenizer; val is used for
early-stop monitoring during P2/P3; the test split is held out and never seen
until the final evaluation. Evaluation = autoregressive generation from a 2-day
seed → 24h windows → AUROC / AUPRC / onset-MAE per complication, then mean
across complications with ≥3 positive windows. See `evaluation.py`. **Published
metrics in `results.tsv` / `status.md` are computed on the held-out test split**
— they are the reportable numbers, not a val-set proxy.

Current best = exp73 (`3eaafa7`): `embed_dim=256, n_layer=4, n_head=4,
time2vec_dim=32, dropout=0.1`. Peak VRAM 9.4 GB on MIMIC-III at the previous data
scale; real-data scale will push this up.

---

## The goal

1. Find the architecture size that maximises `outcome_auroc` and minimises
   `onset_mae_hrs` on real MIMIC-IV, with a decent `outcome_auprc`, subject to
   **peak VRAM ≤ 24 GB** across all three phases. AUROC primary, MAE secondary,
   AUPRC tiebreaker. VRAM over 24 GB → DISCARD (will likely OOM the pod anyway).
2. Re-train the chosen best with QA data (`USE_QA_DATA=True`).
3. Re-infer + evaluate the best with an increasing k-day seed (2 → 6 days,
   controllable from the eval dataset class) to see how inference quality
   scales with context window.

---

## What counts as a sweep dimension

The single most important distinction in this program.

**Architecture (one `status.md` block per unique combo):**
`embed_dim`, `n_layer`, `n_head`, `time2vec_dim`, `dropout`, and anything else
that changes parameter count or data shape.

**Within-size knobs (NOT separate architectures):**
LR (`phase{1,2,3}_learning_rate`, `phase3_backbone_lr_factor`), scheduler
(`lr_warmup_epochs`, `early-stop-patience`, plateau settings, aux
`bce_only_epochs`, `ramp_epochs`), weight decay, batch size +
`grad_accumulation_steps` (keep effective batch constant), aux `aux_fraction_caps`.

Within-size knobs are only allowed as a **response to a diagnosed training
problem**: bumpy loss curves, val diverging from train, flat aux loss, NaNs,
unstable epochs. Don't waste training time on small tunes. Only the final-best
within-size config for an architecture is recorded — earlier tries fold into
that block's training notes.

**Forbidden as its own experiment:** "try LR=2e-4 instead of 3e-4." That is not
a new architecture and does not get a row of its own; it is logged as the
best HP for whichever architecture it stabilised.

---

## What is in scope to modify

- `emr_model/transform_emr/config/model_config.py` — primary edit target.
- `emr_model/transform_emr/config/dataset_config.py` — `USE_QA_DATA` for Phase C.
- `emr_model/transform_emr/*.py` — only if a clear training pathology forces
  it (e.g. NaN at a specific width). Architecture *behaviour* is locked without
  consulting me.

**Out of scope:** `api.py`, `evaluation.py`, `emr_model/data/`, and new
structural ideas (new losses / heads / attention variants).

---

## Running an experiment

Every experiment: **re-read this file first** (VRAM cap and the
architecture-vs-within-size distinction are easy to drift on), then:

1. **Smoke test** with `sample=50` and `phase{1,2,3}_n_epochs=1` in
   `model_config.py`. Run `python api.py > smoke.log 2>&1` and confirm
   `grep "^outcome_auroc:\|^---" smoke.log` produces the summary block.
   Restore the real config. Smoke results are never logged.

2. **Full run**: `python api.py > run.log 2>&1`. Extract with
   `grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^phase2_best_val:\|^phase3_best_val:\|^peak_vram_mb:\|^num_params:" run.log`.
   Empty → crash (`tail -n 50 run.log`). No `---` summary after 120 min → CRASH.

3. **Diagnose** when training looks bumpy or metrics are off:
   `cd emr_model && python -m transform_emr.diagnose > ../diag.log 2>&1 && cd ..`.
   Focus on Report 2 (logit separation), Report 4 (`lambda_max < 0.001` = silent),
   Report 5 (outcome tokens ranked by grad/occ), Δt + outcome-head probes.
   Also inspect `tr_*` / `vl_*` curves in `run.log` across all three phases — a
   flat aux loss across a whole phase is a training pathology; fix with a
   within-size adjustment (dropout / LR / scheduler), not a structural change.

4. **Log**: append a brief row to `results.tsv`
   (`commit\toutcome_auroc\toutcome_auprc\tonset_mae_hrs\tpeak_vram_gb\tstatus\tdescription`).
   `status` ∈ {`KEEP`, `DISCARD`, `CRASH`, `OOM`}; `description` =
   `arch <size_tag>; <within-size knob touched if any>; <one-sentence training observation>`.
   Crashes/OOMs use `0.000000 / 0.00 / 0.0` for metrics. Then update `status.md`
   (next section). Both files are updated before the next experiment starts —
   never batch.

5. **KEEP** iff peak VRAM ≤ 24 GB **and** AUROC strictly above the current
   within-architecture best (or matches within ±0.005 AUROC **and** improves
   MAE by ≥ 2 h) **and** no unresolved training pathology. Otherwise
   **DISCARD** → `git reset --hard <last_keep_commit>`.

---

## `status.md` — per-architecture journal

One block per unique `embed_dim / n_layer / n_head / time2vec_dim / dropout`
combo, updated in place — only the final-best within-size config is recorded.
Top of file: UTC timestamp, `## TL;DR` (best arch / AUROC / AUPRC / MAE / VRAM),
`## Status` (in-flight arch + phase + epoch + ETA), `## Architectures completed`
(blocks below).

Block format:

```
### <size_tag>  (commit <hash>)
- params: <num_params>           peak VRAM: <X.X GB>
- final config (within-size best):
    embed_dim=..., n_layer=..., n_head=..., time2vec_dim=..., dropout=...,
    phase{1,2,3}_lr=..., warmup=..., patience=..., aux_caps={...}
- metrics: AUROC=..., AUPRC=..., MAE=..., max_len%=...
- per-outcome (≥3 pos windows): DEATH=..., RELEASE=..., CARDIO=..., ...

Training notes:
  Phase 1 — <train-loss arc, did Δt converge, any stagnation>
  Phase 2 — <BCE arc, aux loss status, plateau hits>
  Phase 3 — <val_outcome_raw arc, early-stop epoch, head behaviour>
  Within-size adjustments tried: <each knob and why; mark which made the cut>
Verdict: KEEP / DISCARD vs prior best.
```

`status.md` is the journal; `results.tsv` is the ledger. Don't duplicate text.

---

## The sweep plan

Deliberately small. Find the smallest arch that saturates the data and the
largest that fits VRAM, pick the best between them.

### Phase A — Re-baseline on real data

One run at the locked architecture (`embed_dim=256, n_layer=4, n_head=4,
time2vec_dim=32, dropout=0.1`). Grounds the comparison and validates the
pipeline. If training is bumpy, apply the smallest within-size fix before
declaring this the baseline; log the adjustment in the block.

### Phase B — Architecture sweep

Run architectures in order. Each gets one block. For each: smoke → full →
diagnose → (optional within-size tune) → KEEP/DISCARD vs current best.

Starting grid (not a contract — skip a size if the previous failure mode rules
it out; insert at most one missing-middle size if two neighbours disagree):

| Tag          | embed_dim | n_layer | n_head | time2vec_dim | Rationale                              |
|--------------|-----------|---------|--------|--------------|----------------------------------------|
| `S-128`      | 128       | 4       | 4      | 32           | small — does the data want less?       |
| `M-256`      | 256       | 4       | 4      | 32           | baseline                               |
| `M-256-deep` | 256       | 6       | 4      | 32           | more depth at same width               |
| `L-384`      | 384       | 6       | 6      | 48           | wider + deeper, heads scaled           |
| `XL-512`     | 512       | 6       | 8      | 64           | largest credible under 24 GB           |

Skip/replace rules:
- `S-128` within ±0.005 AUROC of `M-256` → prefer `S-128` (smaller wins ties).
- `XL-512` OOMs at current batch → halve batch + double grad-accum **first**
  (within-size). Only if it still OOMs is it a hard exclusion.
- `L-384` beats both neighbours → insert one extra size (e.g. `L-448`). One.

**Stop Phase B** when every credible size has a final block and adding one more
size cannot plausibly improve headline metrics under the VRAM cap.

### Phase C — QA-data re-train

After Phase B picks a best: flip `USE_QA_DATA=True`, re-train that
architecture once, log as `<best_size_tag>-QA`. If it wins on AUROC under VRAM,
it is the final result; otherwise the non-QA best wins.

### Phase D — k-day seed scan

Re-evaluate the final best at k ∈ {2, 3, 4, 5, 6} days of input seed. One
extra block in `status.md` covering how AUROC / MAE / AUPRC scale with k.

### Stop criterion (whole session)

Done when Phases A, B, C, D are each complete. Write a final session report:
best arch, all metrics, per-outcome breakdown, peak VRAM, one paragraph on
what the size sweep revealed.

---

## Process discipline

- **One architecture per commit.** Message prefix: `arch <size_tag>: ...`.
- **Always smoke-test before a full run.** Real-data runs are expensive.
- **VRAM is a hard cap.** Check `peak_vram_mb` every run. Over 24 GB after
  standard memory tricks (gradient checkpointing, BF16 AMP, bucket batching,
  batch halved with grad-accum doubled) → DISCARD.
- **Inspect `tr_*` / `vl_*` across all three phases every run.** Flat aux loss
  across a whole phase is a training failure; if no within-size fix recovers
  it, log it honestly and move on.
- **DISCARD = `git reset --hard <last_keep_commit>`.**
- **Update `status.md` and `results.tsv` before the next experiment.** No batching.
