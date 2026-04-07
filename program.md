# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search on an EMR next-event prediction model.

---

## Background

The model learns to generate the future stream of medical events for a hospital patient given their history. It uses MIMIC-III derived data containing diabetes patients' longitudinal events (lab results, vitals, diagnoses, medications, meals, outcomes such as complications and death).

The architecture is a two-phase pipeline:

- **Phase 1 — EMREmbedding**: learns a compact, time-aware representation of each clinical event using hierarchical token embeddings, Time2Vec, and patient context.
- **Phase 2 — GPT Transformer**: a causal decoder trained over the Phase-1 embeddings to predict the next clinical event in a patient's timeline.

The key clinical targets are 15 *complication* types (e.g. `KIDNEY_COMPLICATION_EVENT`, `CARDIO-VASCULAR_DISORDER_EVENT`). The model must predict both *what* will happen and *when*. These are very sparse, so perfection is not expected, but capturing at least the main ones will be a great win.

---

## Session Handoff

State of research after session 1 (branch `autoresearch/mar31`, 22 experiments). See `session_summary.md` for full details.

**Best result:** `outcome_auroc = 0.722311` — commit `0b196bd`
**Config:** `embed_dim=256, n_layer=4, n_head=4, block_size=512, sample=2000`

### What is already implemented

- Temporal BCE with 12h window (replaces step-based k-window) — biggest single improvement (+0.021)
- Non-overlapping outcome head: BCE=[0,12h] + outcome head=[12h,48h]
- AdaLN-Zero conditioning, SwiGLU MLP, weight-tied LM head
- PRTracker removed

### What has been established

- **Width > depth**: embed_dim 64→256 gave +0.077 AUROC; n_layer increases always hurt, but this might be related to sample size.
- **Temporal BCE is essential**: step-based BCE created contradictory gradients for outcomes; temporal alignment is non-negotiable
- **Outcome head must stay**: removing it costs -0.025 AUROC (gradient flow through backbone matters even if the head's direct predictions are weaker than the LM head)
- **Outcome head is under-powered**: calibrated lambda ≈ 0.0002 — gradient direction is right but magnitude is near-zero
- **RMSNorm regressed** (-0.018): AdaLN-Zero depends on LayerNorm's mean subtraction; do not swap norms

### What has failed (do not repeat)

- n_layer 4→5, 4→6: regression both times
- embed_dim 512: OOM-adjacent, converged too early
- RMSNorm: -0.018 AUROC
- 1h and 3h temporal BCE: too sparse (most positions have zero positives)
- Huber dt loss: neutral, confounded by Phase-1 retrain
- Removing outcome head: -0.025 AUROC

---

## Setup

1. **Agree on a run tag** with the user (e.g. `apr1`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read all in-scope files:**
   - `session_summary.md` — full findings from session 1. Read this first.
   - `prepare.py` — fixed: data loading and evaluation metric. Do NOT modify.
   - `train.py` — primary edit target: model config, training settings, training loop.
   - `emr_model/transform_emr/embedder.py` — Phase-1 embedding model.
   - `emr_model/transform_emr/transformer.py` — Phase-2 GPT model.
   - `emr_model/transform_emr/loss.py` — loss functions.
   - `emr_model/transform_emr/schedulers.py` — auxiliary loss scheduling.
   - `emr_model/transform_emr/utils.py` — masking, targets, penalty calculation.
   - `diagnose.py` — diagnostic script on checkpoints. Run before proposing any experiment.
4. **Verify data exists**: `emr_model/data/source/temporal_data.csv` and `context_data.csv`.
5. **results.tsv already exists** — do not reinitialise. Append to it.
6. **Confirm and go.**

---

## Experimentation

**What you CAN modify:**
- `train.py` — `MODEL_CONFIG`, `TRAINING_SETTINGS`, loss schedules, training logic.
- `emr_model/transform_emr/*.py` — architecture changes.

**What you CANNOT modify:**
- `prepare.py` — fixed ground truth.
- `emr_model/data/` — fixed training data.

**The goal:** maximise `outcome_auroc` — mean per-outcome ROC-AUC across all 15 complication types. 0.5 = random, 1.0 = perfect. The score for each outcome type is: logit for that specific token at each position, label=1 if that complication occurs within 48h of current position's timestamp.

`val_bce_loss` is a secondary sanity signal only — do NOT use it as the keep/discard criterion.
Improvements in aux loss terms is also expected and a sanity signals.

**Simplicity criterion:** a small gain with lots of new code is suspect. Removing code while maintaining performance is always a win.

---

## Running an experiment

```bash
python train.py > run.log 2>&1
```

**Extract the result:**
```bash
grep "^outcome_auroc:\|^val_bce_loss:\|^peak_vram_mb:" run.log
```

If empty — crash. Inspect with `tail -n 50 run.log`.
**Timeout**: kill and treat as crash if no `---` summary after 90 minutes.

---

## Output format

```
---
outcome_auroc:    0.722311
val_bce_loss:     0.557637
phase2_best_val:  ...
phase2_epochs:    ...
total_seconds:    ...
peak_vram_mb:     ...
embed_dim:        256
n_layer:          4
n_head:           4
block_size:       512
num_params:       ...
```

---

## Logging results

Append every completed experiment to `results.tsv` (gitignored — do not commit it).

```
commit	outcome_auroc	val_bce_loss	memory_gb	status	description
```

- commit: 7-char hash
- memory_gb: peak_vram_mb / 1024, 1 decimal
- status: `KEEP`, `DISCARD`, or `CRASH`
- Use `0.000000` / `0.0` for crashes

---

## The experiment loop

**LOOP FOREVER — do NOT stop to ask for permission. The user is away.**

**Before the first experiment of any session: run the gradient diagnosis below.**

```
LOOP:
1. Inspect git state (branch, last commit, results.tsv)
2. Run: python diagnose.py > diag.log 2>&1
3. Read diag.log. Answer these three questions before proposing anything:
   a. Report 4: what are the actual calibrated lambdas for ce, outcome, penalty?
      If any lambda < 0.001, that loss term is near-silent — treat as a bug to fix first.
   b. Report 5: where do outcome tokens rank by gradient utility?
      If they rank in the bottom half of the vocabulary, the loss is not reaching them.
   c. Report 2: what is logit separation (sigmoid[pos] - sigmoid[neg])?
      If separation < 0.05, the model is barely distinguishing positive from negative positions.
   If (a), (b), or (c) show the model is gradient-starved, fix that before any architecture change.
4. Propose and implement ONE experiment targeting the highest-priority issue.
5. git commit
6. python train.py > run.log 2>&1
7. grep "^outcome_auroc:\|^val_bce_loss:\|^peak_vram_mb:" run.log
8. If empty: crash — tail -n 50 run.log, fix once if it's a bug, else log CRASH and move on
9. Append to results.tsv
10. If outcome_auroc improved: KEEP
11. If equal or worse: DISCARD — git reset --hard HEAD~1
```

**Embedder caching**: Phase-1 is skipped automatically when the checkpoint matches `(embed_dim, time2vec_dim, ctx_dim)`. Phase 1 only reruns when one of those three changes. Verify "Config unchanged — loading cached embedder" appears in run.log to confirm the cache was hit.

**Crashes**: fix typos/import errors and retry once. OOM or NaN loss — log as CRASH and move on.

---

## Research directions

Run `python diagnose.py` before every experiment. The three questions in the loop above are the decision gate.

### Priority 0 — Gradient starvation (investigate first, before any other experiment)

The model is known to train with very small gradients across all loss terms. This is the primary suspected bottleneck: the model may be converging to a mediocre local minimum not because the architecture is wrong, but because the loss signal is too weak to move the weights.

**Root cause of the starvation:** The scheduler calibrates all aux lambdas as `cap x (BCE_loss_at_calibration / aux_raw_loss)`. With 12h temporal BCE, BCE loss at calibration is ~0.002. Outcome raw loss is ~13. So:
- `lambda_outcome = 1.0 x 0.002 / 13 = 0.0002`
- `lambda_ce      = 0.20 x 0.002 / 3  = 0.0001`

These are not meaningful gradient contributors. Meanwhile BCE itself is small because temporal targets are dense — many positives per position, so no individual position drives a large loss spike. increasing BCE itself might improve everything, perhaps using different window.

**How to confirm gradient starvation:**
- **run.log** — grep for per-epoch loss components (`ce_loss`, `outcome_loss`, `penalty_loss`). If these are flat from epoch 3 onward while AUROC barely moves, the lambdas are too small to drive learning.
- **Report 4** (lambda calibration) — read the actual `[Phase2] Calibrated lambda` lines. Any lambda < 0.001 is near-silent.
- **Report 5** (gradient utility) — outcome tokens should rank in the top 20% of vocabulary by grad/occ. Bottom half = loss not reaching them.
- **Report 2** (logit separation) — `sigmoid[pos] - sigmoid[neg]` should be > 0.1 for a working model. Near zero means the model has learned almost nothing about outcome timing.

**A gradient-starved model cannot give reliable signal about architecture changes.** All experiments are confounded until this is fixed. Fix it first.

**Why outcome logits are systematically suppressed (and what to do):**
AUROC for outcome `o` reads only `logits[t, o]` {EM} never comparing it to other tokens' logits. This means AUROC doesn't directly penalise a model that keeps outcome logits at -5.0 as long as they are *relatively* higher near approaching complications. However there is an indirect failure mode: BCE training pushes frequent tokens (labs, vitals) up with large gradient every step, while outcome tokens are rare and get tiny gradient. The model learns to over-invest in frequent tokens and barely move outcome logit values. The result is outcome logits that are distinguishable across time (so AUROC > 0.5) but live in a very narrow range with poor separation.

`MaskedSetCE` (the ranking loss) is the correct structural fix: it explicitly compares target tokens against all other tokens within each step, forcing the model to rank outcomes *above* competing tokens when they are in the target set. This is exactly what AUROC measures. But its lambda is near-zero due to calibration starvation. **Fixing CE lambda is therefore the highest-leverage single change.**

**On inference and argmax:** outcomes are so rare that argmax never produces them {EM} at any given position a lab or vital will win. This is expected and not a problem for the current evaluation, which reads `logits[t, outcome_token]` directly from the LM head under teacher forcing. Argmax is irrelevant to AUROC. In clinical deployment the model would use threshold-based inference ("is `logits[t, KIDNEY_COMPLICATION] > threshold`?"), which is exactly what AUROC measures. Do NOT attempt to change `prepare.py` to fix this {EM} the evaluation design is correct. The lever is CE lambda and outcome head gradient, not the inference method.

### Priority 1 — Fix loss signal strength (once gradient starvation is confirmed above)

Fixes in order of invasiveness:

**A. Increase `aux_fraction_caps["outcome"]` from 1.0 to 10–20** (one line in `TRAINING_SETTINGS`).
10–20x more gradient toward the 12–48h window. Risk: backbone overfits to 15 outcome tokens — watch val_bce regression. Try 10 first, then 20.

**B. Increase `aux_fraction_caps["ce"]` from 0.20 to 2–5** (one line in `TRAINING_SETTINGS`).
`MaskedSetCE` is a *ranking* loss — directly aligned with AUROC (which is a ranking metric). Its lambda is also near-zero (~0.0007). BCE trains calibration; CE trains discrimination. Increasing the CE cap gives the ranking signal real magnitude.

**C. Extend BCE window: set `bce_window_hours` from 12 to 24** (one line in `TRAINING_SETTINGS`).
`bce_window_hours` controls the temporal BCE target window; `outcome_window_hi_hours` controls the upper bound of the outcome head window (default 48). The lower bound of the outcome window automatically tracks `bce_window_hours` so the two remain non-overlapping. Run diagnose.py Report 3 before and after to confirm coverage improves without becoming too dense.

**D. Token weights in the tokenizer (`dataset.py`).**
`token_weights` multiplies the focal-alpha for each token in BCE. The hardcoded `10.0`/`15.0` outcome boosts have been removed — outcome upweighting is now done exclusively via `aux_fraction_caps["outcome"]` in `TRAINING_SETTINGS`, which is calibrated and controllable. Current zero-weight tokens: `[PAD]`, `[MASK]`, `ADMISSION_TOKEN` (boundary markers, not prediction targets). `[NULL]` intentionally stays at 1.0 — it is a real sequence token (synthetic 3h gap marker) and the model must learn to predict quiet periods correctly. Do not set `[NULL]` to 0.

**E. Consider replacing the calibration approach entirely.**
The current scheme (lambda = cap × BCE/aux_raw at one epoch) ties all gradient magnitudes to BCE's scale. An alternative: set absolute lambdas directly in `TRAINING_SETTINGS` after reading the raw loss magnitudes from run.log, bypassing the calibration multiplier. This gives explicit control and avoids the compounding effect of small BCE making all aux terms tiny.

### Priority 2 — Architecture

**A. Temporal attention bias**
Time2Vec encodes each token’s absolute timestamp independently, in the embedding. But the attention weights between position i and position j are computed purely from content (Q·K). The model has no direct way to express “attend less to events 48h ago than to events 1h ago” as a function of real time. Adding a learned scalar bias g(Δt_ij) to the pre-softmax attention logits — `attn = softmax(QK^T/√d + g(Δt))` — gives each attention head an explicit temporal decay signal. This is complementary to Time2Vec and separate from legality masking: masking blocks illegal tokens; this bias shapes *how much weight* flows between legal events at different time distances.

Note on position tokens: the current `position_embed` in the embedder is NOT positional in the traditional sense — it encodes _START/_END interval state to support legality masking, not absolute sequence position. Therefore the attention bias must be computed from real `abs_ts` deltas, not token index differences. Implement in `CausalSelfAttention.forward`: compute pairwise Δt = abs_ts[i] - abs_ts[j] (already in normalized hours), apply a small learned function (e.g. a linear layer or a few learnable frequency bins), and add to the pre-softmax logits before masking.

**B. Temporal RoPE (replacing token-index RoPE)**
Standard RoPE rotates Q and K by *token index difference*. For EMR sequences this is nearly meaningless — token index 3 vs 4 might be 0.1h apart or 72h apart. **Temporal RoPE** instead rotates by the actual `abs_ts` difference between positions. This encodes relative real-world time directly inside the attention dot product, which is different from and complementary to Time2Vec: Time2Vec adds absolute time to each token’s embedding; temporal RoPE makes the *similarity* between two tokens decay with real time gap. This is a better inductive bias for irregular clinical time series than index-based RoPE. Implement by replacing the rotation angle from `position_index / 10000^(2i/d)` to `abs_ts_delta / T_scale^(2i/d)` in `CausalSelfAttention.forward`, tuning `T_scale` to the typical event spacing in hours.

**C. Token-type flag embeddings**
The token taxonomy (events, contexts, states, trends, patterns) is too semantically rich for a single flat category. However, a small set of *binary orthogonal flags* can add cheap structural signal without forcing cross-concept tokens into a single category. Proposed flags (each adds a learned vector in R^embed_dim, summed into the token embedding):
- `is_outcome` — 1 for the 15 complication tokens; 0 otherwise. Gives the model a direct signal that this token class is clinically special.
- `is_interval_marker` — 1 for _START and _END tokens. Reinforces interval structure in the embedding space, complementing legality masking.
- `is_trend` — 1 for directional tokens (INCREASE, DECREASE, etc.). Trends are predictively important and currently share embedding space with states.
- `is_treatment_pattern` - Patterns are a way to add intervals to the token space that tells something about the treatment quality based on medical guidelines. A pattern can be an interval of just an event in time (usually if action is missing) and it's values are True (done properly), Partially (done partially well), False (done badly).

Tokens that span multiple categories (e.g. insulin-pattern) get whichever flags apply. This is additive and cheap (~3×embed_dim extra parameters). Implement as a small `flag_embed` in `embedder.py` using a `nn.EmbeddingBag` or sum of per-flag embeddings with a pre-computed flag tensor per token id.

### Lower priority (after above are exhausted)

- `n_head=8` with `embed_dim=256` — 32 dims/head, not yet tested at this width
- `time2vec_dim` sweep — not tested at embed_dim=256
- `block_size` sweep — not tested
- GQA/MQA — more relevant at larger scale
