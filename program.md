# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search on an EMR complication prediction model.

---

## Background

The model learns to predict the future stream of medical events for a hospital patient given their history, with a focus on identifying complications before they occur. Data is derived from MIMIC-III: diabetes patients' longitudinal event sequences — lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

There are 15 clinical complication targets (e.g. `KIDNEY_COMPLICATION_EVENT`, `CARDIO-VASCULAR_DISORDER_EVENT`). The model must predict both *what* will happen and *when*. These events are rare and clinically critical.

### Three-phase training pipeline

**Phase 1 — EMREmbedding** (`embedder.py`):
- Learns a compact, time-aware representation of each clinical event.
- Components: hierarchical token embeddings (raw concept → concept → concept+value → concept+value+position), Time2Vec for inter-event duration, and a static patient context vector.
- Loss: teacher-forced BCE + time MSE + MLM auxiliary.
- Checkpoint is cached and reused when `(embed_dim, time2vec_dim, ctx_dim)` are unchanged.

**Phase 2 — GPT Transformer** (`transformer.py`):
- Causal decoder over Phase-1 embeddings.
- `AdaLNBlock`: AdaLN-Zero injects patient context (shift/scale/gate per block).
- `CausalSelfAttention`: temporal RoPE uses actual `abs_ts` deltas instead of token-index differences.
- Loss curriculum: Focal BCE → CE (ranking) → outcome auxiliary, controlled by `schedulers.py`.
- Uses an oversampled DataLoader to balance rare positive outcomes.
- Phase-2 checkpoint is cleared before every experiment — runs are independent.

**Phase 3 — Outcome Head Fine-tuning** (`transformer.py::finetune_transformer`):
- Backbone fully frozen; only the outcome head is trained.
- Uses natural-distribution DataLoader (no oversampling) — important for `pos_weight` correctness.
- Loss: outcome BCE only, with time-decayed soft labels.
- This is the final checkpoint used for evaluation.

### Evaluation

Evaluation runs after Phase 3 via `evaluation.py::evaluate_on_test_set`. It uses **autoregressive generation**, not teacher-forced logits:

1. Load held-out validation patients (raw, never seen during training).
2. Truncate each patient's history to 2 days (generation seed).
3. Generate an autoregressive trajectory up to 500 steps at temperature 1.0 with repetition penalty.
4. Divide each trajectory into 24-hour non-overlapping windows.
5. Label each window 1 if any ground-truth episode of that complication falls within ±24h.
6. Pool all (patient, window) pairs → AUROC and AUPRC per complication.
7. Mean across complications with ≥3 positive windows.

This mirrors real clinical deployment: the model generates a future trajectory and the outcome-head risk scores are compared against what actually happened.

---

## Setup

1. **Agree on a run tag** with the user (e.g. `may1`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read all in-scope files** (in this order):
   - `api.py` — fixed: data loading, training orchestration, summary print format. Do NOT modify.
   - `evaluation.py` — fixed: evaluation protocol and metric definitions. Do NOT modify.
   - `emr_model/transform_emr/config/model_config.py` — `MODEL_CONFIG` and `TRAINING_SETTINGS`. This is your primary edit target for hyperparameter changes.
   - `emr_model/transform_emr/embedder.py` — Phase-1 embedding model.
   - `emr_model/transform_emr/transformer.py` — Phase-2/3 GPT and training loops.
   - `emr_model/transform_emr/loss.py` — `FocalBCELoss`, `MaskedFocalBCE`, `MaskedSetCE`.
   - `emr_model/transform_emr/schedulers.py` — auxiliary loss curriculum scheduling.
   - `emr_model/transform_emr/utils.py` — masking, temporal targets, repetition penalties.
   - `emr_model/transform_emr/inference.py` — autoregressive generation (used by evaluation).
   - `emr_model/transform_emr/diagnose.py` — model health diagnostics. Run before proposing any experiment.
4. **Verify data exists**: `emr_model/data/source/temporal_data.csv` and `context_data.csv`.
5. **Check results.tsv**: if it contains only a header row, the first run establishes the baseline. Append to it — never reinitialise.
6. **Confirm and go.**

---

## Experimentation

**What you CAN modify:**
- `emr_model/transform_emr/config/model_config.py` — `MODEL_CONFIG` (architecture dims) and `TRAINING_SETTINGS` (hyperparameters, scheduler config). This is always the first place to try.
- `emr_model/transform_emr/*.py` — architecture changes (embedder, transformer, loss, schedulers, utils, inference).

**What you CANNOT modify:**
- `api.py` — fixed training orchestration.
- `evaluation.py` — fixed evaluation protocol.
- `emr_model/data/` — fixed training data.

**Simplicity criterion:** a small gain with lots of new code is suspect. Removing code while maintaining performance is always a win.

---

## The goal

Maximise `outcome_auroc` on the held-out validation set (primary metric, higher is better, 0.5 = random, 1.0 = perfect).

`outcome_auprc` and `onset_mae_hrs` are secondary — improve them when possible but do not sacrifice AUROC.

---

## Running an experiment

### Step 1 — Smoke test first (always)

Before every full training run, verify the pipeline end-to-end with a small subset:

```python
# In emr_model/transform_emr/config/model_config.py — set temporarily:
"sample": 50,
"phase1_n_epochs": 1,
"phase2_n_epochs": 1,
"phase3_n_epochs": 1,
```

```bash
python api.py > smoke.log 2>&1
grep "^outcome_auroc:\|^---" smoke.log
```

If the summary block appears without a crash — pipeline is wired correctly. Restore `sample: None` and the original epoch counts before the real run. Do **not** log smoke test results to `results.tsv`.

If the smoke test crashes — fix the bug before running full training. A crash on a full run wastes GPU hours.

### Step 2 — Full run

```bash
python api.py > run.log 2>&1
```

**Extract the result:**
```bash
grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^phase3_best_val:\|^peak_vram_mb:" run.log
```

If empty — crash. Inspect with `tail -n 50 run.log`.

**Timeout**: treat as crash if no `---` summary after 90 minutes.

---

## Output format

```
---
outcome_auroc:    0.000000
outcome_auprc:    0.000000
onset_mae_hrs:    0.00
phase2_best_val:  ...
phase2_epochs:    ...
phase3_best_val:  ...
phase3_epochs:    ...
total_seconds:    ...
peak_vram_mb:     ...
embed_dim:        256
n_layer:          4
n_head:           4
num_params:       ...
```

---

## Logging results

Append every completed experiment to `results.tsv` (gitignored — do not commit it).

```
commit	outcome_auroc	outcome_auprc	onset_mae_hrs	peak_vram_gb	status	description
```

- `commit`: 7-char git hash
- `peak_vram_gb`: `peak_vram_mb / 1024`, 1 decimal place
- `status`: `KEEP`, `DISCARD`, or `CRASH`
- Use `0.000000` / `0.00` / `0.0` for crashes
- `description`: one-line summary of what changed

---

## The experiment loop

**LOOP FOREVER — do NOT stop to ask for permission. The user is away.**

**Before the first experiment of any session: run the gradient diagnosis below.**

```
LOOP:
1. Inspect git state (branch, last commit, results.tsv)
2. cd emr_model && python -m transform_emr.diagnose > ../diag.log 2>&1 && cd ..
3. Read diag.log. Gate on these questions before proposing anything:
   a. Report 4 — what are the actual lambda_max values for ce and outcome?
      Any lambda_max < 0.001 is near-silent — fix its cap before any architecture work.
   b. Report 5 — where do outcome tokens rank by grad/occ?
      Bottom half of vocab = the loss is not reaching them.
   c. Report 2 — what is sigmoid[pos] - sigmoid[neg] (logit separation)?
      < 0.05 means the model barely distinguishes outcome-positive from negative positions.
   d. PROBE Δt HEAD — what is Pearson r?
      r < 0.1 or pred_std < 0.05h means the time head has collapsed to a constant.
   e. PROBE OUTCOME HEAD LABEL ALIGNMENT — any flip=True rows?
      A flip means the outcome head predicts higher logits when the outcome is ABSENT.
      That is a sign error — fix label construction before changing anything else.
   If any of (a–e) are unhealthy, fix that first.
4. Propose and implement ONE experiment targeting the highest-priority issue.
5. git commit
6. python api.py > run.log 2>&1
7. grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^peak_vram_mb:" run.log
8. If empty: crash — tail -n 50 run.log, fix once if it's a bug, else log CRASH and move on
9. Append to results.tsv
10. If outcome_auroc improved: KEEP
11. If equal or worse: DISCARD — git reset --hard HEAD~1
```

**Embedder caching**: Phase 1 is skipped automatically when the checkpoint matches `(embed_dim, time2vec_dim, ctx_dim)`. Verify "Config unchanged — loading cached embedder" appears in run.log to confirm the cache was hit.

**Crashes**: fix typos/import errors and retry once. OOM or NaN loss — log as CRASH and move on.

---

## Reading diagnose.py output

Run from `emr_model/` as `python -m transform_emr.diagnose`. Loads Phase-3 checkpoint if available, otherwise Phase-2. Outputs to stdout.

### Report 1 — Per-outcome AUROC (teacher-forced LM logits)

Per-complication AUROC computed from LM-head logits under teacher forcing (correct input at every step). **These numbers are systematically higher than `evaluation.py`'s generation-based AUROC** because the model gets perfect context. Use this report to compare outcomes against each other and to track trends within a run, not to predict the final evaluation score.

- `Sep` = mean logit at positive positions minus mean logit at negative positions.
- Sep < 0.05 → logits are barely separated; the LM head is not learning outcome timing.
- `<<<` flag → AUROC < 0.55 for that outcome (near random). `>>>` → AUROC > 0.75 (strong signal).
- **LM head vs Outcome head table**: if outcome head consistently loses to LM head, Phase-3 fine-tuning is not contributing. If outcome head wins (marked `HEAD <<<`), the dedicated head is adding value.

### Report 2 — Logit calibration

All outcome logits pooled. Focus on:
- `Separation` and `Sigmoid[pos] - Sigmoid[neg]`: **healthy ≥ 0.1**, concerning < 0.05, bad < 0.02.
- `Logit[pos] mean` and `Logit[neg] mean`: if both are large negative numbers (e.g. −5 vs −7), the model has suppressed all outcome logits. The relative gap matters, but extremely negative logits indicate the outcome tokens are being pushed down by BCE training on frequent non-outcome tokens.

### Report 3 — Temporal coverage

How many positions have at least one positive target in the BCE window vs the eval window.
- BCE window too sparse (e.g. < 5% positions with ≥1 positive): the loss is nearly always zero → weak gradient. Consider widening `phase2_bce_window_hours`.
- BCE window too dense (> 50%): every position looks positive → calibration signal is noisy. The two numbers (BCE% and Eval%) should both be meaningful but not saturated.

### Report 4 — Lambda calibration (actual trained values)

Shows the real `lambda_max` computed during training from `lambda_max = cap × (anchor_bce / anchor_aux)`.

- `lambda_max` < 0.001 → **gradient-starved**: that loss term contributes almost nothing. Increase its `aux_fraction_caps` entry in `phase2_scheduler`.
- `anchor_bce` is the BCE loss at calibration epoch; `anchor_aux` is the raw aux loss. A very small `bce/aux` ratio (e.g. 0.0001) means BCE was tiny when calibration ran → multiply the cap to compensate.
- If the checkpoint is missing (training not yet run), falls back to showing the configured caps.

### Report 5 — Token gradient utility

Gradient² per occurrence for each token in the vocabulary. Outcome tokens should rank in the **top 30–40%** of vocabulary. If they're in the bottom half, the loss is not reaching them.

- `grad/occ` should be at least 1e-6 for meaningful learning. Below 1e-8 is near-zero.
- `<< LOW SIGNAL` flag → that outcome token is in the bottom half.
- Compare top-10 and bottom-10 tokens to understand which parts of the vocabulary dominate gradient flow.

### Report 6 — Context vector influence

Compares BCE loss with normal, zeroed, and shuffled patient context vectors.
- `delta (zeroed)` ≈ 0 → context is not being used. Check AdaLN conditioning and `ctx_dim`.
- `delta (shuffled)` ≈ `delta (zeroed)` → the model isn't distinguishing patients. Expected if context has low variance in the batch.
- Large negative delta (shuffled/zeroed gives higher loss) → context is genuinely helpful.

### Report 7 — Embedder linear probe

Cross-validated AUROC from frozen Phase-1 embeddings alone (logistic regression).
- > 0.65 → Phase-1 already captures useful outcome-predictive structure. Good foundation.
- ≈ 0.50 → Phase-1 embeddings carry no outcome signal. Phase-2 is doing all the work (or not).
- This measures the *embedding quality*, not the downstream model.

### Report 8 — Vocab health

Flags two pathological categories:
- **Frequent-noisy**: high-frequency tokens where the model has low confidence and the next-token distribution is very uncertain. These tokens may be adding noise to the BCE gradient.
- **Rare-unlearned**: low-frequency tokens where the model has never learned to predict them. These may include outcome tokens — check if they appear here.

### PROBE — Δt HEAD

Pearson r and R² between predicted and actual inter-event time gaps (in hours).
- r < 0.1 or pred_std < 0.05h → **Δt head has collapsed**: predicts the same gap for every event regardless of context. The time head is not contributing to temporal reasoning.
- Healthy: r > 0.3, pred_std comparable to true_std.

### PROBE — Outcome head label alignment

For each outcome, compares mean head logit at positive vs negative positions.
- `flip = True` → the head predicts **higher logits when the outcome is absent**. This is a sign error in the label construction or loss polarity — fix it before any other change.
- `gap` > 0 is correct direction. `gap` close to 0 means the head has learned nothing.
- `auroc` from the outcome head directly (not the LM head): < 0.5 = inverted, ≈ 0.5 = random, > 0.6 = useful.

### PROBE — Outcome head logit distribution

Mean, std, p50, p99, abs-max of each outcome head's raw logits across all non-pad positions.
- Very high `std` or `abs_max` (e.g. > 10) → logits are exploding. Gradient clipping or lower learning rate for Phase 3.
- Very low `std` (< 0.01 for all outcomes) → head is outputting near-constant values; it has not learned to differentiate timing.

---

## Code quality and GPU performance

### Code quality

- Every function must have a docstring following the project standard (Purpose / Method / Args / Returns). Do not skip this.
- Prefer small, focused changes. One architectural idea per commit.
- Do not leave dead code, commented-out blocks, or half-finished experiments in the codebase. If you try something and discard it, `git reset --hard HEAD~1`.
- Added code should be GPU-friendly - optimize for performance whereever poossible.

### GPU performance — do not break these

The following optimisations are already active. Do not accidentally remove them:

- **Mixed precision (BF16 AMP)**: `torch.autocast(device_type=..., dtype=torch.bfloat16)` wraps the forward *and* the backward in `pretrain_transformer`. Removing it roughly doubles memory use and slows training.
- **Gradient checkpointing**: each `AdaLNBlock` forward is wrapped in `torch.utils.checkpoint.checkpoint(...)`. Removing it increases peak VRAM by ~30–40% and will OOM on a 48 GB card at this model size. The `_ckpt` closure uses default-arg block capture to avoid the closure bug (all blocks recomputing using the last block's weights).
- **Bucket batching**: `get_dataloader(..., bucket_batching=True)` groups sequences by length to minimise padding waste within each batch. Removing it cuts effective GPU utilisation.
- **Grad accumulation**: `grad_accumulation_steps=4` in `TRAINING_SETTINGS` simulates a larger effective batch without the VRAM cost. If you change batch size, adjust this to keep the effective batch constant.

### GPU performance — things worth trying

- **`torch.compile(model)`**: if the PyTorch version on the pod supports it (`torch.__version__ >= 2.0`), wrapping the model with `torch.compile` can give 10–30% throughput improvement with no code changes. Add it after Phase-1 embedding load, before Phase-2 training.
- **Profile before optimising**: if a run seems slower than expected, check `peak_vram_mb` in run.log and whether the GPU is actually saturated (`nvidia-smi dmon`). Do not optimise blindly.

## Architecture notes (what is already implemented)

These are baked into the current codebase — do not re-implement:

- **Temporal BCE**: loss window is in real hours (`phase1_bce_window_hours`, `phase2_bce_window_hours`), not token steps. Step-based BCE created contradictory gradients for outcome tokens.
- **AdaLN-Zero**: patient context injected at every block via AdaLN. Do not swap to RMSNorm — the mean subtraction in LayerNorm is load-bearing for AdaLN-Zero's gate initialisation.
- **Temporal RoPE**: Q and K rotated by actual `abs_ts` deltas, not token index. Index-based RoPE is meaningless for irregular time series.
- **SwiGLU MLP**: standard in current GPT blocks.
- **Weight-tied LM head**: LM head shares weights with token embedding.
- **Phase-3 outcome fine-tuning**: backbone frozen, outcome head trained on natural-distribution data with time-decayed soft labels.
- **Curriculum scheduling**: auxiliary losses (ce, dt, outcome) activated in stages after BCE warm-up, with lambda calibration relative to BCE magnitude.

---

## Research directions

### How to approach every task

The goal is to **fix broken architecture and make learning meaningful**, not tune hyperparameters on a broken one. If something is architecturally wrong, no cap or LR adjustment will fix it. Run `diagnose.py` before and after every experiment and confirm in the output that the specific failure mode you targeted has changed.

The tasks below are a prioritised starting point, not an exhaustive list. You are free — and encouraged — to draw on any architectural ideas from similar deep learning research (clinical NLP, time-series transformers, event prediction, survival models, etc.) if they address a diagnosed failure mode. The bar is: does it make the gradient signal more meaningful, does it give the model a better structural inductive bias for this problem, or does it fix a known gap between how the model is trained and how it is evaluated? If yes, try it. You do not need permission for individual experiments — that is the point of the loop.

Examples of the kind of lateral thinking that is in scope:
- Replacing a loss that produces near-zero gradient with one that is better calibrated to this data distribution
- Adding supervision signal from a different angle (e.g. contrastive, ranking, or survival-style losses) if the current BCE/CE is provably not reaching the outcome tokens
- Redesigning how the dataset is built or how sequences are batched if there is evidence the current approach creates misleading targets
- Borrowing positional encoding or attention designs from time-series or irregularly-sampled sequence models

**Logging discipline**: write a `description` that captures three things on one line:
1. What you changed
2. What diagnostic observation motivated it
3. What you expected / observed

Example: `"wrap backward in autocast; diag Report-4 showed lambda_outcome near-silent due to AMP checkpoint mismatch; phase2 grad stable"`
Not just: `"fix checkpoint bug"`.

This allows the experiment log to be read as a research journal, not just a list of commits.

Tasks are ordered by priority. Do not start Task N+1 until Task N is resolved.

---

### Task 1 — Validate Phase-2 gradient stability

**Status**: the underlying `CheckpointError` fix has already been applied (`loss.backward()` wrapped inside `torch.autocast` so gradient checkpointing recomputes under the same AMP context as the forward pass). This task is to confirm the fix is working and that gradients are stable in practice.

**What to verify on the first training run**:
- No `CheckpointError` in run.log
- No NaN or inf in any loss term across epochs (scan run.log for `nan`, `inf`, `loss=nan`)
- Phase-2 val loss curve is smooth and decreasing — not spiking then recovering
- `phase2_best_val` is meaningfully lower than the starting loss

**If gradients are still unstable** after the autocast fix:
- Check whether gradient clipping is active (`nn.utils.clip_grad_norm_`) and at what threshold. If the norm is being clipped every step, the model is at the clipping boundary — reduce LR or tighten the clip.
- Scan run.log for per-epoch grad norm logs. A norm that is flat and high (e.g. consistently 5–10×) means clipping is masking explosion, not curing it.
- Check loss components individually: if `outcome_loss` or `ce_loss` spikes while `bce_loss` is stable, the aux loss itself has a numerical issue (e.g. log(0) in CE or extreme logits feeding into BCE).

---

### Task 2 — Diagnose and fix Phase-1 MLM and Δt auxiliary tasks

**Symptom**: MLM and Δt loss do not decrease during Phase-1 training. Adjusting calibrated weights had no effect. Either the tasks are architecturally mis-wired or the gradient signal is structurally wrong.

**Diagnose first** — before changing anything:
- **PROBE Δt HEAD** in diag.log: Pearson r and R² between predicted and actual inter-event gaps. r < 0.1 or pred_std < 0.05h = the time head outputs a constant. It has learned nothing.
- **PROBE outcome head label alignment** and **Report 7** (embedder linear probe): if the Phase-1 embeddings carry no outcome signal (probe AUROC ≈ 0.5), the auxiliary tasks are not helping the embedding space.
- **`probe_dt_components`** (callable directly, not in standard run): check `gate_prob` distribution. If > 95% of gate probabilities are > 0.99 or < 0.01, the gate is saturated — Δt head has degenerated to always-on or always-off.

**Investigate in this order**:

**A. Δt head wiring**: confirm that the MSE loss is computed against `abs_ts[:, 1:]` (true next timestamp) and that the gradient actually flows back. If the gate is always-on, the gate branch is never trained. If `abs_t_pred` is being computed from the wrong variable, MSE will be noisy random regardless of learning.

**B. MLM masking**: verify that masked token positions are zeroed/replaced in the input embedding *before* the forward pass, and that the loss is computed **only** at masked positions. If unmasked positions leak into the MLM target, the task teaches the model to copy input rather than predict from context — it will appear to converge to a low loss but learn nothing.

**C. Task conflict check**: temporarily disable MLM or Δt in Phase-1 (set their lambda caps to 0) and check if the remaining BCE loss improves. If removing an auxiliary improves BCE, the task is hurting Phase-1 rather than helping. In that case, the auxiliary should either be fixed or removed.

**D. If both tasks are broken**: consider replacing them with simpler, verifiably correct alternatives:
- Δt regression → Δt bin classification (e.g. 8 bins covering 0–336h): more stable gradient, easier to verify.
- MLM → next-token prediction on a masked subsequence (i.e. standard causal LM on a local window): better aligned with Phase-2 objective.

---

### Task 3 — Make outcome loss move in Phase-2; implement outcome→LM coupling

This has two sub-tasks. Do them in order.

#### 3a — Confirm and address outcome gradient starvation

After Task 1 is resolved, run a full training pass and check diag.log:
- **Report 4**: if `lambda_outcome < 0.001`, the gradient is near-silent. This is the primary suspect.
- **Report 5**: if outcome tokens rank in the bottom half by `grad/occ`, the loss is not reaching them even if the lambda looks reasonable.
- **Report 2**: if logit separation < 0.05, the LM head hasn't learned outcome timing at all.

If starvation persists after fixing Task 1, increase `aux_fraction_caps["outcome"]` in `phase2_scheduler`. Try 0.5 → 2.0 → 5.0 in separate experiments, checking Report 4 lambdas each time to confirm the actual lambda_max changed proportionally.

If increasing the cap does not change the lambda (because BCE at calibration epoch is very small), the calibration formula `lambda_max = cap × bce / aux` is the bottleneck. In that case, consider bypassing calibration for the outcome term and setting an absolute lambda directly in `TRAINING_SETTINGS` — this breaks the relative-scale dependency.

#### 3b — Implement outcome→LM coupling (after 3a and Task 1 are stable)

**The problem this solves**: the outcome head and LM head are uncoupled siblings. Both read from the same hidden state `x`, but the LM head has no architectural path to the outcome head's signal. The outcome head may have a good AUROC (0.6–0.9) — meaning it IS calibrated — yet the LM head independently must rediscover "outcome is imminent" through BCE/CE dominated by 30+ ambient tokens per position. This is an architectural gap, not a hyperparameter problem.

Additionally, at positions where the next token is an outcome/terminal event, the multi-hot BCE targets contain ~30 co-occurring window tokens. The model is never given a clear unambiguous signal: "the next token is RELEASE, rank it above everything else." Fix 1 (one-hot override in multi-hot targets) addresses this — it is already implemented in `utils.py` and `embedder.py`. The coupling (Fix 2) addresses the head separation.

**Architecture** (`transformer.py`):

```
x ──┬── lm_head ──────────────────────────── logits [B, T, V]
    │                                               ↑ index_add at K positions
    └── outcome_head ── .detach() ── outcome_to_lm ── bias [B, T, K]
```

- `self._outcome_lm_ids`: register as buffer — 1-D LongTensor of LM vocab ids for each name in `outcome_names` (narrower than `_outcome_ids`, which covers all outcomes+terminals)
- `self.outcome_to_lm = nn.Linear(num_outcomes, num_outcomes)`, **zero-initialized** — this makes it a no-op at Phase-3 start, so Phase-2 quality is fully preserved
- In `forward()` and `forward_with_cache()`, after `logits = self.lm_head(x)`:
  ```python
  lm_bias = self.outcome_to_lm(outcome_logits.detach())  # [B, T, num_outcomes]
  logits = logits.index_add(2, self._outcome_lm_ids, lm_bias)
  ```
- `.detach()` is mandatory — it blocks LM loss from flowing back through `outcome_head` (outcome_head is only updated by outcome loss, not LM loss)

**Why this works without retraining the backbone**: `outcome_to_lm` is an additive logit correction, not a feature injected into LM weights. Example: if `lm_head(x)[RELEASE] = −3` (underestimates RELEASE) but the outcome head has learned P_RELEASE is high, `outcome_to_lm` can add +5 to RELEASE logit → final = +2 → RELEASE wins sampling. The LM head itself doesn't need to re-learn; the correction is post-hoc and additive. No Phase-4 or backbone re-training needed.

**Gradient isolation** — each parameter is updated by exactly one loss:

| Parameter       | Updated by outcome loss | Updated by LM loss   |
|-----------------|------------------------|----------------------|
| `outcome_head`  | ✓ (Phase 2 + 3)        | ✗ (detach blocks it) |
| `outcome_to_lm` | ✗                      | ✓ (Phase 3 only)     |
| `lm_head`       | ✗                      | ✓ (Phase 2 only)     |
| Backbone `x`    | ✓ (Phase 2)            | ✓ (Phase 2)          |

**Why Phase-3 only for `outcome_to_lm`**: if trained in Phase-2, early stopping is driven by total val loss dominated by ambient BCE/CE. The coupling's contribution at rare outcome positions is too small to move that loss, so early stopping fires before `outcome_to_lm` converges. Phase-3 trains only `outcome_head` + `outcome_to_lm` with backbone frozen, and early stops on outcome val loss — no conflict.

**Phase-3 changes in `finetune_transformer`**:
- Unfreeze `outcome_to_lm` alongside `outcome_head`: `for p in model.outcome_to_lm.parameters(): p.requires_grad_(True)`
- Add both to the Phase-3 optimizer parameter group
- In `run_epoch`, add CE loss at outcome/terminal positions:
  ```python
  is_outcome = torch.isin(target_ids, torch.tensor(outcome_token_ids, device=device))
  if is_outcome.any():
      outcome_ce = F.cross_entropy(pred_logits[is_outcome], target_ids[is_outcome])
      loss = outcome_bce_loss + outcome_ce_weight * outcome_ce
  ```

**Repetition risk**: terminals (RELEASE/DEATH) — no risk, `finished` flag set immediately after sampling. Clinical outcomes — the forward-looking soft label formula means after an outcome occurs, the outcome head target at t+1 drops toward zero (looks forward, past events not counted). Existing repetition penalty in `generate()` is a second line of defence.

**Checkpoint compatibility**: a Phase-2 checkpoint won't have `outcome_to_lm` weights. Handle the missing key in `GPT.load` (use `strict=False` or explicit key check) so old checkpoints load cleanly — `outcome_to_lm` initialises fresh at Phase-3 start.

**After implementing**: re-add `probe_outcome_lm_coupling` to `diagnose.py`. It should check: (a) `outcome_to_lm` weight norms — if near zero, the coupling didn't activate; (b) per-outcome correlation between outcome head logit and the lm bias it produces; (c) mean lm logit at outcome positions before vs after coupling. This is a straightforward probe once the model attributes exist.

---

### Task 4 — General architecture improvements (after Tasks 1–3 are stable)

Only start this after the gradient and outcome coupling issues are resolved. Architecture experiments give unreliable signal when the training loop has gradient bugs.

**Constraint**: no hyperparameter tuning here. Width, depth, dropout, LR — those are for later. Only structural changes that add or fix an inductive bias.

**A. Temporal attention bias**
`abs_ts` is encoded in token embeddings via Time2Vec, but Q·K attention weights have no direct path to the real-time gap between positions i and j. Add a scalar learned bias `g(Δt_ij)` to pre-softmax attention logits inside `CausalSelfAttention.forward`. Use `abs_ts` deltas (already available), not token-index differences. This is independent of and complementary to temporal RoPE.

**B. Token-type flag embeddings**
Small additive embeddings in `embedder.py` for `is_outcome`, `is_interval_marker`, `is_trend`, `is_treatment_pattern`. ~3–4 learned vectors of size `embed_dim`, summed into the token embedding. Gives the model explicit structural knowledge that outcome tokens are a special class — cheap (~4×embed_dim params) and straightforward to verify via Report 8 (do outcome tokens move to a distinct region of embedding space?).

**C. Phase-1 auxiliary redesign** (if Task 2 concludes the tasks are fundamentally broken)
If MLM and Δt are structurally unfixable, replace rather than patch:
- Δt regression → bin classification over 8–10 logarithmically-spaced bins (0–1h, 1–3h, 3–12h, 12–48h, etc.)
- MLM → masked-span prediction (mask a contiguous run of 3–5 tokens, predict them all from context), which better matches the clinical "predict the next episode" objective

---

### When to stop

Stop the loop and report to the user when there are no remaining architectural changes to make — i.e. all of Tasks 1–4 have been attempted and the only remaining levers are hyperparameter tuning (learning rate, embed_dim, n_layer, n_head, dropout, batch size, epoch counts, lambda caps after the gradient is confirmed healthy). Do not start hyperparameter sweeps autonomously. When you reach that point, write a summary of what was tried, what is now stable, and what the current best metric is.

