# autoresearch — EMR Event Prediction

Autonomous architecture and hyperparameter search for my thesis's EMR
complication-prediction model, adapted from Karpathy's
[autoresearcher](https://github.com/karpathy/autoresearch) framework.

An AI agent drives a Karpathy-style autonomous loop: edit
`emr_model/transform_emr/`, train, evaluate on a held-out test split, KEEP or
DISCARD per the rules in `program.md`, log to `results/*.tsv`, repeat.

Final best model on MIMIC-IV: **M-256** — AUROC 0.915, AUPRC 0.630, onset MAE
64.98 h. See `status.md` for the full sweep report.

---

## Repository layout

```
api.py                       fixed: data load, training orchestration, eval
evaluation.py                fixed: autoregressive eval (AUROC / AUPRC / MAE)
program.md                   instructions for the autonomous agent
status.md                    sweep narrative report
analysis.ipynb               plots the journey + the size comparison
results/
  results-architecture optimization.tsv   full per-experiment ledger
  results-hyperparameters sweep.tsv       final architecture grid (M-256 family)
emr_model/
  transform_emr/
    config/
      model_config.py        MODEL_CONFIG + TRAINING_SETTINGS (agent edits this)
      dataset_config.py      paths, tokens, USE_QA_DATA flag
    embedder.py              Phase-1 EMREmbedding
    transformer.py           Phase-2/3 GPT + finetune
    dataset.py               DataProcessor, EMRTokenizer, dataloaders
    loss.py / schedulers.py / utils.py / inference.py / diagnose.py
  data/source/               temporal_data.csv + context_data.csv (gitignored)
  checkpoints/                phase{1,2,3}/ckpt_best.pt + tokenizer + scaler
```

`api.py` and `evaluation.py` are the fixed contract. The agent only edits
`model_config.py` (primary) and architecture files under `emr_model/transform_emr/`.

---

## Three-phase model

- **Phase 1 — `EMREmbedding`** — hierarchical token embeddings (raw → concept →
  concept+value → position), Time2Vec for inter-event time, static patient
  context. Loss: teacher-forced BCE + Δt MSE.
- **Phase 2 — `GPT`** — causal decoder over Phase-1 embeddings with AdaLN-Zero
  and temporal RoPE. Curriculum: soft-kernel BCE → next-token CE + Δt → pairwise
  ranking on the outcome head.
- **Phase 3 — outcome-head fine-tune** — backbone differential LR; outcome BCE
  on time-decayed soft labels + P3 ranking. Phase-3 checkpoint is the
  deployed model.

Evaluation runs autoregressive generation from a k-day seed (default k=2),
divides the trajectory into 24 h windows, and computes per-complication AUROC /
AUPRC / onset-MAE on the held-out 15 % test split (split by `PatientId`, seed=42).

---

## Running locally

```bash
pip install -e .
# Place CSVs at emr_model/data/source/{temporal_data,context_data}.csv

# Smoke test (50 patients, 1 epoch per phase, ~1 min on CPU)
# In emr_model/transform_emr/config/model_config.py set sample=50 and phase{1,2,3}_n_epochs=1
python api.py > smoke.log 2>&1
grep "^outcome_auroc:\|^---" smoke.log

# Full run (default config; restore sample=None and original epoch counts)
python api.py > run.log 2>&1
grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^peak_vram_mb:" run.log
```

Outputs go to a final summary block after `---`. The pre-trained final
model is in `emr_model/checkpoints/`; `api.py` will load and reuse it
when the embedder config matches.

---

## Running on a RunPod GPU pod

The agent runs autonomously inside `tmux` so it survives SSH drops.

**One-time setup on a fresh pod:**

```bash
# SSH in (RunPod gives you the host/port on the Connect page)
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519

# Install tmux + Node 20 + Claude Code
apt-get update && apt-get install -y tmux
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs
npm install -g @anthropic-ai/claude-code

# Clone + install Python deps
cd /workspace
git clone git@github.com:shaharoded/AutoResearcher-TransformEMR.git autoresearch
cd autoresearch
pip install -e .

# Create a non-root user (Claude refuses --dangerously-skip-permissions as root)
useradd -m -s /bin/bash agent
cp -r /root/.ssh /home/agent/ && chown -R agent:agent /home/agent/.ssh
chmod -R a+rwX /workspace/autoresearch
```

**SCP the data files (from local PowerShell):**

```powershell
scp -P <PORT> -i ~/.ssh/id_ed25519 emr_model\data\source\temporal_data.csv root@<HOST>:/workspace/autoresearch/emr_model/data/source/
scp -P <PORT> -i ~/.ssh/id_ed25519 emr_model\data\source\context_data.csv  root@<HOST>:/workspace/autoresearch/emr_model/data/source/
```

**Start the agent (each session):**

```bash
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519
su - agent
cd /workspace/autoresearch
tmux new -s claude
claude --dangerously-skip-permissions
```

In the Claude prompt, kick off the loop with something like:

> Read `program.md`. We are on branch `autoresearch-optimization`. Run the experiment loop autonomously — smoke test, full run, KEEP/DISCARD, update `status.md` + `results/*.tsv` after every meaningful step, and commit & push to the branch.

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t claude`.

**Monitoring from your laptop:**

```bash
# Read the live progress without disturbing the agent
ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519 "cat /workspace/autoresearch/status.md"

# Or git pull whenever the agent has pushed
git pull --ff-only
```

**Before stopping the pod:** push the branch from the pod so nothing is lost on
container disk. SCP off `emr_model/checkpoints/` if you want to keep the trained
weights (gitignored, too large for git).

---

## Metrics

All on the held-out 15 % test split via autoregressive generation, per
complication, then averaged across complications with ≥ 3 positive windows.

- **`outcome_auroc`** — primary, ↑. 0.5 = random, 1.0 = perfect.
- **`outcome_auprc`** — secondary, ↑. Average precision; sensitive to false alarms.
- **`onset_mae_hrs`** — tertiary, ↓. Hours between predicted-peak step and
  ground-truth first occurrence of the complication.
