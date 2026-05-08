# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search for a clinical EMR sequence model.

## What this is

An AI agent autonomously experiments on an EMR (Electronic Medical Record) complication-prediction model overnight. Each experiment modifies files under `emr_model/transform_emr/`, trains for a fixed epoch budget, checks if the held-out evaluation metric improved, keeps or discards, and repeats — logging every result in `results.tsv`.

The architecture is a three-phase pipeline (`emr_model/`):

- **Phase 1 — EMREmbedding** — learns compact, time-aware representations of clinical events using hierarchical token embeddings, Time2Vec, and static patient context.
- **Phase 2 — GPT Transformer** — a causal decoder over Phase-1 embeddings; predicts the next clinical event and learns outcome timing via a curriculum of auxiliary losses.
- **Phase 3 — Outcome Fine-tuning** — backbone frozen; outcome head fine-tuned on natural-distribution data to sharpen complication risk scores.

The training data is derived from MIMIC-III and contains diabetes patients' longitudinal event sequences including lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

## Files that matter

| File | Role |
|------|------|
| `api.py` | **Fixed.** Data loading, training orchestration, final evaluation. Do NOT modify. |
| `evaluation.py` | **Fixed.** Post-training evaluation metrics (AUROC/AUPRC/MAE). Do NOT modify. |
| `emr_model/transform_emr/config/model_config.py` | **Agent edits this.** `MODEL_CONFIG` (architecture dims) and `TRAINING_SETTINGS` (hyperparameters, schedulers). |
| `emr_model/transform_emr/` | Model source — agent may modify for architecture changes. |
| `program.md` | **Human edits this.** Instructions and research context for the autonomous agent. |
| `emr_model/data/source/` | Training data (fixed). |
| `results.tsv` | Experiment log (gitignored, untracked by git). |

## Metrics

All metrics are computed by `evaluation.py::evaluate_on_test_set` on the held-out validation set via autoregressive generation — never modified by the agent.

**Primary — `outcome_auroc` (higher is better)**
Mean per-complication AUROC from pooled episode-level AUC. For each generated trajectory, time is divided into 24-hour non-overlapping windows; a window is labelled positive if any ground-truth episode of that complication falls within ±24h of the window edges. AUROC is computed from (window, score) pairs pooled across all patients, then averaged across complications with at least 3 positive windows. Random = 0.5, perfect = 1.0.

**Secondary — `outcome_auprc` (higher is better)**
Mean per-complication average precision (AUPRC) from the same pooled window evaluation. Reflects precision across recall thresholds — more sensitive to false alarms than AUROC.

**Tertiary — `onset_mae_hrs` (lower is better)**
Mean absolute error between the predicted onset time (generated step with peak `P_outcome`) and the ground-truth first occurrence of that complication, in hours. Averaged across patients where the complication occurred.

## Design choices

- **Immutable contract.** `api.py` and `evaluation.py` are the fixed ground truth. The agent edits `model_config.py` (hyperparameters) and files under `emr_model/transform_emr/` (architecture).
- **Embedder caching.** Phase 1 is skipped automatically when `(embed_dim, time2vec_dim, ctx_dim)` are unchanged — saving ~30 min per experiment.
- **Fresh Phase 2 and Phase 3 per experiment.** Those checkpoints are cleared before each run so experiments are independent.
- **Phase 3 for outcome alignment.** The backbone is frozen and only the outcome head is fine-tuned on natural-distribution data — prevents oversampling bias from contaminating risk scores.
- **Generation-based evaluation.** The final metrics are computed from autoregressive generation (not teacher-forced logits), matching real clinical deployment: the model generates a trajectory from 2 days of seed data and its outcome-head risk scores are evaluated against ground-truth future episodes.
- **Epoch budget with early stopping.** Training terminates when validation stops improving (patience configurable), bounded by a maximum epoch count.
- **Data sampling.** `TRAINING_SETTINGS["sample"]` controls how many patients to use. Set to `None` for full training runs; set to a small integer (e.g. `50`) for quick smoke-tests.

## Project structure

```
api.py                  fixed: training orchestration (do not modify)
evaluation.py           fixed: evaluation metrics (do not modify)
program.md              agent instructions (human modifies)
analysis.ipynb          experiment analysis / visualisation
pyproject.toml          dependencies
results.tsv             experiment log (gitignored, untracked)
run.log                 last experiment output (gitignored)
emr_model/
  transform_emr/
    config/
      model_config.py   MODEL_CONFIG + TRAINING_SETTINGS (agent modifies)
      dataset_config.py data paths and special tokens (fixed)
    embedder.py         Phase-1 EMREmbedding
    transformer.py      Phase-2/3 GPT + finetune_transformer
    dataset.py          tokenizer and dataloader
    loss.py             loss functions
    schedulers.py       auxiliary loss curriculum scheduling
    inference.py        autoregressive generation (used by evaluation.py)
    utils.py            masking, targets, penalties
    diagnose.py         model health checks
  data/
    source/
      temporal_data.csv      patient event sequences
      context_data.csv       patient context features
  checkpoints/          saved model weights (gitignored)
```

---

## RunPod: SSH setup

The agent runs on a RunPod GPU pod (A40, 48 GB VRAM).

### First-time SSH key setup (local, run once)

```powershell
ssh-keygen -t ed25519 -C "runpod" -f "$env:USERPROFILE\.ssh\runpod_ed25519"
```

Copy the public key to your clipboard:
```powershell
Get-Content "$env:USERPROFILE\.ssh\runpod_ed25519.pub" | clip
```

Paste it into **RunPod → Settings → SSH Public Keys**.

### Add the pod to your SSH config (local)

Edit `~/.ssh/config` (create if it doesn't exist) and add:

```
Host runpod
    HostName 194.68.245.49
    Port     22036
    User     root
    IdentityFile ~/.ssh/runpod_ed25519
```

Replace the `HostName` and `Port` each time you start a new pod (RunPod shows these on the pod's Connect page under "SSH over exposed TCP").

### Connect from VSCode

1. Install the **Remote - SSH** extension.
2. Press `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → select `runpod`.
3. Once connected: **File → Open Folder** → `/workspace/autoresearch`.
4. Open a terminal and run `claude` to start the agent.

### Connect from PowerShell

```powershell
ssh runpod
```

Or without the config entry:
```powershell
ssh -i "$env:USERPROFILE\.ssh\runpod_ed25519" -p 22036 root@194.68.245.49
```

---

## RunPod: pod lifecycle

### Starting a session

1. Go to [runpod.io](https://runpod.io) → **My Pods** → start the pod (or create a new one).
2. Wait for status to show **Running**.
3. Click **Connect** → copy the "SSH over exposed TCP" address.
4. Update `~/.ssh/config` with the new `HostName` and `Port` if they changed.
5. SSH in and resume work.

### Before stopping the pod (save your work)

The pod's **container disk** is ephemeral — it is wiped when the pod is stopped or deleted. Always save before stopping.

**What needs saving** (copy to your local machine or a RunPod network volume):

| What | Why |
|------|-----|
| `results.tsv` | The full experiment log — not in git |
| `emr_model/checkpoints/tokenizer.pt` | Tokenizer cache — slow to rebuild |
| `emr_model/checkpoints/phase1/ckpt_best.pt` | Cached embedder — saves ~30 min per experiment |
| `run.log` | Last run output (optional) |

**Copy everything to local with SCP** (run from PowerShell locally):

```powershell
# Create a local backup folder
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\runpod_backup"

# Copy the files
scp -P 22036 root@194.68.245.49:/workspace/autoresearch/results.tsv "$env:USERPROFILE\runpod_backup\"
scp -P 22036 root@194.68.245.49:/workspace/autoresearch/emr_model/checkpoints/tokenizer.pt "$env:USERPROFILE\runpod_backup\"
scp -r -P 22036 root@194.68.245.49:/workspace/autoresearch/emr_model/checkpoints/phase1 "$env:USERPROFILE\runpod_backup\"
```

Or use a RunPod **Network Volume** (persistent across pod restarts) and mount it at `/workspace`.

### Restoring to a new pod

```powershell
# Push saved files back to the new pod
scp -P <NEW_PORT> "$env:USERPROFILE\runpod_backup\results.tsv" root@<NEW_HOST>:/workspace/autoresearch/
scp -P <NEW_PORT> "$env:USERPROFILE\runpod_backup\tokenizer.pt" root@<NEW_HOST>:/workspace/autoresearch/emr_model/checkpoints/
scp -r -P <NEW_PORT> "$env:USERPROFILE\runpod_backup\phase1" root@<NEW_HOST>:/workspace/autoresearch/emr_model/checkpoints/
```

Then reinstall dependencies (if not using a network volume):

```bash
pip install scikit-learn tqdm openpyxl joblib matplotlib pandas pyarrow
```

### Stopping the pod

Once files are saved: **RunPod → My Pods → Stop**. This pauses billing. The pod can be restarted later.

To permanently delete: **Terminate** (irreversible — all container data is lost).

---

## Running the agent

Start a Claude Code session in this directory and say:

```
Read program.md and all files listed in its Setup section.
Begin the experiment loop now.
```

The agent will iterate autonomously, logging every result to `results.tsv`.
