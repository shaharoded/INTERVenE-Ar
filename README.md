# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search for a clinical EMR sequence model.

## What this is

An AI agent autonomously experiments on an EMR (Electronic Medical Record) next-event prediction model overnight. Each experiment modifies `train.py`, trains for a fixed epoch budget, checks if the validation metric improved, keeps or discards, and repeats — logging every result in `results.tsv`.

The architecture is a two-phase pipeline (`emr_model/`):

- **Phase 1 — EMREmbedding** — learns compact, time-aware representations of clinical events using hierarchical token embeddings, Time2Vec, and static patient context.
- **Phase 2 — GPT Transformer** — a causal decoder over Phase-1 embeddings, predicting the next clinical event in a patient's timeline.

The training data is derived from MIMIC-III and contains diabetes patients' longitudinal event sequences including lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

## Files that matter

| File | Role |
|------|------|
| `prepare.py` | **Fixed.** Data loading and fixed evaluation metric. Do NOT modify. |
| `train.py` | **Agent edits this.** Model config, training settings, training loop. |
| `program.md` | **Human edits this.** Instructions for the autonomous agent. |
| `emr_model/transform_emr/` | Model source code (agent may modify for architecture changes). |
| `emr_model/data/source/` | Training data (fixed). |
| `results.tsv` | Experiment log (untracked by git). |

## Metric

**Primary — `outcome_f1` (higher is better)**
Time-tolerant F1 for clinical outcome prediction. At each position, the model is considered to have "predicted an outcome" if any outcome token (complication, e.g. `KIDNEY_COMPLICATION_EVENT`) appears in its top-15 predictions. A prediction matches a true outcome if they fall within ±48 hours of each other in the patient timeline. Precision and recall are computed from this matching, then combined into F1. This penalises both missed complications (low recall) and false alarms during quiet periods (low precision).

**Secondary — `val_bce_loss` (lower is better)**
Mean BCE over all token positions (multi-hot, fixed k=5 look-ahead window). Used as a sanity signal to detect model collapse; not the autoresearch keep/discard criterion.

Both are computed in `prepare.py::evaluate_val_metrics` — never modified by the agent.

## Design choices

- **Two-file split.** `prepare.py` is fixed ground truth; `train.py` is fully editable.
- **Embedder caching.** Phase-1 is skipped automatically when `embed_dim`, `time2vec_dim`, and `ctx_dim` are unchanged — saving ~30 min per experiment.
- **Fresh Phase-2 per experiment.** Phase-2 checkpoints are cleared before each run so experiments are independent.
- **Epoch budget with early stopping.** Training terminates when validation stops improving (patience = 10 epochs), bounded by a maximum epoch count.
- **Data sampling.** `TRAINING_SETTINGS["sample"]` controls how many patients to use. Set to `None` for full training runs.

## Project structure

```
prepare.py              fixed data utilities + evaluation metric
train.py                model config + training (agent modifies)
program.md              agent instructions (human modifies)
pyproject.toml          dependencies
results.tsv             experiment log (gitignored, untracked)
run.log                 last experiment output (gitignored)
emr_model/
  transform_emr/        core model source (agent may modify)
    embedder.py         Phase-1 EMREmbedding
    transformer.py      Phase-2 GPT
    dataset.py          tokenizer and dataloader
    loss.py             loss functions
    schedulers.py       auxiliary loss scheduling
    utils.py            utilities
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
Read program.md and all files listed in the Setup section.
The baseline is already logged in results.tsv.
Begin the experiment loop now.
```

The agent will iterate autonomously, logging every result to `results.tsv`.
