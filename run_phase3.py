"""
run_phase3.py — Run Phase 3 fine-tuning from an existing Phase-2 checkpoint.

Use when api.py crashed between Phase 2 and Phase 3 (the checkpoint deletion at
api.py startup would wipe the Phase-2 checkpoint if api.py were restarted).

Loads Phase-1 embedder + Phase-2 GPT from existing checkpoints, runs Phase-3
finetune_transformer, then runs evaluate_on_test_set on the held-out test split.
Prints the same summary block as api.py.

Usage:
    cd /workspace/autoresearch
    python run_phase3.py > run_phase3.log 2>&1
"""

import os
import sys
import time
from pathlib import Path

import torch
from joblib import load as joblib_load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["TQDM_DISABLE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR = os.path.join(PROJECT_ROOT, "emr_model")
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

from transform_emr.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from transform_emr.dataset import collate_emr, get_dataloader
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import GPT, finetune_transformer
from evaluation import evaluate_on_test_set

CHECKPOINT_DIR      = os.path.join(EMR_MODEL_DIR, "checkpoints")
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
PHASE2_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
PHASE3_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt")
PROCESSED_CACHE     = os.path.join(CHECKPOINT_DIR, "processed_datasets.pt")

t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"  training settings : {TRAINING_SETTINGS}")

# Ensure phase3 dir exists
Path(PHASE3_CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)

# Load cached datasets
print("[Data]: Loading processed datasets cache...")
cached    = torch.load(PROCESSED_CACHE, map_location="cpu", weights_only=False)
train_ds  = cached["train_ds"]
val_ds    = cached["val_ds"]
tokenizer = cached["tokenizer"]
test_raw  = cached["test_raw"]
n_train, n_val, n_test = cached["sizes"]
print(f"[Data]: {n_train} train / {n_val} val / {n_test} test patients")

batch_size = TRAINING_SETTINGS["batch_size"]
phase3_train_dl = get_dataloader(train_ds, batch_size=batch_size,
                                 collate_fn=collate_emr, oversample=False, bucket_batching=True)
val_dl          = get_dataloader(val_ds,   batch_size=batch_size,
                                 collate_fn=collate_emr, oversample=False, bucket_batching=True)

# Load Phase-1 embedder
print("[Phase 1]: Loading embedder checkpoint...")
embedder, *_ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
embedder.to(device)

# Load Phase-2 GPT checkpoint as starting point for Phase-3
print("[Phase 2]: Loading Phase-2 best checkpoint...")
_p2_best = Path(PHASE2_CHECKPOINT)
_p2_last = _p2_best.parent / "ckpt_last.pt"
_p2_ckpt = _p2_best if _p2_best.exists() else (_p2_last if _p2_last.exists() else None)
if _p2_ckpt is None:
    raise FileNotFoundError("No Phase-2 checkpoint found. Cannot run Phase-3.")
model_p3, *_ = GPT.load(str(_p2_ckpt), embedder=embedder)
print(f"[Phase 2]: Loaded from {_p2_ckpt}")

# Phase 3 fine-tuning
print("[Phase 3]: Starting outcome head fine-tuning...")
model_p3, _, p3_val_losses = finetune_transformer(
    model             = model_p3,
    train_dl          = phase3_train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = PHASE3_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# Load best Phase-3 checkpoint for evaluation
_p3_path = Path(PHASE3_CHECKPOINT)
if _p3_path.exists():
    best_model, *_ = GPT.load(str(_p3_path), embedder=embedder)
    print(f"[Phase 3]: Loaded best checkpoint from {_p3_path}")
else:
    best_model = model_p3
    print("[Phase 3]: No checkpoint saved — using last in-memory model")

best_model.to(device)

# Final evaluation on held-out test split
test_temporal_raw, test_ctx_raw = test_raw
scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
eval_results = evaluate_on_test_set(
    model=best_model,
    tokenizer=tokenizer,
    val_temporal_raw=test_temporal_raw,
    val_ctx_raw=test_ctx_raw,
    scaler=scaler,
    checkpoint_dir=CHECKPOINT_DIR,
)

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params   = best_model.get_num_params() if hasattr(best_model, "get_num_params") else sum(p.numel() for p in best_model.parameters())
phase3_best  = min(p3_val_losses) if p3_val_losses else float("nan")
phase3_epochs = len(p3_val_losses)

print("---")
print(f"outcome_auroc:    {eval_results['mean_auroc']:.6f}")
print(f"outcome_auprc:    {eval_results['mean_auprc']:.6f}")
print(f"onset_mae_hrs:    {eval_results['mean_mae_hours']:.2f}")
print(f"phase2_best_val:  0.096300 (from smoke_test_L384.log, epoch 39)")
print(f"phase2_epochs:    40")
print(f"phase3_best_val:  {phase3_best:.6f}")
print(f"phase3_epochs:    {phase3_epochs}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"num_params:       {num_params:,}")
