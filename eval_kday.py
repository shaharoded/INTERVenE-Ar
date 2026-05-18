"""
eval_kday.py — Phase D k-day seed scan.

Evaluates the current Phase-3 checkpoint at different input-seed lengths
(k days) by patching evaluation.EVAL_INPUT_DAYS before calling
evaluate_on_test_set.  Does NOT modify evaluation.py.

Usage:
    python eval_kday.py <k>        # e.g. python eval_kday.py 3
    python eval_kday.py <k> > eval_k3.log 2>&1
"""

import os
import sys
import time
from pathlib import Path

import torch
from joblib import load as joblib_load

if len(sys.argv) < 2:
    print("Usage: python eval_kday.py <k_days>", file=sys.stderr)
    sys.exit(1)

K_DAYS = int(sys.argv[1])

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

# Patch EVAL_INPUT_DAYS before evaluation.py is used
import evaluation
evaluation.EVAL_INPUT_DAYS = K_DAYS

from transform_emr.config.model_config import MODEL_CONFIG
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import GPT
from evaluation import evaluate_on_test_set

CHECKPOINT_DIR      = os.path.join(EMR_MODEL_DIR, "checkpoints")
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
PHASE3_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt")
PHASE2_CHECKPOINT   = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
PROCESSED_CACHE     = os.path.join(CHECKPOINT_DIR, "processed_datasets.pt")

print(f"[Phase D] k={K_DAYS} days seed scan")
t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("[Data]: Loading processed datasets cache...")
cached    = torch.load(PROCESSED_CACHE, map_location="cpu", weights_only=False)
tokenizer = cached["tokenizer"]
test_raw  = cached["test_raw"]
test_temporal_raw, test_ctx_raw = test_raw
print(f"[Data]: {cached['sizes'][2]} test patients loaded from cache")

print("[Phase 1]: Loading embedder checkpoint...")
embedder, *_ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
embedder.to(device)

print("[Phase 3]: Loading Phase-3 best checkpoint...")
_p3 = Path(PHASE3_CHECKPOINT)
_p2 = Path(PHASE2_CHECKPOINT)
if _p3.exists():
    best_model, *_ = GPT.load(str(_p3), embedder=embedder)
    print(f"[Phase 3]: Loaded from {_p3}")
elif _p2.exists():
    best_model, *_ = GPT.load(str(_p2), embedder=embedder)
    print(f"[Phase 2]: Loaded from {_p2} (no phase-3 checkpoint)")
else:
    raise FileNotFoundError("Neither phase2 nor phase3 checkpoint found.")

best_model.to(device)

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

print("---")
print(f"k_days:           {K_DAYS}")
print(f"outcome_auroc:    {eval_results['mean_auroc']:.6f}")
print(f"outcome_auprc:    {eval_results['mean_auprc']:.6f}")
print(f"onset_mae_hrs:    {eval_results['mean_mae_hours']:.2f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"num_params:       {num_params:,}")
