"""
EMR Autoresearch training script.

This is the file the agent modifies. Architecture, hyperparameters, and
training settings — everything in here is fair game.

Usage:
    python train.py
    python train.py > run.log 2>&1   (redirect all output to log)

The only file you must NOT modify is prepare.py.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Suppress tqdm progress bars — keeps run.log clean (one line per epoch only)
os.environ["TQDM_DISABLE"] = "1"
# Reduce CUDA memory fragmentation (helps on larger models during backward)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add emr_model to path (prepare.py also does this, but be explicit here)
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR = os.path.join(PROJECT_ROOT, "emr_model")
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

import torch

from prepare import (
    load_data, evaluate_val_metrics,
    CHECKPOINT_DIR, EMBEDDER_CHECKPOINT, TRANSFORMER_CHECKPOINT,
)
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT, train_transformer

# ===========================================================================
# HYPERPARAMETERS — the agent edits this section
# ===========================================================================

# Model architecture
MODEL_CONFIG = {
    "time2vec_dim":  32,    # dimension of each Time2Vec component (>= 2)
    "embed_dim":     256,   # shared embedding dimension for tokens & GPT
    "block_size":    512,   # max sequence length (tokens per patient window)
    "n_head":        4,     # number of attention heads (embed_dim % n_head == 0)
    "n_layer":       4,     # number of transformer decoder blocks
    "dropout":       0.1,   # dropout applied throughout
    "bias":          True,  # use bias in linear layers
}

# Training settings
TRAINING_SETTINGS = {
    # Epoch budgets (early stopping may terminate earlier)
    "phase1_n_epochs": 30,
    "phase2_n_epochs": 50,

    # Phase-2 curriculum masking ramp-up
    "cbm_ramp_epochs":  5,

    # Warm-up: do not save "best" until this many epochs have passed
    "warmup_epochs": 5,

    # Early stopping patience (epochs without improvement)
    "early-stop-patience": 10,

    # Learning rates
    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "weight_decay":         1e-3,

    # Batch size (reduce if OOM; affects training speed not final quality much)
    "batch_size": 64,

    # Data sample: number of patients to use (None = full dataset).
    # Set to e.g. 2000 for faster iteration; use None for final validation runs.
    "sample": 2000,

    # Temporal window for Phase-2 BCE loss (hours).
    # All tokens within this forward window from position t are positive targets.
    # Must be dense enough that most positions have >=1 positive (check diagnose.py Report 3).
    # Tested: 1h (too sparse), 3h (sparse), 12h (best), 24h (untested).
    "bce_window_hours": 12.0,

    # Temporal window for the outcome auxiliary head (hours).
    # Should start where BCE ends to avoid overlap (non-overlapping = complementary, not redundant).
    # outcome_window = (bce_window_hours, outcome_window_hi_hours]
    # The upper bound is fixed at 48h to match the evaluation metric in prepare.py.
    "outcome_window_hi_hours": 48.0,

    # -----------------------------------------------------------------------
    # Phase-1 auxiliary loss scheduler
    # BCE trains alone for bce_only_epochs, then MLM + Δt are activated.
    # Lambda for each aux loss is calibrated once so it contributes at most
    # aux_fraction_caps[key] × BCE_loss at the calibration epoch.
    # -----------------------------------------------------------------------
    "phase1_scheduler": {
        "bce_only_epochs": 3,
        "aux_fraction_caps": {
            "mlm": 0.20,
            "dt":  0.20,
        },
        "order":       [["mlm", "dt"]],
        "ramp_epochs": {"mlm": 1, "dt": 1},
    },

    # -----------------------------------------------------------------------
    # Phase-2 auxiliary loss scheduler (two-stage curriculum)
    #   Stage 0: [ce, dt]   — active after bce_only_epochs
    #   Stage 1: [outcome]  — unlocked on stage-0 plateau
    #   penalty removed: raw penalty flat at 0.72, λ=0.0015, never improves
    # -----------------------------------------------------------------------
    "phase2_scheduler": {
        "bce_only_epochs": 3,
        "aux_fraction_caps": {
            "ce":      2.00,
            "dt":      0.20,
            "outcome": 10.00,
        },
        "order": [["ce", "dt"], ["outcome"]],
        "ramp_epochs": {
            "ce":      1,
            "dt":      1,
            "outcome": 5,
        },
        "plateau_min_delta": 1e-4,
        "plateau_patience":  [3],
    },
}

# ===========================================================================
# Setup
# ===========================================================================

t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Clear Phase-2 checkpoint for a fresh run.
# Phase-1 (embedder) is preserved and reused when the config matches — no need
# to re-train the embedder unless embed_dim / time2vec_dim / ctx_dim changed.
phase2_path = Path(CHECKPOINT_DIR) / "phase2"
if phase2_path.exists():
    shutil.rmtree(phase2_path)
phase2_path.mkdir(parents=True, exist_ok=True)
(Path(CHECKPOINT_DIR) / "phase1").mkdir(parents=True, exist_ok=True)

# Load data — tokenizer is built once and cached
embedder_train_dl, transformer_train_dl, val_dl, tokenizer = load_data(
    sample=TRAINING_SETTINGS.get("sample"),
    batch_size=TRAINING_SETTINGS["batch_size"],
)

# Auto-detect context vector size from the first batch
for _batch in embedder_train_dl:
    MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
    break

print(f"Model config: {MODEL_CONFIG}")

# ===========================================================================
# Phase 1 — Train embedder (token + time + context representations)
# ===========================================================================

# Reuse a cached embedder if the architecture config is unchanged.
# Only the three parameters that change the weight shapes matter here.
_embedder_key = (
    MODEL_CONFIG["embed_dim"],
    MODEL_CONFIG["time2vec_dim"],
    MODEL_CONFIG["ctx_dim"],
)

_cached_ckpt = Path(EMBEDDER_CHECKPOINT)
_embedder_reused = False

if _cached_ckpt.exists():
    try:
        _ckpt_cfg = torch.load(str(_cached_ckpt), map_location="cpu", weights_only=True)["config"]
        _cached_key = (
            _ckpt_cfg["embed_dim"],
            _ckpt_cfg["time2vec_dim"],
            _ckpt_cfg["ctx_dim"],
        )
        if _cached_key == _embedder_key:
            print("[Phase 1]: Config unchanged — loading cached embedder, skipping training.")
            embedder, *_ = EMREmbedding.load(str(_cached_ckpt), tokenizer=tokenizer)
            _embedder_reused = True
    except Exception as e:
        print(f"[Phase 1]: Could not load cached embedder ({e}), retraining.")

if not _embedder_reused:
    embedder = EMREmbedding(
        tokenizer    = tokenizer,
        ctx_dim      = MODEL_CONFIG["ctx_dim"],
        time2vec_dim = MODEL_CONFIG["time2vec_dim"],
        embed_dim    = MODEL_CONFIG["embed_dim"],
        dropout      = MODEL_CONFIG["dropout"],
    )
    embedder, _, _ = train_embedder(
        embedder          = embedder,
        train_loader      = embedder_train_dl,
        val_loader        = val_dl,
        resume            = False,
        checkpoint_path   = EMBEDDER_CHECKPOINT,
        training_settings = TRAINING_SETTINGS,
    )

# ===========================================================================
# Phase 2 — Train GPT transformer over learned embeddings
# ===========================================================================

model = GPT(cfg=MODEL_CONFIG, embedder=embedder)

model, _, val_losses = train_transformer(
    model             = model,
    train_dl          = transformer_train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = TRANSFORMER_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ===========================================================================
# Evaluation — fixed metric from prepare.py (do not change)
# ===========================================================================

# Load best saved checkpoint; fall back to last if best wasn't written
best_ckpt    = Path(TRANSFORMER_CHECKPOINT)
fallback_ckpt = best_ckpt.parent / "ckpt_last.pt"

if best_ckpt.exists():
    best_model, *_ = GPT.load(str(best_ckpt), embedder=embedder)
elif fallback_ckpt.exists():
    best_model, *_ = GPT.load(str(fallback_ckpt), embedder=embedder)
else:
    best_model = model

outcome_auroc, val_bce = evaluate_val_metrics(best_model, val_dl, device=str(device))

# ===========================================================================
# Summary  (grep-friendly format — one key per line)
# ===========================================================================

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params   = best_model.get_num_params() if hasattr(best_model, "get_num_params") else sum(p.numel() for p in best_model.parameters())
phase2_best  = min(val_losses) if val_losses else float("nan")
phase2_epochs = len(val_losses)

print("---")
print(f"outcome_auroc:    {outcome_auroc:.6f}")
print(f"val_bce_loss:     {val_bce:.6f}")
print(f"phase2_best_val:  {phase2_best:.6f}")
print(f"phase2_epochs:    {phase2_epochs}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"block_size:       {MODEL_CONFIG['block_size']}")
print(f"num_params:       {num_params:,}")
