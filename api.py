"""
api.py — EMR Autoresearch immutable contract.

DO NOT MODIFY this file. It defines the fixed training pipeline and evaluation
metrics. To experiment with model architecture or hyperparameters, edit files
under emr_model/transform_emr/ (and its config/ sub-package).

Usage:
    python api.py
    python api.py > run.log 2>&1   (redirect all output to log)

The agent reads program.md for context, edits transform_emr/ files, then runs
this script to train and evaluate. The summary block (after the "---" separator)
is the ground-truth result for each run.

Optimization target (all from the held-out test set, not the val split):
    outcome_auroc  — primary,   higher is better (0.5 = random, 1.0 = perfect)
    outcome_auprc  — secondary, higher is better
    onset_mae_hrs  — tertiary,  lower is better
"""

import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split
import torch

# Force UTF-8 stdout/stderr so Windows cp1252 doesn't choke on Δ, etc. in training logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress tqdm progress bars — keeps run.log clean (one line per epoch only)
os.environ["TQDM_DISABLE"] = "1"
# Reduce CUDA memory fragmentation (helps on larger models during backward)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR = os.path.join(PROJECT_ROOT, "emr_model")
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader
from transform_emr.config.dataset_config import TAK_REPO_PATH
from transform_emr.config.model_config import MODEL_CONFIG, TRAINING_SETTINGS
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT, pretrain_transformer, finetune_transformer

from evaluation import evaluate_on_test_set

# ===========================================================================
# Fixed paths — do not modify
# ===========================================================================

DATA_DIR               = os.path.join(EMR_MODEL_DIR, "data", "source")
TEMPORAL_DATA_FILE     = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT_DATA_FILE      = os.path.join(DATA_DIR, "context_data.csv")

CHECKPOINT_DIR         = os.path.join(EMR_MODEL_DIR, "checkpoints")
EMBEDDER_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
PHASE3_CHECKPOINT      = os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt")
TOKENIZER_PATH         = os.path.join(CHECKPOINT_DIR, "tokenizer.pt")

VAL_SPLIT   = 0.2
RANDOM_SEED = 42

# ===========================================================================
# Fixed API: data loading — do not modify
# ===========================================================================

def load_data(sample=None, batch_size=64):
    """
    Purpose: Load and prepare EMR data from source CSVs into DataLoaders for all three phases.
    Method: Reads source CSVs, fits scaler on the train portion via DataProcessor
            (saved to checkpoints/scaler.pkl), builds/caches tokenizer, splits patients
            into train/val, and keeps the raw val data for post-training evaluation.

    Args:
        sample (int or None): If set, restrict to this many randomly-sampled patients
            (useful for quick smoke-tests; use None for full training).
        batch_size (int): Batch size for all DataLoaders.

    Returns:
        embedder_train_dl (DataLoader): Natural-distribution loader for Phase-1 embedder.
        transformer_train_dl (DataLoader): Oversampled loader for Phase-2 GPT pretraining.
        phase3_train_dl (DataLoader): Natural-distribution loader for Phase-3 fine-tuning.
        val_dl (DataLoader): Natural-distribution validation loader.
        tokenizer (EMRTokenizer): Fitted vocabulary.
        val_raw (tuple): (val_temporal_df_raw, val_ctx_df_raw) — unprocessed val data,
            passed to evaluate_on_test_set() so it can re-process with truncation.
    """
    print("[Data]: Loading source temporal events and context data...")
    temporal_raw = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_raw      = pd.read_csv(CONTEXT_DATA_FILE)

    if sample is not None:
        pids   = temporal_raw["PatientId"].unique()
        rng    = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
        ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(chosen)]

    # Split patient IDs before processing so scaler is fitted on train only
    all_pids  = temporal_raw["PatientId"].unique()
    train_ids, val_ids = train_test_split(all_pids, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    train_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(train_ids)].copy()
    train_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(train_ids)].copy()
    val_temporal_raw   = temporal_raw[temporal_raw["PatientId"].isin(val_ids)].copy()
    val_ctx_raw        = ctx_raw[ctx_raw["PatientId"].isin(val_ids)].copy()

    # Fit scaler on train patients and save to CHECKPOINT_DIR/scaler.pkl
    print("[Data]: Processing train split (fitting scaler)...")
    train_processor = DataProcessor(train_temporal_raw.copy(), train_ctx_raw.copy(),
                                    scaler=None, tak_repo_path=TAK_REPO_PATH,
                                    checkpoint_path=CHECKPOINT_DIR)
    train_temporal_df, train_ctx_df = train_processor.run()

    # Apply fitted scaler to val split
    scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    print("[Data]: Processing val split (applying fitted scaler)...")
    val_processor = DataProcessor(val_temporal_raw.copy(), val_ctx_raw.copy(),
                                  scaler=scaler, tak_repo_path=TAK_REPO_PATH,
                                  checkpoint_path=CHECKPOINT_DIR)
    val_temporal_df, val_ctx_df = val_processor.run()

    # Build / load tokenizer from train data
    tokenizer_path = Path(TOKENIZER_PATH)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        tokenizer = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer (one-time, may take a few minutes)...")
        tokenizer = EMRTokenizer.from_processed_df(train_temporal_df)
        tokenizer.save(str(tokenizer_path))
        print(f"[Data]: Tokenizer saved to {tokenizer_path}")

    train_ds = EMRDataset(train_temporal_df, train_ctx_df, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_temporal_df,   val_ctx_df,   tokenizer=tokenizer)

    print(f"[Data]: {len(train_ids)} train / {len(val_ids)} val patients  "
          f"({len(train_ds.tokens_df):,} train records, {len(val_ds.tokens_df):,} val records)")

    # Phase-1 and Phase-3 use natural-distribution; Phase-2 uses oversampled
    embedder_train_dl    = get_dataloader(train_ds, batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=False, bucket_batching=True)
    phase3_train_dl      = get_dataloader(train_ds, batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=False, bucket_batching=True)
    transformer_train_dl = get_dataloader(train_ds, batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=True,  bucket_batching=True)
    val_dl               = get_dataloader(val_ds,   batch_size=batch_size,
                                          collate_fn=collate_emr, oversample=False, bucket_batching=True)

    return embedder_train_dl, transformer_train_dl, phase3_train_dl, val_dl, tokenizer, (val_temporal_raw, val_ctx_raw)


# ===========================================================================
# Training orchestration
# ===========================================================================

t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Clear Phase-2 and Phase-3 checkpoints for a fresh run.
# Phase-1 (embedder) is preserved and reused when the config matches — no need
# to re-train the embedder unless embed_dim / time2vec_dim / ctx_dim changes.
for _phase in ["phase2", "phase3"]:
    _phase_path = Path(CHECKPOINT_DIR) / _phase
    if _phase_path.exists():
        shutil.rmtree(_phase_path)
    _phase_path.mkdir(parents=True, exist_ok=True)
(Path(CHECKPOINT_DIR) / "phase1").mkdir(parents=True, exist_ok=True)

# Load data — keeps raw val data for post-training evaluation
embedder_train_dl, transformer_train_dl, phase3_train_dl, val_dl, tokenizer, val_raw = load_data(
    sample=TRAINING_SETTINGS.get("sample"),
    batch_size=TRAINING_SETTINGS["batch_size"],
)

# Auto-detect context vector dimension from the first batch
for _batch in embedder_train_dl:
    MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
    break

print(f"Model config: {MODEL_CONFIG}")

# ---------------------------------------------------------------------------
# Phase 1 — Train embedder (token + time + context representations)
# ---------------------------------------------------------------------------

# Reuse cached embedder if the architecture config is unchanged.
_embedder_key = (
    MODEL_CONFIG["embed_dim"],
    MODEL_CONFIG["time2vec_dim"],
    MODEL_CONFIG["ctx_dim"],
)

_cached_ckpt     = Path(EMBEDDER_CHECKPOINT)
_embedder_reused = False

if _cached_ckpt.exists():
    try:
        _ckpt_cfg   = torch.load(str(_cached_ckpt), map_location="cpu", weights_only=True)["config"]
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

# ---------------------------------------------------------------------------
# Phase 2 — Pretrain GPT transformer over learned embeddings
# ---------------------------------------------------------------------------

model = GPT(cfg=MODEL_CONFIG, embedder=embedder)

model, _, val_losses = pretrain_transformer(
    model             = model,
    train_dl          = transformer_train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = TRANSFORMER_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ---------------------------------------------------------------------------
# Phase 3 — Fine-tune outcome head (backbone frozen, natural-distribution data)
# ---------------------------------------------------------------------------

# Load best Phase-2 checkpoint as the starting point for Phase-3.
_p2_best = Path(TRANSFORMER_CHECKPOINT)
_p2_last = _p2_best.parent / "ckpt_last.pt"
_p2_ckpt = _p2_best if _p2_best.exists() else (_p2_last if _p2_last.exists() else None)

if _p2_ckpt is not None:
    model_p3, *_ = GPT.load(str(_p2_ckpt), embedder=embedder)
else:
    model_p3 = model

model_p3, _, p3_val_losses = finetune_transformer(
    model             = model_p3,
    train_dl          = phase3_train_dl,   # natural distribution (no oversampling)
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = PHASE3_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ---------------------------------------------------------------------------
# Final evaluation on held-out test set
# ---------------------------------------------------------------------------

# Prefer Phase-3 best, then Phase-2 best, then last in-memory model
_p3_path = Path(PHASE3_CHECKPOINT)
_p2_path = Path(TRANSFORMER_CHECKPOINT)

if _p3_path.exists():
    best_model, *_ = GPT.load(str(_p3_path), embedder=embedder)
elif _p2_path.exists():
    best_model, *_ = GPT.load(str(_p2_path), embedder=embedder)
else:
    best_model = model_p3

val_temporal_raw, val_ctx_raw = val_raw
scaler = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
eval_results = evaluate_on_test_set(
    model=best_model,
    tokenizer=tokenizer,
    val_temporal_raw=val_temporal_raw,
    val_ctx_raw=val_ctx_raw,
    scaler=scaler,
    checkpoint_dir=CHECKPOINT_DIR,
)

# ===========================================================================
# Summary  (grep-friendly format — one key per line)
# ===========================================================================

t_end         = time.time()
peak_vram_mb  = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params    = best_model.get_num_params() if hasattr(best_model, "get_num_params") else sum(p.numel() for p in best_model.parameters())
phase2_best   = min(val_losses)    if val_losses    else float("nan")
phase2_epochs = len(val_losses)
phase3_best   = min(p3_val_losses) if p3_val_losses else float("nan")
phase3_epochs = len(p3_val_losses)

print("---")
print(f"outcome_auroc:    {eval_results['mean_auroc']:.6f}")
print(f"outcome_auprc:    {eval_results['mean_auprc']:.6f}")
print(f"onset_mae_hrs:    {eval_results['mean_mae_hours']:.2f}")
print(f"phase2_best_val:  {phase2_best:.6f}")
print(f"phase2_epochs:    {phase2_epochs}")
print(f"phase3_best_val:  {phase3_best:.6f}")
print(f"phase3_epochs:    {phase3_epochs}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"num_params:       {num_params:,}")
