"""
Fixed data preparation and evaluation for EMR autoresearch.

This file is NOT modified by the agent. It defines:
  - Data loading (load_data)
  - Fixed evaluation metric (evaluate_val_ce)

The metric is cross-entropy on next-event prediction — lower is better,
and it is independent of auxiliary-loss hyperparameters so experiments
are always fairly compared.

Usage (from root):
    from prepare import load_data, evaluate_val_ce, CHECKPOINT_DIR, ...
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths — all relative to this file's directory
# ---------------------------------------------------------------------------

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR  = os.path.join(PROJECT_ROOT, "emr_model")

# Make transform_emr importable without pip-installing the package
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader
from transform_emr.config.dataset_config import TAK_REPO_PATH, OUTCOMES
from transform_emr.utils import get_multi_hot_targets

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

DATA_DIR               = os.path.join(EMR_MODEL_DIR, "data", "source")
TEMPORAL_DATA_FILE     = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT_DATA_FILE      = os.path.join(DATA_DIR, "context_data.csv")

CHECKPOINT_DIR         = os.path.join(EMR_MODEL_DIR, "checkpoints")
EMBEDDER_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
TOKENIZER_PATH         = os.path.join(CHECKPOINT_DIR, "tokenizer.pt")

VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(sample=None, batch_size=64):
    """
    Load and prepare EMR data from source CSVs.

    Parameters
    ----------
    sample : int or None
        If set, use only this many randomly-sampled patients (useful for
        quick smoke-tests; use None for full training).
    batch_size : int
        Batch size for all dataloaders.

    Returns
    -------
    embedder_train_dl   : DataLoader  (unshuffled, for Phase-1 embedder)
    transformer_train_dl: DataLoader  (oversampled, for Phase-2 GPT)
    val_dl              : DataLoader  (unshuffled validation)
    tokenizer           : EMRTokenizer
    """
    print("[Data]: Loading temporal events and context data...")
    temporal_df = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df      = pd.read_csv(CONTEXT_DATA_FILE)

    if sample is not None:
        pids   = temporal_df["PatientId"].unique()
        rng    = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_df = temporal_df[temporal_df["PatientId"].isin(chosen)]
        ctx_df      = ctx_df[ctx_df["PatientId"].isin(chosen)]

    tokenizer_path = Path(TOKENIZER_PATH)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        processor   = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer   = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer (one-time, may take a few minutes)...")
        processor   = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer   = EMRTokenizer.from_processed_df(temporal_df)
        tokenizer.save(str(tokenizer_path))
        print(f"[Data]: Tokenizer saved to {tokenizer_path}")

    pids = temporal_df["PatientId"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    train_df  = temporal_df[temporal_df.PatientId.isin(train_ids)].copy()
    val_df    = temporal_df[temporal_df.PatientId.isin(val_ids)].copy()
    train_ctx = ctx_df.loc[ctx_df.index.isin(train_ids)]
    val_ctx   = ctx_df.loc[ctx_df.index.isin(val_ids)]

    train_ds  = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
    val_ds    = EMRDataset(val_df,   val_ctx,   tokenizer=tokenizer)

    print(f"[Data]: {len(train_ids)} train / {len(val_ids)} val patients  "
          f"({len(train_ds.tokens_df):,} train records, {len(val_ds.tokens_df):,} val records)")

    embedder_train_dl    = get_dataloader(train_ds, batch_size=batch_size, collate_fn=collate_emr, oversample=False)
    transformer_train_dl = get_dataloader(train_ds, batch_size=batch_size, collate_fn=collate_emr, oversample=True)
    val_dl               = get_dataloader(val_ds,   batch_size=batch_size, collate_fn=collate_emr, oversample=False)

    return embedder_train_dl, transformer_train_dl, val_dl, tokenizer

# ---------------------------------------------------------------------------
# Fixed evaluation metrics (DO NOT CHANGE — this is the ground truth)
# ---------------------------------------------------------------------------

EVAL_BCE_K_WINDOW    = 5     # step-based window for BCE (secondary metric only)
OUTCOME_WINDOW_HOURS = 48.0  # temporal window for AUROC labels
_ABS_TS_SCALE        = 336.0 # abs_ts = TimePoint_in_hours / 336
_OUTCOME_WINDOW_NORM = OUTCOME_WINDOW_HOURS / _ABS_TS_SCALE  # in abs_ts units


def _get_outcome_token_ids(tokenizer):
    """Return a list of (outcome_name, token_id) for all outcomes present in the vocab."""
    return [(name, tokenizer.token2id[name])
            for name in OUTCOMES if name in tokenizer.token2id]


@torch.no_grad()
def evaluate_val_metrics(model, val_dl, device="cuda"):
    """
    Compute two complementary validation metrics in a single forward pass.

    Primary — outcome_auroc (higher is better, 0.5 = random, 1.0 = perfect)
        Mean per-outcome ROC-AUC with a TEMPORAL label, averaged across all
        outcome types present in the validation set.

        For each specific outcome type o (e.g. KIDNEY_COMPLICATION_EVENT):
          - Score at position t  = logit for token o at position t
          - Label at position t  = 1  if outcome o occurs at any future event
                                       whose timestamp is within OUTCOME_WINDOW_HOURS
                                       (48 h) of the current event's timestamp

        The 48 h window is in real patient time (abs_ts units), so label=1 means
        "this specific complication happens within the next 48 hours."  This is
        invariant to event density — a quiet period with sparse events gets the
        same window as a dense ICU period.

        AUROC is computed per outcome type, then averaged.  A model that
        predicts a complication at random scores ≈ 0.5; it must predict the
        *correct* complication *when* it is approaching to score well.

    Secondary — val_bce_loss (lower is better)
        Mean BCE over all token positions (multi-hot, step-based k=5 window).
        Sanity signal — not the keep/discard criterion.

    Returns
    -------
    outcome_auroc : float  (primary — higher is better)
    val_bce       : float  (secondary — lower is better)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    tokenizer    = model.embedder.tokenizer
    padding_idx  = model.embedder.padding_idx
    outcome_list = _get_outcome_token_ids(tokenizer)  # [(name, tid), ...]

    scores_by_outcome = {tid: [] for _, tid in outcome_list}
    labels_by_outcome = {tid: [] for _, tid in outcome_list}

    bce_loss_sum = 0.0
    bce_batches  = 0

    for batch in val_dl:
        batch = {key: v.to(device) if torch.is_tensor(v) else v for key, v in batch.items()}

        logits, _, _, _ = model(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids   =batch["concept_ids"],
            value_ids     =batch["value_ids"],
            position_ids  =batch["position_ids"],
            abs_ts        =batch["abs_ts"],
            context_vec   =batch["context_vec"],
        )

        pred_logits = logits[:, :-1, :].float()   # [B, T-1, V]
        target_ids  = batch["targets"][:, 1:]      # [B, T-1]  true next tokens
        nonpad      = target_ids != padding_idx    # [B, T-1]

        if nonpad.sum() == 0:
            continue

        # Timestamps:
        #   cur_ts[b, t] = timestamp of last seen event at prediction step t
        #                = abs_ts[b, t]  (unshifted — this is what the model
        #                                 has observed up to step t)
        #   fut_ts[b, t] = timestamp of the true future event at step t
        #                = abs_ts[b, t+1] (shifted)
        cur_ts = batch["abs_ts"][:, :-1]   # [B, T-1]
        fut_ts = batch["abs_ts"][:, 1:]    # [B, T-1]

        # time_in_window[b, t, t'] = True if future event at step t' falls
        # strictly after t and within 48 h of t's timestamp.
        # Shape: [B, T-1, T-1]
        dt = fut_ts.unsqueeze(1) - cur_ts.unsqueeze(2)   # [B, T-1, T-1]
        time_in_window = (dt > 0) & (dt <= _OUTCOME_WINDOW_NORM)

        # ------------------------------------------------------------------
        # Primary: per-outcome temporal AUROC
        # ------------------------------------------------------------------
        for _, tid in outcome_list:
            # fut_is_o[b, t'] = 1 if future token at t' is outcome o
            fut_is_o = (target_ids == tid)                          # [B, T-1]

            # label[b, t] = any future event within 48 h that is outcome o
            label = (time_in_window & fut_is_o.unsqueeze(1)).any(dim=2)  # [B, T-1]

            # score = logit for outcome o at each prediction position
            score = pred_logits[:, :, tid]                          # [B, T-1]

            scores_by_outcome[tid].append(score[nonpad].cpu().float().numpy())
            labels_by_outcome[tid].append(label[nonpad].cpu().float().numpy())

        # ------------------------------------------------------------------
        # Secondary: BCE loss (step-based, sanity signal only)
        # ------------------------------------------------------------------
        vocab_size = pred_logits.size(-1)
        multi_hot  = get_multi_hot_targets(
            position_ids=target_ids,
            padding_idx =padding_idx,
            vocab_size   =vocab_size,
            k            =EVAL_BCE_K_WINDOW,
        ).to(device)

        loss_per_elem = F.binary_cross_entropy_with_logits(
            pred_logits, multi_hot, reduction="none"
        )
        valid_mask    = nonpad.unsqueeze(-1).float()
        bce_loss_sum += (
            (loss_per_elem * valid_mask).sum() /
            (valid_mask.sum().clamp(min=1) * vocab_size)
        ).item()
        bce_batches  += 1

    # Average per-outcome AUROC
    aurocs = []
    for name, tid in outcome_list:
        scores = np.concatenate(scores_by_outcome[tid])
        labels = np.concatenate(labels_by_outcome[tid])
        if labels.sum() == 0:
            continue  # outcome not present in val set — skip
        try:
            aurocs.append(roc_auc_score(labels, scores))
        except ValueError:
            pass

    outcome_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    val_bce       = bce_loss_sum / max(bce_batches, 1)
    return outcome_auroc, val_bce


# ---------------------------------------------------------------------------
# Legacy alias — kept so old code that imports evaluate_val_bce still runs
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_val_bce(model, val_dl, device="cuda"):
    """Backward-compatible wrapper. Prefer evaluate_val_metrics."""
    _, val_bce = evaluate_val_metrics(model, val_dl, device=device)
    return val_bce
