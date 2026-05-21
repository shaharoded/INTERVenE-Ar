"""
eval_only.py — Evaluate the deployed checkpoints on the held-out test split.

Use when you want the per-outcome AUROC / AUPRC / window-prevalence numbers
without retraining. Loads `emr_model/checkpoints/{phase1,phase2,phase3}/ckpt_best.pt`
and runs `evaluate_on_test_set` on the same test split that `api.py` would
construct (70 / 15 / 15 by PatientId, seed=42), then prints the same summary
block as `api.py` plus the per-outcome table.

Usage:
    python eval_only.py > eval.log 2>&1
    grep "^outcome_\\|^per_outcome" eval.log

Speedup knob: set `EVAL_SAMPLE` below (e.g. 2000) to evaluate on a random
subsample of patients. The test split is still drawn from the same 15 %
slice as the full run, just truncated. AUROC / AUPRC stabilise well before
the full 8,562 test patients, so a 2000-patient subsample typically gives
the headline within ±0.005 in a fraction of the time.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split
import torch

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

from transform_emr.dataset import EMRTokenizer
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import GPT
from evaluation import evaluate_on_test_set

# Same constants as api.py so the test split is identical.
DATA_DIR           = os.path.join(EMR_MODEL_DIR, "data", "source")
TEMPORAL_DATA_FILE = os.path.join(DATA_DIR, "temporal_data.csv")
CONTEXT_DATA_FILE  = os.path.join(DATA_DIR, "context_data.csv")
CHECKPOINT_DIR     = os.path.join(EMR_MODEL_DIR, "checkpoints")

TEST_SPLIT  = 0.15
VAL_SPLIT   = 0.15
RANDOM_SEED = 42

# Sub-sample knob — set to None for full test split (~8,562 patients), or
# an int (e.g. 2000) for a fast subsample.
EVAL_SAMPLE = 2000


def load_test_raw(sample=None):
    """
    Purpose: Reproduce api.py's 70/15/15 PatientId split and return the raw
             test slice (unprocessed CSVs) for evaluate_on_test_set.
    Method:  Two-stage train_test_split with the same seed as api.py. When
             `sample` is set, randomly subsample test patients before slicing
             the CSVs.

    Args:
        sample (int or None): Subsample N test patients. None = full split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (test_temporal_raw, test_ctx_raw).
    """
    print("[Data]: Loading source CSVs...")
    temporal_raw = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_raw      = pd.read_csv(CONTEXT_DATA_FILE)

    all_pids = temporal_raw["PatientId"].unique()
    trainval_ids, test_ids = train_test_split(all_pids, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
    _, _         = train_test_split(trainval_ids, test_size=val_relative, random_state=RANDOM_SEED)

    if sample is not None and sample < len(test_ids):
        rng       = np.random.RandomState(RANDOM_SEED)
        test_ids  = rng.choice(test_ids, size=sample, replace=False)
        print(f"[Data]: Subsampled to {sample} test patients (of {len(all_pids)*TEST_SPLIT:.0f} held-out)")
    else:
        print(f"[Data]: Full test split: {len(test_ids)} patients")

    test_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
    test_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()
    return test_temporal_raw, test_ctx_raw


def main():
    """
    Purpose: Run the autoregressive evaluation on the deployed M-256 checkpoints
             and emit the api.py-style summary block (mean + per-outcome).
    Method:  Load tokenizer / scaler / embedder / Phase-3 GPT from checkpoints,
             reproduce the test split, call evaluate_on_test_set, print results.
    """
    t_start = time.time()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("[Model]: Loading tokenizer + scaler + checkpoints...")
    tokenizer = EMRTokenizer.load(os.path.join(CHECKPOINT_DIR, "tokenizer.pt"))
    scaler    = joblib_load(os.path.join(CHECKPOINT_DIR, "scaler.pkl"))
    embedder, *_ = EMREmbedding.load(os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt"),
                                     tokenizer=tokenizer, map_location=device)
    best_model, *_ = GPT.load(os.path.join(CHECKPOINT_DIR, "phase3", "ckpt_best.pt"),
                              embedder=embedder, map_location=device)
    num_params = sum(p.numel() for p in best_model.parameters())
    print(f"[Model]: Loaded — {num_params:,} params on {device}")

    test_temporal_raw, test_ctx_raw = load_test_raw(sample=EVAL_SAMPLE)

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

    print("---")
    print(f"outcome_auroc:    {eval_results['mean_auroc']:.6f}")
    print(f"outcome_auprc:    {eval_results['mean_auprc']:.6f}")
    print(f"onset_mae_hrs:    {eval_results['mean_mae_hours']:.2f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"eval_sample:      {EVAL_SAMPLE if EVAL_SAMPLE is not None else 'full'}")
    print(f"num_params:       {num_params:,}")

    print("per_outcome\toutcome\tauroc\tauprc\tn_pos\tn_neg")
    auc_table = eval_results.get("auc_table")
    if auc_table is not None:
        for outcome, row in auc_table.iterrows():
            auroc_s = f"{row['auroc']:.6f}" if not pd.isna(row['auroc']) else "nan"
            auprc_s = f"{row['auprc']:.6f}" if not pd.isna(row['auprc']) else "nan"
            print(f"per_outcome\t{outcome}\t{auroc_s}\t{auprc_s}\t{int(row['n_pos_windows'])}\t{int(row['n_neg_windows'])}")

        try:
            import subprocess
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                              cwd=PROJECT_ROOT).decode().strip()
        except Exception:
            commit = "unknown"
        out_dir  = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(out_dir, exist_ok=True)
        suffix   = f"_n{EVAL_SAMPLE}" if EVAL_SAMPLE is not None else "_full"
        out_path = os.path.join(out_dir, f"per_outcome_{commit}{suffix}.tsv")
        auc_table.to_csv(out_path, sep="\t", index=True, index_label="outcome")
        print(f"per_outcome_csv:  {os.path.relpath(out_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
