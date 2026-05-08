"""
evaluation.py — Fixed post-training evaluation for EMR autoresearch.

DO NOT MODIFY — these metrics define the optimization target for each research round.
The agent should NOT edit this file. Improving these metrics is the goal.

Metrics (computed on the held-out test set, not the training validation split):

  Primary   — mean_auroc : mean per-complication AUROC from pooled episode-level AUC.
                           Higher is better. Random = 0.5, perfect = 1.0.
  Secondary — mean_auprc : mean per-complication AUPRC from the same evaluation.
                           Higher is better. Reflects precision at varying recall thresholds.
  Tertiary  — mean_mae_hours : mean onset-prediction error in hours.
                               Lower is better.

Evaluation protocol (mirrors evaluation.ipynb exactly):
  1. Load held-out test data (data/test/ — never seen during training).
  2. Re-process with the scaler fitted on the training pool.
  3. Build two datasets: full (for ground truth) and truncated (EVAL_INPUT_DAYS-day seed).
  4. Generate one autoregressive trajectory per patient from the truncated seed.
  5. Divide each trajectory into EVAL_WINDOW_HOURS windows.
  6. Label each window: 1 if any ground-truth episode falls within ±EVAL_GRACE_HOURS.
  7. Pool all (patient, window) pairs → single AUROC/AUPRC per complication.
  8. Report mean across all complications that pass MIN_POSITIVES threshold.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR = os.path.join(PROJECT_ROOT, "emr_model")
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

from transform_emr.dataset import DataProcessor, EMRDataset
from transform_emr.config.dataset_config import TAK_REPO_PATH
from transform_emr.inference import generate

# ---------------------------------------------------------------------------
# Fixed evaluation constants (do not change)
# ---------------------------------------------------------------------------

EVAL_INPUT_DAYS  = 2      # days of patient history used as generation seed
EVAL_WINDOW_HOURS = 24.0  # non-overlapping prediction window size
EVAL_GRACE_HOURS  = 24.0  # tolerance added to each window edge for positive labelling
EVAL_MAX_LEN      = 500   # max generated steps per patient
EVAL_TEMPERATURE  = 1.0   # sampling temperature (no top-k filtering)
EVAL_MIN_POSITIVES = 3    # skip an outcome if fewer than this many positive windows exist


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(eval_ds, outcome_names):
    """
    Purpose: Build per-patient first-occurrence ground truth for each outcome.
    Method: Scans each patient's full (untruncated) token sequence from eval_ds.

    Args:
        eval_ds (EMRDataset): Full (untruncated) test dataset.
        outcome_names (list[str]): Outcome token strings to collect.

    Returns:
        dict: {patient_id: {outcome_name: first_time_hours or np.inf}}
    """
    outcome_set = set(outcome_names)
    tok_col     = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: np.inf for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set:
                t = row["TimePoint"]
                if t < patient_gt[tok]:
                    patient_gt[tok] = t
        gt[pid] = patient_gt
    return gt


def extract_ground_truth_episodes(eval_ds, outcome_names):
    """
    Purpose: Build per-patient all-occurrence ground truth (list of times) for each outcome.
    Method: Scans each patient's full (untruncated) token sequence from eval_ds.

    Args:
        eval_ds (EMRDataset): Full (untruncated) test dataset.
        outcome_names (list[str]): Outcome token strings to collect.

    Returns:
        dict: {patient_id: {outcome_name: [t1, t2, ...]}}  (empty list if never occurred)
    """
    outcome_set = set(outcome_names)
    tok_col     = "PositionToken" if "PositionToken" in next(iter(eval_ds.patient_groups.values())).columns else "Token"
    gt = {}
    for pid in eval_ds.patient_ids:
        df = eval_ds.patient_groups[pid]
        patient_gt = {n: [] for n in outcome_names}
        for _, row in df.iterrows():
            tok = row[tok_col]
            if tok in outcome_set:
                patient_gt[tok].append(row["TimePoint"])
        gt[pid] = patient_gt
    return gt


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def pooled_episode_auc(risk_df, gt_labels_episodes, outcome_names,
                        window_hours=EVAL_WINDOW_HOURS,
                        grace_hours=EVAL_GRACE_HOURS,
                        min_positives=EVAL_MIN_POSITIVES):
    """
    Purpose: Compute episode-level AUROC and AUPRC pooled across all patients and time windows.
    Method: Divides each generated trajectory into non-overlapping windows; labels each window
            by whether any ground-truth episode falls within ±grace_hours; pools all pairs.

    For each (patient, window) pair:
      score = max P_<outcome> within the window
      label = 1 if any ground-truth episode of that outcome falls in
              [win_start - grace_hours, win_end + grace_hours]

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        gt_labels_episodes (dict): {pid: {outcome: [t1, t2, ...]}} all episode times in hours.
        outcome_names (list[str]): Outcome names to evaluate.
        window_hours (float): Duration of each evaluation window in hours.
        grace_hours (float): Extra tolerance added to each window edge for positive labelling.
        min_positives (int): Skip outcome if fewer than this many positive windows exist.

    Returns:
        pd.DataFrame: Indexed by outcome, columns: auroc, auprc, n_pos_windows, n_neg_windows.
    """
    gen_df = risk_df[risk_df["IsInput"] == 0].copy()
    p_cols = [f"P_{n}" for n in outcome_names]

    t_min = gen_df["TimePoint"].min()
    gen_df["_win"] = np.floor((gen_df["TimePoint"] - t_min) / window_hours).astype(int)

    # Max risk per (patient, window) — vectorised
    peak = (gen_df
            .groupby(["PatientId", "_win"])[p_cols]
            .max()
            .reset_index())
    peak["_t_start"] = t_min + peak["_win"] * window_hours
    peak["_t_end"]   = peak["_t_start"] + window_hours

    rows = []
    for name in outcome_names:
        pcol   = f"P_{name}"
        scores, labels = [], []
        for _, row in peak.iterrows():
            pid      = row["PatientId"]
            t_lo     = row["_t_start"] - grace_hours
            t_hi     = row["_t_end"]   + grace_hours
            episodes = gt_labels_episodes.get(pid, {}).get(name, [])
            label    = int(any(t_lo <= ep <= t_hi for ep in episodes))
            scores.append(row[pcol])
            labels.append(label)

        labels = np.array(labels)
        scores = np.array(scores)
        n_pos  = int(labels.sum())
        n_neg  = int((1 - labels).sum())

        if n_pos < min_positives:
            rows.append({"outcome": name, "auroc": np.nan, "auprc": np.nan,
                         "n_pos_windows": n_pos, "n_neg_windows": n_neg})
            continue

        rows.append({
            "outcome":       name,
            "auroc":         roc_auc_score(labels, scores),
            "auprc":         average_precision_score(labels, scores),
            "n_pos_windows": n_pos,
            "n_neg_windows": n_neg,
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("auroc", ascending=False)


def time_accuracy(risk_df, gt_labels, outcome_names):
    """
    Purpose: Compute mean absolute error between predicted and actual complication onset time.
    Method: For each patient where a complication occurred, finds the generated step with peak
            outcome-head probability and measures its distance from the ground-truth time.

    Args:
        risk_df (pd.DataFrame): Output of generate() with collect_risk_scores=True.
        gt_labels (dict): {pid: {outcome: first_time_hours or np.inf}}.
        outcome_names (list[str]): Outcome names to evaluate.

    Returns:
        pd.DataFrame: Indexed by outcome, columns: mae_hours, n_patients.
    """
    gen_df = risk_df[risk_df["IsInput"] == 0].copy()
    p_cols = [f"P_{n}" for n in outcome_names]
    idxmax = gen_df.groupby("PatientId")[p_cols].idxmax()

    rows = []
    for name in outcome_names:
        pcol   = f"P_{name}"
        pred_t = gen_df.loc[idxmax[pcol].dropna().astype(int), ["PatientId", "TimePoint"]]
        pred_t = pred_t.set_index("PatientId")["TimePoint"]

        errors = []
        for pid, pt in pred_t.items():
            gt_t = gt_labels.get(pid, {}).get(name, np.inf)
            if gt_t < np.inf:
                errors.append(abs(pt - gt_t))

        rows.append({
            "outcome":    name,
            "mae_hours":  np.mean(errors) if errors else np.nan,
            "n_patients": len(errors),
        })

    return pd.DataFrame(rows).set_index("outcome").sort_values("mae_hours")


# ---------------------------------------------------------------------------
# Main evaluation entry point (called by api.py)
# ---------------------------------------------------------------------------

def evaluate_on_test_set(model, tokenizer, val_temporal_raw, val_ctx_raw, scaler, checkpoint_dir):
    """
    Purpose: Full post-training evaluation on the held-out validation set.
    Method: Re-processes the raw val data twice — once untruncated (for ground truth) and
            once with EVAL_INPUT_DAYS truncation (for generation seed) — then generates
            risk curves and computes episode-level AUROC/AUPRC and onset-time MAE.

    Args:
        model: Trained GPT model (best available checkpoint, already loaded).
        tokenizer (EMRTokenizer): Fitted tokenizer (same as used during training).
        val_temporal_raw (pd.DataFrame): Raw (unprocessed) val temporal events.
        val_ctx_raw (pd.DataFrame): Raw (unprocessed) val context features.
        scaler: Fitted StandardScaler from training (loaded from checkpoints/scaler.pkl).
        checkpoint_dir (str): Path to checkpoints directory.

    Returns:
        dict with keys:
            mean_auroc (float)      : mean per-complication AUROC  [primary, higher is better]
            mean_auprc (float)      : mean per-complication AUPRC  [secondary, higher is better]
            mean_mae_hours (float)  : mean onset-prediction MAE    [tertiary, lower is better]
            auc_table (pd.DataFrame): per-outcome AUROC/AUPRC/n_windows table
            mae_table (pd.DataFrame): per-outcome MAE/n_patients table
    """
    # -- Full dataset (untruncated, for ground truth extraction) --
    print("[Eval] Processing full val sequences (ground truth)...")
    full_proc = DataProcessor(
        val_temporal_raw.copy(), val_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
    )
    full_temporal_df, full_ctx_df = full_proc.run()
    eval_ds_full = EMRDataset(full_temporal_df, full_ctx_df, tokenizer=tokenizer)

    # -- Truncated dataset (EVAL_INPUT_DAYS seed for generation) --
    print(f"[Eval] Processing truncated val sequences ({EVAL_INPUT_DAYS}-day input)...")
    trunc_proc = DataProcessor(
        val_temporal_raw.copy(), val_ctx_raw.copy(),
        scaler=scaler,
        tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=checkpoint_dir,
        max_input_days=EVAL_INPUT_DAYS,
    )
    trunc_temporal_df, trunc_ctx_df = trunc_proc.run()
    eval_ds_input = EMRDataset(trunc_temporal_df, trunc_ctx_df, tokenizer=tokenizer)

    # -- Generate risk curves --
    print("[Eval] Generating risk curves...")
    model.eval()
    risk_df = generate(
        model, eval_ds_input,
        max_len=EVAL_MAX_LEN,
        temperature=EVAL_TEMPERATURE,
        top_k=None,
        rep_decay=0.6,
        collect_risk_scores=True,
    )
    print(f"[Eval] Generated {len(risk_df)} rows for {risk_df['PatientId'].nunique()} patients.")

    outcome_names = model.outcome_names

    # -- Extract ground truth --
    gt_first    = extract_ground_truth(eval_ds_full, outcome_names)
    gt_episodes = extract_ground_truth_episodes(eval_ds_full, outcome_names)

    # -- Compute metrics --
    print("[Eval] Computing episode-level AUC and time accuracy...")
    auc_table = pooled_episode_auc(risk_df, gt_episodes, outcome_names)
    mae_table = time_accuracy(risk_df, gt_first, outcome_names)

    mean_auroc     = float(auc_table["auroc"].mean(skipna=True))
    mean_auprc     = float(auc_table["auprc"].mean(skipna=True))
    mean_mae_hours = float(mae_table["mae_hours"].mean(skipna=True))

    # Summarise per-outcome for the log
    print("[Eval] Per-outcome AUROC:")
    for outcome, row in auc_table.iterrows():
        if not np.isnan(row["auroc"]):
            print(f"  {outcome:<45} AUROC={row['auroc']:.3f}  AUPRC={row['auprc']:.3f}")

    return dict(
        mean_auroc=mean_auroc,
        mean_auprc=mean_auprc,
        mean_mae_hours=mean_mae_hours,
        auc_table=auc_table,
        mae_table=mae_table,
    )
