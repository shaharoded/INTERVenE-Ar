"""
bootstrap_ci.py — Patient-level bootstrap confidence intervals for the headline
AUROC/AUPRC, WITHOUT retraining. Imports evaluation.py's helpers (does not modify
the immutable file), regenerates per-patient (max_P, label) pairs for a given
checkpoint dir, validates the point estimate against the reported headline, then
resamples the test patients with replacement to get 95% CIs.

Usage:
    python bootstrap_ci.py <checkpoint_dir> [n_boot]
The checkpoint dir must contain phase1/, phase3/, processed_datasets.pt, scaler.pkl
(the QA cache's test_raw + tokenizer are reused; greedy decoding, no env overrides,
exactly matching evaluation.py's generate() call).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from pathlib import Path
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score, average_precision_score

from intervene_ar.dataset import DataProcessor, EMRDataset
from intervene_ar.embedder import EMREmbedding
from intervene_ar.transformer import InterveneGPT
from intervene_ar.inference import generate
from intervene_ar.config.dataset_config import TAK_REPO_PATH
import evaluation as ev

ckpt_dir = sys.argv[1].rstrip("/")
N_BOOT   = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
SEED_BOOT = 12345  # fixed so the CI is reproducible

def _pick(p_dir, name):
    best = Path(ckpt_dir) / p_dir / "ckpt_best.pt"
    last = Path(ckpt_dir) / p_dir / "ckpt_last.pt"
    return str(best if best.exists() else last)

print(f"[boot] checkpoint dir: {ckpt_dir} | n_boot={N_BOOT}")
cache = torch.load(os.path.join(ckpt_dir, "processed_datasets.pt"), map_location="cpu", weights_only=False)
tokenizer = cache["tokenizer"]
test_temporal_raw, test_ctx_raw = cache["test_raw"]
scaler = joblib_load(os.path.join(ckpt_dir, "scaler.pkl"))
print(f"[boot] vocab={len(tokenizer.token2id)} | test patients (raw)={test_temporal_raw['PatientId'].nunique()}")

embedder, *_ = EMREmbedding.load(_pick("phase1", "embedder"), tokenizer=tokenizer)
model, *_    = InterveneGPT.load(_pick("phase3", "model"), embedder=embedder)
model.eval()

# Replicate evaluate_on_test_set's data prep (full = GT, truncated = seed).
full_t, full_c = DataProcessor(test_temporal_raw.copy(), test_ctx_raw.copy(), scaler=scaler,
                               tak_repo_path=TAK_REPO_PATH, checkpoint_path=ckpt_dir).run()
eval_ds_full = EMRDataset(full_t, full_c, tokenizer=tokenizer)
trunc_t, trunc_c = DataProcessor(test_temporal_raw.copy(), test_ctx_raw.copy(), scaler=scaler,
                                 tak_repo_path=TAK_REPO_PATH, checkpoint_path=ckpt_dir,
                                 max_input_days=ev.EVAL_INPUT_DAYS).run()
eval_ds_input = EMRDataset(trunc_t, trunc_c, tokenizer=tokenizer)

print("[boot] generating risk curves (greedy, matches eval) ...")
risk_df = generate(model, eval_ds_input, max_len=ev.EVAL_MAX_LEN, temperature=ev.EVAL_TEMPERATURE,
                   top_k=None, rep_decay=0.6, collect_risk_scores=True)
gt_episodes = ev.extract_ground_truth_episodes(eval_ds_full, model.outcome_names)

outcome_names = [n for n in model.outcome_names if n not in ev.AUC_EXCLUDE]
gen_df  = risk_df[risk_df["IsInput"] == 0]
all_pids = list(risk_df["PatientId"].unique())
n_pat = len(all_pids)
p_cols = [f"P_{n}" for n in outcome_names]

# Per-patient max score (0 if no generated rows) — identical to per_patient_max_auc.
maxpp = {pid: {c: 0.0 for c in p_cols} for pid in all_pids}
if len(gen_df):
    g = gen_df.groupby("PatientId")[p_cols].max()
    for pid, row in g.iterrows():
        for c in p_cols:
            maxpp[pid][c] = float(row[c])

S = {n: np.array([maxpp[p][f"P_{n}"] for p in all_pids]) for n in outcome_names}
L = {n: np.array([int(len(gt_episodes.get(p, {}).get(n, [])) > 0) for p in all_pids]) for n in outcome_names}
min_pos = max(1, int(round((ev.OUTCOME_RARE_THRESHOLD_PCT / 100.0) * n_pat)))

def weighted_metrics(idx):
    """n_pos-weighted mean AUROC/AUPRC over outcomes passing min_positives in this sample."""
    aurocs, auprcs, w = [], [], []
    for n in outcome_names:
        lab, sc = L[n][idx], S[n][idx]
        npos, nneg = int(lab.sum()), int((1 - lab).sum())
        if npos < min_pos or nneg < min_pos:
            continue
        aurocs.append(roc_auc_score(lab, sc)); auprcs.append(average_precision_score(lab, sc)); w.append(npos)
    w = np.array(w, float)
    if w.sum() == 0:
        return np.nan, np.nan
    return float((np.array(aurocs) * w).sum() / w.sum()), float((np.array(auprcs) * w).sum() / w.sum())

full_idx = np.arange(n_pat)
pt_auroc, pt_auprc = weighted_metrics(full_idx)
print(f"\n[boot] POINT ESTIMATE  AUROC_w={pt_auroc:.4f}  AUPRC_w={pt_auprc:.4f}  (compare to reported headline)")

# Per-outcome point AUROC (for the per-outcome CI table).
per_out = {}
for n in outcome_names:
    per_out[n] = roc_auc_score(L[n], S[n]) if (L[n].sum() >= min_pos and (1 - L[n]).sum() >= min_pos) else np.nan

rng = np.random.RandomState(SEED_BOOT)
b_auroc, b_auprc = [], []
b_per = {n: [] for n in outcome_names}
for _ in range(N_BOOT):
    idx = rng.randint(0, n_pat, n_pat)  # resample patients with replacement
    a, p = weighted_metrics(idx)
    if not np.isnan(a):
        b_auroc.append(a); b_auprc.append(p)
    for n in outcome_names:
        lab, sc = L[n][idx], S[n][idx]
        if lab.sum() >= min_pos and (1 - lab).sum() >= min_pos:
            b_per[n].append(roc_auc_score(lab, sc))

def ci(arr):
    a = np.array(arr); return np.percentile(a, 2.5), np.percentile(a, 97.5)

lo, hi = ci(b_auroc); plo, phi = ci(b_auprc)
print(f"[boot] AUROC_w  {pt_auroc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (width {hi-lo:.4f})")
print(f"[boot] AUPRC_w  {pt_auprc:.4f}  95% CI [{plo:.4f}, {phi:.4f}]")
print("[boot] per-outcome AUROC 95% CI:")
for n in outcome_names:
    if b_per[n]:
        l, h = ci(b_per[n])
        print(f"   {n:<40} {per_out[n]:.3f}  [{l:.3f}, {h:.3f}]")
print("[boot] DONE")
