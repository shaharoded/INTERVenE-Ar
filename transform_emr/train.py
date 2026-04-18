"""
train.py
=====================
Three-phase transformer training pipeline:
Phase-1 : train_embedder()        →  checkpoints/phase1/ckpt_best.pt
Phase-2 : pretrain_transformer()  →  checkpoints/phase2/ckpt_best.pt
Phase-3 : finetune_transformer()  →  checkpoints/phase3/ckpt_best.pt

Responsibilities:
  prepare_data()  — loads CSVs, runs DataProcessor, builds/loads tokenizer,
                    performs stratified patient split, returns (train_ds, val_ds, tokenizer).
  run_training()  — owns all DataLoader creation (one per phase with correct oversample
                    settings), then calls phase_one / phase_two / phase_three in sequence.

Phase-2 uses oversample=True (WeightedRandomSampler) for balanced rare-outcome batches.
Phase-3 uses oversample=False (natural distribution) so pos_weight in BCEWithLogitsLoss
correctly compensates for class imbalance without double-counting.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT, pretrain_transformer, finetune_transformer
from transform_emr.utils import *
from transform_emr.config.model_config import *
from transform_emr.config.dataset_config import (
    TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE, TAK_REPO_PATH, OUTCOMES, OUTCOME_RARE_THRESHOLD_PCT
)

def summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer):
    """
    Prints summary statistics about your train/val split:
    - Patient counts
    - Record counts
    - Context shapes
    - Event count per patient (min/max/avg)
    - Token coverage (raw, concept, value, position)
    """

    print("✅ Data Split Summary")
    print(f"  - Train patients: {len(train_ids)}")
    print(f"  - Val patients:   {len(val_ids)}")

    print(f"  - Train records:  {len(train_ds.tokens_df):,}")
    print(f"  - Val records:    {len(val_ds.tokens_df):,}")

    # Per-patient record count stats
    train_counts = train_ds.tokens_df.groupby('PatientID').size()
    val_counts = val_ds.tokens_df.groupby('PatientID').size()

    print(f"\n📊 Train patient records:")
    print(f"  - Min:     {train_counts.min()}")
    print(f"  - Max:     {train_counts.max()}")
    print(f"  - Mean:    {train_counts.mean():.1f}")
    print(f"  - Median:  {train_counts.median()}")

    print(f"\n📊 Val patient records:")
    print(f"  - Min:     {val_counts.min()}")
    print(f"  - Max:     {val_counts.max()}")
    print(f"  - Mean:    {val_counts.mean():.1f}")
    print(f"  - Median:  {val_counts.median()}")

    # Token vocab sizes (from tokenizer)
    print(f"\n🧠 Vocabulary sizes:")
    print(f"  - Raw concepts:     {len(tokenizer.rawconcept2id):,}")
    print(f"  - Concepts:         {len(tokenizer.concept2id):,}")
    print(f"  - Concept+Value:    {len(tokenizer.value2id):,}")
    print(f"  - Full Tokens:      {len(tokenizer.token2id):,}")


def prepare_data(sample=False):
    print(f"[Pre-processing]: Reading dataset...")
    temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

    # --- SAMPLE RANDOM PATIENTS ---
    if sample:
        unique_pids = temporal_df["PatientID"].unique()
        rng = np.random.RandomState(42)  # for reproducibility
        sampled_pids = rng.choice(unique_pids, size=sample, replace=False)

        temporal_df = temporal_df[temporal_df["PatientID"].isin(sampled_pids)]
        ctx_df      = ctx_df[ctx_df["PatientID"].isin(sampled_pids)]

    if os.path.exists(os.path.join(CHECKPOINT_PATH, 'tokenizer.pt')):
        print(f"[Pre-processing]: Loading tokenizer from checkpoint...")
        tokenizer = EMRTokenizer.load()

        processor = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()

    else:
        print(f"[Pre-processing]: Building tokenizer...")
        processor = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer = EMRTokenizer.from_processed_df(temporal_df)
        tokenizer.save()

    print(f"[Pre-processing]: Building dataset...")
    pids = temporal_df["PatientID"].unique()

    # Stratified split: each patient's stratum = their rarest outcome by prevalence.
    # Guarantees at least 1 representative of each kept outcome in both train and val.
    patient_tokens = temporal_df.groupby("PatientID")["PositionToken"].apply(set)
    strat_labels = []
    for pid in pids:
        patient_outcomes = [n for n in OUTCOMES if n in patient_tokens.get(pid, set())]
        if patient_outcomes and tokenizer.outcome_patient_ratios:
            rarest = min(patient_outcomes,
                         key=lambda n: tokenizer.outcome_patient_ratios.get(n, 1.0))
            strat_labels.append(rarest)
        else:
            strat_labels.append("__common__")
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42, stratify=strat_labels)

    train_df  = temporal_df[temporal_df.PatientID.isin(train_ids)].copy()
    val_df    = temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx = ctx_df.loc[ctx_df.index.isin(train_ids)]
    val_ctx   = ctx_df.loc[ctx_df.index.isin(val_ids)]

    train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_df, val_ctx, tokenizer=tokenizer)

    summarize_patient_data_split(train_ds, val_ds, train_ids, val_ids, tokenizer)

    MODEL_CONFIG["ctx_dim"] = int(train_ds.context_df.shape[1])
    print(f"[Pre-processing]: Auto-set MODEL_CONFIG['ctx_dim'] = {MODEL_CONFIG['ctx_dim']}")

    return train_ds, val_ds, tokenizer

def phase_one(embedder, train_dl, val_dl, resume=True):
    return train_embedder(
        embedder=embedder,
        train_loader=train_dl,
        val_loader=val_dl,
        resume=resume,
        checkpoint_path=PHASE1_CHECKPOINT,
        training_settings=TRAINING_SETTINGS
    )

def phase_two(model, train_dl, val_dl, resume=True):
    return pretrain_transformer(
                        model=model,
                        train_dl=train_dl,
                        val_dl=val_dl,
                        resume=resume,
                        checkpoint_path=PHASE2_CHECKPOINT,
                        training_settings=TRAINING_SETTINGS
                    )


def phase_three(model, phase3_train_dl, val_dl, resume=True):
    return finetune_transformer(
        model=model,
        train_dl=phase3_train_dl,
        val_dl=val_dl,
        resume=resume,
        checkpoint_path=PHASE3_CHECKPOINT,
        training_settings=TRAINING_SETTINGS,
    )


def _build_or_load_embedder(tokenizer):
    ckpt_last = Path(PHASE1_CHECKPOINT).resolve().parent / "ckpt_last.pt"
    if ckpt_last.exists():
        embedder, *_ = EMREmbedding.load(ckpt_last, tokenizer=tokenizer)
    else:
        embedder = EMREmbedding(
            tokenizer=tokenizer,
            ctx_dim=MODEL_CONFIG.get("ctx_dim"),
            time2vec_dim=MODEL_CONFIG.get("time2vec_dim"),
            embed_dim=MODEL_CONFIG.get("embed_dim"),
        )
    return embedder


def _build_or_load_transformer(embedder):
    ckpt_last = Path(PHASE2_CHECKPOINT).resolve().parent / "ckpt_last.pt"
    if ckpt_last.exists():
        model, *_ = GPT.load(ckpt_last, embedder=embedder)
    else:
        model = GPT(cfg=MODEL_CONFIG, embedder=embedder)
    return model


def run_training():
    """Run the full three-phase training pipeline."""
    train_ds, val_ds, tokenizer = prepare_data()

    bs = TRAINING_SETTINGS["batch_size"]
    train_dl    = get_dataloader(train_ds, batch_size=bs, collate_fn=collate_emr, oversample=False, bucket_batching=True)
    oversampled_train_dl = get_dataloader(train_ds, batch_size=bs, collate_fn=collate_emr, oversample=True, bucket_batching=True)
    val_dl               = get_dataloader(val_ds,   batch_size=bs, collate_fn=collate_emr, oversample=False, bucket_batching=True)

    # Phase 1 — embedder
    embedder = _build_or_load_embedder(tokenizer)
    embedder, _, _ = phase_one(embedder=embedder, train_dl=train_dl, val_dl=val_dl, resume=True)

    # Phase 2 — transformer (oversampled batches for rare-outcome balance)
    model = _build_or_load_transformer(embedder)
    model, _, _ = phase_two(model=model, train_dl=oversampled_train_dl, val_dl=val_dl, resume=True)

    # Phase 3 — outcome head (natural distribution; pos_weight handles class imbalance)
    model, _, _ = phase_three(model=model, phase3_train_dl=train_dl, val_dl=val_dl, resume=True)


if __name__ == "__main__":
    run_training()
