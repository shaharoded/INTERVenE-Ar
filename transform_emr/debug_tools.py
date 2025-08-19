"""
debug_tools.py
==============
Debug helpers for inspecting loss behaviour with MaskedFocalBCE/SetCE.
"""

import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, Any

from transform_emr.utils import (
    get_multi_hot_targets,               # multi-hot next-k targets
    compute_legality_masks_tf,           # legality for BCE masking
    apply_masks_to_logits,               # -inf illegal + bonus
    build_luts,                          # start/end/meal/conflict LUTs
)

from transform_emr.loss import (
    FocalBCELoss,                        # to inspect alpha/weights
    MaskedFocalBCE,                      # masked + pos/neg balanced focal BCE
    MaskedSetCE,                         # soft set-CE (optional comparison)
)

# ---------------------------- token weights snapshot ----------------------------

@torch.no_grad()
def summarize_token_weights(tokenizer):
    """Return quick view of counts, alpha, and manual weights for sanity."""
    crit = FocalBCELoss.from_counts(
        tokenizer.token_counts,
        token_weights=tokenizer.token_weights,
        gamma=1.0, reduction="none"
    )
    alpha   = crit.alpha.cpu()
    counts  = tokenizer.token_counts.cpu()
    weights = tokenizer.token_weights.cpu()

    top  = torch.topk(counts, k=min(10, counts.numel())).indices.tolist()
    rare = torch.topk(-counts, k=min(10, counts.numel())).indices.tolist()

    def rows(ixs):
        out = []
        for i in ixs:
            tok = tokenizer.id2token[i]
            out.append((i, tok, int(counts[i]), float(alpha[i]), float(weights[i])))
        return out

    return {"top_by_count": rows(top), "rare_by_count": rows(rare)}


# ----------------------------- helpers / buckets --------------------------------

def _bucket_preds(ids, tokenizer, luts):
    """Compact histogram of predicted token *types*."""
    is_start = luts["is_start"]; is_end = luts["is_end"]; meal_rank = luts["meal_rank"]
    buckets = {
        "PAD": tokenizer.pad_token_id,
        "MASK": getattr(tokenizer, "mask_token_id", None),
        "CTX": getattr(tokenizer, "ctx_token_id", None),
        "NULL": getattr(tokenizer, "null_token_id", None),
    }
    counter = Counter()
    for v in ids.view(-1).tolist():
        if v == buckets["PAD"]: counter["PAD"] += 1
        elif v == buckets["MASK"]: counter["MASK"] += 1
        elif v == buckets["CTX"]: counter["CTX"] += 1
        elif v == buckets["NULL"]: counter["NULL"] += 1
        elif meal_rank[v] >= 0: counter["MEAL"] += 1
        elif is_start[v]: counter["START"] += 1
        elif is_end[v]: counter["END"] += 1
        else: counter["OTHER"] += 1
    return counter


# --------------------------------- inspectors -----------------------------------

@torch.no_grad()
def inspect_minibatch(model, batch, luts, k_window: int) -> Dict[str, Any]:
    """
    Runs a forward on one minibatch and reports masked focal-BCE components,
    set mass on target set, and distribution diagnostics.
    """
    device = next(model.parameters()).device
    tok = model.embedder.tokenizer
    pad = tok.pad_token_id
    V   = len(tok.token2id)

    # ---- forward (mirrors train loop) ----
    logits, _ = model(
        raw_concept_ids=batch["raw_concept_ids"].to(device),
        concept_ids=batch["concept_ids"].to(device),
        value_ids=batch["value_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        abs_ts=batch["abs_ts"].to(device),
        context_vec=batch["context_vec"].to(device)
    )
    pred_logits = logits[:, 1:, :]           # [B,T,V]
    target_ids  = batch["targets"].to(device)  # [B,T]

    # Legality masks (use predict_block if present; else zeros)
    predict_block = luts.get("predict_block", torch.zeros(V, dtype=torch.bool))
    if predict_block.device != pred_logits.device:
        predict_block = predict_block.to(pred_logits.device)

    illegal, bonus = compute_legality_masks_tf(
        target_ids, luts["is_start"], luts["is_end"], luts["base_id"],
        luts["start_ids_per_base"], luts["end_ids_per_base"],
        luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
        luts["conflict_mat"], predict_block
    )
    pred_logits = apply_masks_to_logits(pred_logits, illegal, bonus)
    pred_ids = pred_logits.argmax(-1)        # [B,T]

    # Multi-hot next-K targets
    multi_hot = get_multi_hot_targets(
        position_ids=target_ids, padding_idx=pad,
        vocab_size=pred_logits.size(-1), k=k_window
    ).masked_fill(illegal, 0.0)

    # Allowed mask: legal classes & non-PAD steps
    allowed = (~illegal) & (target_ids != pad).unsqueeze(-1)  # [B,T,V] bool

    # ---- stats about steps/classes ----
    pos_per_step     = multi_hot.sum(-1)                      # [B,T]
    allowed_per_step = allowed.sum(-1)                        # [B,T]
    nonpad           = (target_ids != pad)

    # percentiles for allowed vocab size (non-PAD only)
    allowed_sz_np = allowed_per_step[nonpad].float().view(-1).cpu()
    if allowed_sz_np.numel():
        p50 = float(torch.quantile(allowed_sz_np, 0.5))
        p90 = float(torch.quantile(allowed_sz_np, 0.9))
    else:
        p50 = p90 = 0.0

    stats = {
        "fraction_pad_steps": float((target_ids == pad).float().mean().cpu()),
        "mean_allowed_vocab": float(allowed_per_step.float().mean().cpu()),
        "allowed_vocab_p50": p50,
        "allowed_vocab_p90": p90,
        "mean_positives_per_step": float(pos_per_step.float().mean().cpu()),
        "frac_zero_positive_steps": float((pos_per_step == 0).float().mean().cpu()),
        "frac_zero_positive_nonpad": float(((pos_per_step == 0) & nonpad).float().sum().cpu() /
                                           (nonpad.float().sum().cpu() + 1e-9)),
        "pred_bucket_hist": _bucket_preds(pred_ids, tok, luts),
    }

    # ---- masked focal BCE (our actual objective now) ----
    crit = MaskedFocalBCE.from_counts(
        counts=tok.token_counts, token_weights=tok.token_weights,
        beta=0.999, min_count=5, clip_max=8.0, gamma=1.0,
        tau=0.5, neg_bounds=(0.05, 0.5), label_smoothing=0.01,
        hard_neg_k=None
    ).to(device)

    loss_bce, info = crit(pred_logits, multi_hot, allowed)  # scalar + diagnostics

    stats.update({
        "loss_bce_masked": float(loss_bce.detach().cpu()),
        "bce_loss_pos": info["loss_pos"],
        "bce_loss_neg": info["loss_neg"],
        "bce_lambda_neg": info["lambda_neg"],
        "bce_Np": info["Np"],
        "bce_Nn": info["Nn"],
    })

    # ---- set probability on target set (diagnostic only) ----
    #   p_set = sum_{v in positives} p(v | allowed)
    logits_allowed = pred_logits.masked_fill(~allowed, float("-inf"))
    logZ  = torch.logsumexp(logits_allowed, dim=-1)                          # [B,T]
    in_set = (multi_hot > 0) & allowed
    logSet = torch.logsumexp(logits_allowed.masked_fill(~in_set, float("-inf")), dim=-1)  # [B,T]
    p_set  = (logSet - logZ).exp()                                           # [B,T] in [0,1]
    if nonpad.any():
        stats["p_set_mean_nonpad"] = float(p_set[nonpad].mean().cpu())
        stats["p_set_p90_nonpad"]  = float(torch.quantile(p_set[nonpad], 0.9).cpu())
    else:
        stats["p_set_mean_nonpad"] = 0.0
        stats["p_set_p90_nonpad"]  = 0.0

    # ---- class-level target mass in this batch (top-15) ----
    B, T, V = pred_logits.shape
    target_sum_per_class = multi_hot.sum(dim=(0,1))
    topk = min(15, V)
    vals, idxs = torch.topk(target_sum_per_class, k=topk)
    stats["batch_target_sum_per_class_top"] = [
        (int(i), tok.id2token[int(i)], int(v)) for v, i in zip(vals.tolist(), idxs.tolist())
    ]

    return stats


@torch.no_grad()
def inspect_epoch(model, loader, k_window: int, max_batches: int = 3):
    """Run inspect_minibatch on a few batches and print concise diagnostics."""
    device = next(model.parameters()).device
    tok = model.embedder.tokenizer

    # LUTs once
    luts = build_luts(tok)
    luts = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in luts.items()}

    # token-weight snapshot (sanity)
    print("=== Token weights / counts snapshot ===")
    w = summarize_token_weights(tok)
    print("Top by count:", w["top_by_count"][:5])
    print("Rare by count:", w["rare_by_count"][:5])

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        stats = inspect_minibatch(model, batch, luts, k_window)
        print(f"\n[Batch {bi}]")
        for k, v in stats.items():
            print(f"  {k}: {v}")