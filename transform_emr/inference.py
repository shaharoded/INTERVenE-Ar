import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.utils import (
    build_luts,
    build_rep_penalty,
    init_legality_state_batched,
    build_illegal_mask_batched,
    update_legality_state_batched,
    build_rep_penalty_batched,
)


def get_token_embedding(embedder, token: str) -> torch.Tensor:
    """
    Returns the embedding vector of a specific token from a trained embedder.

    Args:
        embedder (EMREmbedding): A trained EMREmbedding model.
        token (str): The string token to lookup.

    Returns:
        torch.Tensor: Embedding vector of shape [embed_dim].
    """
    if token not in embedder.token2id:
        raise ValueError(f"Token '{token}' not found in vocabulary.")

    token_id = embedder.tokenizer.token2id[token]
    embedding = embedder.token_embed.weight[token_id].detach()
    return embedding


# ───────── shared generation helpers ────────────────────────────────────── #

# ───────── scalar compatibility wrappers (used by unit tests) ───────────── #
# These thin wrappers adapt the old scalar inference API to the batched utils
# so existing tests continue to work without duplicating logic.

def _build_illegal_mask(luts, open_counts, next_meal_rank, pad_id, mask_id, device):
    """Scalar wrapper around build_illegal_mask_batched (B=1)."""
    oc  = open_counts.unsqueeze(0).long().to(device)   # [1, nb]
    nmr = torch.tensor([next_meal_rank if next_meal_rank is not None else -1],
                       dtype=torch.long, device=device)
    illegal = build_illegal_mask_batched(luts, oc, nmr, pad_id, mask_id)
    return illegal[0]   # [V]


def _update_legality_state(luts, tid: int, open_counts, next_meal_rank, K: int):
    """Scalar wrapper around update_legality_state_batched (B=1).  Mutates open_counts."""
    device   = open_counts.device
    tok_ids  = torch.tensor([tid], dtype=torch.long, device=device)
    oc       = open_counts.unsqueeze(0).long()
    nmr      = torch.tensor([next_meal_rank if next_meal_rank is not None else -1],
                            dtype=torch.long, device=device)
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    update_legality_state_batched(luts, tok_ids, oc, nmr, finished)
    open_counts.copy_(oc[0].to(open_counts.dtype))
    return nmr[0].item() if nmr[0].item() >= 0 else next_meal_rank


def _decode_token_components(tok, token_str: str):
    """Decode a position-token string into (concept_id, value_id)."""
    parts = token_str.split("_")
    concept = (
        "_".join(parts[:-2])
        if len(parts) >= 2 and parts[-2] in ("STATE", "TREND", "CONTEXT", "EVENT", "PATTERN")
        else "_".join(parts)
    )
    value = (
        "_".join(parts[:-1])
        if len(parts) >= 2 and parts[-1] in ("START", "END")
        else "_".join(parts)
    )
    return (
        tok.concept2id.get(concept, tok.mask_token_id),
        tok.value2id.get(value, tok.mask_token_id)
    )


# ───────── batched seed preparation ─────────────────────────────────────── #

def _prepare_batch_seeds(batch_pids, dataset, tok, device):
    """
    Pad a batch of patient seed sequences to the same length.

    Returns
    -------
    pos_ids          : [B, T_max] long
    parent_raw_ids   : [B, T_max, P] long
    concept_ids      : [B, T_max] long
    value_ids        : [B, T_max] long
    abs_ts           : [B, T_max] float  (normalised to [0,1])
    ctx_vecs         : [B, ctx_dim] float
    seed_lens        : [B] long  — real (unpadded) length of each patient's seed
    """
    pad_id = tok.pad_token_id
    P = tok.tokenid2parent_raw_ids.shape[1]

    seqs = []
    for pid in batch_pids:
        df = dataset.patient_groups[pid]
        pos   = torch.tensor(df["PositionID"].tolist(), dtype=torch.long)
        raw   = tok.tokenid2parent_raw_ids[pos]                          # [T, P]
        con   = torch.tensor(df["ConceptID"].tolist(),  dtype=torch.long)
        val   = torch.tensor(df["ValueID"].tolist(),    dtype=torch.long)
        ts    = torch.tensor(df["TimePoint"].tolist(),  dtype=torch.float32) / 336.0
        ctx   = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32)
        seqs.append((pos, raw, con, val, ts, ctx))

    B      = len(seqs)
    T_max  = max(s[0].shape[0] for s in seqs)
    seed_lens = torch.tensor([s[0].shape[0] for s in seqs], dtype=torch.long)

    pos_ids        = torch.full((B, T_max), pad_id, dtype=torch.long)
    parent_raw_ids = torch.full((B, T_max, P), pad_id, dtype=torch.long)
    concept_ids    = torch.full((B, T_max), pad_id, dtype=torch.long)
    value_ids      = torch.full((B, T_max), pad_id, dtype=torch.long)
    abs_ts         = torch.zeros(B, T_max, dtype=torch.float32)
    ctx_vecs       = torch.stack([s[5] for s in seqs], dim=0)

    for i, (pos, raw, con, val, ts, _) in enumerate(seqs):
        Ti = pos.shape[0]
        pos_ids[i, :Ti]        = pos
        parent_raw_ids[i, :Ti] = raw
        concept_ids[i, :Ti]    = con
        value_ids[i, :Ti]      = val
        abs_ts[i, :Ti]         = ts

    return (
        pos_ids.to(device),
        parent_raw_ids.to(device),
        concept_ids.to(device),
        value_ids.to(device),
        abs_ts.to(device),
        ctx_vecs.to(device),
        seed_lens.to(device),
    )


def _sample_tokens(next_logits, temperature, top_k):
    """
    Sample (or argmax) the next token from [B, V] logits.
    Returns LongTensor [B].
    """
    if top_k:
        topv, topi = torch.topk(next_logits, top_k, dim=-1)
        probs = F.softmax(topv / temperature, dim=-1)
        idx   = torch.multinomial(probs, 1).squeeze(-1)          # [B]
        return topi[torch.arange(topi.shape[0], device=topi.device), idx]
    else:
        return torch.argmax(next_logits / temperature, dim=-1)   # [B]


# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def infer_event_stream(model,
                       dataset,
                       max_len=500,
                       temperature=1.0,
                       top_k=None,
                       rep_decay=0.6,
                       batch_size=16,
                       tqdm_position=0,
                       tqdm_desc='Generating'):
    """
    Generates a stream of events for each patient in the dataset using batched
    autoregressive decoding with a KV cache and FP16 autocast.

    Patients are processed in groups of ``batch_size``.  Within each group the
    seed sequences are right-padded to the same length for a single prefill pass,
    then all patients decode in parallel token-by-token.  Finished patients are
    kept as dummy entries (padded with PAD) so the batch shape stays uniform.

    Legality rules (applied per patient):
      • Interval: no END without START, no duplicate STARTs.
      • Concept conflict: no START on conceptX_value1 if another conceptX_value2 is open.
      • Meal cycle: strict cyclic ordering.
      • Repetition: soft penalty on the last ``window`` generated tokens.

    Args:
        model      : Trained GPT model (with .forward_with_cache()).
        dataset    : EMRDataset object.
        max_len    : Maximum new tokens to generate per patient.
        temperature, top_k : Sampling controls (argmax if top_k is None).
        rep_decay  : Repetition penalty strength (0 → disabled).
        batch_size : Number of patients processed in parallel.
        tqdm_position, tqdm_desc : tqdm controls.

    Returns:
        DataFrame with PatientID, Step, Token, TimePoint, IsInput, IsOutcome, IsTerminal.
    """
    autocast_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    device    = next(model.parameters()).device
    tok       = model.embedder.tokenizer
    luts      = build_luts(tok)
    luts      = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    id2token     = tok.id2token
    token2id     = tok.token2id
    outcome_ids  = {token2id[o] for o in OUTCOMES        if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    terminal_set = torch.tensor(sorted(terminal_ids), dtype=torch.long, device=device)
    pad_id       = tok.pad_token_id
    mask_id      = tok.mask_token_id

    rows = []
    all_pids = dataset.patient_ids

    with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda',
                        dtype=autocast_dtype, enabled=torch.cuda.is_available()):

        for batch_start in tqdm(range(0, len(all_pids), batch_size),
                                desc=tqdm_desc, position=tqdm_position,
                                leave=False, dynamic_ncols=True):

            batch_pids = all_pids[batch_start:batch_start + batch_size]
            B = len(batch_pids)

            # ── prepare padded seed tensors ───────────────────────────────
            pos_ids, parent_raw_ids, concept_ids, value_ids, abs_ts, ctx_vecs, seed_lens = \
                _prepare_batch_seeds(batch_pids, dataset, tok, device)

            # ── log input tokens ──────────────────────────────────────────
            for bi, pid in enumerate(batch_pids):
                Ti = seed_lens[bi].item()
                for i in range(Ti):
                    tid = pos_ids[bi, i].item()
                    rows.append({
                        "PatientId":  pid,
                        "Step":       i + 1,
                        "TimePoint":  abs_ts[bi, i].item() * 336.0,
                        "Token":      id2token.get(tid, f"<UNK_{tid}>"),
                        "IsInput":    1,
                        "IsOutcome":  int(tid in outcome_ids),
                        "IsTerminal": int(tid in terminal_ids),
                    })
                    if tid in terminal_ids:
                        break

            # ── mark patients whose seed already ends with a terminal ─────
            last_valid_idx = (seed_lens - 1).clamp(min=0)                   # [B]
            last_toks      = pos_ids[torch.arange(B, device=device), last_valid_idx]  # [B]
            finished = torch.isin(last_toks, terminal_set)                   # [B]

            if finished.all():
                continue

            # ── prefill: one forward pass over all B padded seeds ─────────
            logits_pre, abs_t_pre, _, _, past_kvs = model.forward_with_cache(
                parent_raw_ids=parent_raw_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=pos_ids,
                abs_ts=abs_ts,
                context_vec=ctx_vecs,
            )
            # logits_pre [B, T_max, V]; abs_t_pre [B, T_max]
            # Extract predictions at each patient's last valid seed position
            next_logits    = logits_pre[torch.arange(B, device=device), last_valid_idx, :]   # [B, V]
            current_abs_ts = abs_t_pre[torch.arange(B, device=device), last_valid_idx]       # [B]
            last_seed_ts   = abs_ts[torch.arange(B, device=device), last_valid_idx]          # [B]
            current_abs_ts = torch.maximum(current_abs_ts, last_seed_ts)

            # ── KV-cache mask: starts as the seed pad mask ────────────────
            # True = valid token (not padding)
            cache_mask = (pos_ids != pad_id)   # [B, T_max]

            # ── init batched legality state from seed ─────────────────────
            open_counts, next_meal_rank = init_legality_state_batched(luts, pos_ids)

            # ── repetition state ─────────────────────────────────────────
            last_tokens_batch = [[] for _ in range(B)]

            # ── decode loop ───────────────────────────────────────────────
            steps = 0
            while steps < max_len and not finished.all():

                # Apply illegal mask (per-patient)
                illegal = build_illegal_mask_batched(luts, open_counts, next_meal_rank,
                                                     pad_id, mask_id)
                next_logits = next_logits.masked_fill(illegal, float("-inf"))

                # Repetition penalty
                if rep_decay and rep_decay > 0:
                    rep_vec = build_rep_penalty_batched(last_tokens_batch, V=next_logits.size(-1),
                                                        window=5, strength=rep_decay, device=device)
                    next_logits = next_logits - rep_vec

                # For finished patients, force PAD (output ignored anyway)
                if finished.any():
                    next_logits[finished] = float("-inf")
                    next_logits[finished, pad_id] = 0.0

                # Sample
                next_token_ids = _sample_tokens(next_logits, temperature, top_k)  # [B]

                # Update legality state (skips finished patients)
                open_counts, next_meal_rank = update_legality_state_batched(
                    luts, next_token_ids, open_counts, next_meal_rank, finished)

                # Log and update rep state for each active patient
                for bi in range(B):
                    if finished[bi]:
                        continue
                    tid     = next_token_ids[bi].item()
                    tok_str = id2token.get(tid, f"<UNK_{tid}>")
                    rows.append({
                        "PatientId":  batch_pids[bi],
                        "Step":       seed_lens[bi].item() + steps + 1,
                        "TimePoint":  current_abs_ts[bi].item() * 336.0,
                        "Token":      tok_str,
                        "IsInput":    0,
                        "IsOutcome":  int(tid in outcome_ids),
                        "IsTerminal": int(tid in terminal_ids),
                    })
                    last_tokens_batch[bi].append(tid)

                # Mark newly finished patients
                is_terminal_step = torch.isin(next_token_ids, terminal_set)
                finished = finished | is_terminal_step

                steps += 1
                if finished.all():
                    break

                # ── embed the new tokens for the next decode step ─────────
                # Use LUT for concept/value ids (same logic as _decode_token_components)
                c_ids_new = luts["tok2concept"][next_token_ids].clamp(min=0)
                c_ids_new[luts["tok2concept"][next_token_ids] < 0] = mask_id
                v_ids_new = luts["tok2value"][next_token_ids].clamp(min=0)
                v_ids_new[luts["tok2value"][next_token_ids] < 0] = mask_id

                par_new = tok.tokenid2parent_raw_ids[next_token_ids].to(device)  # [B, P]
                # Finished patients get PAD token input (output ignored)
                par_new[finished] = tok.tokenid2parent_raw_ids[pad_id].to(device)
                c_ids_new[finished] = pad_id
                v_ids_new[finished] = pad_id
                pos_new  = next_token_ids.clone()
                pos_new[finished] = pad_id

                abs_ts_new = current_abs_ts.unsqueeze(1)   # [B, 1]

                # Extend cache mask: new token is always valid
                new_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)
                cache_mask = torch.cat([cache_mask, new_valid], dim=1)  # [B, T_cache+1]

                # Decode forward with KV cache
                logits_dec, abs_t_dec, _, _, past_kvs = model.forward_with_cache(
                    parent_raw_ids=par_new.unsqueeze(1),     # [B, 1, P]
                    concept_ids=c_ids_new.unsqueeze(1),      # [B, 1]
                    value_ids=v_ids_new.unsqueeze(1),        # [B, 1]
                    position_ids=pos_new.unsqueeze(1),       # [B, 1]
                    abs_ts=abs_ts_new,                       # [B, 1]
                    context_vec=ctx_vecs,
                    past_kvs=past_kvs,
                    cache_key_pad_mask=cache_mask,
                )

                next_logits = logits_dec[:, 0, :]           # [B, V]
                new_abs_t   = abs_t_dec[:, 0]               # [B]
                current_abs_ts = torch.maximum(new_abs_t, current_abs_ts)

            # ── fallback: inject terminal for patients that hit max_len ───
            if steps == max_len and len(terminal_ids) > 0:
                for bi in range(B):
                    if finished[bi]:
                        continue
                    term_list  = list(terminal_ids)
                    best_logit = next_logits[bi, term_list]
                    best_tid   = term_list[int(torch.argmax(best_logit))]
                    rows.append({
                        "PatientId":  batch_pids[bi],
                        "Step":       seed_lens[bi].item() + steps + 1,
                        "TimePoint":  current_abs_ts[bi].item() * 336.0,
                        "Token":      id2token[best_tid],
                        "IsInput":    0,
                        "IsOutcome":  1,
                        "IsTerminal": 1,
                    })

    return pd.DataFrame(rows)


@torch.no_grad()
def generate_risk_curves(model,
                         dataset,
                         max_len=500,
                         temperature=1.0,
                         top_k=None,
                         rep_decay=0.6,
                         batch_size=16,
                         tqdm_position=0,
                         tqdm_desc='Risk Curves'):
    """
    Single-trajectory generation with outcome head risk scores at every step.

    Teacher-forced pass over the input seed (batched) supplies outcome probabilities
    for each observed position.  The autoregressive decode loop (batched, KV-cached,
    FP16) then supplies risk scores for each generated position.

    Returns
    -------
    pd.DataFrame with columns:
        PatientId, Step, TimePoint, Token, IsInput, IsTerminal,
        P_<outcome_name>  (one float column per outcome in model.outcome_names)
    """
    autocast_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    device      = next(model.parameters()).device
    tok         = model.embedder.tokenizer
    luts        = build_luts(tok)
    luts        = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    id2token     = tok.id2token
    token2id     = tok.token2id
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    terminal_set = torch.tensor(sorted(terminal_ids), dtype=torch.long, device=device)
    pad_id       = tok.pad_token_id
    mask_id      = tok.mask_token_id
    outcome_cols = [f"P_{n}" for n in model.outcome_names]

    rows     = []
    all_pids = dataset.patient_ids

    with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda',
                        dtype=autocast_dtype, enabled=torch.cuda.is_available()):

        for batch_start in tqdm(range(0, len(all_pids), batch_size),
                                desc=tqdm_desc, position=tqdm_position,
                                leave=False, dynamic_ncols=True):

            batch_pids = all_pids[batch_start:batch_start + batch_size]
            B = len(batch_pids)

            # ── prepare padded seed tensors ───────────────────────────────
            pos_ids, parent_raw_ids, concept_ids, value_ids, abs_ts, ctx_vecs, seed_lens = \
                _prepare_batch_seeds(batch_pids, dataset, tok, device)

            last_valid_idx = (seed_lens - 1).clamp(min=0)   # [B]

            # ── teacher-forced pass for input risk scores ─────────────────
            # Use the regular forward (no cache needed — single full pass).
            _, _, input_outcome_logits, _, _ = model.forward_with_cache(
                parent_raw_ids=parent_raw_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=pos_ids,
                abs_ts=abs_ts,
                context_vec=ctx_vecs,
            )
            # input_outcome_logits [B, T_max, K]
            input_probs = torch.sigmoid(input_outcome_logits)   # [B, T_max, K]

            # ── log input tokens with risk scores ─────────────────────────
            for bi, pid in enumerate(batch_pids):
                Ti = seed_lens[bi].item()
                for i in range(Ti):
                    tid = pos_ids[bi, i].item()
                    row = {
                        "PatientId":  pid,
                        "Step":       i + 1,
                        "TimePoint":  abs_ts[bi, i].item() * 336.0,
                        "Token":      id2token.get(tid, f"<UNK_{tid}>"),
                        "IsInput":    1,
                        "IsTerminal": int(tid in terminal_ids),
                    }
                    for j, col in enumerate(outcome_cols):
                        row[col] = input_probs[bi, i, j].item()
                    rows.append(row)
                    if tid in terminal_ids:
                        break

            # ── mark finished patients ────────────────────────────────────
            last_toks = pos_ids[torch.arange(B, device=device), last_valid_idx]
            finished  = torch.isin(last_toks, terminal_set)

            if finished.all():
                continue

            # ── prefill for generation (reuse forward_with_cache) ─────────
            # We already did the teacher-forced pass above; redo with cache to
            # get the KV state for decoding.
            logits_pre, abs_t_pre, _, _, past_kvs = model.forward_with_cache(
                parent_raw_ids=parent_raw_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=pos_ids,
                abs_ts=abs_ts,
                context_vec=ctx_vecs,
            )

            next_logits    = logits_pre[torch.arange(B, device=device), last_valid_idx, :]
            current_abs_ts = abs_t_pre[torch.arange(B, device=device), last_valid_idx]
            last_seed_ts   = abs_ts[torch.arange(B, device=device), last_valid_idx]
            current_abs_ts = torch.maximum(current_abs_ts, last_seed_ts)

            cache_mask     = (pos_ids != pad_id)                         # [B, T_max]
            open_counts, next_meal_rank = init_legality_state_batched(luts, pos_ids)
            last_tokens_batch = [[] for _ in range(B)]

            # ── decode loop ───────────────────────────────────────────────
            steps = 0
            while steps < max_len and not finished.all():

                illegal = build_illegal_mask_batched(luts, open_counts, next_meal_rank,
                                                     pad_id, mask_id)
                next_logits = next_logits.masked_fill(illegal, float("-inf"))

                if rep_decay and rep_decay > 0:
                    rep_vec = build_rep_penalty_batched(last_tokens_batch, V=next_logits.size(-1),
                                                        window=5, strength=rep_decay, device=device)
                    next_logits = next_logits - rep_vec

                if finished.any():
                    next_logits[finished] = float("-inf")
                    next_logits[finished, pad_id] = 0.0

                next_token_ids = _sample_tokens(next_logits, temperature, top_k)

                open_counts, next_meal_rank = update_legality_state_batched(
                    luts, next_token_ids, open_counts, next_meal_rank, finished)

                # Log generated tokens (outcome probs come from the decode pass below)
                # We record a placeholder row here and fill probs after the forward
                step_row_idx = {}   # bi -> index in rows (filled after forward)
                for bi in range(B):
                    if finished[bi]:
                        continue
                    tid     = next_token_ids[bi].item()
                    tok_str = id2token.get(tid, f"<UNK_{tid}>")
                    row = {
                        "PatientId":  batch_pids[bi],
                        "Step":       seed_lens[bi].item() + steps + 1,
                        "TimePoint":  current_abs_ts[bi].item() * 336.0,
                        "Token":      tok_str,
                        "IsInput":    0,
                        "IsTerminal": int(tid in terminal_ids),
                    }
                    for col in outcome_cols:
                        row[col] = 0.0   # placeholder — filled below
                    step_row_idx[bi] = len(rows)
                    rows.append(row)
                    last_tokens_batch[bi].append(tid)

                is_terminal_step = torch.isin(next_token_ids, terminal_set)
                finished = finished | is_terminal_step

                steps += 1
                if finished.all():
                    break

                # ── embed new tokens, decode forward ──────────────────────
                c_ids_new = luts["tok2concept"][next_token_ids].clamp(min=0)
                c_ids_new[luts["tok2concept"][next_token_ids] < 0] = mask_id
                v_ids_new = luts["tok2value"][next_token_ids].clamp(min=0)
                v_ids_new[luts["tok2value"][next_token_ids] < 0] = mask_id

                par_new = tok.tokenid2parent_raw_ids[next_token_ids].to(device)
                par_new[finished] = tok.tokenid2parent_raw_ids[pad_id].to(device)
                c_ids_new[finished] = pad_id
                v_ids_new[finished] = pad_id
                pos_new  = next_token_ids.clone()
                pos_new[finished] = pad_id

                abs_ts_new = current_abs_ts.unsqueeze(1)
                new_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)
                cache_mask = torch.cat([cache_mask, new_valid], dim=1)

                logits_dec, abs_t_dec, outcome_logits_dec, _, past_kvs = model.forward_with_cache(
                    parent_raw_ids=par_new.unsqueeze(1),
                    concept_ids=c_ids_new.unsqueeze(1),
                    value_ids=v_ids_new.unsqueeze(1),
                    position_ids=pos_new.unsqueeze(1),
                    abs_ts=abs_ts_new,
                    context_vec=ctx_vecs,
                    past_kvs=past_kvs,
                    cache_key_pad_mask=cache_mask,
                )

                next_logits    = logits_dec[:, 0, :]
                new_abs_t      = abs_t_dec[:, 0]
                current_abs_ts = torch.maximum(new_abs_t, current_abs_ts)

                # Fill in outcome probabilities for this step's rows
                step_probs = torch.sigmoid(outcome_logits_dec[:, 0, :])  # [B, K]
                for bi, row_idx in step_row_idx.items():
                    for j, col in enumerate(outcome_cols):
                        rows[row_idx][col] = step_probs[bi, j].item()

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import random
    import joblib
    from pathlib import Path
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT
    from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
    from transform_emr.config.model_config import *
    from transform_emr.config.dataset_config import *


    # Load test data
    print("Loading dataset...")
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Subset: Pick N random patients for this inference batch
    print("Getting subset...")
    patient_ids = df["PatientID"].unique()
    N = 10  # adjust as needed
    selected_ids = sorted(random.sample(list(patient_ids), N))

    df_subset = df[df["PatientID"].isin(selected_ids)].copy()
    ctx_subset = ctx_df.loc[selected_ids].copy()

    # Load tokenizer and scaler
    print("Loading resources...")
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
    scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

    # Run preprocessing for excel file
    print("Building testing dataset...")
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, tak_repo_path=TAK_REPO_PATH)
    df_test, ctx_df_test = processor.run()
    dataset_test = EMRDataset(df_test, ctx_df_test, tokenizer=tokenizer)

    # Run preprocessing for generation
    print("Building input dataset...")
    k_days = 5
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=k_days)
    df_subset, ctx_subset = processor.run()
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    print("Loading model and generating predictions...")
    embedder, _, _, _, _, _, _ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _, _ = GPT.load(PHASE3_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset, temperature=1.0, batch_size=16)

    # Save to Excel with two sheets
    output_path = Path(CHECKPOINT_PATH) / "inference_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, sheet_name="Generated Events", index=False)
        dataset_test.tokens_df.to_excel(writer, sheet_name="Input Events", index=False)

    print(f"Inference results saved to: {output_path}")
