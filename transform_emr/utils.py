"""
utils.py
==============

General util functions for the package
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import Counter


# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import (
    ADMISSION_TOKEN, TERMINAL_OUTCOMES, MEAL_TOKENS
)


def get_multi_hot_targets(position_ids: torch.Tensor,
                          padding_idx: int,
                          vocab_size: int,
                          k: int) -> torch.Tensor:
    """
    For each timestep t, mark all tokens that appear in positions [t+1, t+k]
    with a multi-hot vector (0/1) over the vocabulary.

    Parameters
    ----------
    position_ids : LongTensor, shape [B, T]
        Token ids, right-padded with `padding_idx`.
    padding_idx  : int
        The pad token id. It will be excluded from the targets.
    vocab_size   : int
        Size of the vocabulary.
    k            : int
        Lookahead window size.

    Returns
    -------
    targets : FloatTensor, shape [B, T, V]
        Multi-hot matrix. targets[b, t, v] == 1 if token v appears in
        any of the next k positions after t (inclusive of t+1, exclusive of t).
    """
    B, T = position_ids.shape
    device = position_ids.device

    # One-hot: [B, T, V]
    oh = F.one_hot(position_ids.clamp(min=0), num_classes=vocab_size).to(torch.float32)

    # Cumulative sum over time dimension
    csum = oh.cumsum(dim=1)  # [B, T, V]

    # Shifted cumulative sum by k steps to get counts at t+k
    pad_tail = torch.zeros(B, k, vocab_size, device=device, dtype=csum.dtype)
    csum_k = torch.cat([csum[:, k:], pad_tail], dim=1)  # [B, T, V]

    # Tokens in (t+1 .. t+k]  => difference of cum-sums
    future = (csum_k - csum).clamp_min_(0.0)

    # Remove pad token if present
    if 0 <= padding_idx < vocab_size:
        future[..., padding_idx] = 0.0

    # Convert to binary multi-hot
    targets = (future > 0).to(torch.float32)
    return targets


def build_mlm(ids, tokenizer, p=0.15):
    """
    Embedder MLM Mask Helper.
    Logic: Mask (later predict) tokens important to position that won't hurt the general timeline.
            Mask everything informative but admission, CTX or terminal tokens.
    ids:  (B,T) tensor - raw_concept_ids / concept_ids / value_ids / position_ids
    returns masked_ids (same shape)
    Masking strategy (BERT-style):
        80% → replace with [MASK]
        10% → keep original
        10% → replace with random token (not PAD)
    """
    device = ids.device
    never_mask_ids = {
        tokenizer.pad_token_id,
        tokenizer.ctx_token_id,
        tokenizer.null_token_id,
        tokenizer.token2id.get(ADMISSION_TOKEN),
        *[tokenizer.token2id[tok] for tok in TERMINAL_OUTCOMES],
    }
    keep = torch.zeros_like(ids, dtype=torch.bool)
    for tid in never_mask_ids:
        if tid is not None:
            keep |= (ids == tid)
    mask = (~keep) & (torch.rand_like(ids.float()) < p)
    masked = ids.clone()
    # 80 %
    rand = torch.rand_like(ids.float())
    mask80 = mask & (rand < 0.8)
    masked[mask80] = tokenizer.mask_token_id
    # 10 % random token
    mask10 = mask & (rand >= 0.8) & (rand < 0.9)
    vocab_size = len(tokenizer.token2id)
    random_tokens = torch.randint(1, vocab_size, size=ids.shape, device=device)
    masked[mask10] = random_tokens[mask10]
    # 10 % keep original – already satisfied
    return masked, mask        # mask is the bool tensor of *predict‑me* positions


def set_embedder_frozen(model, freeze: bool):
    for p in model.embedder.parameters():
        p.requires_grad = not freeze
    model.embedder.eval() if freeze else model.embedder.train()
    

def apply_cbm(batch, epoch, total_epochs, tokenizer, forbid_ids, max_p=0.15):
    """
    Transformer CBM (Curriculum by Masking) Helper.
    Logic: Mask tokens that won't hurt the general timeline or conflict with penalties.
        This list adds to the MLM forbidden list, as now we can't mask meals / intervals, 
        as we'll teach the model the contradicting them is OK.
    Randomly masks ratio% of *input* tokens (excluding forbid_ids list),
    replacing them with [MASK] token id and corresponding sub-ids.

    batch: dict of tensors
    tokenizer: EMRTokenizer
    epoch: int, epoch number
    total_epochs: int, total number of epochs from training_config
    forbid_ids: LongTensor of ids that must never be masked (PAD, CTX, ADMISSION, TERMINALS...)
    max_ratio: float, max masking ratio of the input
    """
    def cbm_ratio(epoch, total_epochs, max_p):
    # linear ramp to max_p over whole training (or part of it if you want)
        return max_p * (epoch / max(1, total_epochs - 1))
    
    p = cbm_ratio(epoch, total_epochs, max_p)
    if p <= 1e-6:  # nothing to do
        return batch

    pos_ids = batch["position_ids"]
    device = pos_ids.device
    B, T = pos_ids.shape

    mask_tok = tokenizer.mask_token_id
    pad_id   = tokenizer.pad_token_id

    # Eligible positions: not in forbid list and not PAD
    forbid = torch.zeros(tokenizer.token_weights.numel(), dtype=torch.bool, device=device)
    forbid[pad_id] = True
    forbid[mask_tok] = True
    forbid[forbid_ids] = True

    eligible = ~forbid[pos_ids]
    # sample
    prob = torch.rand(B, T, device=device) < p
    to_mask = prob & eligible

    # random 80/10/10 (like BERT)?
    # Keep it simple: replace all with [MASK]
    pos_ids_masked = pos_ids.clone()
    pos_ids_masked[to_mask] = mask_tok

    # map pos->raw/concept/value for [MASK]
    raw_mask = con_mask = val_mask = tokenizer.mask_token_id

    raw_ids = batch["raw_concept_ids"].clone()
    con_ids = batch["concept_ids"].clone()
    val_ids = batch["value_ids"].clone()

    raw_ids[to_mask] = raw_mask
    con_ids[to_mask] = con_mask
    val_ids[to_mask] = val_mask

    batch["position_ids"]    = pos_ids_masked
    batch["raw_concept_ids"] = raw_ids
    batch["concept_ids"]     = con_ids
    batch["value_ids"]       = val_ids
    return batch


def build_luts(tokenizer):
    """
    Pre-compute all LUTs (lookup tensors) needed for:
      • legality masks (intervals + meals + value-conflict)
      • CBM masking forbid list
      • Inference legality

    Returns
    -------
    luts : dict
        {
        # per-token
        "is_start"          : Bool[V]
        "is_end"            : Bool[V]
        "base_id"           : Long[V]   (-1 if not interval token)
        "meal_rank"         : Long[V]   (-1 non-meal, else 0..K-1)
        "meal_pred_rank"    : Long[V]   (-1 non-meal)

        # per-base (nb = #interval bases)
        "start_ids_per_base": Long[nb]  id of *_START  (‑1 if missing)
        "end_ids_per_base"  : Long[nb]  id of *_END    (‑1 if missing)
        "conflict_mat"      : Bool[nb, nb]  (same concept & different value)

        # misc
        "start_ids"         : Long[*]   all start ids (unordered)
        "end_ids"           : Long[*]   all end ids   (unordered)
        "K_meals"           : Long[]    scalar
        "forbid_mask_ids"   : Long[*]   tokens we never CBM-mask
        }
    """
    V = len(tokenizer.token2id)
    device = torch.device("cpu")  # keep CPU; move to GPU later where needed

    # --- per-token LUTs ------------------------------------------------------
    is_start   = torch.zeros(V, dtype=torch.bool, device=device)
    is_end     = torch.zeros(V, dtype=torch.bool, device=device)
    base_id    = torch.full((V,), -1, dtype=torch.long, device=device)
    tok2concept= torch.full((V,), -1, dtype=torch.long, device=device)
    tok2value  = torch.full((V,), -1, dtype=torch.long, device=device)

    # ---------- detect START/END tokens, map bases, and fill per‑token LUTs ----------
    base2idx = {}
    start_ids_list, end_ids_list = [], []

    for tok, tid in tokenizer.token2id.items():
        # Strip ONLY the suffix for interval tokens
        if tok.endswith("_START"):
            core = tok[:-6]
            is_start[tid] = True
        elif tok.endswith("_END"):
            core = tok[:-4]
            is_end[tid] = True
        else:
            core = tok

        # ----- concept & value ids -----
        parts = core.split("_")
        # concept key = first two chunks if exist, else the whole core (e.g. "GLUCOSE_TREND")
        concept_key = "_".join(parts[:2]) if len(parts) >= 2 else core
        # value key   = the full core (e.g. "GLUCOSE_TREND_INC")
        value_key   = core

        tok2concept[tid] = tokenizer.concept2id.get(concept_key, -1)
        tok2value[tid]   = tokenizer.value2id.get(value_key,   -1)

        # ----- interval base bookkeeping -----
        if is_start[tid] or is_end[tid]:
            base_idx = base2idx.setdefault(core, len(base2idx))
            base_id[tid] = base_idx
            if is_start[tid]:
                start_ids_list.append(tid)
            else:  # END
                end_ids_list.append(tid)

    # tensors of all *_START / *_END ids (unordered)
    start_ids = torch.tensor(start_ids_list, dtype=torch.long, device=device)
    end_ids   = torch.tensor(end_ids_list,   dtype=torch.long, device=device)

    # ---------- per‑base LUTs ----------
    nb = len(base2idx)
    start_ids_per_base = torch.full((nb,), -1, dtype=torch.long, device=device)
    end_ids_per_base   = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_concept       = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_value         = torch.full((nb,), -1, dtype=torch.long, device=device)

    # Fill per‑base arrays (first seen wins; START/END of same base share concept/value)
    for tid in range(V):
        b = base_id[tid].item()
        if b < 0:
            continue
        if is_start[tid]:
            start_ids_per_base[b] = tid
        elif is_end[tid]:
            end_ids_per_base[b] = tid
        if base_concept[b] < 0:
            base_concept[b] = tok2concept[tid]
        if base_value[b] < 0:
            base_value[b] = tok2value[tid]

    # ---------- conflict matrix: same concept, different value ----------
    if nb > 0:
        conf_mat = (base_concept[:, None] == base_concept[None, :]) & \
                (base_value[:,  None]  != base_value[None,  :])
    else:
        conf_mat = torch.zeros(0, 0, dtype=torch.bool, device=device)

    # --- meals ---------------------------------------------------------------
    meal_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    for r, name in enumerate(MEAL_TOKENS):
        tid = tokenizer.token2id.get(name)
        if tid is not None:
            meal_rank[tid] = r
    K = int(meal_rank.max().item()) + 1 if (meal_rank >= 0).any() else 0

    meal_pred_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    if K > 0:
        meal_mask = meal_rank >= 0
        meal_pred_rank[meal_mask] = (meal_rank[meal_mask] - 1) % K

    # ---- forbid list for CBM ----
    forbid = {
        tokenizer.pad_token_id,
        getattr(tokenizer, "ctx_token_id", None),
        getattr(tokenizer, "null_token_id", None),
        tokenizer.token2id.get(ADMISSION_TOKEN),
        *[tokenizer.token2id.get(t) for t in TERMINAL_OUTCOMES],
        *[tokenizer.token2id.get(t) for t in MEAL_TOKENS],
        *start_ids.tolist(),
        *end_ids.tolist(),
    }
    forbid_mask_ids = torch.tensor([tid for tid in forbid if tid is not None],
                                   dtype=torch.long)

    return {
        # per-token
        "is_start": is_start,
        "is_end":   is_end,
        "base_id":  base_id,
        "meal_rank":      meal_rank,
        "meal_pred_rank": meal_pred_rank,

        # per-base
        "start_ids_per_base": start_ids_per_base,
        "end_ids_per_base":   end_ids_per_base,
        "conflict_mat":       conf_mat,

        # misc
        "start_ids": start_ids,
        "end_ids":   end_ids,
        "K_meals":   torch.tensor(K, dtype=torch.long, device=device),
        "forbid_mask_ids": forbid_mask_ids,
    }


@torch.no_grad()
def penalty_interval_structure(pred_ids: torch.LongTensor,
                               gt_ids:   torch.LongTensor,
                               is_start: torch.BoolTensor,
                               is_end:   torch.BoolTensor,
                               base_id:  torch.LongTensor,
                               conflict_mat: torch.BoolTensor) -> torch.Tensor:
    """
    Structural penalty on predictions (scalar ∈ [0,1]).

    Counts three prediction-side violations:
      1. END without an open START         (fsm)             -> Normalized by count(_END)
      2. START while same base already open (dup)            -> Normalized by count(_START)
      3. START that never gets an END       (unclosed)       -> Normalized by count(_START)
      4. START while a *conflicting* base is open (conflict) -> Normalized by count(_START)
    A violation is forgiven once if the SAME violation type for that base (1 & 3 only)
    appears anywhere in the GT (order/time agnostic).

    Vectorized over T for meta lookup; single loop over B to maintain an 'open' set.

    pred_ids, gt_ids : [B,T]
    is_start/is_end/base_id : [V]
    conflict_mat : [nb, nb] (True if (i,j) conflict: same concept & diff value)

    Returns: scalar tensor
    NOTE: This still uses a counter of violation and does not compare the exact violation in GT 
    compared to the exact violation in pred. We can't allow the model to learn that any type X violation 
    is ok if also appear in GT, but only that it's allowed to generate a violating token if the same token 
    caused a violation in GT.
    """
    device = pred_ids.device
    B, T = pred_ids.shape
    nb = conflict_mat.size(0)

    # Meta lookups
    p_s = is_start[pred_ids]         # [B,T] bool
    p_e = is_end[pred_ids]
    p_b = base_id[pred_ids]          # [B,T] long ( -1 if not interval )

    g_s = is_start[gt_ids]
    g_e = is_end[gt_ids]
    g_b = base_id[gt_ids]

    fsm_viol = dup_viol = unclosed_viol = conflict_viol = 0
    tot_end  = tot_start = 0

    for b in range(B):
        # ========== build forgiveness pools from GT ========== #
        gt_open = set()
        gt_fsm, gt_dup, gt_unclosed, gt_conf = [], [], [], []

        for t in range(T):
            bid = g_b[b, t].item()
            if bid < 0:  # not interval
                continue
            if g_s[b, t]:
                if bid in gt_open:
                    gt_dup.append(bid)
                # conflict start?
                if nb > 0 and any(conflict_mat[bid, k] for k in gt_open):
                    gt_conf.append(bid)
                gt_open.add(bid)
            elif g_e[b, t]:
                if bid not in gt_open:
                    gt_fsm.append(bid)
                else:
                    gt_open.remove(bid)
        gt_unclosed.extend(list(gt_open))

        forgive_fsm   = Counter(gt_fsm)
        forgive_dup   = Counter(gt_dup)
        forgive_uncl  = Counter(gt_unclosed)
        forgive_conf  = Counter(gt_conf)

        # ========== check prediction violations ========== #
        pred_open = set()

        for t in range(T):
            bid = p_b[b, t].item()
            if bid < 0:
                continue
            if p_s[b, t]:
                tot_start += 1

                # duplicate
                if bid in pred_open:
                    if forgive_dup[bid] > 0:
                        forgive_dup[bid] -= 1
                    else:
                        dup_viol += 1
                # conflict
                elif nb > 0 and any(conflict_mat[bid, k] for k in pred_open):
                    if forgive_conf[bid] > 0:
                        forgive_conf[bid] -= 1
                    else:
                        conflict_viol += 1

                pred_open.add(bid)

            elif p_e[b, t]:
                tot_end += 1
                if bid not in pred_open:
                    if forgive_fsm[bid] > 0:
                        forgive_fsm[bid] -= 1
                    else:
                        fsm_viol += 1
                else:
                    pred_open.remove(bid)

        # leftover opens → unclosed
        for bid in list(pred_open):
            if forgive_uncl[bid] > 0:
                forgive_uncl[bid] -= 1
            else:
                unclosed_viol += 1

    tot_end   = max(tot_end,   1)
    tot_start = max(tot_start, 1)

    fsm_rate    = fsm_viol      / tot_end
    dup_rate    = dup_viol      / tot_start
    unclosed_r  = unclosed_viol / tot_start
    conflict_r  = conflict_viol / tot_start

    return torch.tensor((fsm_rate + dup_rate + unclosed_r + conflict_r) / 4.0,
                        device=device, dtype=torch.float32)


@torch.no_grad()
def penalty_meal_order(pred_ids: torch.LongTensor,
                       meal_rank: torch.LongTensor) -> torch.Tensor:
    """
    Enforces cyclic order among meals: B→L→D→N→B...
    We ignore non-meal tokens (they can appear anywhere in between).

    Implementation:
      1. Extract only meal ids per sequence (mask ranks>=0).
      2. Compare each consecutive pair in that filtered list.

    pred_ids  : [B,T]
    meal_rank : [V]  (-1 non-meal, else 0..K-1)

    return scalar ∈ [0,1] = (#wrong meal transitions)/(#meal transitions)
    """
    device = pred_ids.device
    ranks = meal_rank[pred_ids]        # [B,T]
    B, T  = ranks.shape
    K     = int(meal_rank.max().item()) + 1 if (meal_rank >= 0).any() else 0

    if K == 0:
        return torch.tensor(0.0, device=device)

    wrong = 0
    total = 0
    for b in range(B):
        seq_r = ranks[b]
        meal_seq = seq_r[seq_r >= 0]          # keep only meals
        if meal_seq.numel() < 2:
            continue
        exp = (meal_seq[:-1] + 1) % K
        total += exp.numel()
        wrong += (meal_seq[1:] != exp).sum().item()

    if total == 0:
        return torch.tensor(0.0, device=device)
    return torch.tensor(wrong / total, dtype=torch.float32, device=device)


def compute_legality_masks_tf(position_ids: torch.LongTensor,
                              is_start: torch.BoolTensor,
                              is_end:   torch.BoolTensor,
                              base_id:  torch.LongTensor,
                              start_ids_per_base: torch.LongTensor,
                              end_ids_per_base:   torch.LongTensor,
                              meal_rank: torch.LongTensor,
                              meal_pred_rank: torch.LongTensor,
                              K_meals: torch.Tensor,
                              conflict_mat: torch.BoolTensor):
    """
    Vectorized legality/bonus masks from GOLD prefix (teacher forcing).

    illegal[B,T,V]  True → forbid v at step t
    bonus  [B,T,V]  True → boost v at step t

    Interval logic (per base):
      • END illegal if base not open yet
      • START illegal if base already open
      • END bonus  if base open
      • START/END of concept_valX is illegal if concept_valY is still open.

    Meal logic:
      cyclic order; meal m illegal if predecessor rank not seen yet, bonus if seen.

    All done without loops over T (only broadcast/cumsums).

    position_ids : [B,T]
    """
    device = position_ids.device
    B, T = position_ids.shape
    V    = is_start.numel()
    nb   = start_ids_per_base.numel()
    K    = int(K_meals.item())

    # ---------------- interval prefix states ----------------
    # map to interval meta
    tok_base = base_id[position_ids]             # [B,T] (-1 if not interval)
    tok_s    = is_start[position_ids]            # [B,T]
    tok_e    = is_end[position_ids]              # [B,T]

    # one-hot over bases (ignore -1)
    base_valid = tok_base >= 0                   # [B,T]
    # flatten indices for scatter
    scatter_idx = tok_base.clone()
    scatter_idx[~base_valid] = 0  # dummy

    start_oh = torch.zeros(B, T, nb, device=device, dtype=torch.int16)
    end_oh   = torch.zeros_like(start_oh)

    b_idx = torch.arange(B, device=device)[:, None]
    t_idx = torch.arange(T, device=device)[None, :]

    start_oh[b_idx, t_idx, scatter_idx] = (tok_s & base_valid)
    end_oh[b_idx, t_idx, scatter_idx]   = (tok_e & base_valid)

    # cumulative open counts up to (and including) t
    starts_cum = start_oh.cumsum(dim=1)          # [B,T,n_b]
    ends_cum   = end_oh.cumsum(dim=1)
    open_cum   = (starts_cum - ends_cum) > 0     # [B,T,n_b] bool

    # Illegal END where not open
    # Illegal_start where already open
    # bonus END where open
    # To paint into [B,T,V], we broadcast with per-base token ids
    illegal = torch.zeros(B, T, V, device=device, dtype=torch.bool)
    bonus   = torch.zeros_like(illegal)

    # Build [1,1,nb] -> [B,T,nb] helpers
    end_tok_ids   = end_ids_per_base.view(1, 1, nb).expand(B, T, nb)
    start_tok_ids = start_ids_per_base.view(1, 1, nb).expand(B, T, nb)

    # Mask matrices
    need_end_closed   = ~open_cum          # END illegal if not open
    need_start_closed =  open_cum          # START illegal if already open
    good_end          =  open_cum          # END bonus if open

    # Scatter to [B,T,V]
    illegal.scatter_(2, end_tok_ids,   need_end_closed)
    illegal.scatter_(2, start_tok_ids, need_start_closed)
    bonus.scatter_(  2, end_tok_ids,   good_end)

    # Value-conflict masking 
    # Conflict_mat[nb,nb]: True if (j,k) conflict (same concept, different value)
    if nb > 0 and conflict_mat.any():
        # For each time step, which bases are conflicting-with-open ones?
        # open_cum: [B,T,nb]; conflict_mat: [nb,nb]
        # -> conflict_active[b,t,j] = OR_k ( open_cum[b,t,k] & conflict_mat[k,j] )
        conflict_active = open_cum @ conflict_mat.T    # bool matmul
        # Forbid START of j when conflict_active==True
        illegal.scatter_(2, start_tok_ids, conflict_active)

    # ---------------- meal prefix states ----------------
    if K > 0:
        mr = meal_rank[position_ids]             # [B,T]
        meal_mask = mr >= 0

        # one-hot of meals per rank
        meal_oh = torch.zeros(B, T, K, device=device, dtype=torch.bool)
        valid_idx = torch.nonzero(meal_mask, as_tuple=False)
        if valid_idx.numel() > 0:
            b_idx = valid_idx[:, 0]
            t_idx = valid_idx[:, 1]
            meal_oh[b_idx, t_idx, mr[meal_mask]] = True

        meal_seen = meal_oh.cumsum(dim=1) > 0    # [B,T,K]

        # for each vocab token v that is a meal:
        #   pred_rank[v] is predecessor rank. At time t, v illegal if meal_seen[b,t,pred_rank[v]] == False
        meal_tok_mask = meal_rank >= 0
        if meal_tok_mask.any():
            pred_rank = meal_pred_rank[meal_tok_mask]        # [Nv]
            tok_ids   = meal_tok_mask.nonzero(as_tuple=False).squeeze(1)  # [Nv]

            # expand meal_seen predecessor info to [B,T,Nv]
            pred_ok = meal_seen[:, :, pred_rank]             # indexing along K -> [B,T,Nv]

            # place into big [B,T,V]
            illegal[:, :, tok_ids] |= ~pred_ok
            bonus[:,   :, tok_ids] |=  pred_ok

    return illegal, bonus


def apply_masks_to_logits(logits, illegal_mask, bonus_mask, bonus_boost=0.2):
    """
    logits: [B,T,V] *after* slicing (no [CTX])
    illegal_mask/bonus_mask: Bool [B,T,V]
    We add/subtract constants (broadcast).
    """
    # illegal → -inf
    logits = logits.masked_fill(illegal_mask, -float("inf"))
    # bonus → add small boost
    if bonus_boost > 0:
        logits = logits + bonus_boost * bonus_mask.float()
    return logits


def plot_losses(train_losses, val_losses):
    """
    Plot train vs. validation loss to inspect training quality.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‑entropy loss")
    plt.title("Training vs. validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_rep_penalty(last_tokens, V, window=5, strength=0.6, device=None):
    """
    Soft repetition discourager on inference.
    last_tokens : list[int]   (already generated, newest at the end)
    V           : vocab size
    window      : how many recent tokens we look back
    strength    : scalar multiplier for the penalty (0..1 typical)
    Returns:
        rep_vec : [V] float tensor, 0 for unseen, higher for very recent repeats
    """
    if not last_tokens or strength <= 0:
        return torch.zeros(V, device=device)
    device = device or torch.device("cpu")
    k = min(window, len(last_tokens))
    # decay weights: newest gets 1.0, then 0.8, 0.6, ...
    decay = torch.linspace(1.0, 0.2, steps=window, device=device)[:k]
    idx = torch.tensor(last_tokens[-k:], device=device)

    rep_vec = torch.zeros(V, device=device)
    # reverse so newest aligns with decay[0]
    rep_vec.index_add_(0, idx.flip(0), decay)
    return rep_vec * strength