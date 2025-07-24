import torch
import pytest

from transform_emr.utils import (
    compute_legality_masks_tf,
    penalty_interval_structure,   # <- your renamed strict version
    build_rep_penalty,
)
from transform_emr.inference import build_luts   # assuming you placed it there
from transform_emr.dataset import EMRTokenizer

# ---------- helpers ----------

def _mini_tokenizer():
    """
    Build a tiny tokenizer with the minimal vocabs needed to test:
      • two interval bases of the same concept but different values  (conflict)
      • a meal cycle of length 3
      • PAD/MASK/CTX
    """
    # fake ids
    toks = [
        "[PAD]", "[MASK]", "[CTX]",
        "ADMISSION",
        "A_VAL1_START", "A_VAL1_END",
        "A_VAL2_START", "A_VAL2_END",   # same concept "A", diff value -> conflict with VAL1
        "B_VAL_START",  "B_VAL_END",
        "MEAL_BREAKFAST", "MEAL_LUNCH", "MEAL_DINNER",
        "TERMINAL"
    ]
    token2id = {t:i for i,t in enumerate(toks)}

    # rawconcept/ concept/ value maps (simplified)
    def concept(tok):
        if tok.endswith("_START") or tok.endswith("_END"):
            base = tok.rsplit("_",1)[0]
        else:
            base = tok
        # Concept = first two parts for these fake examples
        # "A_VAL1" → concept "A"
        return base.split("_",1)[0]

    def value(tok):
        if tok.endswith("_START") or tok.endswith("_END"):
            return tok.rsplit("_",1)[0]     # keep value tag
        return tok

    rawconcept2id = {}
    concept2id    = {}
    value2id      = {}

    for t, i in token2id.items():
        rc = concept(t)
        cv = value(t)
        rawconcept2id.setdefault(rc, len(rawconcept2id))
        concept2id.setdefault(rc, len(concept2id))          # same key as raw here
        value2id.setdefault(cv, len(value2id))

    # special tokens
    special = ["[PAD]", "[MASK]", "[CTX]"]
    pad = token2id["[PAD]"]; mask = token2id["[MASK]"]; ctx = token2id["[CTX]"]

    tk = EMRTokenizer(
        token2id         = token2id,
        rawconcept2id    = rawconcept2id,
        concept2id       = concept2id,
        value2id         = value2id,
        special_tokens   = special,
        token_weights    = torch.ones(len(token2id)),
        important_token_ids = torch.tensor([], dtype=torch.long)
    )
    tk.pad_token_id = pad
    tk.mask_token_id = mask
    tk.ctx_token_id = ctx
    return tk

# ---------- tests ----------
@pytest.mark.order(3)
def test_build_luts_conflict_and_meals():
    tk = _mini_tokenizer()
    l = build_luts(tk)

    V = len(tk.token2id)
    assert l["is_start"].shape == (V,)
    assert l["is_end"].shape   == (V,)
    assert l["base_id"].shape  == (V,)

    nb = l["start_ids_per_base"].numel()
    assert l["conflict_mat"].shape == (nb, nb)

    # we know A_VAL1 and A_VAL2 conflict
    # find their base indices
    a1_b = l["base_id"][tk.token2id["A_VAL1_START"]].item()
    a2_b = l["base_id"][tk.token2id["A_VAL2_START"]].item()
    assert l["conflict_mat"][a1_b, a2_b] and l["conflict_mat"][a2_b, a1_b]

    # meal cycle length 3
    assert l["K_meals"].item() == 3
    assert l["meal_rank"][tk.token2id["MEAL_BREAKFAST"]] == 0
    assert l["meal_pred_rank"][tk.token2id["MEAL_LUNCH"]] == 0  # pred of lunch is breakfast


@pytest.mark.order(4)
def test_legality_masks_tf_basic():
    tk = _mini_tokenizer()
    l = build_luts(tk)
    device = torch.device("cpu")

    # Batch of 1, T=4 ground truth
    # Sequence: A_VAL1_START, B_VAL_START, MEAL_BREAKFAST, A_VAL1_END
    gt = torch.tensor([[ tk.token2id["A_VAL1_START"],
                         tk.token2id["B_VAL_START"],
                         tk.token2id["MEAL_BREAKFAST"],
                         tk.token2id["A_VAL1_END"] ]], dtype=torch.long, device=device)

    illegal, bonus = compute_legality_masks_tf(
        position_ids        = gt,
        is_start            = l["is_start"].to(device),
        is_end              = l["is_end"].to(device),
        base_id             = l["base_id"].to(device),
        start_ids_per_base  = l["start_ids_per_base"].to(device),
        end_ids_per_base    = l["end_ids_per_base"].to(device),
        meal_rank           = l["meal_rank"].to(device),
        meal_pred_rank      = l["meal_pred_rank"].to(device),
        K_meals             = l["K_meals"].to(device),
        base_concept        = l["base_concept"].to(device),
        base_value          = l["base_value"].to(device),
        conflict_mat        = l["conflict_mat"].to(device)
    )

    B,T,V = illegal.shape
    assert (B,T,V) == (1,4,len(tk.token2id))

    # At t=0 (before anything), END of A_VAL1 should be illegal, START of A_VAL1 should be legal
    t0_end_a1 = tk.token2id["A_VAL1_END"]
    assert illegal[0,0,t0_end_a1]

    # After A_VAL1_START, another START of same base illegal
    t1_start_a1 = tk.token2id["A_VAL1_START"]
    assert illegal[0,1,t1_start_a1]

    # Meal LUNCH should be illegal at t=0 (no breakfast yet), legal afterwards
    lunch_id = tk.token2id["MEAL_LUNCH"]
    assert illegal[0,0,lunch_id]
    assert not illegal[0,2,lunch_id]


@pytest.mark.order(5)
def test_penalty_interval_structure():
    tk = _mini_tokenizer()
    l = build_luts(tk)
    device = torch.device("cpu")

    # Two sequences (B=2), same length T=5
    # pred[0]: correct order
    # pred[1]: END without START, START duplicate, conflict start
    A1S = tk.token2id["A_VAL1_START"]
    A1E = tk.token2id["A_VAL1_END"]
    A2S = tk.token2id["A_VAL2_START"]
    A2E = tk.token2id["A_VAL2_END"]
    term= tk.token2id["TERMINAL"]

    pred = torch.tensor([
        [A1S, A1E, term, tk.pad_token_id, tk.pad_token_id],   # no vio
        [A1E, A1S, A2S, term, tk.pad_token_id]                # END first, then START, then conflicting START
    ], dtype=torch.long, device=device)

    gt   = torch.tensor([
        [A1S, A1E, term, tk.pad_token_id, tk.pad_token_id],   # perfect GT
        [A1S, A1E, term, tk.pad_token_id, tk.pad_token_id]    # GT has no violations
    ], dtype=torch.long, device=device)

    p = penalty_interval_structure(
        pred_ids     = pred,
        gt_ids       = gt,
        is_start     = l["is_start"].to(device),
        is_end       = l["is_end"].to(device),
        base_id      = l["base_id"].to(device),
        conflict_mat = l["conflict_mat"].to(device)
    )
    assert p.dim() == 0
    assert 0.0 <= p.item() <= 1.0
    # Second batch has violations; penalty must be > 0
    assert p.item() > 0.0


@pytest.mark.order(6)
def test_build_rep_penalty():
    V = 20
    last = [3, 7, 3]  # repeated
    vec = build_rep_penalty(last, V=V, window=5, strength=0.6, device=torch.device("cpu"))
    assert vec.shape == (V,)
    assert vec[3] > vec[7]   # newest repetition gets bigger penalty
    assert vec.max() <= 0.6