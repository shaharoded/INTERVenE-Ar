"""
THIS MODULE IS INCOMPLETE AND THE TESTED SCENERIO DOES NOT YET REPRESENT THE TRANSFORMER REQUIREMENT
"""

import torch
import pytest

from transform_emr.utils import (
    get_multi_hot_targets,
    build_mlm,
    linear_schedule,
    apply_cbm,
    mix_with_predictions,
    penalty_interval_structure,
    penalty_meal_order,
    build_luts,
    compute_legality_masks_tf,
    apply_masks_to_logits,
    build_rep_penalty
)
from transform_emr.dataset import EMRTokenizer

@pytest.fixture(scope="module")
def mini_tokenizer():
    """
    Simulated tokenizer with a three-level hierarchy:
      - Raw concepts: A (0), MEAL (1), ADMISSION (2), OUTCOME (3)
      - Concepts: A_X (0), A_Y (1), MEAL_B/L/D (2), ADMISSION (3), DEATH/RELEASE (4)
      - Values: VAL1 (0), VAL2 (1) per concept where applicable
      - Position tokens: <concept>_<value>_START/END, plus single tokens for meals/outcomes.
    """
    toks = [
        "[PAD]", "[MASK]", "[CTX]", "[NULL]",
        # Admission context
        "ADMISSION",
        # Intervals for A_X with two values
        "A_STATE_Low_START", "A_STATE_Low_END",
        "A_STATE_High_START", "A_STATE_High_END",
        # Intervals for A_Y with two values
        "A_TREND_dec_START", "A_TREND_dec_END",
        "A_TREND_inc_START", "A_TREND_inc_END",
        # Meals
        "MEAL_B", "MEAL_L", "MEAL_D",
        # Outcomes
        "DEATH", "RELEASE"
    ]
    token2id = {tok: i for i, tok in enumerate(toks)}
    # Raw concept mapping: group by top-level concept
    rawconcept2id = {
        "A": 0,
        "MEAL": 1,
        "ADMISSION": 2,
        "DEATH": 3,
        "RELEASE": 4
    }
    # Concept-level mapping
    concept2id = {
        "A_STATE": 0,
        "A_TREND": 1,
        "MEAL": 2,
        "ADMISSION": 3,
        "DEATH": 4,
        "RELEASE": 5
    }
    # Value-level mapping (e.g., high/low categories)
    value2id = {
        "A_STATE_Low": 0,
        "A_STATE_High": 1,
        "A_TREND_dec": 2,
        "A_TREND_inc": 3,
        "MEAL_B": 4,
        "MEAL_L": 5,
        "MEAL_D": 6,
        "ADMISSION": 7,
        "DEATH": 8,
        "RELEASE": 9
    }
    special_tokens = ["[PAD]", "[MASK]", "[CTX]", "[NULL]"]
    token_weights = torch.ones(len(toks))
    important_token_ids = torch.tensor([], dtype=torch.long)

    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=special_tokens,
        token_weights=token_weights,
        important_token_ids=important_token_ids
    )
    # assign special attributes
    tk.pad_token_id  = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.ctx_token_id  = token2id['[CTX]']
    tk.null_token_id = token2id['[NULL]']
    return tk


def test_multi_hot_targets_visual_and_assert():
    """
    For each t in a longer sequence, print the
    true future IDs vs. the multi-hot IDs, then
    assert they exactly match.

    Expectation: At every position t, the targets are the positions t+1 up to t+k, until the first padding (0) token.
    """
    # --- Setup a toy sequence: 1..10 then two PADs (0) ---
    seq = torch.tensor([[1,2,3,4,5,6,7,8,9,10,0,0]])
    B, T = seq.shape
    V    = seq.max().item() + 1  # 11 = tokens 0..10
    k    = 5

    # --- Compute multi-hot targets ---
    mh = get_multi_hot_targets(seq, padding_idx=0, vocab_size=V, k=k)
    assert mh.shape == (B, T, V)

    # --- For each timestep, print & assert correctness ---
    for t in range(T):
        # curr
        curr = seq[0, t]
        # ground-truth future slice
        future = seq[0, t+1 : t+1+k].tolist()
        # drop pads & dedupe
        expected = sorted({x for x in future if x != 0})

        # what the function actually marked
        hot_ids = mh[0, t].nonzero(as_tuple=False).squeeze(-1).tolist()
        hot_ids.sort()

        # print for human verification
        print(f"t={t}, curr={curr} | future={future} | hot_ids={hot_ids}")

        # pytest assertion
        assert hot_ids == expected, (
            f"At t={t}, expected {expected} but got {hot_ids}"
        )


def test_linear_schedule_visual_and_assert():
    """
    For a linear ramp over warmup epochs:
    - When epoch <= warmup: value = (epoch / warmup) * max
    - When epoch >  warmup: value = max
    """
    warmup = 5
    maxv   = 1.0
    # test a range of epochs before, at, and after warmup
    for epoch in [0, 1, 2, 5, 6, 10]:
        val = linear_schedule(epoch, warmup, maxv)
        expected = min(epoch / warmup, 1.0) * maxv
        # print for visual inspection
        print(f"epoch={epoch} | expected={expected:.3f} | actual={val:.3f}")
        # assert correctness
        assert val == pytest.approx(expected)


def test_build_mlm_masking_visual_and_assert():
    """
    Print each position’s token, mask flag, and new ID,
    then assert forbidden tokens stay unchanged and eligible positions are flagged.
    """
    tk = mini_tokenizer()

    # One-row batch covering forbidden & eligible IDs
    seq_ids = [
        tk.pad_token_id,                      # forbidden
        tk.ctx_token_id,                      # forbidden
        tk.null_token_id,                     # forbidden
        tk.token2id["ADMISSION"],             # forbidden
        tk.token2id["DEATH"],                 # forbidden
        tk.token2id["RELEASE"],               # forbidden
        tk.token2id["A_STATE_High_START"],    # eligible
        tk.token2id["A_TREND_inc_START"],     # eligible
        tk.token2id["MEAL_B"],                # eligible
        tk.token2id["A_TREND_inc_END"],       # eligible
        tk.token2id["A_STATE_High_END"],      # eligible
    ]
    ids = torch.tensor([seq_ids], dtype=torch.long)

    masked, mask = build_mlm(ids, tokenizer=tk, p=1.0)

    forbidden = {
        tk.pad_token_id,
        tk.ctx_token_id,
        tk.null_token_id,
        tk.token2id["ADMISSION"],
        tk.token2id["DEATH"],
        tk.token2id["RELEASE"],
    }

    for pos, orig in enumerate(seq_ids):
        token    = tk.id2token[orig]
        was_mask = bool(mask[0, pos].item())
        new_id   = masked[0, pos].item()

        print(f"pos={pos:<2} token={token:<24}"
              f"orig={orig:<2} masked={was_mask:<5} new={new_id}")

        if orig in forbidden:
            # these must never be masked or changed
            assert not was_mask, f"❌ Forbidden {token} was masked"
            assert new_id == orig, f"❌ Forbidden {token} changed to {new_id}"
        else:
            # eligible positions must have mask flag True
            assert was_mask, f"❌ Expected {token} to be masked"

    # (We no longer assert new_id != orig, since 10% of the time BERT-style
    # keeps the original even when flagged masked.)


def test_apply_cbm_visual_and_assert():
    """
    With p=1.0 (epoch == warmup), verify that:
      – forbidden IDs (pad, mask, forbid_mask_ids) never change
      – eligible IDs (everything else) are always replaced by mask_token_id
    """
    tk   = mini_tokenizer()
    luts = build_luts(tk)

    pad   = tk.pad_token_id
    msk   = tk.mask_token_id
    forbid_ids = set(luts["forbid_mask_ids"].tolist()) | {pad, msk}

    # Find one eligible ID
    V = len(tk.token_weights)
    eligible = next((i for i in range(V) if i not in forbid_ids), None)
    if eligible is None:
        pytest.skip("No eligible tokens in vocab to test CBM masking")

    # Build a 1×3 batch: [MASK], [PAD], [eligible]
    in_seq = torch.tensor([[msk, pad, eligible]], dtype=torch.long)
    batch = {
        "position_ids":    in_seq.clone(),
        "raw_concept_ids": in_seq.clone(),
        "concept_ids":     in_seq.clone(),
        "value_ids":       in_seq.clone(),
    }

    # Force p = 1.0 by epoch == warmup_epochs
    out = apply_cbm(
        batch.copy(),
        epoch=10,
        warmup_epochs=10,
        tokenizer=tk,
        forbid_ids=torch.tensor(sorted(luts["forbid_mask_ids"].tolist()), dtype=torch.long),
        max_p=1.0
    )
    print("MASKED BATCH:\n", out)

    out_seq = out["position_ids"][0]

    for j, orig in enumerate(in_seq[0]):
        orig_id = int(orig)
        new_id  = int(out_seq[j])
        is_forb = (orig_id in forbid_ids)
        print(f"pos={j} | orig={orig_id:<3} | new={new_id:<3} | forbidden={is_forb}")

        if is_forb:
            # Forbidden must remain exactly the same
            assert new_id == orig_id, f"❌ Forbidden {orig_id} was changed → {new_id}"
        else:
            # Eligible must become mask_token_id
            assert new_id == msk, f"❌ Eligible {orig_id} not masked (got {new_id})"


def test_mix_with_predictions_visual_and_assert():
    """
    Protected GT tokens stay, unprotected are replaced by pred.
    """
    tk = mini_tokenizer()

    # Single sequence of length 3
    gt   = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[9, 9, 9]])
    prot = torch.zeros(len(tk.token2id), dtype=torch.bool)
    prot[1] = True  # protect ID=1

    mixed, mask = mix_with_predictions(
        gt, pred,
        epoch=5,
        warmup_epochs=5,
        protected_ids=prot,
        max_rate=1.0
    )
    print("MIXED BATCH:\n", mixed)

    for j, (g, p, m_flag) in enumerate(zip(gt[0], pred[0], mask[0])):
        g_id     = int(g.item())
        p_id     = int(p.item())
        mixed_id = int(mixed[0,j].item())

        print(f"pos={j} | gt={g_id} | pred={p_id} | mask={bool(m_flag.item())} | mixed={mixed_id}")

        if prot[g_id]:
            assert mixed_id == g_id, f"❌ Protected {g_id} was replaced"
            assert not m_flag,       f"❌ Protected {g_id} should not be masked"
        else:
            assert mixed_id == p_id, f"❌ Unprotected {g_id} not replaced by {p_id}"
            assert m_flag,           f"❌ Unprotected {g_id} should be masked"


def test_build_luts_and_legality():
    tk = mini_tokenizer()
    l = build_luts(tk)
    # simple interval sequence START, END
    s = tk.token2id['A_START']; e = tk.token2id['A_END']
    seq = torch.tensor([[s,e,0]])
    illegal,bonus = compute_legality_masks_tf(
        seq, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['meal_rank'],l['meal_pred_rank'],l['K_meals'],l['conflict_mat']
    )
    # no illegal at correct order
    assert not illegal.any()
    # reversed order is illegal
    seq2=torch.tensor([[e,s,0]])
    illegal2,_= compute_legality_masks_tf(
        seq2, **{k: l[k] for k in ['is_start','is_end','base_id',
        'start_ids_per_base','end_ids_per_base','meal_rank','meal_pred_rank','K_meals','conflict_mat']}
    )
    assert illegal2[0,0,e]


def test_penalty_interval_and_meal_order(mini_tokenizer):
    tk = mini_tokenizer
    l = build_luts(tk)
    # interval penalty: unclosed start
    s = tk.token2id['A_START']
    seq = torch.tensor([[s,0]])
    p = penalty_interval_structure(seq, seq, **l)
    assert p >= 0
    # meal order: B->L->D sequence ok vs bad
    b= tk.token2id['MEAL_B']; lmk=tk.token2id['MEAL_L']; d=tk.token2id['MEAL_D']
    ok = torch.tensor([[b,lmk,d,0]])
    bad= torch.tensor([[d,lmk,b,0]])
    assert penalty_meal_order(ok,l['meal_rank'])==0
    assert penalty_meal_order(bad,l['meal_rank'])>0


def test_apply_masks_and_rep_penalty():
    # logits and masks
    logits = torch.zeros(1,2,4)
    illegal = torch.zeros_like(logits).bool()
    bonus   = torch.zeros_like(logits).bool()
    # mark V=3 as illegal at t=1
    illegal[0,1,3]=True
    bonus[0,0,2]=True
    out = apply_masks_to_logits(logits,illegal,bonus,bonus_boost=0.5)
    assert out[0,1,3]==-float('inf')
    assert out[0,0,2]==pytest.approx(0.5)

    # rep penalty
    last = [1,2,1,3]
    V=5
    vec = build_rep_penalty(last,V,window=3,strength=0.5)
    # vector length and max <= strength
    assert vec.shape[0]==V
    assert vec.max() <= 0.5
