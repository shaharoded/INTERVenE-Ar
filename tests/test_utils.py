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
    # Very small vocab: PAD, MASK, CTX, ADMISSION, A_X intervals, MEAL tokens, TERMINAL
    toks = ["[PAD]","[MASK]","[CTX]","[NULL]","ADMISSION",
            "A_X_START","A_X_END","MEAL_B","MEAL_L","MEAL_D","TERMINAL"]
    token2id = {t:i for i,t in enumerate(toks)}
    rawconcept2id = {"A_X":0,"MEAL":1,"ADMISSION":2,"TERMINAL":3}
    concept2id    = rawconcept2id.copy()
    value2id      = rawconcept2id.copy()
    special_tokens = ["[PAD]","[MASK]","[CTX]","[NULL]"]
    token_weights = torch.ones(len(toks))
    important_ids = torch.tensor([], dtype=torch.long)
    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=special_tokens,
        token_weights=token_weights,
        important_token_ids=important_ids
    )
    tk.pad_token_id  = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.ctx_token_id  = token2id['[CTX]']
    return tk


def test_multi_hot_targets_basic(mini_tokenizer):
    # Build a longer sequence: values 1–10, then two PADs (0)
    seq = torch.tensor([[1,2,3,4,5,6,7,8,9,10,0,0]])
    B, T = seq.shape
    V    = seq.max().item() + 1  # vocab size = 11 (0–10)
    k    = 4

    mh = get_multi_hot_targets(seq, padding_idx=0, vocab_size=V, k=k)
    assert mh.shape == (B, T, V)

    # For each timestep, compute the “expected” hot ids via slicing
    for t in range(T):
        # lookahead window in Python
        future = seq[0, t+1 : t+1+k].tolist()
        # remove PADs and dedupe
        expected = sorted({x for x in future if x != 0})
        # pull out all nonzero entries in mh
        hot_ids = mh[0, t].nonzero(as_tuple=False).squeeze(-1).tolist()
        hot_ids.sort()
        assert hot_ids == expected, (
            f"At t={t}, expected hot={expected} but got {hot_ids}"
        )


def test_build_mlm_masks(mini_tokenizer):
    tk = mini_tokenizer
    # ids includes PAD, CTX, ADMISSION, TERMINAL, and an interval
    ids = torch.tensor([[tk.pad_token_id, tk.ctx_token_id, tk.token2id['ADMISSION'],
                         tk.token2id['TERMINAL'], tk.token2id['A_X_START']]], dtype=torch.long)
    masked, mask = build_mlm(ids, tokenizer=tk, p=1.0)
    # forbidden tokens should not be masked
    for idx in [0,1,2,3]:
        assert mask[0,idx] == False and masked[0,idx] == ids[0,idx]
    # only the interval should be masked
    assert mask[0,4] == True
    assert masked[0,4] == tk.mask_token_id or masked[0,4] != ids[0,4]


def test_build_mlm_forbidden_and_masking(mini_tokenizer):
    tk = mini_tokenizer
    # create ids with forbidden tokens in front
    ids = torch.tensor([[tk.pad_token_id, tk.ctx_token_id, tk.null_token_id, 
                         mini_tokenizer.token2id['A_START'], mini_tokenizer.token2id['MEAL_B']]])
    masked, mask = build_mlm(ids, tokenizer=tk, p=1.0)
    # forbidden ids not masked
    for idx in [0,1,2]:
        assert mask[0,idx] == False
        assert masked[0,idx] == ids[0,idx]
    # remaining positions must be flagged
    assert mask[0,3] or mask[0,4]


def test_linear_schedule_boundaries():
    # ramp 0->max over warmup
    assert linear_schedule(0, 5, 1.0) == pytest.approx(0.0)
    assert linear_schedule(5, 5, 1.0) == pytest.approx(1.0)
    assert linear_schedule(10,5,1.0) == pytest.approx(1.0)


def test_apply_cbm_and_mix(mini_tokenizer):
    tk = mini_tokenizer
    # build LUTs and forbid_ids
    luts = build_luts(tk)
    forbid = luts['forbid_mask_ids']
    # batch of 2 sequences
    batch = {
        'position_ids':    torch.tensor([[1,2,3],[2,3,4]]),
        'raw_concept_ids': torch.tensor([[1,2,3],[2,3,4]]),
        'concept_ids':     torch.tensor([[1,2,3],[2,3,4]]),
        'value_ids':       torch.tensor([[1,2,3],[2,3,4]]),
    }
    out = apply_cbm(batch.copy(), epoch=5, warmup_epochs=10, tokenizer=tk, forbid_ids=forbid, max_p=1.0)
    # ensure forbidden tokens remain
    for fid in forbid.tolist():
        assert not (out['position_ids']==fid).all()

    # test mix_with_predictions at epoch end
    gt = torch.tensor([[1,2,3]])
    pred= torch.tensor([[9,9,9]])
    prot= torch.zeros(len(tk.token2id),dtype=torch.bool)
    prot[1]=True
    mixed,mask = mix_with_predictions(gt,pred,epoch=5,warmup_epochs=5,protected_ids=prot,max_rate=1.0)
    # protected id 1 not replaced
    assert mixed[0,0]==1 and mask[0,0]==False
    # some replacement elsewhere
    assert mask[0,1] or mask[0,2]


def test_build_luts_and_legality(mini_tokenizer):
    tk = mini_tokenizer
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
