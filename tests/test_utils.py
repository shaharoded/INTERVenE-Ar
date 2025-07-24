"""
THIS MODULE IS INCOMPLETE AND THE TESTED SCENERIO DOES NOT YET REPRESENT THE TRANSFORMER REQUIREMENT
"""

import torch
import math
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
    compute_legality_masks_tf
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
    # Sequence of 4 tokens with pad at end
    seq = torch.tensor([[1,2,3,0]])  # assume 0=PAD
    mh = get_multi_hot_targets(seq, padding_idx=0, vocab_size=len(mini_tokenizer.token2id), k=2)
    # At t=0, look at tokens at [1,2] → ids 2 and 3
    assert mh.shape == (1,4,len(mini_tokenizer.token2id))
    assert mh[0,0,2] == 1 and mh[0,0,3] == 1
    # pad index never hot
    assert torch.all(mh[...,0] == 0)


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

@pytest.mark.parametrize("epoch,warmup,maxv,exp",[
    (0,10,0.5,0.0),(5,10,0.5,0.25),(10,10,0.5,0.5)
])
def test_linear_schedule(epoch,warmup,maxv,exp):
    val = linear_schedule(epoch, warmup, maxv)
    assert math.isclose(val, exp, rel_tol=1e-5)


def test_apply_cbm_and_forbid(mini_tokenizer):
    tk = mini_tokenizer
    luts = build_luts(tk)
    # create batch with 1..5
    batch = {k: torch.arange(5).unsqueeze(0).clone() for k in ['position_ids','raw_concept_ids','concept_ids','value_ids']}
    forbid = luts['forbid_mask_ids']
    out = apply_cbm(batch.copy(), epoch=5, warmup_epochs=10, tokenizer=tk, forbid_ids=forbid, max_p=1.0)
    # ensure forbidden ids remain
    for fid in forbid.tolist():
        assert torch.any(out['position_ids']==fid)


def test_mix_with_predictions_protect(mini_tokenizer):
    tk = mini_tokenizer
    gt = torch.tensor([[0,1,2,3,4]])
    pred = torch.tensor([[9,9,9,9,9]])
    # protect id 1 and 2
    prot = torch.zeros(len(tk.token2id), dtype=torch.bool)
    prot[1] = True; prot[2] = True
    mixed, mask = mix_with_predictions(gt, pred, epoch=5, warmup_epochs=10, protected_ids=prot)
    # positions 1 and 2 unchanged
    assert mixed[0,1] == 1 and mixed[0,2] == 2
    # some other positions likely replaced
    assert mask.dtype == torch.bool


def test_legality_masks_and_penalties(mini_tokenizer, capsys):
    tk  = mini_tokenizer
    l   = build_luts(tk)
    # Build a simple sequence: START then END
    s = tk.token2id['A_X_START']; e = tk.token2id['A_X_END']
    gt = torch.tensor([[s, e, 0, 0]])
    # a pred that violates FSM: END first, then START
    pred = torch.tensor([[e, s, 0, 0]])
    # compute masks
    illegal_gt, bonus_gt = compute_legality_masks_tf(gt, l['is_start'], l['is_end'], l['base_id'],
                                                     l['start_ids_per_base'], l['end_ids_per_base'],
                                                     l['meal_rank'], l['meal_pred_rank'], l['K_meals'],
                                                     l['conflict_mat'])
    illegal_pred, bonus_pred = compute_legality_masks_tf(pred, l['is_start'], l['is_end'], l['base_id'],
                                                        l['start_ids_per_base'], l['end_ids_per_base'],
                                                        l['meal_rank'], l['meal_pred_rank'], l['K_meals'],
                                                        l['conflict_mat'])
    # print for debug
    print("illegal_gt:\n", illegal_gt[0,:, [s,e]])
    print("illegal_pred:\n", illegal_pred[0,:, [s,e]])
    # GT should have no illegal
    assert not illegal_gt.any()
    # pred should mark the first position illegal (END without START)
    assert illegal_pred[0,0,e]

    # penalty should be > 0 for pred vs gt
    p = penalty_interval_structure(pred, gt, **l, window=1)
    assert p.item() > 0

    # test meal order
    b = tk.token2id['MEAL_B']; l_id = tk.token2id['MEAL_L']; d = tk.token2id['MEAL_D']
    seq_ok  = torch.tensor([[b, l_id, d, 0]])
    seq_bad = torch.tensor([[d, l_id, b, 0]])
    p_ok = penalty_meal_order(seq_ok, l['meal_rank'])
    p_bad= penalty_meal_order(seq_bad,l['meal_rank'])
    assert p_ok.item() == 0.0
    assert p_bad.item() > 0.0

