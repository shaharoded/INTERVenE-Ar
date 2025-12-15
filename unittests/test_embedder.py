import torch
import pytest

from transform_emr.embedder import EMREmbedding
from transform_emr.dataset import EMRTokenizer

@pytest.fixture(scope="module")
def mini_tokenizer():
    # Minimal tokenizer for testing embedder
    toks = ["[PAD]", "[MASK]", "[CTX]", "[NULL]", "A_START", "A_END"]
    token2id = {t:i for i,t in enumerate(toks)}
    rawconcept2id = {"A":0, "[NULL]":1}
    concept2id    = {"A":0, "[NULL]":1}
    value2id      = {"A":0, "[NULL]":1}
    special_tokens = ["[PAD]","[MASK]","[CTX]","[NULL]"]
    token_weights = torch.ones(len(toks))
    important_ids = torch.tensor([], dtype=torch.long)
    token_counts = torch.tensor([], dtype=torch.long)

    # Dummy parent raw mapping
    vocab_size = len(token2id)
    tokenid2parent_raw_ids = torch.zeros((vocab_size, 1), dtype=torch.long)
    parent_pad_len = 1

    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=special_tokens,
        token_weights=token_weights,
        important_token_ids=important_ids,
        token_counts = token_counts,
        tokenid2parent_raw_ids=tokenid2parent_raw_ids,
        parent_pad_len=parent_pad_len
    )
    # set special token attributes
    tk.pad_token_id  = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.ctx_token_id  = token2id['[CTX]']
    tk.null_token_id = token2id['[NULL]']
    return tk

@pytest.mark.order(2)
def test_embedder_initialization(mini_tokenizer):
    cfg = {"ctx_dim":2, "time2vec_dim":2, "embed_dim":8}
    model = EMREmbedding(
        tokenizer=mini_tokenizer,
        ctx_dim=cfg['ctx_dim'],
        time2vec_dim=cfg['time2vec_dim'],
        embed_dim=cfg['embed_dim']
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == cfg['embed_dim']

@pytest.mark.order(3)
def test_embedder_forward_and_mask_predict(mini_tokenizer):
    tokenizer = mini_tokenizer
    ctx_dim = 2
    embed_dim = 7
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=ctx_dim,
        time2vec_dim=2,
        embed_dim=embed_dim
    )
    B, T = 2, 5
    dummy = {
        'parent_raw_ids':  torch.zeros(B, T, 1, dtype=torch.long),
        'concept_ids':     torch.zeros(B, T, dtype=torch.long),
        'value_ids':       torch.zeros(B, T, dtype=torch.long),
        'position_ids':    torch.zeros(B, T, dtype=torch.long),
        'abs_ts':          torch.zeros(B, T),
        'patient_contexts': torch.zeros(B, ctx_dim)
    }
    # forward without mask
    seq = model(**dummy)
    assert seq.shape == (B, T+1, embed_dim)
    # forward with mask
    seq2, mask = model.forward(**dummy, return_mask=True)
    assert seq2.shape == (B, T+1, embed_dim)
    assert mask.shape == (B, T+1)

    # test predict_time
    pred_t = model.predict_time(dummy['abs_ts'])
    assert pred_t.shape == (B, T, 1)
    assert (pred_t >= 0).all() and (pred_t <= 1).all()

@pytest.mark.order(4)
def test_forward_with_decoder_logits(mini_tokenizer):
    tokenizer = mini_tokenizer
    ctx_dim = 2
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=ctx_dim,
        time2vec_dim=2,
        embed_dim=8
    )
    B, T = 2, 4
    dummy = {
        'parent_raw_ids':  torch.zeros(B, T, 1, dtype=torch.long),
        'concept_ids':     torch.zeros(B, T, dtype=torch.long),
        'value_ids':       torch.zeros(B, T, dtype=torch.long),
        'position_ids':    torch.zeros(B, T, dtype=torch.long),
        'abs_ts':          torch.zeros(B, T),
        'patient_contexts': torch.zeros(B, ctx_dim)
    }
    batch = {
     "parent_raw_ids":  dummy['parent_raw_ids'],
     "concept_ids":     dummy['concept_ids'],
     "value_ids":       dummy['value_ids'],
     "position_ids":    dummy['position_ids'],
     "abs_ts":          dummy['abs_ts'],
     # note: forward_with_decoder pulls patient_contexts from "context_vec"
     "context_vec":     dummy['patient_contexts'],
    }
    logits = model.forward_with_decoder(
        batch
    )
    # forward_with_decoder predicts next-token logits: [B, T, vocab_size]
    vocab_size = len(tokenizer.token2id)
    assert logits.shape == (B, T, vocab_size)
