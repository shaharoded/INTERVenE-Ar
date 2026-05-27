"""
ar_ft.py
==============

I5 / P-AR-FT — Autoregressive fine-tuning of the Phase-3 outcome head on
model-generated trajectories.

Background / rationale
----------------------
The eval scores the outcome head's per-position risk curve over a trajectory
that the model *generates* from a short (2-day) input seed. Phase 3, however,
trains the outcome head only on ground-truth (GT) sequences, so at eval time
the head sees inputs it never trained on (its own roll-outs). I5 closes that
train/eval distribution gap: once per Phase-3 run we roll the just-trained
model forward from a 2-day seed for every train patient, cache those
trajectories, and mix them into Phase-3 training. The outcome-head labels at
generated positions are NOT read from the generated tokens (the LM may not
emit outcome tokens at all) — they are derived from each patient's GT outcome
events via the same future-only soft kernel the GT path uses
(`get_future_outcome_targets`). The backbone is frozen so the head learns to
read frozen features off model-generated context and predict the real labels.

Everything here is gated behind `training_settings["phase3_ar_ft"]`; when that
flag is False the Phase-3 path is unchanged.
"""

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transform_emr.dataset import EMRDataset, collate_emr, BucketBatchSampler
from transform_emr import inference
from transform_emr.config.dataset_config import OUTCOMES


_ABS_TS_SCALE = 336.0


# ───────────────────────── GT-derived soft labels ─────────────────────────── #

@torch.no_grad()
def get_outcome_targets_from_gt_times(
    query_abs_ts: torch.Tensor,     # [B, T_q] query timestamps (normalised, /336)
    gt_outcome_idx: torch.Tensor,   # [B, M] long — index into model.outcome_names (pad = -1)
    gt_outcome_times: torch.Tensor, # [B, M] float — GT occurrence times (normalised, /336)
    tau,                            # scalar OR [K] tensor — decay constant (normalised units)
    horizon: float,                 # max lookahead horizon (normalised units = hours / 336)
    K: int,                         # number of outcomes the head predicts
) -> torch.Tensor:
    """
    Future-only soft-kernel outcome targets built from a patient's GT outcome
    events, for use on MODEL-GENERATED query positions.

    For each query step t and outcome k:
        target[b, t, k] = clamp_{0..1}( sum_{m : idx_m == k, 0 < dt <= horizon}
                                        exp(-dt / tau_k) )
        where dt = gt_outcome_times[b, m] - query_abs_ts[b, t].

    This is exactly the time-decayed branch of `get_future_outcome_targets`,
    but with the "matches" matrix coming from a sparse (idx, time) list of GT
    events instead of from the (possibly generated) token sequence. Same
    future-only direction (0 < dt), same per-outcome tau, same horizon, same
    clamp[0,1] — so feeding a GT sequence's own (idx, time) events through this
    helper reproduces the GT path's targets (verified in __main__ below).

    Returns FloatTensor [B, T_q, K] in [0, 1].
    """
    B, T_q = query_abs_ts.shape
    device = query_abs_ts.device
    M = gt_outcome_idx.size(1)

    out = torch.zeros(B, T_q, K, device=device, dtype=torch.float32)
    if M == 0:
        return out

    valid = gt_outcome_idx >= 0                                   # [B, M]
    # dt[b, t, m] = future offset from query t to GT event m. Positive = future.
    dt = gt_outcome_times.unsqueeze(1) - query_abs_ts.unsqueeze(2)  # [B, T_q, M]
    in_horizon = (dt > 0) & (dt <= horizon) & valid.unsqueeze(1)    # [B, T_q, M]

    # Per-event tau lookup: tau_k for the outcome each event belongs to.
    is_per_k_tau = torch.is_tensor(tau) and tau.dim() == 1 and tau.numel() == K
    safe_idx = gt_outcome_idx.clamp(min=0)                          # [B, M]
    if is_per_k_tau:
        tau_per_m = tau[safe_idx]                                   # [B, M]
    else:
        tau_per_m = torch.as_tensor(tau, device=device, dtype=torch.float32) \
                        .expand(B, M)
    # masked_fill (not multiply) so exp() overflow on out-of-horizon dt never
    # produces 0*inf = NaN — mirrors get_future_outcome_targets' per-k branch.
    decay = torch.exp(-dt / tau_per_m.unsqueeze(1).clamp(min=1e-6))  # [B, T_q, M]
    decay = decay.masked_fill(~in_horizon, 0.0)

    # Scatter-add each event's decay into its outcome column.
    col_idx = safe_idx.unsqueeze(1).expand(B, T_q, M)               # [B, T_q, M]
    out.scatter_add_(2, col_idx, decay)
    return out.clamp(0.0, 1.0)


# ───────────────────────── GT outcome extraction ──────────────────────────── #

def extract_gt_outcomes(tokens_df: pd.DataFrame, outcome_names):
    """
    For each patient, extract their GT outcome events as
    (index-into-outcome_names, time-in-hours) pairs from rows whose
    PositionToken is one of the head's outcomes.

    Returns dict[pid] -> list[(k_idx, time_hours)], aligned to `outcome_names`.
    """
    name2k = {n: k for k, n in enumerate(outcome_names)}
    keep = tokens_df[tokens_df["PositionToken"].isin(name2k)]
    gt = {}
    for pid, grp in keep.groupby("PatientId", sort=False):
        gt[pid] = [
            (name2k[tok], float(tp))
            for tok, tp in zip(grp["PositionToken"].tolist(), grp["TimePoint"].tolist())
        ]
    return gt


# ───────────────────────── generated-trajectory dataset ───────────────────── #

class GeneratedTrajectoryDataset(Dataset):
    """
    Wraps cached model-generated trajectories as an EMRDataset-compatible
    Dataset: __getitem__ returns the same tensor-dict keys EMRDataset does
    (parent_raw_ids / concept_ids / value_ids / position_ids / abs_ts /
    context_vec / targets), re-encoded from the generated PositionToken strings
    through the tokenizer's LUTs (the same lookups EMRDataset / inference use).

    Each example additionally carries its patient's GT outcome (idx, time)
    events and a patient-level multi-hot label, so the GT-from-times targets
    and the P4 pool label can be built downstream.

    gen_traj: dict[pid] -> {"tokens": list[str], "times": list[float hours]}
    """

    def __init__(self, gen_traj, context_df, tokenizer, gt_outcomes, num_outcomes):
        self.tokenizer = tokenizer
        self.context_df = context_df
        self.num_outcomes = num_outcomes
        self.gt_outcomes = gt_outcomes
        # Only keep patients that produced at least one usable token and have
        # context features available.
        self.patient_ids = [
            pid for pid, tr in gen_traj.items()
            if len(tr["tokens"]) > 0 and pid in context_df.index
        ]
        self.gen_traj = {pid: gen_traj[pid] for pid in self.patient_ids}
        # BucketBatchSampler buckets by len(patient_groups[pid]); expose the
        # generated token list so length-aware batching works (len == #tokens).
        self.patient_groups = {pid: self.gen_traj[pid]["tokens"] for pid in self.patient_ids}

        tok = tokenizer
        self._pad_id = tok.pad_token_id
        self._mask_id = tok.mask_token_id
        # tok2concept / tok2value LUTs (id -> concept/value id, -1 if absent),
        # same as inference.build_luts; -1 maps to [MASK] like the decode loop.
        from transform_emr.utils import build_luts
        luts = build_luts(tok)
        self._tok2concept = luts["tok2concept"]
        self._tok2value = luts["tok2value"]
        self._t2parent = tok.tokenid2parent_raw_ids   # [V, P]

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        tr = self.gen_traj[pid]
        tokens = tr["tokens"]
        times = tr["times"]

        pos = torch.tensor(
            [self.tokenizer.token2id.get(t, self._mask_id) for t in tokens],
            dtype=torch.long,
        )                                                            # [T]
        parent_raw_ids = self._t2parent[pos]                         # [T, P]

        concept = self._tok2concept[pos].clone()
        concept[concept < 0] = self._mask_id
        value = self._tok2value[pos].clone()
        value[value < 0] = self._mask_id

        abs_ts = torch.tensor(times, dtype=torch.float32) / _ABS_TS_SCALE  # [T]
        context_vec = torch.tensor(self.context_df.loc[pid].values, dtype=torch.float32)

        # GT outcome events (idx into outcome_names, time normalised /336).
        gt = self.gt_outcomes.get(pid, [])
        if len(gt) > 0:
            gt_idx = torch.tensor([g[0] for g in gt], dtype=torch.long)
            gt_time = torch.tensor([g[1] for g in gt], dtype=torch.float32) / _ABS_TS_SCALE
        else:
            gt_idx = torch.zeros(0, dtype=torch.long)
            gt_time = torch.zeros(0, dtype=torch.float32)

        # Patient-level multi-hot label: 1 where the patient has any GT
        # occurrence of outcome k (used by the P4 pool aux).
        patient_label = torch.zeros(self.num_outcomes, dtype=torch.float32)
        if len(gt) > 0:
            patient_label[gt_idx] = 1.0

        return {
            "parent_raw_ids": parent_raw_ids,
            "concept_ids": concept.long(),
            "value_ids": value.long(),
            "position_ids": pos,
            "abs_ts": abs_ts,
            "context_vec": context_vec,
            "targets": pos.clone(),
            "gt_outcome_idx": gt_idx,
            "gt_outcome_time": gt_time,
            "patient_label": patient_label,
        }


def collate_generated(batch, pad_token_id=0):
    """
    Collate generated examples: reuse collate_emr for the standard tensor keys,
    then pad the variable-length GT-outcome (idx, time) lists and stack the
    patient-level labels.
    """
    base = collate_emr(batch, pad_token_id=pad_token_id)
    B = len(batch)
    M = max((x["gt_outcome_idx"].numel() for x in batch), default=0)

    gt_idx = torch.full((B, M), -1, dtype=torch.long)
    gt_time = torch.zeros((B, M), dtype=torch.float32)
    for i, x in enumerate(batch):
        m = x["gt_outcome_idx"].numel()
        if m > 0:
            gt_idx[i, :m] = x["gt_outcome_idx"]
            gt_time[i, :m] = x["gt_outcome_time"]

    base["gt_outcome_idx"] = gt_idx
    base["gt_outcome_time"] = gt_time
    base["patient_label"] = torch.stack([x["patient_label"] for x in batch])
    base["__gen__"] = True
    return base


# ───────────────────────── generation + caching ───────────────────────────── #

def _config_signature(training_settings, n_patients, num_outcomes):
    """Stable hash of the AR-FT-relevant config so a re-run with the same
    settings reloads the cache instead of regenerating."""
    keys = ["phase3_ar_seed_hours", "phase3_ar_K", "outcome_horizon_hours", "sample"]
    payload = {k: training_settings.get(k) for k in keys}
    payload["n_patients"] = int(n_patients)
    payload["num_outcomes"] = int(num_outcomes)
    raw = repr(sorted(payload.items())).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _build_seed_dataset(train_ds, seed_hours):
    """
    Build an EMRDataset whose sequences are truncated to the seed window
    (TimePoint <= seed_hours), WITHOUT touching api.py: slice the existing
    train dataset's tokens_df and re-wrap it with the same tokenizer/context.
    """
    tdf = train_ds.tokens_df
    trunc = tdf[tdf["TimePoint"] <= float(seed_hours)].copy()
    return EMRDataset(trunc, train_ds.context_df, tokenizer=train_ds.tokenizer)


@torch.no_grad()
def generate_and_cache_trajectories(model, train_ds, training_settings, cache_dir):
    """
    Generate K trajectories per train patient from a `phase3_ar_seed_hours`
    seed and cache them to `cache_dir / ar_ft_cache.pt`. Reloads on a config
    match.

    Cache format (torch.save dict):
        {
          "signature":   str,
          "outcome_names": list[str],
          "gen_traj":   dict[pid] -> {"tokens": list[str], "times": list[float hours]},
          "gt_outcomes": dict[pid] -> list[(k_idx, time_hours)],
          "config":     {seed_hours, K, horizon_hours},
        }

    Returns (gen_traj, gt_outcomes).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "ar_ft_cache.pt"

    n_patients = len(train_ds.patient_ids)
    signature = _config_signature(training_settings, n_patients, model.num_outcomes)

    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("signature") == signature \
               and cached.get("outcome_names") == list(model.outcome_names):
                print(f"[AR-FT]: Reloading cached trajectories from {cache_path} "
                      f"({len(cached['gen_traj'])} patients).")
                return cached["gen_traj"], cached["gt_outcomes"]
            print(f"[AR-FT]: Cache signature mismatch — regenerating ({cache_path}).")
        except Exception as e:  # noqa: BLE001 — stale/corrupt cache → regenerate
            print(f"[AR-FT]: Failed to read cache ({e}); regenerating.")

    seed_hours = float(training_settings.get("phase3_ar_seed_hours", 48.0))
    K = int(training_settings.get("phase3_ar_K", 1))
    horizon_hours = float(training_settings.get("outcome_horizon_hours", 48.0))

    gt_outcomes = extract_gt_outcomes(train_ds.tokens_df, model.outcome_names)

    seed_ds = _build_seed_dataset(train_ds, seed_hours)
    print(f"[AR-FT]: Generating {K} trajectory/ies per patient from a "
          f"{seed_hours:.0f}h seed for {len(seed_ds.patient_ids)} patients...")

    was_training = model.training
    model.eval()
    gen_traj = {}
    for k in range(K):
        # generate() is already batched, no_grad, KV-cached and moves results
        # to CPU as a DataFrame; one call rolls out every patient in seed_ds.
        df = inference.generate(
            model,
            seed_ds,
            temperature=1.0,
            batch_size=training_settings.get("batch_size", 16),
            collect_risk_scores=False,
            tqdm_desc=f"[AR-FT] gen {k + 1}/{K}",
        )
        # Group generated rows back to per-patient token/time sequences. The
        # full sequence (input seed + generated continuation) is the context
        # the outcome head must learn to read.
        df = df.sort_values(["PatientId", "Step"])
        for pid, grp in df.groupby("PatientId", sort=False):
            entry = gen_traj.setdefault(pid, {"tokens": [], "times": []})
            # K>1: keep the longest roll-out per patient (a single example per
            # patient keeps the generated/GT mix balanced and the cache small).
            toks = grp["Token"].tolist()
            tms = grp["TimePoint"].tolist()
            if len(toks) > len(entry["tokens"]):
                entry["tokens"] = toks
                entry["times"] = tms

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if was_training:
        model.train()

    torch.save(
        {
            "signature": signature,
            "outcome_names": list(model.outcome_names),
            "gen_traj": gen_traj,
            "gt_outcomes": gt_outcomes,
            "config": {"seed_hours": seed_hours, "K": K, "horizon_hours": horizon_hours},
        },
        cache_path,
    )
    print(f"[AR-FT]: Cached {len(gen_traj)} generated trajectories to {cache_path}.")
    return gen_traj, gt_outcomes


# ───────────────────────── mixed Phase-3 loader ───────────────────────────── #

class MixedPhase3Loader:
    """
    Iterable that interleaves GT batches (from the existing `train_dl`) with
    generated batches (from a DataLoader over GeneratedTrajectoryDataset).

    Each epoch, a fraction `gen_fraction(epoch)` of the yielded batches are
    generated batches; the rest are GT batches. GT batches are tagged
    "__gen__": False, generated batches "__gen__": True, so run_epoch can pick
    the right outcome-target path. len() reports GT + the sampled generated
    batches so loss averaging matches the number of batches actually iterated.
    """

    def __init__(self, gt_loader, gen_dataset, batch_size, collate_fn,
                 frac_start, frac_end, n_epochs):
        self.gt_loader = gt_loader
        self.gen_dataset = gen_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.frac_start = float(frac_start)
        self.frac_end = float(frac_end)
        self.n_epochs = max(int(n_epochs), 1)
        self._epoch = 0
        self._cur_gen_batches = 0

        # Number of full GT batches per epoch (bucket sampler defines this).
        try:
            self._n_gt_batches = len(gt_loader)
        except TypeError:
            self._n_gt_batches = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def gen_fraction(self, epoch):
        if self.n_epochs <= 1:
            return self.frac_start
        t = min(max((epoch - 1) / (self.n_epochs - 1), 0.0), 1.0)
        return self.frac_start + (self.frac_end - self.frac_start) * t

    def _make_gen_loader(self):
        # Bucket-batch the generated dataset the same way GT is batched, then
        # take only as many batches as the target generated fraction needs.
        sampler = BucketBatchSampler(self.gen_dataset, batch_size=self.batch_size)
        return DataLoader(
            self.gen_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    def __len__(self):
        return self._n_gt_batches + self._cur_gen_batches

    def __iter__(self):
        frac = self.gen_fraction(self._epoch)
        n_gt = self._n_gt_batches
        # Number of generated batches to draw so generated ≈ frac of the total.
        if frac <= 0.0 or len(self.gen_dataset) == 0:
            n_gen_target = 0
        elif frac >= 1.0:
            n_gen_target = max(n_gt, 1)
        else:
            n_gen_target = int(round(n_gt * frac / (1.0 - frac)))

        gen_batches = []
        if n_gen_target > 0:
            for b in self._make_gen_loader():
                gen_batches.append(b)
                if len(gen_batches) >= n_gen_target:
                    break
        self._cur_gen_batches = len(gen_batches)

        # Interleave: tag GT batches, then splice generated batches at roughly
        # even intervals so the head sees a steady mix through the epoch.
        gt_batches = []
        for b in self.gt_loader:
            b["__gen__"] = False
            gt_batches.append(b)

        total = len(gt_batches) + len(gen_batches)
        if not gen_batches:
            for b in gt_batches:
                yield b
            return

        # Even interleave by index position.
        gen_positions = set()
        if total > 0:
            step = total / len(gen_batches)
            gen_positions = {int(i * step) for i in range(len(gen_batches))}

        gi = 0
        gj = 0
        for pos in range(total):
            if pos in gen_positions and gj < len(gen_batches):
                yield gen_batches[gj]
                gj += 1
            elif gi < len(gt_batches):
                yield gt_batches[gi]
                gi += 1
            elif gj < len(gen_batches):
                yield gen_batches[gj]
                gj += 1


# ───────────────────────── numeric equivalence check ──────────────────────── #

if __name__ == "__main__":
    # Verify get_outcome_targets_from_gt_times reproduces
    # get_future_outcome_targets when fed a GT sequence's own outcome events.
    from transform_emr.utils import get_future_outcome_targets

    torch.manual_seed(0)
    K = 3
    outcome_ids = [10, 11, 12]            # arbitrary token ids for the K outcomes
    B, T = 2, 8
    tau = torch.tensor([0.04, 0.07, 0.10])  # per-outcome, normalised units
    horizon = 48.0 / _ABS_TS_SCALE

    # Random GT token sequence containing some outcome tokens; rising times.
    seq = torch.randint(0, 20, (B, T))
    # force a few outcome occurrences
    seq[0, 3] = 10; seq[0, 6] = 12
    seq[1, 2] = 11; seq[1, 5] = 10
    abs_ts = torch.cumsum(torch.rand(B, T) * 0.02, dim=1)  # increasing, normalised

    query = abs_ts[:, :-1]
    ref = get_future_outcome_targets(
        target_ids=seq, outcome_ids=outcome_ids,
        all_abs_ts=abs_ts, query_abs_ts=query, tau=tau, horizon=horizon,
    )  # [B, T-1, K]

    # Build (idx, time) GT events from the same sequence.
    id2k = {tid: k for k, tid in enumerate(outcome_ids)}
    M = T
    gt_idx = torch.full((B, M), -1, dtype=torch.long)
    gt_time = torch.zeros(B, M)
    for b in range(B):
        m = 0
        for t in range(T):
            tid = int(seq[b, t])
            if tid in id2k:
                gt_idx[b, m] = id2k[tid]
                gt_time[b, m] = abs_ts[b, t]
                m += 1

    mine = get_outcome_targets_from_gt_times(
        query_abs_ts=query, gt_outcome_idx=gt_idx, gt_outcome_times=gt_time,
        tau=tau, horizon=horizon, K=K,
    )

    max_err = (ref - mine).abs().max().item()
    print(f"max |ref - mine| = {max_err:.3e}")
    assert max_err < 1e-5, "GT-from-times helper does not match get_future_outcome_targets!"
    print("OK: get_outcome_targets_from_gt_times matches get_future_outcome_targets.")
