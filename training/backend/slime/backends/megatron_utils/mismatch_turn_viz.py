"""Per-turn train/rollout off-policy token fraction, logged to wandb as a figure.

Enabled with MISMATCH_TURN_VIZ=1 (default OFF -> zero impact on other runs).

A token counts as "off-policy" when |logp_train - logp_rollout| > tau
(MISMATCH_TURN_THRESH, default 0.1 -- i.e. the per-token IS weight
exp(logp_train - logp_rollout) deviates from 1 by more than ~10%).

Turn = contiguous run of loss_mask==1 in the FULL response mask (same notion as
the viz/turn_* entropy plots and _opd_early_stop_gap_masks). Because at the
advantage-computation stage log_probs / rollout_log_probs are CP-sliced while
loss_masks are full-length, each rank maps its local tokens back to global
response positions with the same offset math as advantage normalization
(get_logits_and_tokens_offset_with_cp), accumulates per-(sample, turn) counts,
and the counts are all-reduced over the CP group. Only global rank 0 logs:

  * viz/turn_offpolicy_frac   -- x = turn index (1-based), y = macro-averaged %
                                 of off-policy tokens in that turn (per-sample
                                 turn %, then mean over samples); right axis
                                 gray bars = #samples contributing to the turn.
  * train/offpolicy_token_frac -- overall micro fraction across all tokens.

Aggregation is over this DP rank's samples only (global rank 0's shard); for a
monitoring curve on 32+ samples per step that is statistically plenty.
Any failure degrades to a one-time warning; training is never interrupted.
"""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_MAX_TURNS = 64
_warned: set[str] = set()
_fallback_step = [0]


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def _turn_ids_from_mask(full_mask: torch.Tensor) -> torch.Tensor:
    """Full-length int tensor: 0-based turn id where mask==1, -1 elsewhere."""
    m = full_mask.bool()
    prev = torch.cat([m.new_zeros(1), m[:-1]])
    starts = (m & ~prev).long()
    tid = torch.cumsum(starts, 0) - 1
    return torch.where(m, tid, torch.full_like(tid, -1))


def _local_turn_ids(tid_full: torch.Tensor, total_length: int, response_length: int, cp_size: int) -> torch.Tensor:
    """Slice the full-length (response-space) turn-id array to this CP rank's
    local chunk layout -- exactly mirroring the loss_mask slicing used by
    advantage normalization in loss.py."""
    if cp_size == 1:
        return tid_full
    from .cp_utils import get_logits_and_tokens_offset_with_cp

    prompt_len = total_length - response_length
    _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_length, response_length)
    s0, e0 = token_offsets[0]
    s1, e1 = token_offsets[1]
    res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
    res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)
    parts = []
    if res_e0 > res_s0:
        parts.append(tid_full[res_s0:res_e0])
    if res_e1 > res_s1:
        parts.append(tid_full[res_s1:res_e1])
    if not parts:
        return tid_full.new_empty(0)
    return torch.cat(parts)


def maybe_log_mismatch_turn_viz(args, rollout_data) -> None:
    if os.environ.get("MISMATCH_TURN_VIZ", "0") != "1":
        return
    try:
        _run(args, rollout_data)
    except Exception as e:  # never break training for a monitoring plot
        _warn_once("fail", f"mismatch_turn_viz failed (plot skipped): {e!r}")


def _run(args, rollout_data) -> None:
    if getattr(args, "use_rollout_logprobs", False):
        _warn_once("bypass", "mismatch_turn_viz: use_rollout_logprobs is set; no independent train logprobs -> skip.")
        return
    log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    loss_masks = rollout_data.get("loss_masks")
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")
    if not log_probs or not rollout_log_probs or not loss_masks:
        return

    from megatron.core import mpu

    cp_size = mpu.get_context_parallel_world_size()
    thresh = float(os.environ.get("MISMATCH_TURN_THRESH", "0.1"))

    n = len(log_probs)
    device = log_probs[0].device
    # counts[i, k, 0] = tokens of sample i in turn k (this rank);  [.., 1] = off-policy tokens
    counts = torch.zeros((n, _MAX_TURNS, 2), dtype=torch.float32, device=device)

    for i in range(n):
        full_mask = loss_masks[i]
        tid_full = _turn_ids_from_mask(full_mask)
        tid_local = _local_turn_ids(tid_full, int(total_lengths[i]), int(response_lengths[i]), cp_size).to(device)
        lp_t = log_probs[i]
        lp_r = rollout_log_probs[i].to(device)
        L = min(lp_t.shape[0], lp_r.shape[0], tid_local.shape[0])
        if L == 0:
            continue
        if max(lp_t.shape[0], lp_r.shape[0], tid_local.shape[0]) - L > 1:
            # >1-token disagreement is a real layout mismatch, not an off-by-one pad
            _warn_once(
                "len",
                f"mismatch_turn_viz: length mismatch (train={lp_t.shape[0]} rollout={lp_r.shape[0]} "
                f"turn_ids={tid_local.shape[0]}); truncating to {L}. Check CP slicing assumptions.",
            )
        tids = tid_local[:L]
        valid = (tids >= 0) & (tids < _MAX_TURNS)
        if not bool(valid.any()):
            continue
        tids_v = tids[valid]
        anom = ((lp_t[:L] - lp_r[:L]).abs() > thresh)[valid].to(torch.float32)
        counts[i, :, 0].index_add_(0, tids_v, torch.ones_like(anom))
        counts[i, :, 1].index_add_(0, tids_v, anom)

    if cp_size > 1 and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=mpu.get_context_parallel_group())

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    counts = counts.cpu()
    tok = counts[:, :, 0]
    anom = counts[:, :, 1]
    total_tok = float(tok.sum().item())
    if total_tok == 0:
        return
    overall = float(anom.sum().item()) / total_tok

    # macro over samples: per-(sample, turn) fraction, then mean over samples having that turn
    has = tok > 0
    frac = torch.where(has, anom / tok.clamp(min=1.0), torch.zeros_like(tok))
    n_contrib = has.sum(0)  # [T] samples contributing per turn
    turn_pct, turn_ks, turn_cnt = [], [], []
    for k in range(_MAX_TURNS):
        c = int(n_contrib[k].item())
        if c == 0:
            continue
        turn_ks.append(k + 1)  # 1-based for display
        turn_pct.append(float(frac[has[:, k], k].mean().item()) * 100.0)
        turn_cnt.append(c)

    step = rollout_data.get("rollout_id")
    if step is None:
        step = _fallback_step[0]
        _fallback_step[0] += 1
        _warn_once("step", "mismatch_turn_viz: rollout_id not in rollout_data; using internal counter as wandb step.")

    try:
        import wandb

        if getattr(wandb, "run", None) is None:
            _warn_once("wandb", "mismatch_turn_viz: wandb.run is None on rank 0; plot skipped.")
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
        ax2 = ax.twinx()
        ax2.bar(turn_ks, turn_cnt, color="0.85", width=0.8, zorder=1, label="#samples")
        ax2.set_ylabel("#samples with turn", color="0.5")
        ax.plot(turn_ks, turn_pct, "o-", color="#d62728", ms=3.5, lw=1.5, zorder=3)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xlabel("turn index")
        ax.set_ylabel(f"off-policy token % (|Δlogp| > {thresh:g})")
        ax.set_title(
            f"train↔rollout off-policy tokens per turn — rollout {step}\n"
            f"overall {overall:.2%} of {int(total_tok)} tokens (macro over {counts.shape[0]} samples/rank0-DP)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        wandb.log(
            {"viz/turn_offpolicy_frac": wandb.Image(fig), "train/offpolicy_token_frac": overall},
            step=step,
            commit=False,
        )
        plt.close(fig)
    except Exception as e:
        _warn_once("plot", f"mismatch_turn_viz: wandb/matplotlib logging failed: {e!r}")
