"""
Span-level DRO utilities for DRTO.

Functions:
- construct_spans:  cutpoint-based span construction with strict L_min, no fixed-length fallback.
- build_span_losses: from token-level PPO losses → flat span_losses + uniform p0 over responses.
- chi2_span_dro_loss: χ² span-DRO objective.
- kl_span_dro_loss:   KL span-DRO objective with eta search via grid + LSE.

Notes:
- The original RTO reward is unchanged; advantages are still computed from r_RTO upstream.
- p0_{i,s} = 1/n * 1/S_i (response-balanced uniform).
- No disagreement weighting, no two-level DRO, no fixed-length fallback.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

import torch


__all__ = [
    "construct_spans",
    "build_span_losses",
    "chi2_span_dro_loss",
    "kl_span_dro_loss",
]


# ---------------------------------------------------------------------------
# Span construction
# ---------------------------------------------------------------------------

def construct_spans(
    probs: torch.Tensor,
    H: int,
    span_min_len: int,
    span_max_num: int,
    span_cut_prob: float,
) -> List[Tuple[int, int]]:
    """
    Sequentially construct contiguous spans on a single response.

    Inputs:
        probs:            [H] token probabilities exp(old_log_probs[h])
        H:                number of valid tokens
        span_min_len:     L_min, lower bound on span length (unless H < L_min)
        span_max_num:     S_max, upper bound on number of spans
        span_cut_prob:    p_cut, only h with probs[h] < p_cut may be cutpoints

    Output:
        list of (start, end) inclusive index pairs covering [0, H-1] exactly once.

    Algorithm:
        - If H <= L_min: one span covering all valid tokens.
        - Otherwise sequentially pick eligible cutpoints with the LOWEST probability,
          subject to:
              h >= start + L_min - 1
              h <= H - L_min - 1
              probs[h] < p_cut
        - Stop when no eligible cutpoint remains, or len(spans) reaches S_max - 1,
          or remaining tokens < L_min.
        - Always close out with final span [start, H-1].
        - No fixed-length fallback. If no cutpoint, response stays as one span.
    """
    L_min = max(1, int(span_min_len))
    S_max = max(1, int(span_max_num))

    if H <= L_min:
        return [(0, H - 1)]

    spans: List[Tuple[int, int]] = []
    start = 0

    while start < H and len(spans) < S_max - 1:
        lo = start + L_min - 1
        hi = H - L_min - 1            # inclusive upper bound
        if lo > hi:
            break

        # eligible cutpoints
        slice_probs = probs[lo : hi + 1]
        mask = slice_probs < span_cut_prob
        if not bool(mask.any()):
            break

        # pick argmin probability among eligible (lowest-prob token)
        masked_probs = slice_probs.clone()
        masked_probs[~mask] = float("inf")
        local_idx = int(torch.argmin(masked_probs).item())
        h_star = lo + local_idx

        spans.append((start, h_star))
        start = h_star + 1

        if H - start < L_min:
            break

    # final span
    if start <= H - 1:
        spans.append((start, H - 1))

    if not spans:
        spans = [(0, H - 1)]

    return spans


# ---------------------------------------------------------------------------
# Build flat span loss + p0 vectors
# ---------------------------------------------------------------------------

def build_span_losses(
    token_loss: torch.Tensor,
    old_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    span_min_len: int,
    span_max_num: int,
    span_cut_prob: float,
    log_stats: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Convert token-level PPO loss [B, T] into a flat vector of per-span mean losses
    along with response-balanced uniform p0.

    Inputs:
        token_loss:       [B, T] token PPO surrogate loss (already action-masked)
        old_log_probs:    [B, T] rollout logprobs
        action_mask:      [B, T] 1 = valid generated token, 0 = padding
        span_min_len:     L_min
        span_max_num:     S_max
        span_cut_prob:    p_cut
        log_stats:        if True, return debug stats dict

    Outputs:
        span_losses:      [M] mean token loss per span (gradient flows)
        p0:               [M] response-balanced uniform: p0_{i,s} = 1/B / S_i
                          (no_grad)
        stats:            dict (empty unless log_stats=True)
    """
    B = token_loss.size(0)
    device = token_loss.device
    dtype = token_loss.dtype

    span_losses_list: List[torch.Tensor] = []
    p0_list: List[float] = []

    span_lens: List[int] = []
    spans_per_resp: List[int] = []

    # exp on detached old logprobs only (used for boundary picking, no gradient)
    with torch.no_grad():
        old_probs_full = torch.exp(old_log_probs.detach()).clamp(min=0.0, max=1.0)

    for i in range(B):
        mask_i = action_mask[i].bool()
        H_i = int(mask_i.sum().item())
        if H_i == 0:
            # No valid tokens — emit a zero loss as a single span placeholder.
            zero = token_loss.new_zeros(())  # scalar with grad-capable dtype
            span_losses_list.append(zero)
            p0_list.append(1.0 / max(1, B))
            spans_per_resp.append(1)
            span_lens.append(0)
            continue

        valid_idx = mask_i.nonzero(as_tuple=True)[0]
        loss_i = token_loss[i].index_select(0, valid_idx)
        probs_i = old_probs_full[i].index_select(0, valid_idx)

        spans = construct_spans(
            probs_i, H_i, span_min_len, span_max_num, span_cut_prob
        )
        S_i = len(spans)
        spans_per_resp.append(S_i)

        for (s, e) in spans:
            seg = loss_i[s : e + 1]
            span_losses_list.append(seg.mean())
            p0_list.append(1.0 / B / S_i)
            span_lens.append(e - s + 1)

    span_losses = torch.stack(span_losses_list).to(dtype=dtype, device=device)
    p0 = torch.tensor(p0_list, dtype=torch.float32, device=device)
    # Renormalize for numerical safety (handles edge cases like B=0 → already protected)
    p0 = p0 / p0.sum().clamp_min(1e-12)

    stats: Dict[str, float] = {}
    if log_stats:
        spans_per_resp_t = torch.tensor(spans_per_resp, dtype=torch.float32)
        span_lens_t = torch.tensor(span_lens, dtype=torch.float32)
        stats = {
            "avg_num_spans": float(spans_per_resp_t.mean().item()) if spans_per_resp else 0.0,
            "max_num_spans": int(spans_per_resp_t.max().item()) if spans_per_resp else 0,
            "min_span_len": int(span_lens_t.min().item()) if span_lens else 0,
            "avg_span_len": float(span_lens_t.mean().item()) if span_lens else 0.0,
            "frac_one_span": float((spans_per_resp_t == 1).float().mean().item()) if spans_per_resp else 0.0,
            "p0_sum": float(p0.sum().item()),
        }

    return span_losses, p0, stats


# ---------------------------------------------------------------------------
# χ² span-DRO loss
# ---------------------------------------------------------------------------

def chi2_span_dro_loss(
    span_losses: torch.Tensor,
    p0: torch.Tensor,
    rho: float,
    delta: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    χ² span-DRO objective:
        L = E_p0[L_span] + sqrt(rho) * sqrt(Var_p0(L_span) + delta)
    """
    p0_d = p0.detach().to(span_losses.dtype)
    mean = (p0_d * span_losses).sum()
    diff = span_losses - mean
    var = (p0_d * diff.pow(2)).sum()
    loss = mean + math.sqrt(max(rho, 0.0)) * torch.sqrt(var + delta)
    stats = {
        "span_loss_mean": float(mean.detach().item()),
        "span_loss_var": float(var.detach().item()),
        "chi2_var_penalty": float(
            (math.sqrt(max(rho, 0.0)) * torch.sqrt(var.detach() + delta)).item()
        ),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# KL span-DRO loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def _kl_eta_search(
    span_losses_det: torch.Tensor,
    log_p0: torch.Tensor,
    rho: float,
    eta_min: float,
    eta_max: float,
    eta_iters: int,
) -> float:
    """
    Grid search eta on log scale to minimize psi_eta + rho/eta with
        psi_eta = (1/eta) * logsumexp(log_p0 + eta * L)
    """
    if rho <= 0.0 or span_losses_det.numel() < 2:
        return max(eta_min, 1e-6)

    losses = span_losses_det.float()
    log_p0_f = log_p0.float()

    log_grid = torch.linspace(
        math.log(max(eta_min, 1e-12)),
        math.log(max(eta_max, eta_min * 1.001)),
        steps=max(2, int(eta_iters)),
        device=losses.device,
    )

    best_eta = float(eta_min)
    best_obj = float("inf")
    for log_eta in log_grid:
        eta = float(log_eta.exp().item())
        if eta <= 0.0:
            continue
        # logsumexp over flat span dimension
        psi = float(torch.logsumexp(log_p0_f + eta * losses, dim=0).item()) / eta
        obj = psi + rho / eta
        if obj < best_obj:
            best_obj = obj
            best_eta = eta

    return best_eta


def kl_span_dro_loss(
    span_losses: torch.Tensor,
    p0: torch.Tensor,
    rho: float,
    eta_min: float = 1e-6,
    eta_max: float = 1e6,
    eta_iters: int = 30,
    detach_q: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    KL span-DRO objective:
        eta* = argmin_eta { (1/eta) logsumexp(log p0 + eta L) + rho / eta }
        q_j  ∝ p0_j * exp(eta* * L_j)
        L    = sum_j q_j L_j

    `detach_q=True` matches the original KL-DRTO convention where adversarial
    weights do not receive gradients.
    """
    p0_d = p0.detach().to(span_losses.dtype)
    log_p0 = torch.log(p0_d.clamp_min(1e-12))

    eta_star = _kl_eta_search(
        span_losses.detach(), log_p0, rho, eta_min, eta_max, eta_iters
    )

    logits = log_p0 + eta_star * span_losses
    if detach_q:
        q = torch.softmax(logits.detach(), dim=0)
    else:
        q = torch.softmax(logits, dim=0)

    loss = (q * span_losses).sum()
    stats = {
        "kl_eta_star": float(eta_star),
        "span_loss_mean": float((p0_d * span_losses).sum().detach().item()),
    }
    return loss, stats
