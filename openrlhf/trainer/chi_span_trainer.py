"""
χ²-DRTO trainer with span-level DRO.

Subclasses `wPPOTrainer` (chi_trainer.py).  The original RTO reward is
unchanged — advantages are still derived from r_RTO upstream — only the
per-token PPO surrogate loss is reduced via span-level χ² DRO instead of
trajectory-level χ² DRO when --span_dro is set.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

from .chi_trainer import wPPOTrainer
from .span_dro_utils import (
    build_span_losses,
    chi2_span_dro_loss,
    kl_span_dro_loss,
)


class ChiSpanTrainer(wPPOTrainer):
    """χ²-DRTO with span-level DRO over correctly-cut spans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Span-DRO config (read from args)
        self.span_dro: bool = getattr(self.args, "span_dro", True)
        self.span_min_len: int = int(getattr(self.args, "span_min_len", 16))
        self.span_max_num: int = int(getattr(self.args, "span_max_num", 8))
        self.span_cut_prob: float = float(getattr(self.args, "span_cut_prob", 0.05))
        self.log_span_stats: bool = bool(getattr(self.args, "log_span_stats", False))

        # The χ² radii are inherited from wPPOTrainer (self.dro_rho, self.dro_delta).
        # span_dro_type: "chi2" | "kl"
        self.span_dro_type: str = getattr(self.args, "span_dro_type", "chi2")
        if self.strategy.is_rank_0():
            self.strategy.print(
                f"[χ²-Span-DRTO] span_dro={self.span_dro} "
                f"L_min={self.span_min_len} S_max={self.span_max_num} "
                f"p_cut={self.span_cut_prob} | rho={self.dro_rho} delta={self.dro_delta} "
                f"type={self.span_dro_type}"
            )

    # ------------------------------------------------------------------
    # Override actor step
    # ------------------------------------------------------------------
    def training_step_actor(self, experience) -> Dict[str, float]:
        """
        Per-token PPO loss → cutpoint spans → χ² span-DRO objective.

        RTO reward is upstream (in advantages); we do NOT modify it.
        """
        self.actor.train()

        # ─── Unpack (mirrors wPPOTrainer.training_step_actor) ───
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            if isinstance(experience.action_mask, list):
                action_mask = torch.cat(experience.action_mask, dim=0).unsqueeze(0)
            else:
                action_mask = experience.action_mask
            num_actions = [v.numel() for v in experience.values]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            action_mask = experience.action_mask
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # ─── Token-level PPO surrogate loss ───
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        r_clip = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
        token_loss_raw = -(ratio * advantages)
        token_loss_clip = -(r_clip * advantages)
        token_loss = torch.maximum(token_loss_raw, token_loss_clip) * action_mask

        # ─── Reduce via span DRO or trajectory DRO ───
        if self.span_dro:
            span_losses, p0, span_stats = build_span_losses(
                token_loss=token_loss,
                old_log_probs=old_action_log_probs,
                action_mask=action_mask,
                span_min_len=self.span_min_len,
                span_max_num=self.span_max_num,
                span_cut_prob=self.span_cut_prob,
                log_stats=self.log_span_stats,
            )
            if self.span_dro_type == "kl":
                actor_loss, dro_stats = kl_span_dro_loss(
                    span_losses=span_losses,
                    p0=p0,
                    rho=self.dro_rho,
                )
            else:  # default: chi2
                actor_loss, dro_stats = chi2_span_dro_loss(
                    span_losses=span_losses,
                    p0=p0,
                    rho=self.dro_rho,
                    delta=self.dro_delta,
                )
            L_mean = torch.tensor(dro_stats["span_loss_mean"], device=actor_loss.device)
            dro_penalty = torch.tensor(dro_stats["chi2_var_penalty"], device=actor_loss.device)
            dro_var = torch.tensor(dro_stats["span_loss_var"], device=actor_loss.device)
            extra_log = {**dro_stats, **span_stats}
        else:
            # Fallback: trajectory-level χ² DRO (original wPPOTrainer behavior).
            counts = action_mask.sum(dim=1).clamp_min(1)
            L_i = token_loss.sum(dim=1) / counts                                # [B]
            L_stats = self._all_gather_with_grad(L_i) if self.dro_global_stats else L_i
            L_mean = L_stats.mean()
            if self.dro_rho > 0.0 and L_stats.numel() > 1:
                dro_var = (L_stats - L_mean).pow(2).mean()
                dro_penalty = math.sqrt(self.dro_rho) * torch.sqrt(dro_var + self.dro_delta)
            else:
                dro_var = torch.tensor(0.0, device=L_mean.device)
                dro_penalty = torch.tensor(0.0, device=L_mean.device)
            actor_loss = L_mean + dro_penalty
            extra_log = {}

        # ─── Aux + backward ───
        aux_loss = output.aux_loss if self.aux_loss else 0.0
        base_loss = actor_loss + aux_loss * self.args.aux_loss_coef

        self.strategy.backward(base_loss, self.actor, self.actor_optim)

        # PTX
        ptx_loss_value = 0.0
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask_ptx = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask_ptx.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )
            output_ptx = self.actor(inputs, attention_mask=attention_mask_ptx, return_output=True)
            ptx_logits = output_ptx["logits"]
            ptx_loss = self.ptx_loss_fn(ptx_logits, label)
            ptx_loss_value = float(ptx_loss.item())
            aux_loss_ptx = output_ptx.aux_loss if self.aux_loss else 0.0
            ptx_total = ptx_loss + aux_loss_ptx * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * ptx_total, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # ─── Status ───
        status = {
            "policy_loss": float(actor_loss.item()),
            "actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
            "dro_penalty": float(dro_penalty.item()),
            "dro_var": float(dro_var.item()),
            "L_mean": float(L_mean.item()),
            "dro_rho": float(self.dro_rho),
            "span_dro": int(self.span_dro),
        }
        for k, v in extra_log.items():
            status[k] = float(v) if not isinstance(v, (int, str)) else v
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss_value
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum()
                    / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status
