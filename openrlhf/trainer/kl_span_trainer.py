"""
KL-DRTO trainer with span-level DRO.

Subclasses `KL_Trainer` (kl_trainer.py).  RTO reward unchanged; only the
per-token PPO surrogate loss is reduced via span-level KL DRO when --span_dro is set.
"""

from __future__ import annotations

from typing import Dict

import torch

from .kl_trainer import KL_Trainer
from .span_dro_utils import build_span_losses, kl_span_dro_loss


class KLSpanTrainer(KL_Trainer):
    """KL-DRTO with span-level DRO over correctly-cut spans."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.span_dro: bool = getattr(self.args, "span_dro", True)
        self.span_min_len: int = int(getattr(self.args, "span_min_len", 16))
        self.span_max_num: int = int(getattr(self.args, "span_max_num", 8))
        self.span_cut_prob: float = float(getattr(self.args, "span_cut_prob", 0.05))
        self.span_kl_rho: float = float(getattr(self.args, "span_kl_rho", 0.01))
        self.span_kl_eta_min: float = float(getattr(self.args, "span_kl_eta_min", 1e-6))
        self.span_kl_eta_max: float = float(getattr(self.args, "span_kl_eta_max", 1e6))
        self.span_kl_eta_iters: int = int(getattr(self.args, "span_kl_eta_iters", 30))
        self.log_span_stats: bool = bool(getattr(self.args, "log_span_stats", False))

        if self.strategy.is_rank_0():
            self.strategy.print(
                f"[KL-Span-DRTO] span_dro={self.span_dro} "
                f"L_min={self.span_min_len} S_max={self.span_max_num} "
                f"p_cut={self.span_cut_prob} | rho={self.span_kl_rho} "
                f"eta_grid=[{self.span_kl_eta_min:g},{self.span_kl_eta_max:g}] iters={self.span_kl_eta_iters}"
            )

    def training_step_actor(self, experience) -> Dict[str, float]:
        if not self.span_dro:
            # delegate to original KL_Trainer behavior
            return super().training_step_actor(experience)

        self.actor.train()

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

        ratio = torch.exp(action_log_probs - old_action_log_probs)
        r_clip = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
        token_loss_raw = -(ratio * advantages)
        token_loss_clip = -(r_clip * advantages)
        token_loss = torch.maximum(token_loss_raw, token_loss_clip) * action_mask

        span_losses, p0, span_stats = build_span_losses(
            token_loss=token_loss,
            old_log_probs=old_action_log_probs,
            action_mask=action_mask,
            span_min_len=self.span_min_len,
            span_max_num=self.span_max_num,
            span_cut_prob=self.span_cut_prob,
            log_stats=self.log_span_stats,
        )
        actor_loss, dro_stats = kl_span_dro_loss(
            span_losses=span_losses,
            p0=p0,
            rho=self.span_kl_rho,
            eta_min=self.span_kl_eta_min,
            eta_max=self.span_kl_eta_max,
            eta_iters=self.span_kl_eta_iters,
        )

        aux_loss = output.aux_loss if self.aux_loss else 0.0
        base_loss = actor_loss + aux_loss * self.args.aux_loss_coef

        self.strategy.backward(base_loss, self.actor, self.actor_optim)

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

        status = {
            "policy_loss": float(actor_loss.item()),
            "actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
            "span_kl_eta_star": float(dro_stats["kl_eta_star"]),
            "span_loss_mean": float(dro_stats["span_loss_mean"]),
            "span_kl_rho": float(self.span_kl_rho),
            "span_dro": 1,
        }
        for k, v in span_stats.items():
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
