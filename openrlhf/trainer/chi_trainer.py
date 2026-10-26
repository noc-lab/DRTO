import math
import os.path
import os
from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import ray
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

# Try relative import first (package mode), else fallback (script mode)
try:
    from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer
except Exception:
    from ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class wPPOTrainer(ABC):
    """
    Trainer for PPO algorithm with PURE χ²-DRO replacement on the actor objective.

    PURE χ² surrogate per minibatch:
        J(θ) = L̄(θ) + √ρ · √(Var(L(θ)) + δ),
    where L is the per-sequence PPO clipped surrogate loss.

    Notes:
    - If args.dro_global_stats=True and distributed is initialized, we compute mean/var on the
      global (all-ranks) minibatch using autograd-safe all_gather when available.
    - Otherwise we compute mean/var on the local batch (still a valid minibatch objective).
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        # RTO
        dpo_model: Optional[Actor] = None,
        dpo_ref: Optional[Actor] = None,
        rl_config: Dict[str, Any] = {},
        **generate_kwargs,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()
        self.eps_clip = eps_clip

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
            dpo_model,
            dpo_ref,
            rl_config,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        # ---- PURE χ²-DRO knobs ----
        self.dro_rho = getattr(self.args, "dro_rho", 0.0)            # ρ ≥ 0
        self.dro_delta = getattr(self.args, "dro_delta", 1e-8)       # δ ≥ 0
        self.dro_global_stats = getattr(self.args, "dro_global_stats", False)

        self.strategy.print(
            f"[χ²-DRO] Pure replacement: J=L̄+√ρ√(Var+δ). ρ={self.dro_rho}, δ={self.dro_delta}, global_stats={self.dro_global_stats}"
        )

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
        )
        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)

        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes
        if args.save_steps == -1:
            args.save_steps = float("inf")

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        steps = consumed_samples // args.rollout_batch_size * update_timesteps + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                experience = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)

                if steps % update_timesteps == 0:
                    output = self.tokenizer.batch_decode(
                        experience.sequences[0].unsqueeze(0), skip_special_tokens=True
                    )
                    self.strategy.print(output)
                self.replay_buffer.append(experience)

                if steps % update_timesteps == 0:
                    global_steps = steps // update_timesteps

                    torch.cuda.empty_cache()
                    self.replay_buffer.normalize("advantages", self.strategy)
                    status = self.ppo_train(global_steps)
                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()

                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    pbar.set_postfix(status)

                    client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                    self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}
                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }
                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]
                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

            if self.args.save_epoch_checkpoints:
                tag = f"global_step{global_steps}_epoch{epoch + 1}"
                client_states = {"consumed_samples": global_steps * self.args.rollout_batch_size}
                self._save_checkpoint(self.args, tag, client_states)
                if self.strategy.is_rank_0():
                    self.strategy.print(f"Saved epoch checkpoint: {tag}")

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience)
        status.update(self.training_step_critic(experience))
        return status

    @torch.no_grad()
    def _token_counts_per_sample(self, action_mask: torch.Tensor) -> torch.Tensor:
        counts = action_mask.sum(dim=1).clamp_min(1)
        return counts

    def _all_gather_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Autograd-safe all_gather for 1D tensors [B].
        Falls back to local stats if not available.
        """
        if (not dist.is_initialized()) or (dist.get_world_size() == 1):
            return x
        try:
            from torch.distributed.nn.functional import all_gather
            xs = all_gather(x)
            return torch.cat(xs, dim=0)
        except Exception:
            if self.strategy.is_rank_0():
                self.strategy.print(
                    "[χ²-DRO] dro_global_stats=True but autograd all_gather not available; falling back to local DRO stats."
                )
            return x

    def _per_sequence_ppo_surrogate(
        self,
        new_logp: torch.Tensor,          # [B, T]
        old_logp: torch.Tensor,          # [B, T]
        advantages: torch.Tensor,        # [B, T]
        action_mask: torch.Tensor,       # [B, T] 1 on answer tokens
        return_components: bool = False, # if True, also return (ell_raw, ell_clip)
    ):
        """
        Compute per-sequence PPO clipped surrogate loss vector L_i(θ) in the standard token-wise way:
            L_i = (1/|T_i|) Σ_t max( -r_t A_t, -clip(r_t) A_t )  over answer tokens

        This corresponds exactly to the PPO clipped objective:
            -Σ min(r_t A_t, clip(r_t) A_t).
        """
        assert new_logp.shape == old_logp.shape == advantages.shape == action_mask.shape

        ratio = torch.exp(new_logp - old_logp)                          # [B, T]
        r_clipped = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)

        # Tokenwise negative surrogates
        token_raw = -(ratio * advantages)                               # [B, T]
        token_clip = -(r_clipped * advantages)                          # [B, T]
        token_loss = torch.maximum(token_raw, token_clip)               # tokenwise max  <=>  -min(...)

        token_loss = token_loss * action_mask

        counts = self._token_counts_per_sample(action_mask)             # [B]
        L_i = token_loss.sum(dim=1) / counts                            # [B]

        if return_components:
            ell_raw = (token_raw * action_mask).sum(dim=1) / counts
            ell_clip = (token_clip * action_mask).sum(dim=1) / counts
            return L_i, ell_raw, ell_clip
        return L_i

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        """
        Actor step with PURE χ²-DRO replacement:
            actor_loss = mean(L_i) + √ρ √(Var(L_i)+δ)
        where L_i is the per-sequence PPO clipped surrogate loss.
        """
        self.actor.train()

        # Packed sequences handling
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)

            # IMPORTANT: ensure action_mask is also packed consistently
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

        # Per-sequence PPO loss vector L_i(θ)
        L_i, ell_raw, ell_clip = self._per_sequence_ppo_surrogate(
            new_logp=action_log_probs,
            old_logp=old_action_log_probs,
            advantages=advantages,
            action_mask=action_mask,
            return_components=True,
        )  # [B]

        # Select stats batch: local or global (with grad)
        L_stats = L_i
        if self.dro_global_stats:
            L_stats = self._all_gather_with_grad(L_i)

        L_mean = L_stats.mean()

        dro_penalty = torch.tensor(0.0, device=L_mean.device)
        dro_var = torch.tensor(0.0, device=L_mean.device)

        if self.dro_rho > 0.0 and L_stats.numel() > 1:
            dro_var = (L_stats - L_mean).pow(2).mean()  # unbiased=False
            dro_penalty = math.sqrt(self.dro_rho) * torch.sqrt(dro_var + self.dro_delta)

        actor_loss = L_mean + dro_penalty

        # Aux loss (Mixtral)
        aux_loss = output.aux_loss if self.aux_loss else 0.0
        base_loss = actor_loss + aux_loss * self.args.aux_loss_coef

        # Backward
        self.strategy.backward(base_loss, self.actor, self.actor_optim)

        # PTX loss
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
            ptx_total_loss = ptx_loss + aux_loss_ptx * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * ptx_total_loss, self.actor, self.actor_optim)

        # Optim step
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        # EMA
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # Status logging
        status = {
            "policy_loss": float(actor_loss.item()),
            "actor_lr": float(self.actor_scheduler.get_last_lr()[0]),
            "dro_penalty": float(dro_penalty.item()),
            "dro_var": float(dro_var.item()),
            "L_mean": float(L_mean.item()),
            "ell_raw_mean_local": float(ell_raw.mean().item()),
            "ell_clip_mean_local": float(ell_clip.mean().item()),
            "dro_rho": float(self.dro_rho),
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss_value

        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()

        return status

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:
        self.critic.train()

        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_values = torch.cat(experience.values, dim=0).unsqueeze(0)
            returns = torch.cat(experience.returns, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.values]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_values = experience.values
            returns = experience.returns
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask if not isinstance(experience.action_mask, list)
            else torch.cat(experience.action_mask, dim=0).unsqueeze(0),
        )

        aux_loss = output.aux_loss if self.aux_loss else 0.0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef

        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        status = {
            "critic_loss": float(critic_loss.item()),
            "values": float(masked_mean(values, experience.action_mask).item()),
            "critic_lr": float(self.critic_scheduler.get_last_lr()[0]),
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        if global_step % args.eval_steps == 0:
            pass

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        self.strategy.save_ckpt(
            self.actor.model,
            os.path.join(args.ckpt_path, "_actor"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            client_states,
        )
        self.strategy.save_ckpt(
            self.critic, os.path.join(args.ckpt_path, "_critic"), tag, args.max_ckpt_num, args.max_ckpt_mem
        )
