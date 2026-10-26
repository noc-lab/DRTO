import math
import os.path
import os
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class KL_Trainer(ABC):
    """
        Trainer for PPO algorithm (with KL-DRO option).

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (nn.Module): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
        remote_rm_url (str, optional): function for reward model api
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
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
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

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
        # keep eps_clip for our per-sequence surrogate (KL-DRO computation)
        self.eps_clip = eps_clip

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
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

        # ---- KL-DRO knobs (exponential reweighting of per-sequence PPO losses) ----
        # Base η (KL radius): 0 disables DRO; try 0.1 ~ 10.0 or use adaptive mode
        self.dro_kl_eta = getattr(self.args, "dro_kl_eta", 0.0)
        # numerical stability for standardization
        self.dro_eps = getattr(self.args, "dro_eps", 1e-8)
        
        # Adaptive η scheduler (if enabled)
        self.dro_adaptive = getattr(self.args, "dro_adaptive", False)
        if self.dro_adaptive:
            # Target variance level (normalized by mean loss magnitude)
            self.dro_target_var_ratio = getattr(self.args, "dro_target_var_ratio", 0.1)
            # Learning rate for dual variable update
            self.dro_eta_lr = getattr(self.args, "dro_eta_lr", 1e-2)
            # Learnable η (initialized to dro_kl_eta if set, else 1.0)
            self.dro_eta_dual = self.dro_kl_eta if self.dro_kl_eta > 0 else 1.0
            # EMA of variance ratio for stability
            self.dro_var_ratio_ema = None
            self.dro_ema_beta = 0.9
            self.strategy.print(f"[DRO Adaptive] Enabled with target_var_ratio={self.dro_target_var_ratio}, eta_lr={self.dro_eta_lr}")
        else:
            self.dro_eta_dual = None

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

        # Initialize TensorBoard writer if wandb is not available
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

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
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
                # print prompt/answer in each update step
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

                    # logs/checkpoints
                    client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                    self.save_logs_and_checkpoints(args, global_steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
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

                # for DP
                # weighted mean for kl
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
        """
        action_mask: [B, T] with 1s on answer tokens; returns per-sample counts [B]
        """
        counts = action_mask.sum(dim=1).clamp_min(1)
        return counts

    def _per_sequence_ppo_surrogate(
        self,
        new_logp: torch.Tensor,          # [B, T] log π_θ(a_t|s_t)
        old_logp: torch.Tensor,          # [B, T] log π_old(a_t|s_t)
        advantages: torch.Tensor,        # [B, T]
        action_mask: torch.Tensor,       # [B, T] 1 on answer tokens, else 0
        return_both: bool = False,       # If True, return (ell_raw, ell_clip)
    ) -> torch.Tensor:
        """
        Returns per-sequence PPO loss (NEGATIVE surrogate) as a vector [B].
        Aggregates tokens ONLY over answer tokens with length normalization.
        
        If return_both=True, returns (ell_raw, ell_clip) for pre-clip variance computation.
        Otherwise returns max(ell_raw, ell_clip) for standard PPO.
        """
        assert new_logp.shape == old_logp.shape == advantages.shape == action_mask.shape
        ratio = torch.exp(new_logp - old_logp)                          # [B, T]
        r_clipped = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
        
        # Raw (unclipped) surrogate
        token_loss_raw = -(ratio * advantages) * action_mask            # [B, T]
        counts = self._token_counts_per_sample(action_mask)             # [B]
        ell_raw = token_loss_raw.sum(dim=1) / counts                    # [B]
        
        # Clipped surrogate
        token_loss_clip = -(r_clipped * advantages) * action_mask       # [B, T]
        ell_clip = token_loss_clip.sum(dim=1) / counts                  # [B]
        
        if return_both:
            return ell_raw, ell_clip
        else:
            # Standard PPO: use max(raw, clipped) for conservative update
            return torch.maximum(ell_raw, ell_clip)

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        """
        Enhanced actor training step with KL-DRO (Algorithm 2).
        Implements standardized log-sum-exp with adaptive η dual variable.
        """
        self.actor.train()

        # Handle packed sequences
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.values]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_action_log_probs = experience.action_log_probs
            advantages = experience.advantages
            num_actions = experience.action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # Forward pass
        action_log_probs, output = self.actor(
            sequences,
            num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )

        # === Auxiliary Loss (for Mixtral) ===
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0

        # === KL-DRO Implementation (Algorithm 2) ===
        # Get both raw and clipped per-sequence losses for variance computation
        ell_raw, ell_clip = self._per_sequence_ppo_surrogate(
            new_logp=action_log_probs,
            old_logp=old_action_log_probs,
            advantages=advantages,
            action_mask=experience.action_mask,
            return_both=True,
        )                              # [B], [B]

        # Gather ell_raw across all GPUs for proper distributed variance computation
        if torch.distributed.is_initialized():
            ell_raw_list = [torch.zeros_like(ell_raw) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(ell_raw_list, ell_raw)
            ell_raw_global = torch.cat(ell_raw_list, dim=0)
        else:
            ell_raw_global = ell_raw
        
        # Standard PPO: mean of max(ell_raw, ell_clip)
        ell = torch.maximum(ell_raw, ell_clip)                   # [B]
        base_ppo_loss = ell.mean()

        kl_dro_loss = None
        dro_stats = {}
        
        if self.dro_kl_eta > 0.0 and ell_raw_global.numel() > 1:
            # Step 1: Standardize pre-clip losses (using global statistics)
            mean_raw = ell_raw_global.mean()
            std_raw = ell_raw_global.std() + self.dro_eps
            ell_standardized = (ell_raw_global - mean_raw) / std_raw    # [B_global]
            
            # Step 2: Stable log-sum-exp
            m = ell_standardized.max()
            log_mean_exp = torch.log(torch.mean(torch.exp(self.dro_kl_eta * (ell_standardized - m))))
            L_KL = m + (1.0 / self.dro_kl_eta) * log_mean_exp
            
            # Step 3: Final objective: mean(max(ell_raw, ell_clip)) + L_KL
            kl_dro_loss = base_ppo_loss + L_KL
            
            # Track variance ratio for adaptive η
            if self.dro_adaptive:
                with torch.no_grad():
                    var_raw = ell_raw_global.var()
                    var_ratio = (std_raw ** 2) / (mean_raw.abs() ** 2 + self.dro_eps)
                    # EMA update
                    if self.dro_var_ratio_ema is None:
                        self.dro_var_ratio_ema = var_ratio.item()
                    else:
                        self.dro_var_ratio_ema = (
                            0.9 * self.dro_var_ratio_ema + 0.1 * var_ratio.item()
                        )
                    
                    # Adaptive η update: dual ascent
                    # If variance ratio > target, increase η (more robust)
                    # If variance ratio < target, decrease η (less penalty)
                    grad_eta = self.dro_var_ratio_ema - self.dro_target_var_ratio
                    self.dro_eta_dual = max(0.0, min(100.0, self.dro_eta_dual + self.dro_eta_lr * grad_eta))
                    self.dro_kl_eta = self.dro_eta_dual
                    
                    dro_stats = {
                        "dro_var_ratio": var_ratio.item(),
                        "dro_var_ratio_ema": self.dro_var_ratio_ema,
                        "dro_eta": self.dro_kl_eta,
                    }
            
            base_loss = kl_dro_loss + aux_loss * self.args.aux_loss_coef
        else:
            # Use standard mean of per-sequence losses
            base_loss = base_ppo_loss + aux_loss * self.args.aux_loss_coef

        # === Backward Pass ===
        self.strategy.backward(base_loss, self.actor, self.actor_optim)

        # === PTX Loss (Pretraining Loss) ===
        ptx_loss_value = 0
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # Compute PTX loss
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            ptx_loss_value = ptx_loss.item()

            # Add auxiliary loss for Mixtral
            if self.aux_loss:
                aux_loss_ptx = output.aux_loss
            else:
                aux_loss_ptx = 0

            ptx_total_loss = ptx_loss + aux_loss_ptx * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * ptx_total_loss, self.actor, self.actor_optim)

        # === Optimizer Step ===
        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")

        # === EMA Update ===
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # === Status Logging ===
        status = {
            "policy_loss": kl_dro_loss.item() if kl_dro_loss is not None else base_ppo_loss.item(),
            "actor_lr": self.actor_scheduler.get_last_lr()[0]
        }
        if self.dro_kl_eta > 0.0 and ell_raw.numel() > 1:
            if kl_dro_loss is not None:
                status["dro_kl_loss"] = kl_dro_loss.item()
            status["dro_eta"] = self.dro_kl_eta
            status.update(dro_stats)  # Add variance ratio and EMA stats
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss_value

        # Add experience statistics
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

        # TODO: this is a bad indicator to say that data is packed...
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

        # critic loss
        values, output = self.critic(
            sequences,
            num_actions=num_actions,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        # loss function
        critic_loss = self.critic_loss_fn(
            values,
            old_values,
            returns,
            action_mask=experience.action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.critic, self.critic_optim)
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")

        # status
        status = {
            "critic_loss": critic_loss.item(),
            "values": masked_mean(values, experience.action_mask).item(),
            "critic_lr": self.critic_scheduler.get_last_lr()[0],
        }
        return status

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
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