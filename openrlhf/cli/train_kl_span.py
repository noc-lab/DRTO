import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import pandas as pd  # Added for CSV support
from datasets import Dataset, load_dataset  # Added for CSV and GSM8K support
import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import KLSpanTrainer
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    critic = get_llm_for_sequence_regression(
        args.critic_pretrain,
        "critic",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
        init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
    )

    if not args.remote_rm_url:
        reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=args.value_head_prefix,
        )
        get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    else:
        reward_model = None

    strategy.print("reward normalization status: {}".format(args.normalize_reward))
    strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    strategy.print(actor)
    strategy.print(critic)

    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )
    get_tokenizer(args.pretrain, initial_model.model, "left", strategy)

    if args._use_dpo:
        dpo_model = Actor(
            args.dpo_pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        strategy.print(dpo_model)
    else:
        dpo_model = None

    if args._use_dpo_ref:
        dpo_ref = Actor(
            args.dpo_ref,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=False),
        )
        strategy.print(dpo_ref)
    else:
        dpo_ref = None

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(offload=True),
        )
    else:
        ema_model = None

    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    # Modified prompts_data loading to support CSV and GSM8K repo:config notation
    if args.prompt_data.startswith("openai/gsm8k"):
        # Special handling for GSM8K with optional :config notation
        if ":" in args.prompt_data:
            repo, cfg = args.prompt_data.split(":", 1)
        else:
            repo, cfg = "openai/gsm8k", "main"
        prompts_data = load_dataset(repo, cfg, split=args.prompt_split)
    elif args.prompt_data.endswith('.csv'):
        # Load CSV file directly
        df = pd.read_csv(args.prompt_data)
        prompts_data = Dataset.from_pandas(df)
        # Apply max_samples limit if specified
        if args.max_samples and len(prompts_data) > args.max_samples:
            prompts_data = prompts_data.select(range(args.max_samples))
    else:
        # Use original HuggingFace dataset loading
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
            train_split=args.prompt_split,
        )
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))

    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)

    # Modified pretrain_data loading to support CSV and GSM8K repo:config notation
    if args.pretrain_data:
        if args.pretrain_data.startswith("openai/gsm8k"):
            # Special handling for GSM8K with optional :config notation
            if ":" in args.pretrain_data:
                repo, cfg = args.pretrain_data.split(":", 1)
            else:
                repo, cfg = "openai/gsm8k", "main"
            pretrain_data = load_dataset(repo, cfg, split=args.pretrain_split)
        elif args.pretrain_data.endswith('.csv'):
            # Load CSV file directly
            df = pd.read_csv(args.pretrain_data)
            pretrain_data = Dataset.from_pandas(df)
        else:
            # Use original HuggingFace dataset loading
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
            )
        
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
            tokenizer,
            pretrain_max_len,
            strategy,
            pretrain_mode=True,
        )

    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)
    if args.pretrain_data:
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,
                    args.micro_train_batch_size,
                    True,
                    True,
                    pretrain_dataset.collate_fn,
                )
            )
        )
    else:
        pretrain_dataloader = None

    num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "cosine_with_min_lr",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    critic_scheduler = get_scheduler(
        "cosine_with_min_lr",
        critic_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
    )

    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        dpo_model,
        dpo_ref,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        dpo_model,
        dpo_ref,
        is_rlhf=True,
    )

    if ema_model:
        ema_model._offload = True
        ema_model = strategy.prepare(ema_model, is_rlhf=True)

    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
        _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"))
        strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"))
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    trainer = KLSpanTrainer(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        ema_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ema_beta=0.992,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote_rm_url=args.remote_rm_url,
        dpo_model=dpo_model,
        dpo_ref=dpo_ref,
        rl_config=args._rl_config,
        dro_kl_eta=args.dro_kl_eta,
        dro_adaptive=args.dro_adaptive,
        dro_target_var_ratio=args.dro_target_var_ratio,
        dro_eta_lr=args.dro_eta_lr,
        dro_eps=args.dro_eps,
    )

    trainer.fit(args, prompts_dataloader, pretrain_dataloader, consumed_samples, num_update_steps_per_episodes)

    strategy.save_model(
        ema_model if args.enable_ema else actor,
        tokenizer,
        args.save_path,
        merge_lora=args.merge_lora,
    )

    if args.save_value_network:
        strategy.save_model(
            critic,
            tokenizer,
            args.save_path + "_critic",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RTO
    parser.add_argument("--dpo_pretrain", type=str, default=None, help="HF model name or path, activate RTO")
    parser.add_argument("--dpo_ref", type=str, default=None, help="HF model name or path. If unset, reuse reference")
    parser.add_argument("--dpo_reward_scale", type=float, default=1.0, help="beta_1")
    parser.add_argument("--dpo_reward_clip", type=float, default=0, help="clip abs value of token rewards")

    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=float, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=512)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens per prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="(deprecated) overall max length")
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-PTX loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--gamma", type=float, default=1.0, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable reward normalization")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument("--n_samples_per_prompt", type=int, default=1, help="Responses per prompt in generation")
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # KL-DRO robustness
    parser.add_argument("--dro_kl_eta", type=float, default=0.0, 
                       help="KL-DRO radius parameter η. 0 disables DRO, try 0.1-10.0 for regularization")
    parser.add_argument("--dro_adaptive", action="store_true", default=False,
                       help="Enable adaptive η tuning via dual variable optimization")
    parser.add_argument("--dro_target_var_ratio", type=float, default=0.1,
                       help="Target variance ratio for adaptive DRO (typically 0.05-0.15)")
    parser.add_argument("--dro_eta_lr", type=float, default=1e-2,
                       help="Learning rate for adaptive η dual ascent")
    parser.add_argument("--dro_eps", type=float, default=1e-8,
                       help="Epsilon for numerical stability in standardization")

    # ── Span-level DRO ──
    parser.add_argument("--span_dro", action="store_true", default=False,
                        help="Enable span-level DRO over correctly-cut spans")
    parser.add_argument("--span_min_len", type=int, default=16,
                        help="L_min: minimum span length")
    parser.add_argument("--span_max_num", type=int, default=8,
                        help="S_max: maximum number of spans per response")
    parser.add_argument("--span_cut_prob", type=float, default=0.05,
                        help="p_cut: only tokens with prob < p_cut are eligible cutpoints")
    parser.add_argument("--span_dro_type", type=str, default="kl", choices=["chi2", "kl"],
                        help="Type of span-DRO objective (kl only used by KLSpanTrainer)")
    parser.add_argument("--span_kl_rho", type=float, default=0.01,
                        help="KL robustness radius for span-DRO")
    parser.add_argument("--span_kl_eta_min", type=float, default=1e-6,
                        help="Lower bound of η grid search")
    parser.add_argument("--span_kl_eta_max", type=float, default=1e6,
                        help="Upper bound of η grid search")
    parser.add_argument("--span_kl_eta_iters", type=int, default=30,
                        help="Number of grid points (log-scale) for η search")
    parser.add_argument("--log_span_stats", action="store_true", default=False,
                        help="Log per-step span statistics")

    # DeepSpeed / infra
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0.0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--merge_lora", action="store_true", default=False,
                        help="Merge LoRA adapter into the base model before saving")

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--prompt_data_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument("--pretrain_data_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--pretrain_split", type=str, default="train")
    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False, help="Use HF chat template")

    # wandb
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_kl_span_drto")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain
        else:
            args.critic_pretrain = args.pretrain

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    args._use_dpo = args.dpo_pretrain is not None
    args._use_dpo_ref = args._use_dpo and args.dpo_ref is not None and args.dpo_ref != args.pretrain
    args._rl_config = {}
    for argument in ["dpo_reward_scale", "dpo_reward_clip"]:
        args._rl_config[argument] = getattr(args, argument)

    train(args)