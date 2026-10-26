#!/bin/bash
# KL-DRTO with span-level DRO — math (Llama3-8B, OpenMathInstruct prompts).
#
#   bash scripts/train_kl_drto_span_math.sh
#
# Override any setting from the environment, e.g.  RHO=0.02 MAX_SAMPLES=5000 bash scripts/...
set -e
source "$(dirname "$0")/common.sh"

# Models
SFT=${SFT:-OpenRLHF/Llama-3-8b-sft-mixture}
DPO=${DPO:-RTO-RL/Llama3-8B-DPO}
RM=${RM:-RTO-RL/Llama3.2-1B-RewardModel}

# Data
DATASET=${DATASET:-$REPO_ROOT/data/openmath_100k.csv}
MAX_SAMPLES=${MAX_SAMPLES:-10000}
SEED=${SEED:-42}

# KL-DRTO:  q* = softmax(log p0 + eta* L),  eta* found by a log-spaced grid search
RHO=${RHO:-0.01}
ETA_MIN=${ETA_MIN:-1e-6}
ETA_MAX=${ETA_MAX:-1e6}
ETA_ITERS=${ETA_ITERS:-30}

# Span construction
SPAN_MIN_LEN=${SPAN_MIN_LEN:-16}
SPAN_MAX_NUM=${SPAN_MAX_NUM:-8}
SPAN_CUT_PROB=${SPAN_CUT_PROB:-0.05}

SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/ckpt/math/kl_drto_span}
mkdir -p "$SAVE_DIR/checkpoints" "$REPO_ROOT/logs"

deepspeed --master_port ${MASTER_PORT:-29504} --module openrlhf.cli.train_kl_span \
    --pretrain $SFT \
    --critic_pretrain $RM \
    --reward_pretrain $RM --normalize_reward \
    --dpo_pretrain $DPO \
    --prompt_data $DATASET --prompt_split train --input_key question \
    --pretrain_data $DATASET --pretrain_split train \
    --max_samples $MAX_SAMPLES \
    --seed $SEED \
    --prompt_max_len 512 --generate_max_len 4096 \
    --rollout_batch_size 512 --train_batch_size 256 \
    --micro_rollout_batch_size 8 --micro_train_batch_size 8 \
    --init_kl_coef 0.005 --actor_learning_rate 1e-5 --critic_learning_rate 5e-5 \
    --dpo_reward_scale 0.05 --dpo_reward_clip 0.05 \
    --zero_stage 2 --adam_offload --flash_attn --gradient_checkpointing --bf16 \
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --ptx_coef 0.05 \
    --max_epochs 1 \
    --span_dro --span_dro_type kl \
    --span_min_len $SPAN_MIN_LEN \
    --span_max_num $SPAN_MAX_NUM \
    --span_cut_prob $SPAN_CUT_PROB \
    --span_kl_rho $RHO \
    --span_kl_eta_min $ETA_MIN \
    --span_kl_eta_max $ETA_MAX \
    --span_kl_eta_iters $ETA_ITERS \
    --log_span_stats \
    --merge_lora \
    --ckpt_path "$SAVE_DIR/checkpoints" \
    --save_path "$SAVE_DIR" \
    2>&1 | tee "$REPO_ROOT/logs/kl_drto_span_math.log"

echo "Model saved to $SAVE_DIR"
