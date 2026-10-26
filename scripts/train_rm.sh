#!/bin/bash
# Reward model (Llama-3.2-1B) trained on UltraFeedback preference pairs.
# Its output directory is used as RM in the DRTO scripts:  RM=ckpt/rm bash scripts/train_chi_drto_span_math.sh
set -e
source "$(dirname "$0")/common.sh"

BASE=${BASE:-unsloth/Llama-3.2-1B-Instruct}
DATASET=${DATASET:-HuggingFaceH4/ultrafeedback_binarized}
SAVE_DIR=${SAVE_DIR:-$REPO_ROOT/ckpt/rm}
mkdir -p "$SAVE_DIR" "$REPO_ROOT/logs"

deepspeed --master_port ${MASTER_PORT:-29502} --module openrlhf.cli.train_rm \
    --pretrain $BASE \
    --dataset $DATASET --train_split train_prefs --eval_split test_prefs \
    --apply_chat_template \
    --tokenizer_chat_template "$(cat "$REPO_ROOT/scripts/llama3_chat_template.jinja")" \
    --chosen_key chosen --rejected_key rejected \
    --max_len 2048 \
    --train_batch_size 128 --micro_train_batch_size 4 \
    --learning_rate 9e-6 --max_epochs 1 \
    --zero_stage 2 --bf16 --flash_attn --gradient_checkpointing \
    --seed 42 \
    --save_path "$SAVE_DIR" \
    2>&1 | tee "$REPO_ROOT/logs/rm.log"

echo "Reward model saved to $SAVE_DIR"
