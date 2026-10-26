#!/bin/bash
# Shared environment for the DRTO training scripts (sourced, not executed).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200000

# GPUs used for training (override with e.g. CUDA_VISIBLE_DEVICES=0,1,2,3).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Optional: point the Hugging Face cache to a larger disk.
#   export HF_HOME=/path/to/huggingface
