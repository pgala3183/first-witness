#!/usr/bin/env bash
# Fine-tune Qwen3-4B-Instruct-2507 on the First Witness call-agent data with Fireworks.
# Commands follow https://docs.fireworks.ai/fine-tuning/supervised-fine-tuning
# Run step by step; each step prints what to do next.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="accounts/fireworks/models/qwen3-4b-instruct-2507"
TRAIN_DS="first-witness-call-train"
VAL_DS="first-witness-call-val"
OUT_MODEL="first-witness-call-4b"

step="${1:-help}"
case "$step" in
  check)    # 0. confirm the base model is tunable (look for Tunable: true)
    firectl model get -a fireworks qwen3-4b-instruct-2507 ;;
  data)     # 1. build + upload datasets
    python3 finetune/build_dataset.py
    firectl dataset create "$TRAIN_DS" finetune/data/train_sft.jsonl
    firectl dataset create "$VAL_DS" finetune/data/val_sft.jsonl ;;
  train)    # 2. start the SFT job (LoRA). ~1.7M tokens/epoch.
    firectl sftj create --base-model "$BASE" --dataset "$TRAIN_DS" \
      --evaluation-dataset "$VAL_DS" --output-model "$OUT_MODEL" \
      --epochs 2 --lora-rank 16 ;;
  status)   # 3. watch it; then confirm the model exists
    firectl sftj list; firectl model list ;;
  deploy)   # 4. deploy the fine-tuned model (dedicated GPU, billed while up)
    firectl deployment create "$OUT_MODEL" --deployment-shape default ;;
  *)
    echo "usage: finetune/run_fireworks.sh check|data|train|status|deploy" ;;
esac
