#!/usr/bin/env bash
# End-to-end QLoRA text-to-SQL pipeline for Apple Silicon.
# Run steps individually while iterating:  ./run_all.sh quantize-base
set -euo pipefail

BASE_REPO="${BASE_REPO:-Qwen/Qwen2.5-Coder-7B-Instruct}"
BASE_4BIT="${BASE_4BIT:-models/qwen2.5-coder-7b-4bit}"
FUSED="${FUSED:-models/qwen-sql-fused}"
FUSED_4BIT="${FUSED_4BIT:-models/qwen-sql-4bit}"
SPIDER_ROOT="${SPIDER_ROOT:-spider_data}"
CONFIG="${CONFIG:-configs/qlora_m3pro_18gb.yaml}"
EVAL_N="${EVAL_N:-200}"

step="${1:-all}"

quantize_base() {
  echo ">>> Quantising base to 4-bit"
  mlx_lm.convert --hf-path "$BASE_REPO" --mlx-path "$BASE_4BIT" \
      -q --q-bits 4 --q-group-size 64
}

prep() {
  echo ">>> Building training data from Spider"
  python src/prepare_data.py --spider-root "$SPIDER_ROOT" --out data --eval-size 500
}

train() {
  echo ">>> QLoRA fine-tune ($CONFIG)"
  mlx_lm.lora --config "$CONFIG" 2>&1 | tee train.log
}

fuse() {
  echo ">>> Fusing adapter into bf16 weights"
  mlx_lm.fuse --model "$BASE_4BIT" --adapter-path adapters --save-path "$FUSED"
}

quantize_ft() {
  echo ">>> Quantising fine-tuned model to 4-bit"
  mlx_lm.convert --hf-path "$FUSED" --mlx-path "$FUSED_4BIT" \
      -q --q-bits 4 --q-group-size 64
}

bench() {
  echo ">>> Benchmarking"
  python src/benchmark.py \
      --manifest data/eval_manifest.json \
      --variant "base-4bit=$BASE_4BIT" \
      --variant "finetuned-4bit=$FUSED_4BIT" \
      --variant "finetuned-adapter=$BASE_4BIT,adapters" \
      --limit "$EVAL_N" --out results
}

case "$step" in
  quantize-base) quantize_base ;;
  prep)          prep ;;
  train)         train ;;
  fuse)          fuse ;;
  quantize-ft)   quantize_ft ;;
  bench)         bench ;;
  all)           quantize_base; prep; train; fuse; quantize_ft; bench ;;
  *) echo "usage: $0 {quantize-base|prep|train|fuse|quantize-ft|bench|all}"; exit 1 ;;
esac
