#!/bin/bash
# Bias-instruction detection = knowledge retrieval + constitution.
# One pass records four responses per item: baseline (direct prompting),
# +general safety reminder, +fine-grained safety reminder, +constitution (BlueCodeAgent).
# Override MODEL / CONSTITUTION_MODEL / TOPK / KNOWLEDGE / TEST / OUTPUT via env.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-gpt4o}"
CONSTITUTION_MODEL="${CONSTITUTION_MODEL:-gpt4o}"
TOPK="${TOPK:-3}"
KNOWLEDGE="${KNOWLEDGE:-$REPO/dataset/bias/bias_knowledge.json}"
TEST="${TEST:-$REPO/dataset/bias/bias_test.json}"
OUTPUT="${OUTPUT:-$REPO/results/bias/result_${MODEL}_top${TOPK}.json}"
mkdir -p "$(dirname "$OUTPUT")"

echo "🛡️  Bias detection | model=$MODEL constitution=$CONSTITUTION_MODEL topk=$TOPK"
python -u "$REPO/src/blue_team/blue_team_bias_code_prompt_constitution.py" \
  --knowledge_path "$KNOWLEDGE" \
  --test_path "$TEST" \
  --output_path "$OUTPUT" \
  --topk "$TOPK" \
  --model "$MODEL" \
  --constitution_model "$CONSTITUTION_MODEL"
