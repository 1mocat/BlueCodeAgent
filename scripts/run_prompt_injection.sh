#!/bin/bash
# Prompt-injection detection = knowledge retrieval + constitution (yes/no judgment).
# Records baseline / +general reminder / +fine-grained reminder / +constitution per item.
# Override MODEL / CONSTITUTION_MODEL / TOPK / KNOWLEDGE / TEST / OUTPUT via env.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-gpt4o}"
CONSTITUTION_MODEL="${CONSTITUTION_MODEL:-gpt4o}"
TOPK="${TOPK:-3}"
KNOWLEDGE="${KNOWLEDGE:-$REPO/dataset/prompt_injection/prompt_injection_knowledge.json}"
TEST="${TEST:-$REPO/dataset/prompt_injection/prompt_injection_test.json}"
OUTPUT="${OUTPUT:-$REPO/results/prompt_injection/result_${MODEL}_top${TOPK}.json}"
mkdir -p "$(dirname "$OUTPUT")"

echo "🛡️  Prompt-injection detection | model=$MODEL constitution=$CONSTITUTION_MODEL topk=$TOPK"
python -u "$REPO/src/blue_team/blue_team_prompt_injection.py" \
  --knowledge_path "$KNOWLEDGE" \
  --test_path "$TEST" \
  --output_path "$OUTPUT" \
  --topk "$TOPK" \
  --model "$MODEL" \
  --constitution_model "$CONSTITUTION_MODEL"
