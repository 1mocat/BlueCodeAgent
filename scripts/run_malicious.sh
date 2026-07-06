#!/bin/bash
# Malicious-instruction detection = knowledge retrieval + constitution.
# Runs both malicious subsets (RMCBench-derived + RedCode-Gen-derived).
#
# ⚠️  GATED DATA: the malicious knowledge/test JSONs are NOT distributed with this
# repo (see dataset/malicious_rmc/README.md and dataset/malicious_redcode/README.md).
# Obtain / regenerate them first, then run this script.
# Override MODEL / CONSTITUTION_MODEL / TOPK via env.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-gpt4o}"
CONSTITUTION_MODEL="${CONSTITUTION_MODEL:-gpt4o}"
TOPK="${TOPK:-3}"

for SUB in malicious_rmc malicious_redcode; do
  KNOWLEDGE="$REPO/dataset/$SUB/${SUB}_knowledge.json"
  TEST="$REPO/dataset/$SUB/${SUB}_test.json"
  OUTPUT="$REPO/results/$SUB/result_${MODEL}_top${TOPK}.json"
  if [ ! -f "$KNOWLEDGE" ] || [ ! -f "$TEST" ]; then
    echo "⏭️  Skipping $SUB — gated data not found (see dataset/$SUB/README.md)."
    continue
  fi
  mkdir -p "$(dirname "$OUTPUT")"
  echo "🛡️  Malicious detection [$SUB] | model=$MODEL constitution=$CONSTITUTION_MODEL topk=$TOPK"
  python -u "$REPO/src/blue_team/blue_team_malicious_code_prompt_constitution.py" \
    --knowledge_path "$KNOWLEDGE" \
    --test_path "$TEST" \
    --output_path "$OUTPUT" \
    --topk "$TOPK" \
    --model "$MODEL" \
    --constitution_model "$CONSTITUTION_MODEL"
done
