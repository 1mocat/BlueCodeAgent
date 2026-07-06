#!/bin/bash
# Vulnerable-code detection = knowledge retrieval + constitution + dynamic testing
# (Algorithm 1). Produces baseline / +code-examples / +constitution / +dynamic
# judgments in one pass.
#
# NOTE: dynamic testing executes generated tests in an ISOLATED DOCKER SANDBOX
# (image `red-teaming-code-agent`, auto-built from environment/). A running Docker
# daemon is required. The released dataset is already embedded.
#
# Override any of MODEL / CONSTITUTION_MODEL / TOPK / KNOWLEDGE / TEST / RESULT via env.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL="${MODEL:-gpt4o}"
CONSTITUTION_MODEL="${CONSTITUTION_MODEL:-gpt4o}"
TOPK="${TOPK:-3}"
KNOWLEDGE="${KNOWLEDGE:-$REPO/dataset/vulnerability/vulnerability_knowledge.json}"
TEST="${TEST:-$REPO/dataset/vulnerability/vulnerability_test.json}"
RESULT="${RESULT:-$REPO/results/vulnerability/result_${MODEL}_top${TOPK}.json}"
mkdir -p "$(dirname "$RESULT")"

echo "🛡️  Vulnerable-code detection | model=$MODEL constitution=$CONSTITUTION_MODEL topk=$TOPK"
echo "📚 knowledge=$KNOWLEDGE"
echo "📥 test=$TEST"
echo "📤 result=$RESULT"

python -u "$REPO/src/blue_team/blue_team_red_help_blue_test_all.py" \
  --knowledge_file "$KNOWLEDGE" \
  --test_file "$TEST" \
  --result_file "$RESULT" \
  --topk "$TOPK" \
  --model "$MODEL" \
  --constitution_model "$CONSTITUTION_MODEL"
