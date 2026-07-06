#!/bin/bash
# Utility: compute text-embedding-3-small embeddings for the `prompt` field
# (bias / malicious / prompt-injection data).
# The released dataset/ files are ALREADY embedded — only needed for NEW data.
#   usage: scripts/embed_prompt.sh <input.json> <output.json>
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INPUT="${1:?usage: embed_prompt.sh <input.json> <output.json>}"
OUTPUT="${2:?usage: embed_prompt.sh <input.json> <output.json>}"
python -u "$REPO/src/blue_team/calculate_embed_for_prompt.py" \
  --input_file "$INPUT" --record_save_file "$OUTPUT"
