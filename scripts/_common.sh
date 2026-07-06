#!/bin/bash
# Shared setup sourced by every run script.
# - Resolves the repo root (scripts live in <repo>/scripts/).
# - Puts src/ on PYTHONPATH so `from models.client import ...` / `from utils.* import ...` resolve.
# - Loads API keys from <repo>/.env if present (copy .env.example -> .env and fill in).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi
