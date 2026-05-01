#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV="../.disertatie/bin/activate"
if [[ ! -f "$VENV" ]]; then
  echo "Virtualenv not found at $VENV — activate it manually or adjust run.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD_FLAG="${RELOAD:-1}"

ARGS=(backend.main:app --host "$HOST" --port "$PORT")
if [[ "$RELOAD_FLAG" == "1" ]]; then
  ARGS+=(--reload)
fi

exec uvicorn "${ARGS[@]}"
