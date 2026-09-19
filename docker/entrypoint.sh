#!/usr/bin/env bash
# Apply migrations before serving, so the schema is never behind the code.
set -euo pipefail

echo "[entrypoint] applying database migrations"
alembic upgrade head

echo "[entrypoint] starting uvicorn on ${HOST:-0.0.0.0}:${PORT:-8000}"
# --factory: src.main deliberately builds no app at import time.
exec uvicorn src.main:create_app \
    --factory \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --log-level "$(echo "${LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')"
