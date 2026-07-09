#!/usr/bin/env bash
# End-to-end Docker smoke test for the titer-prediction inference service.
#
# Builds the image, runs it two ways, and asserts the full contract over HTTP:
#   1. WITH the model mounted  -> /health model_loaded=true, /predict 200,
#      and an invalid payload -> 400 (per the OpenAPI spec).
#   2. WITHOUT the model       -> /health model_loaded=false, /predict 503.
#
# Requires a running Docker daemon (Docker Desktop on Windows). Run from anywhere:
#   bash scripts/smoke_docker.sh
set -euo pipefail

# Stop git-bash/MSYS from rewriting container-side paths (e.g. /app/..., URLs).
export MSYS_NO_PATHCONV=1

IMAGE="datahow-titer-service"
MODEL_CONTAINER="titer_smoke_model"
NOMODEL_CONTAINER="titer_smoke_nomodel"
PORT=8000
BASE="http://localhost:${PORT}"

# Windows-style path (C:/...) via `pwd -W` so native docker.exe / curl.exe accept
# it for the build context, the -v mount, and --data @file.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && { pwd -W 2>/dev/null || pwd; })"
PAYLOAD="${REPO_DIR}/scripts/sample_payload.json"

cleanup() {
  docker rm -f "$MODEL_CONTAINER" "$NOMODEL_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# POST and echo the HTTP status code; the response body is left in $LAST_BODY.
# Captures the code via -w on stdout (no -o /dev/null, which native curl.exe on
# Windows cannot write to under MSYS_NO_PATHCONV).
# Splits the response into $RESP_CODE / $RESP_BODY. Runs in the current shell (not
# a command-substitution subshell), so the globals survive the call.
RESP_CODE=""
RESP_BODY=""
post() {  # post URL [curl-args...]
  local url="$1"; shift
  local resp
  resp="$(curl -s -w $'\n%{http_code}' -X POST -H 'Content-Type: application/json' "$@" "$url")"
  RESP_CODE="${resp##*$'\n'}"   # after the final newline (the status code)
  RESP_BODY="${resp%$'\n'*}"    # before it (the response body)
}

wait_for_health() {  # wait_for_health EXPECT_LOADED(true|false)
  local expect="$1" i body
  for i in $(seq 1 60); do
    body="$(curl -s "${BASE}/health" 2>/dev/null || true)"
    if echo "$body" | grep -q "\"model_loaded\":${expect}"; then
      echo "  /health -> $body"
      return 0
    fi
    sleep 1
  done
  fail "timed out waiting for /health model_loaded=${expect} (last: ${body:-<none>})"
}

echo "==> Building image ${IMAGE}"
docker build -t "$IMAGE" "$REPO_DIR"

# --- 1. WITH the model mounted --------------------------------------------
echo "==> Running WITH model mounted"
cleanup
docker run -d --name "$MODEL_CONTAINER" -p "${PORT}:8000" \
  -e MODEL_PATH=/app/artifacts/xgb_best.joblib \
  -v "${REPO_DIR}/artifacts:/app/artifacts:ro" "$IMAGE" >/dev/null
wait_for_health true

post "${BASE}/predict" --data "@${PAYLOAD}"
[[ "$RESP_CODE" == "200" ]] || fail "/predict with valid payload expected 200, got $RESP_CODE (body: $RESP_BODY)"
echo "  /predict (valid) -> 200; body: $RESP_BODY"

post "${BASE}/predict" --data '{"timestamps":[0,0],"values":{}}'
[[ "$RESP_CODE" == "400" ]] || fail "/predict with invalid payload expected 400, got $RESP_CODE (body: $RESP_BODY)"
echo "  /predict (invalid) -> 400"

docker rm -f "$MODEL_CONTAINER" >/dev/null

# --- 2. WITHOUT the model --------------------------------------------------
echo "==> Running WITHOUT model (no mount)"
docker run -d --name "$NOMODEL_CONTAINER" -p "${PORT}:8000" "$IMAGE" >/dev/null
wait_for_health false

post "${BASE}/predict" --data "@${PAYLOAD}"
[[ "$RESP_CODE" == "503" ]] || fail "/predict without a model expected 503, got $RESP_CODE (body: $RESP_BODY)"
echo "  /predict (no model) -> 503"

echo "==> SMOKE OK: all Docker service checks passed."
