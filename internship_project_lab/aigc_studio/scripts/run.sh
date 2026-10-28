#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SOURCE_DIR/scripts/env.sh"
cd "$SOURCE_DIR"
COMMAND="${1:-serve}"
shift || true
case "$COMMAND" in
  build) npm run build --prefix frontend ;;
  format) npm run format --prefix frontend ;;
  test)
    "$AIGC_DATA_ROOT/venv/bin/python" -B -m pytest -q -p no:cacheprovider --basetemp="$AIGC_DATA_ROOT/checks/pytest" tests "$@"
    ;;
  browser-install)
    frontend/node_modules/.bin/playwright install --no-shell chromium
    ;;
  browser-test)
    node frontend/check-ui.mjs "$@"
    ;;
  preview)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m uvicorn tests.preview:factory --factory --host 127.0.0.1 --port 18799
    ;;
  serve)
    mkdir -p "$AIGC_DATA_ROOT/run"
    exec 9>"$AIGC_DATA_ROOT/run/api.lock"
    flock -n 9 || { printf 'Studio API is already running for this data directory.\n'; exit 1; }
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m uvicorn backend.app:factory --factory --host 127.0.0.1 --port "${AIGC_PORT:-18790}" --workers 1 "$@"
    ;;
  models)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B scripts/prepare_models.py "$@"
    ;;
  download-progress)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B scripts/download_progress.py "$@"
    ;;
  worker)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B scripts/worker.py "$@"
    ;;
  diagnose-comfy)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m scripts.check_comfy "$@"
    ;;
  eval-faults)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m evaluation.faults --output "$AIGC_DATA_ROOT/evaluations/acceptance-v1/faults" "$@"
    ;;
  evaluate)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m evaluation.run "$@"
    ;;
  eval-check)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m evaluation.check_harness
    ;;
  eval-status)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m evaluation.run --stage status "$@"
    ;;
  eval-review)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B -m http.server 18813 --bind 127.0.0.1 --directory "$AIGC_DATA_ROOT/evaluations/acceptance-v1"
    ;;
  stop-checks)
    exec "$AIGC_DATA_ROOT/venv/bin/python" -B scripts/stop_checks.py
    ;;
  start|start-download|start-worker|start-evaluation)
    mkdir -p "$AIGC_DATA_ROOT/logs"
    SERVICE_ARGS=()
    if [[ "$COMMAND" == start ]]; then
      UNIT=aigc-studio-api
      SUBCOMMAND=(serve)
    elif [[ "$COMMAND" == start-download ]]; then
      UNIT=aigc-studio-download
      SUBCOMMAND=(models --model all)
      SERVICE_ARGS=(--property=Restart=on-failure --property=RestartSec=60 --property=RestartPreventExitStatus=2 --property=StartLimitIntervalSec=0)
    elif [[ "$COMMAND" == start-worker ]]; then
      UNIT=aigc-studio-worker
      SUBCOMMAND=(worker --gpu auto --port 18791 --wait)
    else
      UNIT=aigc-studio-evaluation
      SUBCOMMAND=(evaluate --stage full "$@")
    fi
    ENV_ARGS=("--setenv=AIGC_DATA_ROOT=$AIGC_DATA_ROOT")
    for ENV_NAME in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy HF_ENDPOINT; do
      if [[ -v "$ENV_NAME" ]]; then ENV_ARGS+=("--setenv=$ENV_NAME=${!ENV_NAME}"); fi
    done
    systemd-run --user --collect --unit="$UNIT" --working-directory="$SOURCE_DIR" \
      --property=Nice=10 --property=IOSchedulingClass=idle \
      --property="StandardOutput=append:$AIGC_DATA_ROOT/logs/$UNIT.log" \
      --property="StandardError=append:$AIGC_DATA_ROOT/logs/$UNIT.log" \
      "${SERVICE_ARGS[@]}" "${ENV_ARGS[@]}" /usr/bin/bash "$SOURCE_DIR/scripts/run.sh" "${SUBCOMMAND[@]}"
    ;;
  status)
    systemctl --user --no-pager --full status aigc-studio-api.service aigc-studio-download.service aigc-studio-worker.service || true
    curl --noproxy '*' --connect-timeout 3 --max-time 5 --fail --silent --show-error http://127.0.0.1:18790/api/health
    ;;
  stop)
    systemctl --user stop aigc-studio-api.service
    ;;
  stop-worker)
    systemctl --user stop aigc-studio-worker.service
    ;;
  stop-download)
    systemctl --user stop aigc-studio-download.service
    ;;
  stop-evaluation)
    systemctl --user stop aigc-studio-evaluation.service
    ;;
  *) printf 'Usage: bash scripts/run.sh {build|test|serve|start|stop|status|models|start-download|download-progress|worker|browser-install|browser-test|diagnose-comfy|eval-faults|evaluate|start-evaluation|eval-status|eval-review|stop-evaluation}\n'; exit 2 ;;
esac
