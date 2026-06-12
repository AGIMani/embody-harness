#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DISPLAY_ARG="${DISPLAY:-:0}"
MODEL_PATH="${MODEL_PATH:-/home/whf/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.5}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/vr_stack}"
PYTHON_BIN="${PYTHON_BIN:-}"
WEB_MODE="${WEB_MODE:-image}"
START_CLOUDXR=1
START_WEB=1
START_VOICE=1
START_VR_OUTPUT=1
CHECK_ONLY=0
REEXEC_DOCKER_GROUP="${HARNESS_VR_PREREQS_REEXEC_DOCKER:-0}"
ORIGINAL_ARGS=("$@")
WITH_SCENE=0
SCENE_BACKEND="${SCENE_BACKEND:-gpu}"
SCENE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/run_add_scene_vr_prereqs.sh [options]

Starts the add_scene VR prerequisites in one terminal:
  1. python -m isaacteleop.cloudxr --accept-eula
  2. scripts/run_cloudxr_web_client.sh
  3. scripts/run_quest_voice_command_bridge.sh
  4. scripts/run_add_scene_vr_output.sh

Options:
  --display :N[.S]       X11 display to capture for Genesis/VR output (default: $DISPLAY or :0)
  --model-path PATH      Vosk model path
  --min-confidence N     Voice command min confidence (default: 0.5)
  --log-dir PATH         Log directory (default: logs/vr_stack)
  --python PATH          Python executable (default: PYTHON_BIN, genesis env python, then python3)
  --web-mode MODE        Pass --mode MODE to run_cloudxr_web_client.sh (default: image)
  --skip-cloudxr         Do not start CloudXR runtime
  --skip-web             Do not start CloudXR web client
  --skip-voice           Do not start Quest voice bridge
  --skip-vr-output       Do not start VR sim-screen/XR output
  --with-scene           Also start add_scene_glb.py in this terminal
  --scene-backend cpu|gpu
                         Backend passed to add_scene_glb.py when --with-scene is used (default: gpu)
  --check-only           Check basic prerequisites and exit
  --                     Extra arguments passed to add_scene_glb.py with --with-scene
  -h, --help             Show this help

Without --with-scene, after this script is running, start the scene in another terminal:
  python add_scene_glb.py --backend gpu --enable-vr-teleop
EOF
}

log() { printf '[vr-prereqs] %s\n' "$*"; }
ok() { log "ok: $*"; }
warn() { log "warn: $*" >&2; }
err() { log "error: $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --display) DISPLAY_ARG="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --min-confidence) MIN_CONFIDENCE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --web-mode) WEB_MODE="$2"; shift 2 ;;
        --skip-cloudxr) START_CLOUDXR=0; shift ;;
        --skip-web) START_WEB=0; shift ;;
        --skip-voice) START_VOICE=0; shift ;;
        --skip-vr-output) START_VR_OUTPUT=0; shift ;;
        --with-scene) WITH_SCENE=1; shift ;;
        --scene-backend) SCENE_BACKEND="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --) shift; SCENE_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1"; usage >&2; exit 2 ;;
    esac
done

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

if [[ -z "${PYTHON_BIN}" && -x "/home/whf/anaconda3/envs/genesis/bin/python3" ]]; then
    PYTHON_BIN="/home/whf/anaconda3/envs/genesis/bin/python3"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || true)"
fi

failures=0
check_ok() { ok "$*"; }
check_warn() { warn "$*"; }
check_error() { err "$*"; failures=$((failures + 1)); }

shell_quote() {
    printf '%q' "$1"
}

reexec_with_docker_group_if_possible() {
    if [[ "${REEXEC_DOCKER_GROUP}" == "1" ]]; then
        return 1
    fi
    if ! command -v sg >/dev/null 2>&1; then
        return 1
    fi
    if ! getent group docker | grep -Eq "(^|[:,])${USER}([,]|$)"; then
        return 1
    fi
    if ! sg docker -c "docker ps >/dev/null 2>&1"; then
        return 1
    fi

    local cmd
    cmd="cd $(shell_quote "${REPO_ROOT}") && HARNESS_VR_PREREQS_REEXEC_DOCKER=1 exec $(shell_quote "$0")"
    for arg in "$@"; do
        cmd+=" $(shell_quote "${arg}")"
    done
    warn "current shell has not refreshed docker group; re-executing this script with 'sg docker'"
    exec sg docker -c "${cmd}"
}

if [[ -n "${PYTHON_BIN}" && -x "${PYTHON_BIN}" ]]; then
    check_ok "python=${PYTHON_BIN}"
else
    check_error "missing python executable; pass --python /path/to/python"
fi

if [[ ! -f "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    check_error "voice model path not found: ${MODEL_PATH}"
else
    check_ok "voice model=${MODEL_PATH}"
fi

if [[ -z "${DISPLAY_ARG}" ]]; then
    check_error "display is empty; pass --display :0"
else
    check_ok "capture display=${DISPLAY_ARG}"
fi

if command -v ffmpeg >/dev/null 2>&1; then
    check_ok "ffmpeg=$(command -v ffmpeg)"
else
    check_error "missing ffmpeg"
fi

if command -v docker >/dev/null 2>&1; then
    check_ok "docker=$(command -v docker)"
    if docker ps >/dev/null 2>&1; then
        check_ok "docker daemon is accessible"
    else
        if reexec_with_docker_group_if_possible "${ORIGINAL_ARGS[@]}"; then
            :
        else
            check_error "docker daemon is not accessible by current shell"
            check_warn "if you just joined the docker group, run: newgrp docker"
            check_warn "or fully log out and log back in, then retry this script"
        fi
    fi
else
    check_error "missing docker"
fi

if [[ -e /dev/video44 ]]; then
    check_ok "sim screen device=/dev/video44"
else
    check_error "sim screen device missing: /dev/video44"
    check_warn "create once with: sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2"
fi

if [[ ! -x "${SCRIPT_DIR}/run_cloudxr_web_client.sh" ]]; then
    check_error "missing executable: scripts/run_cloudxr_web_client.sh"
fi
if [[ ! -x "${SCRIPT_DIR}/run_quest_voice_command_bridge.sh" ]]; then
    check_error "missing executable: scripts/run_quest_voice_command_bridge.sh"
fi
if [[ ! -x "${SCRIPT_DIR}/run_add_scene_vr_output.sh" ]]; then
    check_error "missing executable: scripts/run_add_scene_vr_output.sh"
fi
if [[ "${WITH_SCENE}" -eq 1 && ! -f "${REPO_ROOT}/add_scene_glb.py" ]]; then
    check_error "missing add_scene_glb.py"
fi
if [[ "${SCENE_BACKEND}" != "cpu" && "${SCENE_BACKEND}" != "gpu" ]]; then
    check_error "--scene-backend must be cpu or gpu"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    if [[ "${failures}" -eq 0 ]]; then
        check_ok "preflight passed"
        exit 0
    fi
    check_error "preflight failed with ${failures} problem(s)"
    exit 1
fi

if [[ "${failures}" -ne 0 ]]; then
    err "fix preflight errors before starting VR prerequisites"
    exit 1
fi

PIDS=()
NAMES=()

start_bg() {
    local name="$1"
    shift
    local log_path="${LOG_DIR}/${name}.log"
    : > "${log_path}"
    log "starting ${name}; log=${log_path}"
    "$@" >"${log_path}" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    NAMES+=("${name}")
    ok "${name} pid=${pid}"
}

stop_all() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        warn "stopping background services..."
        for idx in "${!PIDS[@]}"; do
            local pid="${PIDS[$idx]}"
            local name="${NAMES[$idx]}"
            if kill -0 "${pid}" >/dev/null 2>&1; then
                warn "stopping ${name} pid=${pid}"
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done
        for pid in "${PIDS[@]}"; do
            wait "${pid}" >/dev/null 2>&1 || true
        done
    fi
    exit "${exit_code}"
}
trap stop_all EXIT INT TERM

wait_for_socket() {
    local socket_path="$1"
    local label="$2"
    local timeout_s="$3"
    local started
    started="$(date +%s)"
    while true; do
        if [[ -S "${socket_path}" ]]; then
            ok "${label} is ready: ${socket_path}"
            return 0
        fi
        if (( $(date +%s) - started >= timeout_s )); then
            err "${label} did not become ready within ${timeout_s}s: ${socket_path}"
            return 1
        fi
        sleep 1
    done
}

wait_for_port() {
    local port="$1"
    local label="$2"
    local timeout_s="$3"
    local started
    started="$(date +%s)"
    while true; do
        if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
            ok "${label} is listening on port ${port}"
            return 0
        fi
        if (( $(date +%s) - started >= timeout_s )); then
            err "${label} did not listen on port ${port} within ${timeout_s}s"
            return 1
        fi
        sleep 1
    done
}

if [[ "${START_CLOUDXR}" -eq 1 ]]; then
    start_bg "cloudxr_runtime" "${PYTHON_BIN}" -m isaacteleop.cloudxr --accept-eula
    wait_for_socket "${NV_CXR_RUNTIME_DIR:-${HOME}/.cloudxr/run}/ipc_cloudxr" "CloudXR runtime" 90
else
    warn "skipping CloudXR runtime"
fi

if [[ "${START_WEB}" -eq 1 ]]; then
    start_bg "cloudxr_web_client" "${SCRIPT_DIR}/run_cloudxr_web_client.sh" --mode "${WEB_MODE}"
    wait_for_port 8443 "CloudXR web client" 120
else
    warn "skipping CloudXR web client"
fi

if [[ "${START_VOICE}" -eq 1 ]]; then
    start_bg "quest_voice_bridge" \
        "${SCRIPT_DIR}/run_quest_voice_command_bridge.sh" \
        --model-path "${MODEL_PATH}" \
        --no-tls \
        --min-confidence "${MIN_CONFIDENCE}"
    wait_for_port 8766 "Quest voice bridge" 60
else
    warn "skipping Quest voice bridge"
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-127.0.0.1}"
ok "Quest web page: https://${HOST_IP}:8443/"
ok "CloudXR certificate page if needed: https://${HOST_IP}:48322/"
if [[ "${WITH_SCENE}" -eq 1 ]]; then
    ok "scene will start in this terminal"
else
    ok "Next terminal for scene: python add_scene_glb.py --backend gpu --enable-vr-teleop"
fi

if [[ "${WITH_SCENE}" -eq 1 ]]; then
    if [[ "${START_VR_OUTPUT}" -eq 1 ]]; then
        start_bg "add_scene_vr_output" "${SCRIPT_DIR}/run_add_scene_vr_output.sh" --display "${DISPLAY_ARG}"
        sleep 3
    else
        warn "--with-scene used with --skip-vr-output; add_scene_glb.py may wait for overlay hand samples"
    fi
    log "starting add_scene_glb.py in foreground; press Ctrl+C here to stop the whole stack"
    "${PYTHON_BIN}" "${REPO_ROOT}/add_scene_glb.py" \
        --backend "${SCENE_BACKEND}" \
        --enable-vr-teleop \
        --vr-input-source overlay-log \
        "${SCENE_ARGS[@]}"
elif [[ "${START_VR_OUTPUT}" -eq 1 ]]; then
    log "starting VR output in foreground; press Ctrl+C here to stop the whole stack"
    "${SCRIPT_DIR}/run_add_scene_vr_output.sh" --display "${DISPLAY_ARG}"
else
    warn "skipping VR output; background services are running, press Ctrl+C to stop"
    while true; do
        sleep 3600
    done
fi
