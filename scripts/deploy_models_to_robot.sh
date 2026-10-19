#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOT_HOST="${ROBOT_HOST:-nerv}"
ROBOT_PROJECT="${ROBOT_PROJECT:-/home/nerv/Voice}"
LOCAL_MODELS="${PROJECT_ROOT}/data/models"
REMOTE_MODELS="${ROBOT_PROJECT}/data/models"

if [[ ! -d "${LOCAL_MODELS}" ]]; then
    echo "Local model directory does not exist: ${LOCAL_MODELS}" >&2
    exit 1
fi

ssh "${ROBOT_HOST}" "mkdir -p '${REMOTE_MODELS}'"
rsync -a "${LOCAL_MODELS}/" "${ROBOT_HOST}:${REMOTE_MODELS}/"

echo "Models synchronized to ${ROBOT_HOST}:${REMOTE_MODELS}"
