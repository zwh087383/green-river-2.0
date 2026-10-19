#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROBOT_HOST="${ROBOT_HOST:-nerv}"
ROBOT_PROJECT="${ROBOT_PROJECT:-/home/nerv/Voice}"

rsync -a --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.venv-*/' \
    --exclude 'data/' \
    --exclude 'build/' \
    --exclude 'install/' \
    --exclude 'log/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.pytest-tmp*/' \
    "${PROJECT_ROOT}/" "${ROBOT_HOST}:${ROBOT_PROJECT}/"

echo "Project synchronized to ${ROBOT_HOST}:${ROBOT_PROJECT}"
echo "Install and build on the robot with:"
echo "  ssh ${ROBOT_HOST} 'bash ${ROBOT_PROJECT}/ros2/voice/scripts/install_linux.sh'"
