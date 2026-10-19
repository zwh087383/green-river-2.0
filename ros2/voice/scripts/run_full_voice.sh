#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
FULL_PYTHON="${RAG_SYSTEM_PYTHON:-${PROJECT_ROOT}/.venv-linux/bin/python}"

if [[ ! -x "${FULL_PYTHON}" ]]; then
    echo "Full Linux environment is missing: ${FULL_PYTHON}" >&2
    echo "Run ${PROJECT_ROOT}/scripts/install_all_linux.sh first." >&2
    exit 1
fi

export RAG_SYSTEM_PYTHON="${FULL_PYTHON}"
exec "${SCRIPT_DIR}/run_voice.sh" enable_local_audio:=true "$@"
