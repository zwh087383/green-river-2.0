#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_python="$project_root/.venv-linux/bin/python"

if [[ -n "${RAG_SYSTEM_PYTHON:-}" ]]; then
    python_bin="$RAG_SYSTEM_PYTHON"
elif [[ -x "$local_python" ]]; then
    python_bin="$local_python"
else
    echo "找不到本项目的 Linux Python 环境：$local_python" >&2
    echo "请先创建 .venv-linux 并安装项目全部可选依赖。" >&2
    exit 1
fi

python_path="$project_root/src"
export PYTHONPATH="$python_path"
export RAG_SYSTEM_PORT="${RAG_SYSTEM_PORT:-8001}"
cd "$project_root"

case "${1:-web}" in
    web)
        "$python_bin" -c 'import fastapi, uvicorn, multipart, chromadb, docling, unstructured, edge_tts, aiohttp'
        exec "$python_bin" -m rag_system.web
        ;;
    web-strict)
        "$python_bin" -c 'import fastapi, uvicorn, multipart, chromadb, docling, unstructured, edge_tts, aiohttp'
        export RAG_SYSTEM_STRICT_NO_FALLBACK=1
        exec "$python_bin" -m rag_system.web
        ;;
    doctor)
        exec "$python_bin" -m rag_system.module_test doctor
        ;;
    test)
        export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
        exec "$python_bin" -m pytest -q -p no:cacheprovider --basetemp=/tmp/rag-system-pytest
        ;;
    probe)
        shift
        exec "$python_bin" -m rag_system.time_probe "$@"
        ;;
    *)
        echo "用法: $0 [web|web-strict|doctor|test|probe]" >&2
        exit 2
        ;;
esac
