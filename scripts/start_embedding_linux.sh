#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_root/.venv-linux/bin/python"
runtime_dir="${RAG_DATA_DIR:-$project_root/data}/runtime"
model_dir="${EMBEDDINGGEMMA_MODEL_DIR:-$project_root/data/models/embeddinggemma-2}"
mkdir -p "$runtime_dir"
exec 9>"$runtime_dir/embedding-start.lock"
flock -x 9
cd "$project_root"

check_service() {
    "$python_bin" - <<'PY' >/dev/null 2>&1
import json
from urllib.request import urlopen
with urlopen('http://127.0.0.1:8765/health', timeout=2) as response:
    payload = json.load(response)
    assert payload.get('status') == 'ready'
    assert payload.get('model') == 'google/embeddinggemma-2'
    assert payload.get('dimensions') == 768
PY
}

if "$python_bin" - <<'PY'
from rag_system.settings import SettingsStore
import os
from pathlib import Path
url = SettingsStore(Path(os.environ.get('RAG_DATA_DIR', 'data')) / 'settings.json').load().embedding_url
raise SystemExit(0 if url.rstrip('/') not in {'http://127.0.0.1:8765/v1/embeddings', 'http://localhost:8765/v1/embeddings'} else 1)
PY
then
    echo "使用自定义向量接口，跳过本地模型启动。"
    exit 0
fi
if check_service; then
    echo "EmbeddingGemma 2 已就绪：http://127.0.0.1:8765"
    exit 0
fi
if "$python_bin" - <<'PY'
import socket
with socket.socket() as sock:
    raise SystemExit(0 if sock.connect_ex(('127.0.0.1', 8765)) == 0 else 1)
PY
then
    echo "8765 端口被其他服务占用，未启动向量服务。" >&2
    exit 1
fi
if [[ ! -f "$model_dir/model.safetensors" ]]; then
    echo "缺少模型，请先运行 .venv-linux/bin/python -m rag_system.indexing.embedding_server --download" >&2
    exit 1
fi
nohup setsid "$python_bin" -m rag_system.indexing.embedding_server --model-dir "$model_dir" \
    >"$runtime_dir/embeddinggemma2.log" 2>&1 </dev/null 9>&- &
model_pid=$!
printf '%s\n' "$model_pid" > "$runtime_dir/embeddinggemma2.pid"
for _attempt in $(seq 1 120); do
    if ! kill -0 "$model_pid" 2>/dev/null; then
        tail -35 "$runtime_dir/embeddinggemma2.log" >&2
        exit 1
    fi
    if check_service; then
        echo "EmbeddingGemma 2 已启动：http://127.0.0.1:8765（PID $model_pid）"
        exit 0
    fi
    sleep 0.5
done
echo "向量服务仍在加载，请查看 $runtime_dir/embeddinggemma2.log" >&2
exit 1
