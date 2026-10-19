#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_root/data/runtime"
pid_file="$runtime_dir/web.pid"
log_file="$runtime_dir/web.log"
host="${RAG_SYSTEM_HOST:-127.0.0.1}"
port="${RAG_SYSTEM_PORT:-8000}"
probe_host="$host"
if [[ "$probe_host" == "0.0.0.0" || "$probe_host" == "::" ]]; then
    probe_host="127.0.0.1"
fi

mkdir -p "$runtime_dir"

if [[ -f "$pid_file" ]]; then
    existing_pid="$(tr -cd '0-9' < "$pid_file")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "服务已运行：http://$host:$port（PID $existing_pid）"
        exit 0
    fi
fi

cd "$project_root"
RAG_SYSTEM_HOST="$host" RAG_SYSTEM_PORT="$port" \
    nohup setsid "$project_root/scripts/run_linux.sh" web >"$log_file" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"

for _attempt in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "服务启动失败：" >&2
        sed -n '1,160p' "$log_file" >&2
        exit 1
    fi
    if "$project_root/.venv-linux/bin/python" - "$probe_host" "$port" <<'PY' >/dev/null 2>&1
import json
import sys
from urllib.request import urlopen

with urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/api/health", timeout=1) as response:
    payload = json.load(response)
    if response.status != 200 or payload.get("ok") is not True:
        raise SystemExit(1)
PY
    then
        # A stale service may already own the port. Only report success when
        # the process started by this script is still alive after the probe.
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "服务启动失败，端口可能已被其他进程占用：" >&2
            sed -n '1,160p' "$log_file" >&2
            exit 1
        fi
        sleep 0.2
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "服务启动失败，端口可能已被其他进程占用：" >&2
            sed -n '1,160p' "$log_file" >&2
            exit 1
        fi
        echo "服务已启动：http://$host:$port（PID $server_pid）"
        echo "日志：$log_file"
        exit 0
    fi
    sleep 0.25
done

echo "服务仍在启动，请查看日志：$log_file" >&2
exit 1
