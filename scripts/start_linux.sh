#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${RAG_DATA_DIR:-$project_root/data}/runtime"
pid_file="$runtime_dir/web.pid"
log_file="$runtime_dir/web.log"
host="${RAG_SYSTEM_HOST:-127.0.0.1}"
port="${RAG_SYSTEM_PORT:-8001}"
probe_host="$host"
if [[ "$probe_host" == "0.0.0.0" || "$probe_host" == "::" ]]; then
    probe_host="127.0.0.1"
fi

mkdir -p "$runtime_dir"
# Serialize even standalone web starts; the detached server must not inherit it.
exec 9>"$runtime_dir/web-start.lock"
flock -x 9

if [[ -f "$pid_file" ]]; then
    existing_pid="$(tr -cd '0-9' < "$pid_file")"
    if [[ -n "$existing_pid" ]] && /usr/bin/python3 - "$existing_pid" "$project_root" <<'PY_CHECK'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / 'scripts'))
from robot_runtime import read_process, role
process = read_process(int(sys.argv[1]))
raise SystemExit(0 if process and process.state != 'Z' and role(process, (Path(sys.argv[2]),)) == 'web' else 1)
PY_CHECK
    then
        echo "服务已运行：http://$host:$port（PID $existing_pid）"
        exit 0
    fi
fi

cd "$project_root"
bash "$project_root/scripts/start_embedding_linux.sh"
RAG_SYSTEM_HOST="$host" RAG_SYSTEM_PORT="$port" \
    nohup setsid "$project_root/scripts/run_linux.sh" web >"$log_file" 2>&1 </dev/null 9>&- &
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
