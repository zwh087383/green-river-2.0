#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_file="$project_root/data/runtime/web.pid"

if [[ ! -f "$pid_file" ]]; then
    echo "没有记录正在运行的服务。"
    exit 0
fi

server_pid="$(tr -cd '0-9' < "$pid_file")"
if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    for _attempt in $(seq 1 20); do
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 0.25
    done
fi
rm -f "$pid_file"
echo "服务已停止。"
