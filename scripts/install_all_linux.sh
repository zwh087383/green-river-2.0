#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$project_root/.venv-linux"
python_bin="${RAG_SYSTEM_PYTHON:-python3}"
ros_distro="${ROS_DISTRO:-humble}"

cd "$project_root"

if command -v apt-get >/dev/null 2>&1 && ! command -v tesseract >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
        apt_prefix=()
    elif command -v sudo >/dev/null 2>&1; then
        apt_prefix=(sudo)
    else
        echo "缺少 Tesseract，且当前用户没有 sudo。" >&2
        echo "请由管理员安装：tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim" >&2
        exit 1
    fi
    "${apt_prefix[@]}" apt-get update
    "${apt_prefix[@]}" apt-get install -y \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim
fi

if [[ ! -x "$venv_dir/bin/python" ]]; then
    venv_options=()
    if [[ -f "/opt/ros/${ros_distro}/setup.bash" ]]; then
        venv_options+=(--system-site-packages)
    fi
    "$python_bin" -m venv "${venv_options[@]}" "$venv_dir"
fi

"$venv_dir/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv_dir/bin/python" -m pip install -e '.[test,web,indexing,pdf,ocr,html,heavy,voice,embedding-local]'
# Keep compatibility with optional ROS/legacy packages if they already exist
# in an environment that is being upgraded in place.
"$venv_dir/bin/python" -m pip install 'setuptools>=68,<80' 'wrapt<2'

"$venv_dir/bin/python" -m pip check
if [[ -f "/opt/ros/${ros_distro}/setup.bash" ]]; then
    set +u
    source "/opt/ros/${ros_distro}/setup.bash"
    set -u
    "$venv_dir/bin/python" -c 'import rclpy'
fi
PYTHONPATH=src "$venv_dir/bin/python" -m rag_system.module_test doctor || true
