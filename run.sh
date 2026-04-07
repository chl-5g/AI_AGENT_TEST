#!/usr/bin/env bash
# 一键启动 Web 服务（适用于 Linux / macOS / WSL / Git Bash）
# 用法：chmod +x run.sh && ./run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"

# 优先 python3，没有则用 python
PY="${PYTHON:-python3}"
if ! command -v "$PY" &>/dev/null; then
  PY=python
fi
if ! command -v "$PY" &>/dev/null; then
  echo "未找到 python3/python，请先安装 Python 3.10+" >&2
  exit 1
fi

# 虚拟环境（只在仓库根 .venv，不污染系统）
if [[ ! -d .venv ]]; then
  echo ">>> 创建虚拟环境: $ROOT/.venv"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo ">>> 安装依赖（requirements.txt）…"
pip install -r requirements.txt -q

# 没有 .env 时从模板复制（仍需你填入真实 Key，否则问答会报错）
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo ">>> 已生成 .env（来自 .env.example），请用编辑器打开仓库根目录的 .env 填入 Key 后保存，再重新执行 ./run.sh"
    exit 0
  else
    echo "缺少 .env 且没有 .env.example，无法继续" >&2
    exit 1
  fi
fi

# 减少中文乱码（终端支持 UTF-8 时）
export PYTHONUTF8="${PYTHONUTF8:-1}"

cd "$ROOT/script"
echo ">>> 服务地址: http://127.0.0.1:8000  （Ctrl+C 停止）"
exec uvicorn main:app --reload --host 127.0.0.1 --port 8000
