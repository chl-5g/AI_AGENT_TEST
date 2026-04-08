"""
deploy.py — 一键部署入口（包方式/源码方式均可用）
"""

from __future__ import annotations

import argparse
import os

import uvicorn

try:
    from .config import PROJECT_ROOT
except ImportError:  # 兼容直接在 script/ 下运行
    from config import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 RAG Web 服务")
    parser.add_argument("--host", default=os.environ.get("RAG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RAG_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args(argv)

    # 允许用户从任意目录启动：显式告诉应用项目根（含 .env / data / static）
    os.environ.setdefault("AI_AGENT_HOME", str(PROJECT_ROOT))
    uvicorn.run("script.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
