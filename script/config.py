"""
config.py — 独立配置模块：项目根、.env 加载、运行时参数（与业务代码解耦）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

def _detect_project_root() -> Path:
    """
    计算项目根目录，支持源码运行与安装后包运行两种形态。

    优先级：
    1) AI_AGENT_HOME 环境变量（显式指定）
    2) 当前文件在仓库 script/ 下，且上一级有 data/ 或 static/（源码运行）
    3) 当前工作目录（安装后从项目根执行命令）
    4) 当前文件目录（兜底）
    """
    env_home = (os.environ.get("AI_AGENT_HOME") or "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()

    this_dir = Path(__file__).resolve().parent
    parent = this_dir.parent
    if this_dir.name == "script" and ((parent / "data").exists() or (parent / "static").exists()):
        return parent

    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() or (cwd / "static").exists():
        return cwd
    return this_dir


PROJECT_ROOT = _detect_project_root()


def load_project_env() -> None:
    """从 PROJECT_ROOT/.env 加载环境变量（不覆盖已存在于 os.environ 的键）。"""
    load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_data_dir(root: Path) -> Path:
    """
    从 .env 读取知识库目录（RAG_KB_DIR），默认 data/。

    - 相对路径：相对 PROJECT_ROOT 解析
    - 绝对路径：按绝对路径使用
    """
    raw = (os.environ.get("RAG_KB_DIR") or "data").strip()
    if not raw:
        raw = "data"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


@dataclass(frozen=True)
class Settings:
    """从环境变量读取的应用配置（每次 get_settings() 重新解析，便于开发时改 .env）。"""

    chroma_path: str
    collection_name: str
    data_dir: Path
    chunk_chars: int
    chunk_overlap: int
    chroma_add_batch_size: int
    rag_top_k: int
    rag_debug: bool
    cors_origins: str
    # 异步入库时 to_thread 并发度（对 Chroma add 仍建议保持 1，避免客户端线程安全问题）
    ingest_io_workers: int

    @property
    def chroma_add_batch(self) -> int:
        return self.chroma_add_batch_size


def get_settings() -> Settings:
    load_project_env()
    root = PROJECT_ROOT
    chunk_chars = max(50, _env_int("CHUNK_CHARS", 400))
    chunk_overlap = max(0, min(chunk_chars - 1, _env_int("CHUNK_OVERLAP", 50)))
    batch = _env_int("CHROMA_ADD_BATCH_SIZE", 256)
    return Settings(
        chroma_path=str(root / "chroma_data"),
        collection_name=(os.environ.get("CHROMA_COLLECTION_NAME") or "support_docs").strip()
        or "support_docs",
        data_dir=_resolve_data_dir(root),
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        chroma_add_batch_size=batch,
        rag_top_k=max(1, _env_int("RAG_TOP_K", 5)),
        rag_debug=_env_bool("RAG_DEBUG", False),
        cors_origins=(os.environ.get("CORS_ORIGINS") or "*").strip() or "*",
        ingest_io_workers=max(1, _env_int("INGEST_IO_WORKERS", 4)),
    )
