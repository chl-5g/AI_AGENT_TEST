"""
config.py — 独立配置模块：项目根、.env 加载、运行时参数（与业务代码解耦）
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_THIS_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_FILE_DIR.parent if _THIS_FILE_DIR.name == "script" else _THIS_FILE_DIR


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
        data_dir=root / "data",
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        chroma_add_batch_size=batch,
        rag_top_k=max(1, _env_int("RAG_TOP_K", 5)),
        rag_debug=_env_bool("RAG_DEBUG", False),
        cors_origins=(os.environ.get("CORS_ORIGINS") or "*").strip() or "*",
        ingest_io_workers=max(1, _env_int("INGEST_IO_WORKERS", 4)),
    )
