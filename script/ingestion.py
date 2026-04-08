"""
ingestion.py — 数据入库与「嵌入客户端」构造（供 Chroma 与 bot 共用）

职责划分：
    1. load_project_env / PROJECT_ROOT：见 config.py（此处 re-export 保持旧 import 兼容）。
    2. openai_embedding_function()：按环境变量选择嵌入网关。
    3. 文本切片：chunking.split_text；PDF/Docx 由 doc_parse 按页或全文抽取并切片。
    4. initialize() / initialize_async()：收集 data/ 下 txt/md/pdf/docx，删建集合，分批 add。

重要说明（避免踩坑）：
    - DeepSeek 官方对话 API 通常不提供 OpenAI 兼容的 /v1/embeddings；若仅配 DeepSeek
      而不配 OpenRouter / OpenAI 嵌入 Key，initialize 会因无法向量而失败。推荐混搭见 .env.example。
    - normalize_api_base() 被 bot._openai_chat_client 复用，保持与嵌入侧 URL 处理一致。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

try:
    from .config import PROJECT_ROOT, get_settings, load_project_env
    from .doc_parse import parse_document
except ImportError:  # 兼容直接在 script/ 下运行
    from config import PROJECT_ROOT, get_settings, load_project_env
    from doc_parse import parse_document

# ---------------------------------------------------------------------------
# 与历史代码兼容：bot / main 可能仍从 ingestion 引用下列名
# ---------------------------------------------------------------------------
CHROMA_PATH = str(PROJECT_ROOT / "chroma_data")
COLLECTION_NAME = "support_docs"
CHUNK_CHARS = 400
CHUNK_OVERLAP = 50
CHROMA_ADD_BATCH_SIZE = 256


def normalize_api_base(url: str | None) -> str | None:
    """
    规范化 OpenAI 兼容网关的 Base URL。

    - None / 空白 → None（调用方决定是否使用 SDK 默认官方地址）。
    - 非空 → strip 后去掉末尾 '/'，避免部分网关对「/v1」与「/v1/」处理不一致。

    与 bot 模块共享此函数，避免两处 rstrip/strip 逻辑漂移。
    """
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    return u.rstrip("/")


def openai_embedding_function():
    """
    构造 Chroma 使用的 OpenAIEmbeddingFunction（底层走 OpenAI SDK 的 embeddings 接口）。

    决策顺序（简图）：
        A. 配置了 OPENROUTER_API_KEY，或 OPENAI_EMBEDDING_BASE_URL 含 openrouter
           → 嵌入走 OpenRouter（默认 https://openrouter.ai/api/v1），模型名常用
           OPENROUTER_EMBEDDING_MODEL / OPENAI_EMBEDDING_MODEL（如带厂商前缀）。
        B. 显式设置了 OPENAI_EMBEDDING_BASE_URL（非 OpenRouter）
           → 使用该 base + OPENAI_EMBEDDING_API_KEY 或 OPENAI_API_KEY。
        C. 对话 URL 含 deepseek 且未走 A/B
           → 嵌入默认改走 https://api.openai.com/v1，需 OPENAI_EMBEDDING_API_KEY
           （混搭：DeepSeek 聊天 + OpenAI 嵌入），或你改用 OpenRouter 做嵌入。
        D. 其它
           → OPENAI_BASE_URL + OPENAI_EMBEDDING_API_KEY 或 OPENAI_API_KEY（同一网关既聊天又嵌入时）。

    可选请求头（OpenRouter 文档建议）：OPENROUTER_HTTP_REFERER、OPENROUTER_X_TITLE。

    若 api_base 仍指向 deepseek 且未配合理嵌入模型，会 RuntimeError 提示改用 OpenRouter 等。
    """
    load_project_env()
    emb_key = (os.environ.get("OPENAI_EMBEDDING_API_KEY") or "").strip()
    openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    primary_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    chat_url = (os.environ.get("OPENAI_BASE_URL") or "").lower()
    is_deepseek_chat = "deepseek" in chat_url
    emb_base_raw = (os.environ.get("OPENAI_EMBEDDING_BASE_URL") or "").strip()

    use_openrouter = bool(openrouter_key) or ("openrouter" in emb_base_raw.lower())
    if use_openrouter:
        if emb_base_raw and "openrouter" in emb_base_raw.lower():
            api_base = normalize_api_base(emb_base_raw)
        else:
            or_base = (os.environ.get("OPENROUTER_EMBEDDING_BASE_URL") or "").strip()
            api_base = normalize_api_base(or_base) if or_base else "https://openrouter.ai/api/v1"
        api_key = openrouter_key or emb_key
        if not api_key:
            raise RuntimeError(
                "已启用 OpenRouter 嵌入，但未配置 Key。请设置 OPENROUTER_API_KEY，"
                "或把 OpenRouter 的 Key 写在 OPENAI_EMBEDDING_API_KEY 中。"
                f" 项目根：{PROJECT_ROOT / '.env'}"
            )
        model_raw = (
            os.environ.get("OPENROUTER_EMBEDDING_MODEL")
            or os.environ.get("OPENAI_EMBEDDING_MODEL")
            or ""
        ).strip()
        model_name = model_raw or "openai/text-embedding-3-small"
        headers: dict[str, str] = {}
        referer = (os.environ.get("OPENROUTER_HTTP_REFERER") or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        x_title = (os.environ.get("OPENROUTER_X_TITLE") or "").strip()
        if x_title:
            headers["X-Title"] = x_title
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            default_headers=headers if headers else None,
        )

    if emb_base_raw:
        api_base = normalize_api_base(emb_base_raw)
        api_key = emb_key or primary_key
    elif is_deepseek_chat:
        api_base = "https://api.openai.com/v1"
        if not emb_key:
            raise RuntimeError(
                "检测到对话走 DeepSeek，且未配置 OpenRouter / 其它嵌入网关。\n"
                "推荐混搭：在 .env 设置 OPENROUTER_API_KEY（嵌入走 https://openrouter.ai/api/v1 ，见 .env.example）。\n"
                "或设置 OPENAI_EMBEDDING_API_KEY=OpenAI 官方 Key，嵌入将走 api.openai.com。\n"
                "也可设置 OPENAI_EMBEDDING_BASE_URL 指向已支持 /v1/embeddings 的地址。"
            )
        api_key = emb_key
    else:
        api_base = normalize_api_base(os.environ.get("OPENAI_BASE_URL"))
        api_key = emb_key or primary_key

    if not api_key:
        raise RuntimeError(
            "未检测到用于嵌入的 API Key（OPENAI_API_KEY / OPENAI_EMBEDDING_API_KEY / OPENROUTER_API_KEY 至少其一需可用于嵌入）。"
            f" 项目根：{PROJECT_ROOT / '.env'}"
        )

    model_raw = (os.environ.get("OPENAI_EMBEDDING_MODEL") or "").strip()
    base_lower = (api_base or "").lower()
    if "deepseek" in base_lower:
        if not model_raw:
            raise RuntimeError(
                "嵌入 Base 指向 DeepSeek 时，必须在 .env 设置 OPENAI_EMBEDDING_MODEL=官方 embedding 模型 ID。\n"
                "若 /v1/models 无 embedding 模型，请改用 OPENROUTER_API_KEY 做向量。"
            )
        model_name = model_raw
    else:
        model_name = model_raw or "text-embedding-3-small"

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
    )


def _collect_source_files() -> list[str]:
    """
    递归收集 data/ 下 .txt、.md、.pdf、.docx 的绝对路径（稳定排序）。
    """
    settings = get_settings()
    data_dir = settings.data_dir
    if not data_dir.is_dir():
        return []
    seen: set[Path] = set()
    for pattern in ("*.txt", "*.md", "*.pdf", "*.docx"):
        for p in data_dir.rglob(pattern):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(str(x) for x in seen)


def _meta_page_value(page: str) -> str:
    p = (page or "").strip()
    return p if p else "-"


def _build_corpus() -> tuple[list[str], list[str], list[dict[str, str]], int]:
    """
    扫描 data/，解析为 ids、documents、metadatas（不写库）。
    多文件时使用线程池并行解析（INGEST_IO_WORKERS）；metadata 含 source、page、file_type。
    返回最后一项为参与解析的文件数。
    """
    paths = _collect_source_files()
    if not paths:
        return [], [], [], 0
    ids, docs, meta = _parse_many_files_parallel(paths)
    return ids, docs, meta, len(paths)


def has_indexed_documents() -> bool:
    """
    判断本地是否已有「非空」向量索引，供 main 启动时决定是否跳过 initialize()。
    """
    load_project_env()
    settings = get_settings()
    if not Path(settings.chroma_path).is_dir():
        return False
    try:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        ef = openai_embedding_function()
        col = client.get_collection(name=settings.collection_name, embedding_function=ef)
        return col.count() > 0
    except Exception:
        return False


def _add_documents_batched(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str]],
    batch_size: int,
) -> None:
    n = len(documents)
    if n == 0:
        return

    batch = batch_size
    if batch <= 0 or n <= batch:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return

    for start in range(0, n, batch):
        end = min(start + batch, n)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


def initialize() -> dict[str, int]:
    """
    全量重建索引：读 data/（含 PDF/Docx）→ 切片 → 删除同名集合 → create_collection → 分批 add。

    返回：files（文件数）、chunks（向量条数）。
    """
    load_project_env()
    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_EMBEDDING_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    ):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY（或 OPENAI_EMBEDDING_API_KEY / OPENROUTER_API_KEY）。"
            f"请在项目根目录创建或编辑 .env：{PROJECT_ROOT / '.env'} "
            "（可复制 .env.example），并写入有效 Key；勿在 Key 两侧加引号。"
        )

    settings = get_settings()
    all_ids, all_docs, all_meta, nfiles = _build_corpus()

    embedding_fn = openai_embedding_function()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        client.delete_collection(settings.collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
    )
    if all_docs:
        _add_documents_batched(
            collection,
            all_ids,
            all_docs,
            all_meta,
            settings.chroma_add_batch_size,
        )

    return {"files": nfiles, "chunks": len(all_docs)}


async def initialize_async() -> dict[str, int]:
    """
    异步封装的全量入库：在线程池中执行 initialize()，避免长时间阻塞 FastAPI 事件循环。
    大批量时内部分批 add 逻辑与 initialize() 相同。
    """
    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(initialize)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, initialize)


def _parse_many_files_parallel(paths: list[str]) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """
    多文件并行解析（ThreadPoolExecutor，并发度 INGEST_IO_WORKERS），输出顺序与 sorted(paths) 一致。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    settings = get_settings()
    workers = min(settings.ingest_io_workers, max(1, len(paths)))
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, str]] = []

    def one(path: str) -> tuple[str, list[tuple[str, str, dict[str, str]]]]:
        path_p = Path(path)
        try:
            rel = str(path_p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            rel = os.path.normpath(path)
        chunks = parse_document(
            path_p,
            chunk_chars=settings.chunk_chars,
            chunk_overlap=settings.chunk_overlap,
        )
        rows: list[tuple[str, str, dict[str, str]]] = []
        for idx, sc in enumerate(chunks):
            rid = f"{rel}#{idx}"
            meta = {
                "source": rel,
                "page": _meta_page_value(sc.page),
                "file_type": sc.file_type,
            }
            rows.append((rid, sc.text, meta))
        return rel, rows

    by_rel: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, p): p for p in paths}
        for fut in as_completed(futs):
            rel, rows = fut.result()
            by_rel[rel] = rows

    for path in sorted(paths):
        path_p = Path(path)
        try:
            rel = str(path_p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            rel = os.path.normpath(path)
        for rid, doc, meta in by_rel.get(rel, []):
            all_ids.append(rid)
            all_docs.append(doc)
            all_meta.append(meta)

    return all_ids, all_docs, all_meta
