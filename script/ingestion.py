"""
ingestion.py — 数据入库与「嵌入客户端」构造（供 Chroma 与 bot 共用）

职责划分：
    1. load_project_env / PROJECT_ROOT：统一从项目根加载 .env，保证无论从哪个 cwd 启动
       uvicorn、python cli.py、或 import 本模块，相对路径（data/、chroma_data/）一致。
    2. openai_embedding_function()：按环境变量选择 OpenRouter、显式 OPENAI_EMBEDDING_BASE_URL、
       或「DeepSeek 对话 + OpenAI 官方嵌入」等组合，返回 Chroma 可用的 OpenAIEmbeddingFunction。
    3. 文本切片：优先按 \\n\\n 分段，超长段再按 CHUNK_CHARS 窗口 + CHUNK_OVERLAP 滑动。
    4. initialize()：收集 data/*.txt、*.md，删建集合，分批 add（受 CHROMA_ADD_BATCH_SIZE 控制）。

重要说明（避免踩坑）：
    - DeepSeek 官方对话 API 通常不提供 OpenAI 兼容的 /v1/embeddings；若仅配 DeepSeek
      而不配 OpenRouter / OpenAI 嵌入 Key，initialize 会因无法向量而失败。推荐混搭见 .env.example。
    - normalize_api_base() 被 bot._openai_chat_client 复用，保持与嵌入侧 URL 处理一致。
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 项目根目录（本文件所在目录即仓库/项目根）
# ---------------------------------------------------------------------------
# .env、data/、chroma_data/ 均相对此路径解析，避免「从其它工作目录启动」时：
#   - load_dotenv 找不到 .env
#   - _collect_source_files 扫不到 data
#   - PersistentClient 把向量库建到意外目录
PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env() -> None:
    """
    从 PROJECT_ROOT/.env 加载环境变量（不覆盖已存在的环境变量，符合 python-dotenv 默认）。

    可被多次调用：main.py 启动时一次，bot/_get_collection、openai_embedding_function 内
    也会调用，便于开发中改 .env 后下一轮请求生效（取决于进程是否重新读文件；dotenv
    默认不覆盖已设置的 os.environ，故生产环境仍以进程启动前注入的环境为准）。
    """
    load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Chroma 持久化路径与集合名（bot.py 必须 import 相同常量）
# ---------------------------------------------------------------------------
# 若 bot 使用硬编码路径而此处改 CHROMA_PATH，会出现写入与读取不一致。
CHROMA_PATH = str(PROJECT_ROOT / "chroma_data")
COLLECTION_NAME = "support_docs"

# 单段（按 \\n\\n 切出的块）超过该字符数时，在段内做滑动窗口切分。
CHUNK_CHARS = 400
# 相邻窗口共享尾部若干字符，减轻「关键句落在两个 chunk 边界」导致检索漏召回。
CHUNK_OVERLAP = 50

# 入库时 col.add 分批大小。资料很多时一次 add 可能触发嵌入 API 超时或限流，分批更稳。
# 设为 0 或负数时，在 _add_documents_batched 中按「整包一次 add」处理。
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
    # 用于「未显式指定嵌入 base」时的分支：DeepSeek 只做对话，嵌入需另找网关
    is_deepseek_chat = "deepseek" in chat_url
    emb_base_raw = (os.environ.get("OPENAI_EMBEDDING_BASE_URL") or "").strip()

    # ---------- 分支 A：OpenRouter（POST /v1/embeddings，OpenAI 兼容）----------
    use_openrouter = bool(openrouter_key) or ("openrouter" in emb_base_raw.lower())
    if use_openrouter:
        if emb_base_raw and "openrouter" in emb_base_raw.lower():
            api_base = normalize_api_base(emb_base_raw)
        else:
            or_base = (os.environ.get("OPENROUTER_EMBEDDING_BASE_URL") or "").strip()
            api_base = normalize_api_base(or_base) if or_base else "https://openrouter.ai/api/v1"
        # OpenRouter Key 优先；否则允许把 Key 只写在 OPENAI_EMBEDDING_API_KEY
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
        # OpenRouter 模型 ID 常为 厂商/模型，如 openai/text-embedding-3-small、qwen/qwen3-embedding-8b
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

    # ---------- 分支 B：显式嵌入 Base（非 OpenRouter 已在上面 return）----------
    if emb_base_raw:
        api_base = normalize_api_base(emb_base_raw)
        api_key = emb_key or primary_key
    elif is_deepseek_chat:
        # ---------- 分支 C：对话是 DeepSeek，嵌入未单独指定 URL ----------
        # 公共 DeepSeek API 无标准 OpenAI embeddings，故默认指向 OpenAI 官方嵌入端点
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
        # ---------- 分支 D：同一套 BASE_URL 既对话又嵌入（如全 OpenAI）----------
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
        # 若用户误把嵌入 base 指到 DeepSeek，必须显式模型 ID；且多数情况下该端点无 embedding
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


def _split_long_paragraph(para: str) -> list[str]:
    """
    对单个段落（已无 \\n\\n 内部分段）做长度控制。

    - 长度 <= CHUNK_CHARS：整段作为一个 chunk。
    - 否则：从 0 开始每次取 CHUNK_CHARS 字符，下一步起点为 end - CHUNK_OVERLAP，
      形成重叠窗口，减少边界切割损失。
    """
    para = para.strip()
    if not para:
        return []
    if len(para) <= CHUNK_CHARS:
        return [para]

    chunks: list[str] = []
    i = 0
    n = len(para)
    while i < n:
        end = min(i + CHUNK_CHARS, n)
        piece = para[i:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        i = end - CHUNK_OVERLAP
        if i < 0:
            i = 0
    return chunks


def _split_text(text: str) -> list[str]:
    """
    全文切片入口：先按双换行分段（保留文档结构），再对每段调用 _split_long_paragraph。

    空文本返回 []；段之间空块跳过。
    """
    text = text.strip()
    if not text:
        return []

    out: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        out.extend(_split_long_paragraph(block))
    return out


def _collect_source_files() -> list[str]:
    """
    递归收集 PROJECT_ROOT/data 下所有 .txt、.md 文件的绝对路径排序列表。

    使用 resolve() 去重（同一文件不同相对路径），按字符串排序保证入库顺序稳定。
    """
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.is_dir():
        return []
    seen: set[Path] = set()
    for pattern in ("*.txt", "*.md"):
        for p in data_dir.rglob(pattern):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(str(p) for p in seen)


def initialize() -> dict[str, int]:
    """
    全量重建索引：读 data/ → 切片 → 删除同名集合（若存在）→ create_collection → add。

    返回统计字典：
        files：参与入库的文件数；
        chunks：切片条数（与 Chroma 中向量条数一致，除非 add 失败）。

    前置条件：至少一种 Key 可被嵌入逻辑使用（与 bot 侧检查口径一致）。

    ID 规则：「相对项目根的路径」+ # + 段序号，例如 data/foo.md#0，便于溯源。
    metadata：每条含 source 字段为相对路径字符串。

    delete_collection 失败时忽略（例如首次运行本就没有集合），随后 create_collection。
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

    paths = _collect_source_files()
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, str]] = []

    for path in paths:
        path_p = Path(path)
        try:
            rel = str(path_p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            # 极少数路径不在 PROJECT_ROOT 下时退回 normpath
            rel = os.path.normpath(path)
        raw = path_p.read_text(encoding="utf-8")
        for idx, chunk in enumerate(_split_text(raw)):
            all_ids.append(f"{rel}#{idx}")
            all_docs.append(chunk)
            all_meta.append({"source": rel})

    embedding_fn = openai_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # 集合不存在或首次运行：无需向上抛出
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    if all_docs:
        _add_documents_batched(collection, all_ids, all_docs, all_meta)

    return {"files": len(paths), "chunks": len(all_docs)}


def _add_documents_batched(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str]],
) -> None:
    """
    将 ids/documents/metadatas 写入 Chroma；根据 CHROMA_ADD_BATCH_SIZE 决定是否分批。

    - batch <= 0 或总条数 <= batch：单次 collection.add。
    - 否则：按 [start:end) 切片多次 add，降低单次请求体大小与嵌入批处理压力。
    """
    n = len(documents)
    if n == 0:
        return

    batch = CHROMA_ADD_BATCH_SIZE
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
