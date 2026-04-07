"""
bot.py — RAG 问答：Chroma 向量检索 + OpenAI 兼容「对话」API

设计要点：
    - 「嵌入」与「对话」可以走不同网关：例如对话用 DeepSeek，嵌入用 OpenRouter。
    - Chroma 路径、集合名与 ingestion 一致（来自 config.get_settings()）。
    - 检索结果携带 metadata（source、page、file_type），接口返回 sources 供前端溯源展示。
"""

from __future__ import annotations

import os
import sys
from typing import Any

import chromadb
from openai import OpenAI

from config import PROJECT_ROOT, get_settings, load_project_env
from ingestion import normalize_api_base, openai_embedding_function

_SYSTEM_LINE = """你是项目管理规范助手，必须依据下方【背景资料】作答，并遵守：

【引用与溯源】当答案直接依据某段背景时，请在对应句子或段末用简短括号标注来源，格式示例：（来源：文件名，第N页）或（来源：文件名）（无页码时省略「第N页」）。背景中若标注了「文件名 / 页码」，须与之一致，勿编造页码。

【多件事要拆开答】用户若一次问了多个要点（例如「谁审批」和「最晚何时回复」），请**分项**回答：
- 资料里**有**的：必须明确写出（可归纳，不必逐字照抄）。
- 资料里**没有**的：单独写「资料未规定……」，可建议用户以 OA 流程节点为准或咨询人工 PM。
- **禁止**因为其中某一项资料没写，就拒绝回答其他项，也**禁止**因此整段套用文末「固定道歉语」。

【写「未提及」前必查】在声称「资料未提及某时限/部门/路径/比例」之前，必须再扫一遍【背景资料】全文；
若文中已有对应内容（例如「人力资源部在 2 个工作日内给出初审意见」），**必须照答**，不得写未提及。

【用户表述边界 — 禁止替用户编场景】**不得编造【问题】里没出现的信息。**
- 不得虚构用户负责的系统名称、业务领域（如「薪资管理系统」）、单位性质等；**只能复述用户已写明的字句**作为前提。
- 若用户**未说明**是否为「政府类项目」或其它关键前提，**禁止**写「您负责的是××」「您是非政府项目」等断言；应使用**条件句**：「条文规定：**政府类项目**须提交《等保三级测评报告》；**非政府类**则未要求该报告。请您按实际项目是否属于政府类自行对照。」
- 若缺少前提导致无法唯一结论，可明确写：「您未在问题中说明是否政府类项目，我无法替您认定；若属于政府类则需……，否则只需……」

【固定道歉语】仅当【背景资料】与用户问题在**所有**要点上均无任何关联、整段资料都无法支撑任何一句有用答复时，
才允许**一字不差**输出下面整句，且**不要**再加其它解释：
抱歉，在现有的项目管理规范中未找到相关规定，请咨询人工PM。

【格式】回答请使用 **Markdown**（如 `##` 小标题、**加粗**、`-` 列表、`` `术语` `` 行内代码），便于网页端渲染阅读。

【其它】不要编造资料中不存在的制度；不要用外部常识冒充本公司条文；不要替用户捏造其未提及的项目背景。"""


def _openai_chat_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY。请在项目根目录的 .env 中配置："
            f"{PROJECT_ROOT / '.env'}"
        )
    base = normalize_api_base(os.environ.get("OPENAI_BASE_URL"))
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    return OpenAI(api_key=api_key)


def _get_collection():
    load_project_env()
    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_EMBEDDING_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    ):
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY（或 OPENAI_EMBEDDING_API_KEY / OPENROUTER_API_KEY）。"
            f"请检查：{PROJECT_ROOT / '.env'}"
        )
    settings = get_settings()
    embedding_fn = openai_embedding_function()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_collection(
        name=settings.collection_name,
        embedding_function=embedding_fn,
    )


def _excerpt(doc: str, max_len: int = 160) -> str:
    t = doc.strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def chat_with_sources(query: str) -> dict[str, Any]:
    """
    返回 {"answer": str, "sources": list[dict]}，sources 与检索命中顺序一致，便于 UI 展示溯源。
    """
    col = _get_collection()
    settings = get_settings()
    top_k = settings.rag_top_k
    res = col.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    docs = [d for d in (docs or []) if isinstance(d, str) and d.strip()]
    metas = (res.get("metadatas") or [[]])[0] or []
    ids = (res.get("ids") or [[]])[0] or []

    if settings.rag_debug:
        print("RAG_DEBUG query:", query, file=sys.stderr, flush=True)
        print("RAG_DEBUG top_k:", top_k, file=sys.stderr, flush=True)
        print("RAG_DEBUG documents:", docs, file=sys.stderr, flush=True)
        print("RAG_DEBUG metadatas:", metas, file=sys.stderr, flush=True)

    if not docs:
        return {
            "answer": (
                "（系统提示）检索未返回任何文档片段，无法根据知识库作答。"
                "请确认已执行过入库：在网页调用 POST /init，或删除 chroma_data 后重启服务，"
                "或在 script 目录执行 python cli.py --init。"
                "若已入库仍为空，请设置环境变量 RAG_DEBUG=1 查看服务端日志中的检索结果。"
            ),
            "sources": [],
        }

    sources: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else None
        meta = meta if isinstance(meta, dict) else {}
        chunk_id = ids[i] if i < len(ids) else ""
        page = str(meta.get("page") or "-")
        sources.append(
            {
                "source": str(meta.get("source") or ""),
                "page": page,
                "file_type": str(meta.get("file_type") or ""),
                "chunk_id": str(chunk_id) if chunk_id is not None else "",
                "excerpt": _excerpt(doc),
            }
        )

    context = "\n\n".join(docs)
    model = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    client = _openai_chat_client()
    user_content = (
        f"{_SYSTEM_LINE}\n\n"
        f"【背景资料】\n{context}\n\n"
        f"【问题】\n{query}"
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
    )
    msg = completion.choices[0].message
    answer = (msg.content or "").strip()
    return {"answer": answer, "sources": sources}


def chat(query: str) -> str:
    """兼容旧调用：仅返回答案字符串。"""
    return str(chat_with_sources(query).get("answer") or "")
