"""
bot.py — RAG 问答：Chroma 向量检索 + OpenAI 兼容「对话」API

设计要点：
    - 「嵌入」与「对话」可以走不同网关：例如对话用 DeepSeek（OPENAI_BASE_URL 指向
      api.deepseek.com），嵌入用 OpenRouter（OPENROUTER_API_KEY + 嵌入模型）。
      具体分支见 ingestion.openai_embedding_function()。
    - Chroma 集合路径、集合名必须与 ingestion 一致（CHROMA_PATH、COLLECTION_NAME），
      否则会出现「写入 A 目录、读取 B 目录」或集合不存在。
    - API Base URL 的规范化（去空白、去尾斜杠）统一使用 ingestion.normalize_api_base，
      避免与 ingestion 内逻辑不一致。

依赖环境变量（节选，完整说明见 .env.example）：
    - 对话：OPENAI_API_KEY、可选 OPENAI_BASE_URL、OPENAI_CHAT_MODEL
    - 检索嵌入：由 openai_embedding_function() 读取 OPENROUTER_* / OPENAI_EMBEDDING_* 等
"""

from __future__ import annotations

import os
import sys

import chromadb
from openai import OpenAI

from ingestion import (
    CHROMA_PATH,
    COLLECTION_NAME,
    PROJECT_ROOT,
    load_project_env,
    normalize_api_base,
    openai_embedding_function,
)

# 默认检索条数；可通过环境变量 RAG_TOP_K 覆盖（资料长或总答「未找到」时可试 8～12）。
_DEFAULT_TOP_K = 5


def _top_k() -> int:
    raw = (os.environ.get("RAG_TOP_K") or "").strip()
    if not raw:
        return _DEFAULT_TOP_K
    try:
        k = int(raw)
        return k if k >= 1 else _DEFAULT_TOP_K
    except ValueError:
        return _DEFAULT_TOP_K


# 系统约束写进单条 user 消息（部分网关对 system 角色支持不一致时仍用 user 承载）。
# 目标：避免「多问题里有一问缺条文就整段道歉」、避免把资料里已有的时限/路径说成「未提及」。
_SYSTEM_LINE = """你是项目管理规范助手，必须依据下方【背景资料】作答，并遵守：

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
    """
    构造用于「聊天补全」的 OpenAI SDK 客户端（非嵌入）。

    - OPENAI_API_KEY：对话网关的 Key（DeepSeek 场景下为 DeepSeek 的 Key）。
    - OPENAI_BASE_URL：可选；DeepSeek 通常为 https://api.deepseek.com/v1。
      若未设置，则使用 OpenAI SDK 默认官方地址（一般仅在你用 OpenAI 官方对话时）。

    注意：不要把对话用的 base_url 与嵌入用的 OPENAI_EMBEDDING_BASE_URL 混为一谈；
    DeepSeek 公共 API 不提供与 OpenAI 兼容的 /v1/embeddings，嵌入需走 OpenRouter 等。
    """
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
    """
    获取已持久化的 Chroma 集合，并绑定与入库时相同的嵌入函数。

    每次调用会 load_project_env()，确保在 CLI 或长驻进程中 .env 被修改后能重新加载
    （与 ingestion 内 openai_embedding_function 开头的 load 行为一致）。

    Key 检查：至少需要一种可用于嵌入配置的 Key（与 initialize() 口径一致），否则
    openai_embedding_function() 或后续 Chroma 调用会失败，此处提前给出明确报错。
    """
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
    embedding_fn = openai_embedding_function()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def chat(query: str) -> str:
    """
    对外主入口：根据用户问题检索相关文档段，再调用对话模型生成回答。

    步骤：
        1. _get_collection() 取得集合（嵌入与入库一致）；
        2. col.query(query_texts=[query], n_results=top_k)：Chroma 用同一嵌入模型
           将 query 向量化并做相似度检索；
        3. 将返回的 documents 用双换行拼接为 context；
        4. OPENAI_CHAT_MODEL（默认 gpt-4o-mini）+ _openai_chat_client() 发起
           chat.completions.create，把系统约束、背景资料、问题放在一条 user 内容里。

    返回：模型回复文本（strip 后）；若 content 为空则返回空字符串。
    """
    col = _get_collection()
    top_k = _top_k()
    res = col.query(query_texts=[query], n_results=top_k)
    # Chroma 返回结构为嵌套列表：documents[0] 对应当前唯一 query 的若干条
    docs = (res.get("documents") or [[]])[0]
    docs = [d for d in (docs or []) if isinstance(d, str) and d.strip()]
    # 调试：环境变量 RAG_DEBUG=1 时打印检索片段（排查「答非所问」是检索问题还是模型问题）
    _dbg = (os.environ.get("RAG_DEBUG") or "").strip().lower()
    if _dbg in ("1", "true", "yes", "on"):
        print("RAG_DEBUG query:", query, file=sys.stderr, flush=True)
        print("RAG_DEBUG top_k:", top_k, file=sys.stderr, flush=True)
        print("RAG_DEBUG documents:", docs, file=sys.stderr, flush=True)
    if not docs:
        return (
            "（系统提示）检索未返回任何文档片段，无法根据知识库作答。"
            "请确认已执行过入库：在网页调用 POST /init，或删除 chroma_data 后重启服务，"
            "或在 script 目录执行 python cli.py --init。"
            "若已入库仍为空，请设置环境变量 RAG_DEBUG=1 查看服务端日志中的检索结果。"
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
    return (msg.content or "").strip()
