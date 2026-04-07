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

import chromadb
from openai import OpenAI

from ingestion import CHROMA_PATH, COLLECTION_NAME, load_project_env, normalize_api_base, openai_embedding_function

# 每次 query 返回最相近的文档段数量；过大易撑满上下文，过小可能漏掉关键句。
TOP_K = 5

# 系统约束写进单条 user 消息前缀（部分网关对 system 角色支持不一致时仍可用）。
# 要求模型严格依据【背景资料】回答，资料没有则固定话术，减少幻觉。
_SYSTEM_LINE = (
    "你是一个严格的客服助手。仅限使用下方提供的【背景资料】回答问题。"
    "如果资料中未提及相关内容，请礼貌地回答："
    "'抱歉，在现有的项目管理规范中未找到相关规定，请咨询人工PM。' "
    "严禁发挥或引用外部知识。"
)


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
            "未检测到 OPENAI_API_KEY。请在项目根目录的 .env 中配置（与 main.py、ingestion.py 同级目录）。"
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
            "未检测到 OPENAI_API_KEY（或 OPENAI_EMBEDDING_API_KEY / OPENROUTER_API_KEY）。请检查项目根目录 .env 是否已保存。"
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
        2. col.query(query_texts=[query], n_results=TOP_K)：Chroma 用同一嵌入模型
           将 query 向量化并做相似度检索；
        3. 将返回的 documents 用双换行拼接为 context；
        4. OPENAI_CHAT_MODEL（默认 gpt-4o-mini）+ _openai_chat_client() 发起
           chat.completions.create，把系统约束、背景资料、问题放在一条 user 内容里。

    返回：模型回复文本（strip 后）；若 content 为空则返回空字符串。
    """
    col = _get_collection()
    res = col.query(query_texts=[query], n_results=TOP_K)
    # Chroma 返回结构为嵌套列表：documents[0] 对应当前唯一 query 的 TOP_K 条
    docs = (res.get("documents") or [[]])[0]
    context = "\n\n".join(docs) if docs else ""

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
