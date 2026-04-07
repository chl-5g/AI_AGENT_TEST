"""
main.py — FastAPI 应用入口（路由 + 启动时自动入库 + 简易问答页）

职责概览：
    - 模块加载时尽早调用 load_project_env()，从「项目根」读取 .env，避免在非项目目录
      执行 `uvicorn main:app` 时读不到 OPENAI_* / OPENROUTER_*。
    - lifespan：应用启动阶段同步执行 ingestion.initialize()，完成「扫描 data/ → 切片 →
      写入 Chroma」；失败时记录完整异常栈，但服务仍会启动，便于你打开 / 页查看说明或
      修正 .env 后重启（不必因入库失败而进程直接退出）。
    - 路由：GET / 返回 static/index.html；POST /chat 调用 bot.chat；POST /init 可手动
      全量重建索引；GET /health 供探活。

与 bot / ingestion 的关系：
    - 入库逻辑只在 ingestion.initialize()；问答检索在 bot.chat()；本文件不重复实现业务。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from bot import chat
from ingestion import initialize, load_project_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 环境变量：必须在定义 CORS、创建路由依赖的配置之前加载
# ---------------------------------------------------------------------------
# 尽早从「项目根」加载 .env（路径与 ingestion.PROJECT_ROOT 一致），使下方
# CORS_ORIGINS、lifespan 内 initialize()、以及子模块再次 load_dotenv 时
# 行为一致。注意：lifespan 内不再重复调用 load_project_env()，避免冗余。
load_project_env()

# 静态页：使用 __file__ 解析目录，不依赖「当前工作目录 cwd」，避免从别的路径启动
# uvicorn 时找不到 index.html。
_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期（启动 / 关闭）。

    启动阶段：
        同步执行 initialize()：扫描 data/ 下 .txt/.md、切片、删除并重建 Chroma 集合、
        写入向量。这是「极简 RAG」的自动入库入口，用户一般无需再手动 POST /init。

    失败处理：
        任意异常会被 logger.exception 打出完整栈；服务进程仍继续运行，yield 后正常
        提供 HTTP。这样你可以在浏览器访问 / 看到前端提示，或根据日志修正 Key、
        data/ 路径、嵌入网关配置后再重启。

    关闭阶段：
        当前无额外清理逻辑；Chroma 数据已持久化在 chroma_data/。
    """
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        stats = initialize()
        logger.info("启动时自动入库完成: %s", stats)
    except Exception:
        logger.exception(
            "启动时自动入库失败：请检查 OPENAI_API_KEY、嵌入相关配置，以及 data/ 下是否有 .txt/.md"
        )
    yield


app = FastAPI(title="极简 RAG 客服", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS（跨域资源共享）
# ---------------------------------------------------------------------------
# 浏览器从不同「源」（协议+域名+端口）访问本 API 时，会先发预检请求；若未配置
# CORSMiddleware，fetch 可能被浏览器拦截。Postman、curl 不依赖 CORS。
# 环境变量 CORS_ORIGINS：
#   - 未设置或 "*"：允许任意 Origin（开发方便，生产慎用）。
#   - 逗号分隔多个具体 URL，例如 http://127.0.0.1:5173,http://localhost:3000
# allow_credentials=False：与 allow_origins=["*"] 组合时浏览器规范要求如此。
_cors = (os.environ.get("CORS_ORIGINS") or "*").strip()
if _cors == "*":
    _allow_origins = ["*"]
else:
    _allow_origins = [o.strip() for o in _cors.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """
    POST /chat 的请求体（JSON）。

    使用 Pydantic + Field 后，FastAPI 会自动：
        - 校验 JSON 结构；缺字段、类型错误返回 422。
        - query 为空字符串时因 min_length=1 同样 422。
    路由内无需手写 json.loads 或手动判断空串。
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"query": "根据 SOP，需求变更要怎么走流程？"}]},
    )

    query: str = Field(..., min_length=1, description="用户提问，非空")


@app.get("/", response_class=HTMLResponse)
def index():
    """
    返回带输入框的简易问答页（static/index.html）。

    前端通过 fetch POST /chat，请求体 {"query": "..."}，展示返回的 answer 字段。
    若 static/index.html 缺失（误删或未同步仓库），返回 500 与简短 HTML 说明。
    """
    if not _STATIC_INDEX.is_file():
        return HTMLResponse(
            "<p>缺少 static/index.html，请从仓库恢复该文件。</p>",
            status_code=500,
        )
    return HTMLResponse(_STATIC_INDEX.read_text(encoding="utf-8"))


@app.post("/init")
def post_init():
    """
    可选：手动再次全量重建索引。

    典型场景：你刚更新了 data/ 下的文档，且不想重启 uvicorn，可调用本接口触发
    与启动时相同的 initialize()（删集合、重建、重新嵌入）。

    说明：正常流程下 lifespan 已执行过一次 initialize()，多数时候不必调用 /init。
    失败时（缺 Key、嵌入网关错误等）返回 503，detail 为 RuntimeError 的文案。
    """
    try:
        stats = initialize()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"ok": True, **stats}


@app.post("/chat")
def post_chat(body: ChatRequest):
    """
    客服问答接口。

    流程简述（详见 bot.chat）：
        1. 使用与入库相同的嵌入函数打开 Chroma 集合并向量检索 TOP_K 段文本；
        2. 将检索结果拼成【背景资料】，与用户问题一并发给 OPENAI_CHAT_MODEL
          （如 DeepSeek）生成回答。

    配置类错误（缺 Key、集合打不开等）在 bot 层抛 RuntimeError，此处转为 503，
    便于前端区分「服务不可用」与未捕获的 500。
    """
    try:
        answer = chat(body.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"answer": answer}


@app.get("/health")
def health():
    """负载均衡或本地探活：返回最小 JSON，不做业务逻辑。"""
    return {"status": "ok"}
