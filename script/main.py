"""
main.py — FastAPI 应用入口（路由 + 启动时自动入库 + 简易问答页）

职责概览：
    - 模块加载时尽早调用 load_project_env()，从「项目根」读取 .env，避免在非项目目录
      执行 `uvicorn main:app` 时读不到 OPENAI_* / OPENROUTER_*。
    - lifespan：若本地 Chroma 已有非空索引则**跳过**入库（省 Token、避免 --reload 反复嵌入）；
      否则执行 ingestion.initialize()。全量重建请 POST /init 或删 chroma_data/ 后重启。
      失败时记录完整异常栈，但服务仍会启动。
    - 路由：GET / 返回 static/index.html；POST /chat 调用 bot.chat_with_sources；POST /init
      异步触发全量重建索引；GET /health 供探活。

与 bot / ingestion 的关系：
    - 入库逻辑在 ingestion.initialize() / initialize_async()；问答在 bot.chat_with_sources()。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from .bot import chat_with_sources
    from .config import PROJECT_ROOT, get_settings, load_project_env
    from .ingestion import has_indexed_documents, initialize, initialize_async
except ImportError:  # 兼容直接在 script/ 下运行
    from bot import chat_with_sources
    from config import PROJECT_ROOT, get_settings, load_project_env
    from ingestion import has_indexed_documents, initialize, initialize_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 环境变量：必须在定义 CORS、创建路由依赖的配置之前加载
# ---------------------------------------------------------------------------
# 尽早从「项目根」加载 .env（路径与 config.PROJECT_ROOT 一致），使下方
# CORS_ORIGINS、lifespan 内 initialize()、以及子模块再次 load_dotenv 时
# 行为一致。注意：lifespan 内不再重复调用 load_project_env()，避免冗余。
load_project_env()

# 静态页在仓库根目录 static/（与 config.PROJECT_ROOT 一致）
_STATIC_INDEX = PROJECT_ROOT / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期（启动 / 关闭）。

    启动阶段：
        若 has_indexed_documents() 为 True（已有非空集合），则**不**调用 initialize()，
        避免 --reload 每次重启都重新打 Embedding。否则执行 initialize() 完成首次入库。
        更新资料后请 POST /init，或删除 chroma_data/ 后重启，或 CLI `python script/cli.py --init`。

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
        if has_indexed_documents():
            logger.info(
                "检测到已有 Chroma 索引（非空），跳过启动时入库；"
                "更新 data/ 后请 POST /init 或删除 chroma_data/ 后重启"
            )
        else:
            stats = initialize()
            logger.info("启动时自动入库完成: %s", stats)
    except Exception:
        logger.exception(
            "启动时自动入库失败：请检查 OPENAI_API_KEY、嵌入相关配置，以及 data/ 下是否有可解析资料（.txt/.md/.pdf/.docx）"
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
_cors = get_settings().cors_origins.strip()
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


class SourceItem(BaseModel):
    """单条检索命中溯源信息（与 Chroma metadata 对齐）。"""

    source: str = Field(default="", description="相对项目根的文件路径")
    page: str = Field(default="-", description="PDF 页码；无页码时为 -")
    file_type: str = Field(default="", description="txt / md / pdf / docx")
    chunk_id: str = Field(default="", description="向量库中的文档 ID")
    excerpt: str = Field(default="", description="片段摘要")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)


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
async def post_init():
    """
    可选：手动再次全量重建索引。

    典型场景：你刚更新了 data/ 下的文档，且不想重启 uvicorn，可调用本接口触发
    与启动时相同的 initialize()（删集合、重建、重新嵌入）。

    说明：若启动时因已有索引跳过了入库，改完 data/ 后应调用本接口或删库后重启。
    失败时（缺 Key、嵌入网关错误等）返回 503，detail 为 RuntimeError 的文案。
    """
    try:
        stats = await initialize_async()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"ok": True, **stats}


@app.post("/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest):
    """
    客服问答接口。

    流程简述（详见 bot.chat_with_sources）：
        1. 使用与入库相同的嵌入函数打开 Chroma 集合并向量检索 TOP_K 段文本；
        2. 将检索结果拼成【背景资料】，与用户问题一并发给 OPENAI_CHAT_MODEL
          （如 DeepSeek）生成回答；
        3. 同时返回 sources 列表供前端展示引用溯源。
    """
    try:
        raw = chat_with_sources(body.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    src_list = raw.get("sources") or []
    sources = [SourceItem(**s) for s in src_list if isinstance(s, dict)]
    return ChatResponse(answer=str(raw.get("answer") or ""), sources=sources)


@app.get("/health")
def health():
    """负载均衡或本地探活：返回最小 JSON，不做业务逻辑。"""
    return {"status": "ok"}
