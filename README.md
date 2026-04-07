# 极简 RAG 客服 Agent

**思路**：LLM 只根据检索到的纯文本资料回答，不凭空编造。第一版不做 PDF/Docx 解析、不做引用溯源、不做复杂配置模块。

---

## 技术栈（怎么省事怎么来）

| 选型 | 说明 |
|------|------|
| 向量库 | **ChromaDB** 持久化到本地目录，无需单独装数据库、无账号密码 |
| LLM / 向量 | **OpenAI 官方 SDK**（`base_url` 可指向 DeepSeek 等兼容网关），手写 `messages=[...]` |
| 密钥 / 端点 | **`.env`**：**对话** DeepSeek（`OPENAI_API_KEY` + `OPENAI_BASE_URL`）；**嵌入**推荐 **OpenRouter**（`OPENROUTER_API_KEY` + `OPENROUTER_EMBEDDING_MODEL`，如 `openai/text-embedding-3-small`）。详见 [OpenRouter Embeddings](https://openrouter.ai/docs/api-reference/embeddings) 与 `.env.example` |

---

## 仓库结构（4 个路径）

| 路径 | 说明 |
|------|------|
| `data/` | 只放 **`.txt` / `.md`**（先把资料手工整理成纯文本，例如 `context.txt`） |
| `ingestion.py` | **`initialize()`**：`glob` 扫描 `data/` → 优先按 `\n\n` 分段，过长再按字数切（重叠 50 字）→ `delete` + `create` 后 `Chroma.add()` |
| `bot.py` | **`chat(query)`**：`Chroma.query` → 拼 Prompt → `OpenAI.chat.completions.create` |
| `main.py` | **FastAPI**：启动时**自动**入库；`GET /` 简易问答页；`POST /chat` API；可选 `POST /init` 手动重建索引 |
| `static/index.html` | 浏览器问答页：输入框 + 调用 `/chat` |
| `cli.py` | **命令行**提问：支持一句话参数或交互输入（可不打开浏览器） |

运行后会在项目下生成 **`chroma_data/`**（Chroma 持久化目录，可视作本地「库文件」）。

---

## 数据流（两个函数闭环）

1. **`initialize()`** — 扫描 `data/**/*.txt` 与 `data/**/*.md` → 先按 `\n\n` 成段，单段超过约 400 字再切，块间重叠 **50** 字 → `delete_collection` 后 `create_collection` → 写入向量。  
2. **`chat(query)`** — 与入库共用 **`ingestion` 中的 `CHROMA_PATH` / `COLLECTION_NAME`**，保证 Chroma 路径一致 → 相似检索（默认 Top **5**）→ 防御性客服 Prompt + 背景资料 + 问题 → 聊天模型（默认 **`gpt-4o-mini`**）。

---

## 第一版刻意不做的事

- 复杂 Parser（PDF/Docx）：先手工复制到 `.txt` / `.md`  
- 异步/多线程入库：文档少时同步足够  
- 引用页码/段落溯源：只要能按资料答即可  
- 独立 `config` 包：只认 `.env`  

---

## 运行

```bash
pip install -r requirements.txt
cd d:\ai_agent_test
copy .env.example .env
# 编辑「项目根目录」的 .env 填入 OPENAI_API_KEY（与 main.py 同级；勿只改 .env.example）
uvicorn main:app --reload
```

**说明**：程序会从 **`ingestion.py` 所在目录**（项目根）固定加载 `.env`，并在此目录下查找 `data/`、写入 `chroma_data/`，**与你在哪个文件夹执行 `uvicorn` 无关**。若仍报未设置 Key，请确认文件名是 `.env`（无多余扩展名）、且 Key 行无引号、已保存。

- 把资料放进 **`data/`** 后启动服务：**启动时会自动执行 `initialize()`**，无需再手动调 **`POST /init`**（更新资料后可重启服务，或仍可调 **`POST /init`** 热重建）。  
- 浏览器打开 **http://127.0.0.1:8000/** ，在输入框里提问即可。  
- **只用 CMD、不开浏览器**：在项目根目录执行  
  `python cli.py 你的问题一句话`  
  或 `python cli.py` 进入交互模式（输入问题回车，**空行**退出）。  
  索引为空时会**自动**建库；改完 `data/` 后可 `python cli.py --init 任意问题` 强制重建。  
- 也可直接调 **`POST /chat`**，JSON：`{"query": "你的问题"}`（不合法时返回 **422**）。  
- 若用其它前端域名调 API，请在 `.env` 配置 **`CORS_ORIGINS`**（示例见 `.env.example`）。  
- 网页里点「发送」若出现 **Failed to fetch**：多为 **uvicorn 未运行**、或 **用 IDE 内置浏览器打开页面**（请改用 **Chrome/Edge** 访问 `http://127.0.0.1:8000/`），或 **用 file:// 打开了 static 文件**。可先访问 **`/health`** 是否返回 `{"status":"ok"}`。  

---

## API 摘要

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/` | 简易网页：输入框问答 |
| POST | `/init` | （可选）再次执行 `initialize()`，返回 `files` / `chunks` |
| POST | `/chat` | 执行 `chat(query)`，返回 `answer` |
| GET | `/health` | 健康检查 |
