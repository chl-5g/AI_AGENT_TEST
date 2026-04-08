# AI_AGENT_TEST — 极简 RAG 客服

基于 **ChromaDB** + **OpenAI 兼容 API** 的检索增强问答：模型只依据 `data/` 中的纯文本作答，不引用未入库内容。无 LangChain，无独立向量服务。

**本仓库布局**：应用代码在 **`script/`**；配置与资料在**仓库根**（`.env`、`data/`、`static/`）；运行期生成 **`chroma_data/`**（已 `.gitignore`，勿提交）。

---

## 技术栈

| 组件 | 说明 |
|------|------|
| 向量库 | ChromaDB，持久化目录 `chroma_data/` |
| HTTP | FastAPI + Uvicorn |
| SDK | `openai`（对话与嵌入均可设 `base_url`） |
| 配置 | 根目录 `.env` + `script/config.py`（`get_settings()` 集中读取） |
| 文档解析 | `pypdf`（PDF）、`python-docx`（Word） |

**典型配置**：对话走 **DeepSeek**（`OPENAI_API_KEY` + `OPENAI_BASE_URL`）；嵌入走 **OpenRouter**（`OPENROUTER_API_KEY` + `OPENROUTER_EMBEDDING_MODEL`）。详见 [.env.example](.env.example) 与 [OpenRouter Embeddings](https://openrouter.ai/docs/api-reference/embeddings)。

---

## 目录说明

```
仓库根/
├── run.bat               # Windows 一键启动（CMD / PowerShell 双击或运行）
├── run.sh                # Linux / macOS / WSL / Git Bash 一键启动（无需 chmod 时可直接 bash run.sh）
├── .env.example          # 环境变量模板（勿提交真实 .env）
├── requirements.txt      # Python 依赖
├── pyproject.toml        # 打包配置（支持 pip install . 与命令行入口）
├── data/                 # 知识库：*.txt / *.md / *.pdf / *.docx
├── static/index.html     # 浏览器问答页（含「引用溯源」展示）
├── chroma_data/          # 本地向量库（运行后生成，已忽略）
└── script/
    ├── __init__.py       # 包标识
    ├── deploy.py         # 一键部署入口（python -m script.deploy）
    ├── main.py           # FastAPI 入口、生命周期、路由
    ├── config.py         # 独立配置：PROJECT_ROOT、get_settings()、load_project_env()
    ├── chunking.py       # 纯文本切片（字符窗口 + 重叠）
    ├── doc_parse.py      # PDF（pypdf）/ Docx（python-docx）抽取与按页 metadata
    ├── ingestion.py      # 并行解析多文件、嵌入、initialize() / initialize_async()
    ├── bot.py            # Chroma 检索 + 溯源字段 + chat.completions
    └── cli.py            # 命令行提问
```

**路径约定**：`config.PROJECT_ROOT` 在代码位于 `script/` 时指向**上一级**（仓库根），`.env`、`data/`、`chroma_data/`、`static/` 均在根目录解析。

---

## 切片、入库与检索（与代码一致）

- **支持格式**：`data/` 下递归扫描 `.txt`、`.md`、`.pdf`、`.docx`。PDF 按**页**抽取文本，切片 metadata 带 `page`（1 起）；纯文本与 Docx 的 `page` 在库中为 `-`。
- **知识库目录可配置**：`.env` 可设 `RAG_KB_DIR`（默认 `data/`），支持相对路径（相对项目根）或绝对路径。
- **分段**：优先按 `\n\n`，单段超过 **`CHUNK_CHARS`**（默认 400）再滑窗，重叠 **`CHUNK_OVERLAP`**（默认 50），可在 `.env` 覆盖（见 `.env.example`）。
- **大批量入库**：解析阶段对多文件使用线程池（**`INGEST_IO_WORKERS`**，默认 4）；写入 Chroma 仍按 **`CHROMA_ADD_BATCH_SIZE`** 分批 `add`，避免单次请求过大。
- **`POST /init`**：路由为 **async**，内部通过 **`initialize_async()`**（`asyncio.to_thread(initialize)`）执行全量重建，避免长时间阻塞事件循环（嵌入本身仍在工作线程中同步调用）。
- **检索**：默认 **Top 5**；`.env` 中 **`RAG_TOP_K`** 覆盖（由 `config.get_settings()` 读取）。
- **溯源**：每条向量 metadata 含 `source`（相对仓库根路径）、`page`、`file_type`；`POST /chat` 返回 **`sources`** 列表（含片段摘要），网页端同步展示；Prompt 要求模型在正文中括号标注来源。
- **启动**：若已有**非空** Chroma 集合，**跳过**全量 `initialize()`；更新资料后需 **`POST /init`**、删 `chroma_data/` 或 **`python cli.py --init`**。

---

## 运行方式

### 一键启动（最少步骤）

**Windows（PowerShell / CMD）**不要用 `chmod`（那是 Linux 命令，会报「无法识别」）。在仓库根目录执行：

```powershell
.\run.bat
```

或资源管理器中**双击 `run.bat`**。

**Linux / macOS / WSL / Git Bash** 任选其一：

```bash
./run.sh
# 若提示没有执行权限，再执行：chmod +x run.sh
# 或直接：bash run.sh
```

脚本会：创建 **`./.venv`** → 若该环境里**已能导入** chromadb / openai 等则**跳过 pip**（避免每次启动都联网检查）；否则再执行 `pip install -r requirements.txt`。**依赖若只装在系统 Python 而未装入 `.venv`，第一次仍会安装到 `.venv`。** → 若没有 **`.env`** 则从 **`.env.example`** 复制并提示填 Key → 启动 **http://127.0.0.1:8000**。

**仍需你完成的一件事**：在仓库根编辑 **`.env`** 写入真实 API Key。`run.bat` / `run.sh` 首次若因复制 `.env` 而结束，填好 Key 后再运行一次即可。

---

### 打包安装（可选）

项目已支持 Python 包方式安装，安装后可直接用命令行入口启动：

```bash
pip install .
ai-rag-serve --reload --host 127.0.0.1 --port 8000
ai-rag-cli --init 需求变更怎么走
```

若不在仓库根目录启动，可显式指定项目根（用于定位 `.env` / `data` / `static`）：

```bash
AI_AGENT_HOME=/path/to/AI_AGENT_TEST ai-rag-serve --host 0.0.0.0 --port 8000
```

---

### 手动运行

**1. 安装依赖**（建议在虚拟环境中，于仓库根执行）：

```bash
pip install -r requirements.txt
```

若报错 **`No matching distribution found for chromadb>=...`**，多半是用了 **Python 3.7** 自带的 `pip`（`pip -V` 会显示路径）。请改用 **Python 3.10+**：例如先 `./run.sh` 让脚本创建 `.venv` 并升级 pip，或执行 `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`。本项目的 `openai` v1、`fastapi` 新版本同样依赖较新的 Python，不建议继续用 3.7 跑全套服务。

**2. 配置环境变量**（仓库根）：

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env` 填入真实 Key（勿提交 `.env`）。

**3. 启动 Web 服务**（推荐统一入口）：

```bash
python -m script.deploy --reload --host 127.0.0.1 --port 8000
```

浏览器访问：**http://127.0.0.1:8000/**

**4. 命令行提问**：

```bash
python -m script.cli 你的问题一句话
python -m script.cli              # 交互模式，空行结束
python -m script.cli --init 任意  # 强制重建索引后再问
```

**Windows 中文**：资料读取为 UTF-8；CLI 会尝试将标准输出设为 UTF-8。若终端仍乱码，可先执行 `chcp 65001` 或设置环境变量 `PYTHONUTF8=1`。

---

## 调试

| 现象 | 建议 |
|------|------|
| 回答「不知道」或明显偏题 | 查集合 `count()` 是否为 0；确认是否因「已有索引」跳过了入库 |
| 怀疑检索质量 | 设置 `RAG_DEBUG=1`（见 `.env.example`），查看 stderr 中的检索片段 |
| 片段不相关 | 检查嵌入模型、`api_base`、切片参数 |
| 片段相关但答案错 | 检查对话模型与 Prompt（`bot.py` 中 `_SYSTEM_LINE`） |

---

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 简易网页问答 |
| POST | `/chat` | JSON：`{"query": "..."}` → `{"answer": "...", "sources": [{ "source", "page", "file_type", "chunk_id", "excerpt" }]}` |
| POST | `/init` | 异步执行全量重建索引，返回 `ok`、`files`、`chunks` |
| GET | `/health` | `{"status":"ok"}` |

---