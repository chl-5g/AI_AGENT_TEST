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
| 配置 | 根目录 `.env`（复制 `.env.example`） |

**典型配置**：对话走 **DeepSeek**（`OPENAI_API_KEY` + `OPENAI_BASE_URL`）；嵌入走 **OpenRouter**（`OPENROUTER_API_KEY` + `OPENROUTER_EMBEDDING_MODEL`）。详见 [.env.example](.env.example) 与 [OpenRouter Embeddings](https://openrouter.ai/docs/api-reference/embeddings)。

---

## 目录说明

```
仓库根/
├── run.bat               # Windows 一键启动（CMD / PowerShell 双击或运行）
├── run.sh                # Linux / macOS / WSL / Git Bash 一键启动（无需 chmod 时可直接 bash run.sh）
├── .env.example          # 环境变量模板（勿提交真实 .env）
├── requirements.txt      # Python 依赖
├── data/                 # 知识库：*.txt / *.md（示例含脱敏 SOP）
├── static/index.html     # 浏览器问答页
├── chroma_data/          # 本地向量库（运行后生成，已忽略）
└── script/
    ├── main.py           # FastAPI 入口、启动生命周期
    ├── ingestion.py      # 切片、嵌入、initialize()、has_indexed_documents()
    ├── bot.py            # Chroma 检索 + chat.completions
    └── cli.py            # 命令行提问
```

**路径约定**：`ingestion` 中 `PROJECT_ROOT` 在代码位于 `script/` 时指向**上一级**（仓库根），因此 `.env`、`data/`、`chroma_data/`、`static/` 均在根目录解析。

---

## 切片与检索（与代码一致）

- 分段：优先按 `\n\n`，单段超过约 **400** 字符再滑窗，重叠 **50** 字符（`CHUNK_CHARS` / `CHUNK_OVERLAP`）。
- 检索：默认 **Top 5**；可在 `.env` 设 **`RAG_TOP_K`** 覆盖。
- 启动：若已有**非空** Chroma 集合，**跳过**全量 `initialize()`（避免 `uvicorn --reload` 反复打 Embedding）；更新资料后需 **`POST /init`**、删 `chroma_data/` 或 **`python cli.py --init`**。

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

### 手动运行

**1. 安装依赖**（建议在虚拟环境中，于仓库根执行）：

```bash
pip install -r requirements.txt
```

**2. 配置环境变量**（仓库根）：

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

编辑 `.env` 填入真实 Key（勿提交 `.env`）。

**3. 启动 Web 服务**（必须先进入 `script/`，否则 `import bot` 会失败）：

```bash
cd script
uvicorn main:app --reload
```

浏览器访问：**http://127.0.0.1:8000/**

**4. 命令行提问**（同样在 `script/` 下）：

```bash
cd script
python cli.py 你的问题一句话
python cli.py              # 交互模式，空行结束
python cli.py --init 任意  # 强制重建索引后再问
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
| POST | `/chat` | JSON：`{"query": "..."}` → `{"answer": "..."}` |
| POST | `/init` | 全量重建索引，返回 `ok`、`files`、`chunks` |
| GET | `/health` | `{"status":"ok"}` |

---

## 刻意未实现（第一版）

- PDF/Docx 自动解析、引用溯源、异步大批量入库、独立配置模块等。

---

## 远程仓库

示例：`https://github.com/chl-5g/AI_AGENT_TEST.git`

```bash
git add -A
git commit -m "你的说明"
git push origin main
```
