"""
cli.py — 命令行向 RAG 客服提问（无需浏览器、无需先开 uvicorn）

用法示例（在 **script/** 目录执行；**.env 在仓库根**，与 data/ 同级）：

    一次性提问（整句作为参数，多个词会拼成一句）：
        python cli.py 需求变更要走什么流程？

    交互模式（多轮输入，单独一行空行结束）：
        python cli.py

    强制按当前 data/ 全量重建索引后再问：
        python cli.py --init 今天周几？

    禁止在空库时自动入库（若从未启动过服务建索引，后续 chat 可能失败）：
        python cli.py --no-auto-init

行为说明：
    - 默认若检测到本地 Chroma 集合不存在或 count()==0，会自动执行一次 initialize()，
      便于只开 CMD、不启动 FastAPI 也能直接提问。
    - 若你希望「索引必须由服务启动或显式 --init 建立」，可传 --no-auto-init。
    - main() 开头 load_project_env()：与 main.py 一致，从项目根加载 .env。
"""

from __future__ import annotations

import argparse
import sys

try:
    from .bot import chat
    from .ingestion import has_indexed_documents, initialize, load_project_env
except ImportError as e:
    # 仅在“直接执行 script/cli.py（无包上下文）”时回退到旧导入方式；
    # 若是依赖缺失（如 chromadb 未安装）应保留原始异常，避免误导。
    if __package__ in (None, "") and "attempted relative import" in str(e):
        from bot import chat
        from ingestion import has_indexed_documents, initialize, load_project_env
    else:
        raise


def _configure_stdio_utf8() -> None:
    """
    Windows 控制台默认编码常为 GBK/系统 ANSI，不处理时 print 中文或读入参数可能乱码。
    将 stdout/stderr 设为 UTF-8，与 ingestion 中 read_text(encoding='utf-8') 一致。
    （data/ 文件读取已在 ingestion 显式 utf-8；此处负责终端输出侧。）
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _ensure_index(*, force: bool, auto: bool) -> None:
    """
    保证在调用 bot.chat 之前索引可用。

    :param force: True 时无论当前是否有数据，都执行 initialize()（删除旧集合、
        扫描 data/、重新嵌入并写入）。对应命令行 --init。
    :param auto: False 时若索引为空也不自动建库；用户会可能在 chat 时收到 Chroma
        或 RuntimeError。对应 --no-auto-init。

    控制台信息输出到 stderr，便于管道只捕获「模型回答」（stdout）。
    """
    if force:
        stats = initialize()
        print("索引已重建:", stats, file=sys.stderr)
        return
    if not auto:
        return
    if has_indexed_documents():
        return
    stats = initialize()
    print("已自动建立索引（首次或空库）:", stats, file=sys.stderr)


def _run_query(text: str) -> None:
    """
    调用 bot.chat(text) 并将回答打印到标准输出。

    空串或仅空白：忽略并提示 stderr，避免无意义 API 调用。
    """
    text = text.strip()
    if not text:
        print("（空问题已忽略）", file=sys.stderr)
        return
    print(chat(text))


def main(argv: list[str] | None = None) -> int:
    _configure_stdio_utf8()
    # 与 FastAPI 入口一致：优先从项目根加载 .env（路径由 ingestion.PROJECT_ROOT 决定）
    load_project_env()

    parser = argparse.ArgumentParser(description="命令行向 RAG 客服提问")
    parser.add_argument(
        "--init",
        action="store_true",
        help="强制根据 data/ 重建 Chroma 索引后再处理提问",
    )
    parser.add_argument(
        "--no-auto-init",
        action="store_true",
        help="禁止在索引为空时自动入库（若未先启动过服务或建过索引，chat 可能失败）",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="问题正文；省略则进入交互模式，在 CMD 里多行输入直至空行退出",
    )
    args = parser.parse_args(argv)

    auto_init = not args.no_auto_init
    try:
        _ensure_index(force=args.init, auto=auto_init)
    except Exception as e:
        print("建立索引失败:", e, file=sys.stderr)
        return 1

    # Windows CMD：python cli.py 我 的 问题 → 多个 argv 拼成一句
    joined = " ".join(args.query).strip()
    if joined:
        _run_query(joined)
        return 0

    # 交互：每行一问，单独空行退出；Ctrl+Z 后回车（Windows）触发 EOFError 同样退出
    print("输入问题后回车；仅回车（空行）退出。Ctrl+Z 后回车（Windows）也可结束。", file=sys.stderr)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        _run_query(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
