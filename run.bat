@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo 未找到 python，请先安装 Python 3.10+ 并勾选「Add to PATH」
    pause
    exit /b 1
)

if not exist ".venv" (
    echo ^>^>^> 创建虚拟环境 .venv ...
    python -m venv .venv
)

call "%~dp0.venv\Scripts\activate.bat"
if errorlevel 1 (
    echo 无法激活虚拟环境，请检查 .venv 是否完整
    pause
    exit /b 1
)

python -c "import chromadb, openai, fastapi, uvicorn, dotenv" 2>nul
if errorlevel 1 (
    echo ^>^>^> 当前 .venv 里还缺包，正在 pip install -r requirements.txt ...
    echo     （若你装在「系统 Python」，.venv 仍是空的，必须装这一遍；首次可能较慢）
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    echo ^>^>^> 依赖已在 .venv 中就绪，跳过 pip。更新依赖后可删除 .venv 再运行本脚本。
)

if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo ^>^>^> 已生成 .env，请用编辑器打开本目录下的 .env 填入 API Key 后，再次双击或运行 run.bat
        pause
        exit /b 0
    ) else (
        echo 缺少 .env 与 .env.example
        pause
        exit /b 1
    )
)

set PYTHONUTF8=1
cd /d "%~dp0"
echo ^>^>^> 服务地址 http://127.0.0.1:8000  （Ctrl+C 停止）
python -m script.deploy --reload --host 127.0.0.1 --port 8000

endlocal
