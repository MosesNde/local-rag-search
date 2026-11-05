#!/bin/bash
# Launch LUCY on macOS without reinstalling dependencies.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-lucy/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "LUCY has not been installed here. First run: bash setup-macos.sh"
  read -r -p "Press Enter to close..." _
  exit 1
fi
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if [[ -d /Applications/Ollama.app ]]; then
    open -a Ollama
  elif command -v ollama >/dev/null 2>&1; then
    nohup ollama serve >/tmp/lucy-ollama.log 2>&1 </dev/null &
  else
    echo "Ollama is missing. Run: bash setup-macos.sh"
    read -r -p "Press Enter to close..." _
    exit 1
  fi
fi
for i in {1..30}; do
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
  sleep 1
done
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama did not start. Open Ollama and try again."
  read -r -p "Press Enter to close..." _
  exit 1
fi
if curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  open http://localhost:8501
  exit 0
fi
(open http://localhost:8501 >/dev/null 2>&1) &
exec "$PYTHON" -m streamlit run "$ROOT/src/web_app.py" --server.address 127.0.0.1
