#!/usr/bin/env bash
# Launch LUCY on Ubuntu Linux without reinstalling dependencies.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-lucy/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "LUCY is not installed here. First run: bash setup-linux.sh" >&2
  exit 1
fi
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if command -v systemctl >/dev/null 2>&1 && systemctl --user list-unit-files ollama.service --no-legend 2>/dev/null | grep -q ollama.service; then
    systemctl --user start ollama || true
  fi
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    command -v ollama >/dev/null 2>&1 || { echo 'Ollama is missing; run setup-linux.sh' >&2; exit 1; }
    nohup ollama serve >/tmp/lucy-ollama.log 2>&1 </dev/null &
  fi
fi
ready=0
for ((i=0; i<30; i++)); do
  if curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
[[ "$ready" == 1 ]] || { echo 'Ollama did not start' >&2; exit 1; }
if curl -fsS --max-time 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  if command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:8501 >/dev/null 2>&1 || true; fi
  exit 0
fi
if command -v xdg-open >/dev/null 2>&1; then (sleep 3; xdg-open http://localhost:8501 >/dev/null 2>&1 || true) & fi
exec "$PYTHON" -m streamlit run "$ROOT/src/web_app.py" --server.address 127.0.0.1
