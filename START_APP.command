#!/bin/bash
# Double-click in Finder after one-time chmod +x START_APP.command
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd -P)"
PY="$ROOT/.venv/bin/python"
APP="$ROOT/src/web_app.py"
RUN_DIR="$ROOT/.run"
PID_FILE="$RUN_DIR/streamlit.pid"
LOG_FILE="$RUN_DIR/streamlit.log"

if [[ ! -x "$PY" ]]; then
  echo "Missing .venv/bin/python. Create the macOS virtual environment and install dependencies first."
  read -r -p "Press Return to close..." _
  exit 1
fi
if [[ ! -f "$APP" ]]; then
  echo "Missing src/web_app.py"
  read -r -p "Press Return to close..." _
  exit 1
fi
mkdir -p "$RUN_DIR"
is_our_process() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  local command
  command="$(ps -p "$pid" -o command= 2>/dev/null)" || return 1
  [[ "$command" == *"streamlit"* && "$command" == *"$APP"* ]]
}
if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if is_our_process "$old_pid"; then
    echo "Local RAG is already running (PID $old_pid)."
    open "http://localhost:8501"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

# Never silently pick a different port or open another process's site.
if ! "$PY" -c 'import socket; s=socket.socket(); x=s.connect_ex(("127.0.0.1",8501)); s.close(); raise SystemExit(0 if x != 0 else 1)'; then
  echo "Port 8501 is already in use. Stop the other service before starting Local RAG."
  read -r -p "Press Return to close..." _
  exit 1
fi

nohup "$PY" -m streamlit run "$APP" --server.headless true --server.port 8501 \
  > "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if is_our_process "$pid"; then
  echo "Local RAG started (PID $pid)."
  echo "Log: $LOG_FILE"
  open "http://localhost:8501"
  exit 0
fi
echo "The app failed to start. Last log lines:"
tail -n 25 "$LOG_FILE"
rm -f "$PID_FILE"
read -r -p "Press Return to close..." _
exit 1
