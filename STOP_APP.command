#!/bin/bash
# Double-click in Finder after one-time chmod +x STOP_APP.command
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd -P)"
PID_FILE="$ROOT/.run/streamlit.pid"
APP="$ROOT/src/web_app.py"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No launcher-owned Local RAG process recorded."
  exit 0
fi
pid="$(cat "$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID file; refusing to stop any process."
  exit 1
fi
command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
if [[ -z "$command" ]]; then
  echo "Local RAG is not running. Cleaning stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi
if [[ "$command" != *"streamlit"* || "$command" != *"$APP"* ]]; then
  echo "PID $pid is not this project's Streamlit app; refusing to stop it."
  exit 1
fi
kill -TERM "$pid" || exit 1
for ((i=0;i<20;i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Local RAG stopped."
    exit 0
  fi
  sleep 0.25
done
echo "Stop signal sent. Process is still shutting down; no other processes were affected."
exit 0
