#!/usr/bin/env bash
# One-command source installation and launch for macOS (Apple Silicon / Intel).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS. On Windows, run setup-windows.ps1." >&2
  exit 1
fi

log() { printf '\n[LUCY] %s\n' "$*"; }
fail() { printf '\n[LUCY] ERROR: %s\n' "$*" >&2; exit 1; }

# Do not install into the system Python or overwrite existing .venv.
if ! command -v brew >/dev/null 2>&1; then
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  else
    log "Homebrew is required to install Python and Ollama."
    log "The official Homebrew installer may request administrator approval."
    read -r -p "Install Homebrew now? [y/N] " approve
    [[ "$approve" =~ ^[Yy]$ ]] || fail "Setup cancelled; nothing was removed."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ -x /opt/homebrew/bin/brew ]]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -x /usr/local/bin/brew ]]; then
      eval "$(/usr/local/bin/brew shellenv)"
    else
      fail "Homebrew installed but isn't accessible; follow the installer's PATH instructions, then rerun."
    fi
  fi
fi

log "Checking Python 3.12"
if ! command -v python3.12 >/dev/null 2>&1; then
  brew install python@3.12
fi
command -v python3.12 >/dev/null 2>&1 || fail "Python 3.12 isn't accessible after installation."

VENV="$ROOT/.venv-lucy"
if [[ -f "$VENV/pyvenv.cfg" ]] && ! grep -qi 'version = 3\.12\.' "$VENV/pyvenv.cfg"; then
  fail ".venv-lucy exists but is not Python 3.12. Rename it yourself, then rerun."
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating isolated Python 3.12 environment"
  python3.12 -m venv "$VENV"
fi

log "Installing LUCY Python dependencies"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

if ! command -v ollama >/dev/null 2>&1; then
  log "Installing Ollama for macOS"
  brew install --cask ollama
fi
command -v ollama >/dev/null 2>&1 || fail "Ollama isn't on PATH. Open a new terminal and rerun setup."

if ! curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  log "Starting Ollama"
  if [[ -d /Applications/Ollama.app ]]; then
    open -a Ollama
  else
    (nohup ollama serve > /tmp/lucy-ollama.log 2>&1 < /dev/null &)
  fi
fi

log "Checking Ollama server"
ready=0
for ((i=0; i<45; i++)); do
  if curl -fsS --max-time 3 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]] || fail "Ollama didn't start. Open the Ollama app, then rerun setup."

log "Downloading any missing models (first run can download several gigabytes)"
for model in nomic-embed-text qwen3:8b qwen3-vl:8b; do
  log "Checking model: $model"
  ollama pull "$model"
done

log "Starting LUCY at http://localhost:8501"
exec "$VENV/bin/python" -m streamlit run "$ROOT/src/web_app.py"
