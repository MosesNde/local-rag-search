
#!/usr/bin/env bash
# LUCY source setup for Ubuntu 24.04 LTS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

log() {
  printf '\n[LUCY] %s\n' "$*"
}

fail() {
  printf '\n[LUCY] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "This script requires Linux."
fi

if [[ ! -f /etc/os-release ]]; then
  fail "Cannot identify the Linux distribution."
fi

# Identify supported Ubuntu release.
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  fail "Supported Linux release: Ubuntu 24.04 LTS. Detected: ${PRETTY_NAME:-unknown}."
fi

case "$(uname -m)" in
  x86_64|aarch64) ;;
  *) fail "Supported architectures: x86_64 and aarch64." ;;
esac

command -v apt-get >/dev/null 2>&1 ||
  fail "apt-get was not found."

if [[ "$EUID" -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null 2>&1 ||
    fail "sudo is needed for system package installation."
  SUDO=(sudo)
fi

log "Checking system dependencies"

packages=()

command -v curl >/dev/null 2>&1 ||
  packages+=(curl)

command -v python3.12 >/dev/null 2>&1 ||
  packages+=(python3.12)

dpkg-query -W -f='${Status}' python3.12-venv 2>/dev/null |
  grep -q 'install ok installed' ||
  packages+=(python3.12-venv)

if ((${#packages[@]} > 0)); then
  log "Installing missing Ubuntu packages"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" apt-get install -y "${packages[@]}"
fi

command -v python3.12 >/dev/null 2>&1 ||
  fail "Python 3.12 is unavailable."

# Never overwrite the user's existing .venv.
VENV="$ROOT/.venv-lucy"

if [[ -f "$VENV/pyvenv.cfg" ]]; then
  if ! grep -Eq '^version = 3\.12\.' "$VENV/pyvenv.cfg"; then
    fail ".venv-lucy uses another Python version. Rename it manually."
  fi
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  log "Creating Python 3.12 environment"
  python3.12 -m venv "$VENV"
fi

log "Installing Python dependencies"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

# Install Ollama if missing, using the official installer.
if ! command -v ollama >/dev/null 2>&1; then
  log "Ollama is not installed."
  log "The official installer may require administrator privileges."

  if [[ ! -t 0 ]]; then
    fail "Interactive approval is required to install Ollama."
  fi

  read -r -p "Install Ollama now? [y/N] " approve

  [[ "$approve" =~ ^[Yy]$ ]] ||
    fail "Setup cancelled."

  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT

  curl -fsSL https://ollama.com/install.sh -o "$installer"
  sh "$installer"

  rm -f "$installer"
  trap - EXIT
fi

command -v ollama >/dev/null 2>&1 ||
  fail "Ollama installation completed but its command was not found."

ollama_ready() {
  curl -fsS --max-time 3 \
    http://127.0.0.1:11434/api/tags \
    >/dev/null 2>&1
}

if ! ollama_ready; then
  log "Starting Ollama"

  if command -v systemctl >/dev/null 2>&1 &&
     systemctl list-unit-files ollama.service --no-legend 2>/dev/null |
       grep -q 'ollama.service'; then
    "${SUDO[@]}" systemctl start ollama
  else
    nohup ollama serve \
      > /tmp/lucy-ollama.log 2>&1 \
      < /dev/null &
  fi
fi

log "Waiting for Ollama server"

ready=0

for ((i=0; i<45; i++)); do
  if ollama_ready; then
    ready=1
    break
  fi
  sleep 2
done

[[ "$ready" -eq 1 ]] ||
  fail "Ollama did not start. Check its service or startup log."

log "Checking required AI models"

for model in nomic-embed-text qwen3:8b qwen3-vl:8b; do
  log "Checking model: $model"

  if ! ollama list | awk 'NR>1 {print $1}' |
       grep -Fxq "$model"; then
    log "Downloading $model"
    ollama pull "$model"
  else
    log "$model is already installed"
  fi
done

log "Starting LUCY"
log "Open http://localhost:8501"

exec "$VENV/bin/python" -m streamlit run \
  "$ROOT/src/web_app.py"
