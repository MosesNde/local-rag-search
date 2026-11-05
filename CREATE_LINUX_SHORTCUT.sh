#!/usr/bin/env bash
# Run once after Linux setup, from any directory.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP="$HOME/Desktop"
mkdir -p "$DESKTOP"
# Spaces and shell metacharacters in paths are quoted safely via printf %q.
printf -v START '%q' "$ROOT/START_APP.sh"
cat > "$DESKTOP/LUCY.desktop" <<DESKTOP_ENTRY
[Desktop Entry]
Version=1.0
Type=Application
Name=LUCY
Comment=Open local RAG search
Exec=bash -c "$START"
Path=$ROOT
Terminal=true
Categories=Utility;
DESKTOP_ENTRY
chmod +x "$DESKTOP/LUCY.desktop" "$ROOT/START_APP.sh"
echo "Created $DESKTOP/LUCY.desktop"
echo "On Ubuntu you may need to right-click it and select Allow Launching."
