
"""Shared application paths for development and packaged Windows builds."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_data_root():
    """Return a persistent location for application data."""

    # Explicit override, useful for testing and future installers.
    override = os.environ.get("LOCAL_RAG_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Installed executable: use the current user's local AppData.
    if getattr(sys, "frozen", False):
        local_appdata = os.environ.get("LOCALAPPDATA")

        if local_appdata:
            base = Path(local_appdata)
        else:
            base = Path.home() / "AppData" / "Local"

        return base / "LocalRAGSearch" / "data"

    # Development: preserve the existing repository data directory.
    return PROJECT_ROOT / "data"


DATA_ROOT = get_data_root()


# Indexed source material
DOCUMENTS_DIR = DATA_ROOT / "documents"
IMAGES_DIR = DATA_ROOT / "images"

# Persistent databases
DB_PATH = DATA_ROOT / "chroma"
CHAT_DB_PATH = DATA_ROOT / "chat_history.sqlite3"

# Webpage indexing
URLS_FILE = DATA_ROOT / "urls.txt"

# PDF extraction and validation
FIGURES_DIR = DATA_ROOT / "figures"
FIGURE_TESTS_DIR = DATA_ROOT / "figure_tests"
TABLE_TESTS_DIR = DATA_ROOT / "table_tests"

# Evaluation and generated outputs
RETRIEVAL_TESTS_DIR = DATA_ROOT / "retrieval_tests"
OUTPUTS_DIR = DATA_ROOT / "outputs"
