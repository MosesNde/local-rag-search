
import os


# --------------------------------------------------
# Ollama configuration
# --------------------------------------------------

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

ANSWER_MODEL = os.getenv(
    "ANSWER_MODEL",
    "qwen3:8b",
)

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen3-vl:8b",
)

VISION_CONTEXT_SIZE = int(
    os.getenv("VISION_CONTEXT_SIZE", "8192")
)

VISION_MAX_RETRIES = int(
    os.getenv("VISION_MAX_RETRIES", "1")
)
