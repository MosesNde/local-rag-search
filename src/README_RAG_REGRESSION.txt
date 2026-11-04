LOCAL RAG SEARCH ENGINE — CORE REGRESSION TESTS

Installation
1. Copy test_rag_core_regression.py into C:\Users\yirim\local-rag-search\src\
2. Activate the existing .venv and run from the project root:
   python src\test_rag_core_regression.py

This tests current source functions without running Streamlit or connecting to
Ollama, ChromaDB, or SQLite. No existing file or database needs replacement.

Tests:
- LangGraph new-topic, follow-up, and recall dispatch with a fake RAG backend
- Preservation of full recall_history and citations across the graph
- Cumulative memory threshold, append-only segments, message numbers
- No watermark advancement if Ollama fails or summary capacity is exceeded
- Example-request recognition
- Separate hypothetical / reasoning / recall and document-evidence labels

Mocked behavior is not a substitute for end-to-end verification of actual
model responses, document retrieval quality, PDF highlighting, or database
persistence. Those need separate integration or live tests.

Any failing test produces a nonzero exit code. The LangGraph test is skipped
if LangGraph is not installed in the active Python environment.
