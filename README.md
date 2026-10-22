# Local RAG Search Engine

A locally running **retrieval-augmented generation (RAG)** search application built with Python, [Ollama](https://ollama.com/), [ChromaDB](https://www.trychroma.com/), and [Streamlit](https://streamlit.io/).

Index local TXT/PDF documents, images, and webpages, then ask questions using a local language model. The system supports source-aware table lookup, PDF figure descriptions, and neighboring-chunk context.

> **Status:** Functional research/development prototype. Not production-hardened; retrieved sources do not guarantee that every generated claim is correct.

## Features

- **Inputs:** TXT, PDF, PNG, JPG/JPEG, WebP, and webpages.
- **Local models:** `nomic-embed-text` embeddings, `qwen3:8b` answers, and `qwen3-vl:8b` image descriptions (configurable).
- **Persistent knowledge base:** ChromaDB collection `documents` in `data/chroma/`.
- **PDF support:** Text extraction, figure cropping/description/validation, and structured table extraction and validation.
- **Source-aware table references:** Supports both newer numeric references (such as Table 3) and legacy Roman numerals (such as Table II).
- **Exact table operations:** Certain structured table listings and numerical rankings are handled directly in Python rather than reconstructed by a language model.
- **Neighboring-chunk retrieval:** Adds nearby text passages when a relevant explanation spans several chunks.
- **Interfaces:** Streamlit web app and command-line questions.
- **Diagnostics:** Regression testing, retrieval evaluation, and optional profiling.

Sources can include filenames, pages, table/figure numbers, and chunk references. They are not yet verified claim-level citations.

## Architecture

### Indexing

```text
TXT/PDF text and webpages ---> Text extraction and chunking ----┐
Images ---------------------> Vision-generated descriptions ----┤
PDF figures ----------------> Crop, describe, validate ----------┤
PDF tables -----------------> Extract, structure, validate ------┤
                                                                 v
                                                        nomic-embed-text
                                                                 |
                                                                 v
                                                        ChromaDB documents
```

### Question answering

```text
Question
   |
   +-- Explicit table/figure reference? --> Source-aware lookup
   |                                        (some exact table answers in Python)
   |
   +-- Otherwise --> Embedding --> Semantic search
                                      |
                                      v
                              Neighboring text chunks
   |
   v
Retrieved evidence --> qwen3:8b --> Answer and retrieved sources
```

## Requirements

- Python and `pip` (development used Python 3.14; other versions have not been comprehensively tested).
- [Ollama](https://ollama.com/) installed and running.
- Sufficient system memory for the selected models.
- Internet connectivity for initial downloads and webpage indexing.

Pull the configured models:

```powershell
ollama pull nomic-embed-text
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
```

Image and PDF-figure processing can be substantially more resource-intensive than text retrieval.

## Installation

Clone the repository:

```powershell
git clone https://github.com/MosesNde/local-rag-search.git
cd local-rag-search
```

On **Windows PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On **macOS or Linux**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start Ollama before using the application. The default setup assumes Ollama is available locally.

### Model configuration

`src/config.py` reads these environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint setting |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Text embeddings |
| `ANSWER_MODEL` | `qwen3:8b` | Answer generation |
| `VISION_MODEL` | `qwen3-vl:8b` | Image/figure descriptions |
| `VISION_CONTEXT_SIZE` | `8192` | Vision context configuration |
| `VISION_MAX_RETRIES` | `1` | Vision retry configuration |

## Add content

| Content | Local path |
| --- | --- |
| TXT and PDF files | `data/documents/` |
| PNG, JPG/JPEG, WebP images | `data/images/` |
| Webpage URLs (one per line) | `data/urls.txt` |

Example `data/urls.txt`:

```text
https://docs.python.org/3/tutorial/introduction.html
```

Run the main indexer from the project root:

```powershell
python .\src\index_documents.py
```

The indexer creates local data directories and stores records in the persistent ChromaDB collection `documents`. Sources already indexed are normally skipped; modified and deleted source files are **not** automatically synchronized.

### PDF table and figure pipelines

The codebase includes dedicated modules:

- **Figures:** `src/pdf_figures.py`, `src/pdf_pipeline.py`, and `src/pdf_validation.py` detect/crop, describe, and validate PDF figures.
- **Tables:** `src/pdf_tables.py`, `src/pdf_table_pipeline.py`, and `src/pdf_table_validation.py` extract and validate structured records.
- **Legacy/specialized scripts:** `src/index_pdf_figures.py` and `src/index_pdf_tables.py` were developed for earlier indexing workflows; use the integrated pipeline for ordinary indexing.

For an individual PDF figure-processing run:

```powershell
python .\src\pdf_pipeline.py --pdf example.pdf
```

For table extraction/validation **without ChromaDB writes**:

```powershell
python .\src\pdf_table_pipeline.py example.pdf
```

For explicit table indexing:

```powershell
python .\src\pdf_table_pipeline.py example.pdf --index
```

Place `example.pdf` in `data/documents/` or supply a valid path. Extraction and validation are heuristic and do not guarantee perfect correspondence with the rendered PDF.

## Web interface

Start Streamlit:

```powershell
python -m streamlit run src/web_app.py
```

Open **http://localhost:8501** if the browser does not open automatically. The UI supports uploads, URL entry, indexing, questions, expandable retrieved sources, and clearing session chat history.

On Windows, double-click `START_APP.vbs` to launch without a visible terminal (requires `.venv`). Use `STOP_APP.vbs` to request shutdown.

The web interface is intended for **trusted local use**, not exposure directly to the public internet.

## Command-line questions

```powershell
python .\src\query_documents.py
```

Example questions (which require the relevant documents to have been indexed):

```text
What is retrieval-augmented generation?
According to Table II of the OWASP vulnerability dataset paper, list all OWASP categories and their most affected programming languages.
What does Figure 3 in agentchaos.pdf show?
```

When sufficient evidence exists, the output includes an answer and source references. For some exact structured table requests, Python produces the answer without an LLM call.

## Evaluation and diagnostics

Run the regression tests:

```powershell
python .\src\test_rag_regression.py --live
```

At the October 2026 development checkpoint, **3 offline tests and 4 live tests passed** using the development machine's existing indexed corpus. Live tests depend on local indexed documents and models; this is not a general benchmark of RAG accuracy.

Evaluate evidence retrieval:

```powershell
python .\src\evaluate_retrieval_quality.py
```

Generated reports are saved under `data/retrieval_tests/`. Some evaluator tests use generic semantic retrieval, so their scores are not equivalent to testing the application's full table-aware route.

Enable optional per-stage profiling:

```powershell
$env:RAG_PROFILE = "1"
python .\src\query_documents.py
```

## Privacy and data handling

Ollama inference and ChromaDB indexing/storage operate locally by default. Downloading packages/models and indexing webpages require external network access. Only retrieve and index content you are authorized to process.

Local documents, images, ChromaDB records, extracted figures/tables, and generated reports are excluded from Git using `.gitignore`. **Cloning this repository does not download the development corpus or its database.**

## Known limitations

- PDF layouts, figures, tables, and vision-generated descriptions may be incomplete or inaccurate.
- Structural validation does not certify every extracted cell against the original PDF.
- Chunking is simple; neighboring-chunk expansion reduces but does not eliminate missing-context problems.
- Grounded answers can still be incomplete or incorrect; source listings are not verified per claim.
- Ambiguous table references across documents may require a clearer source name.
- Existing ChromaDB records are not automatically updated when source files change or are removed.
- Local model inference may be slow on CPU-only machines.
- Chat messages are displayed within the Streamlit session but are not used as conversational retrieval context.

## Roadmap

- Claim-level citation/evidence attribution.
- Safer refresh and deletion of outdated indexed records.
- More figure/table regression coverage and multi-document retrieval tests.
- Stronger evaluations of factual answer quality.
- Improved layout extraction and ambiguity handling.

## Project status

**Functional research prototype:** CLI and Streamlit access, local multimodal indexing, structured PDF retrieval, and automated regression tests.
