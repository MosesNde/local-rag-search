## One-command source setup (Windows, macOS, and Ubuntu Linux)

The repository includes an assisted first-run setup that installs a compatible
Python environment and required AI models, starts Ollama, and launches LUCY.
An internet connection and enough disk space for multiple large model downloads
are needed the first time. The setup may ask for OS administrator approval.

**Windows 10/11 (PowerShell; winget required):**

```powershell
git clone https://github.com/MosesNde/local-rag-search.git
cd local-rag-search
powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1
```

**macOS (Terminal):**

```bash
git clone https://github.com/MosesNde/local-rag-search.git
cd local-rag-search
bash setup-macos.sh
```

On macOS, the setup can install Homebrew after asking for permission; it then
installs Python 3.12 and the Ollama app if needed. On Windows, it uses the
Windows Package Manager (`winget`) to install Python 3.14 and Ollama if missing.
Both platforms install `requirements.txt` in `.venv-lucy`, pull the
`nomic-embed-text`, `qwen3:8b` and `qwen3-vl:8b` Ollama models, and launch
Streamlit at <http://localhost:8501>.

Re-run the same setup command on future launches. Already installed Python
packages and Ollama models will be reused; Ollama may check for model updates.

**Existing user data is preserved:** the setup never removes `data/chroma/`,
`data/documents/`, `data/images/`, or SQLite chat history. It also does not
modify or overwrite an existing `.venv` or `.venv312`. The new environment is
`.venv-lucy`.

**Limitations:** This is a source-install experience, not a signed macOS app or
Windows installer. Security restrictions, unavailable Homebrew/winget services,
unsupported hardware, or insufficient storage can still require user action.
The full 8B models can be slow on CPU-only computers. The installer scripts
should be tested on clean machines before claiming universal compatibility.


**Ubuntu 24.04 LTS (Terminal; x86-64 or ARM64):**

```bash
git clone https://github.com/MosesNde/local-rag-search.git
cd local-rag-search
bash setup-linux.sh
```

The Linux setup installs required dependencies, configures Python 3.12,
creates .venv-lucy, starts Ollama, downloads missing AI models,
and launches LUCY at http://localhost:8501.

Linux support currently targets Ubuntu 24.04 LTS.
Other distributions have not been validated.

Existing documents, indexes, and chat history are preserved.

### Windows Python environment

Windows setup uses Python 3.14 and creates an isolated
environment named .venv-lucy.

If .venv-lucy already exists with another Python version,
the script preserves it and uses .venv-lucy-314 instead.

The setup has been tested successfully on Windows 11
with Python 3.14.
