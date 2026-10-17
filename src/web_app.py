
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DB_PATH = PROJECT_ROOT / "data" / "chroma"
URLS_FILE = PROJECT_ROOT / "data" / "urls.txt"

INDEX_SCRIPT = PROJECT_ROOT / "src" / "index_documents.py"
QUERY_SCRIPT = PROJECT_ROOT / "src" / "query_documents.py"

DOCUMENT_TYPES = {".txt", ".pdf"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}

for directory in [DOCUMENTS_DIR, IMAGES_DIR, DB_PATH]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_record_count():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(name="documents")
    return collection.count()


# --------------------------------------------------
# File and URL handling
# --------------------------------------------------

def save_uploaded_files(uploaded_files):
    saved = []
    skipped = []

    for uploaded_file in uploaded_files:
        filename = (
            uploaded_file.name
            .replace("\\", "/")
            .split("/")[-1]
        )

        suffix = Path(filename).suffix.lower()

        if (
            not filename
            or filename in (".", "..")
            or suffix not in DOCUMENT_TYPES | IMAGE_TYPES
        ):
            skipped.append(filename or "Unnamed file")
            continue

        destination = (
            DOCUMENTS_DIR / filename
            if suffix in DOCUMENT_TYPES
            else IMAGES_DIR / filename
        )

        if destination.exists():
            skipped.append(filename)
            continue

        try:
            with destination.open("xb") as file:
                file.write(uploaded_file.getbuffer())
        except FileExistsError:
            skipped.append(filename)
            continue

        saved.append(filename)

    return saved, skipped


def save_website_url(url):
    url = url.strip()

    if not url:
        return None

    parsed = urlparse(url)

    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Enter a valid HTTP or HTTPS webpage URL.")

    existing_urls = []

    if URLS_FILE.exists():
        existing_urls = [
            line.strip()
            for line in URLS_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    if url in existing_urls:
        return False

    with URLS_FILE.open("a", encoding="utf-8") as file:
        if URLS_FILE.stat().st_size > 0:
            file.write("\n")
        file.write(url + "\n")

    return True


# --------------------------------------------------
# Existing RAG scripts
# --------------------------------------------------

def run_indexer():
    return subprocess.run(
        [sys.executable, str(INDEX_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )


def run_query(question):
    return subprocess.run(
        [sys.executable, str(QUERY_SCRIPT)],
        input=question + "\n",
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


def parse_query_output(output):
    if "Answer:" not in output:
        return output.strip(), []

    answer_section = output.split("Answer:", 1)[1]

    if "Retrieved sources:" in answer_section:
        answer_text, sources_text = answer_section.split(
            "Retrieved sources:", 1
        )
    else:
        answer_text = answer_section
        sources_text = ""

    sources = []

    for line in sources_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            sources.append(line[2:])

    return answer_text.strip(), sources


# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Local RAG Search",
    page_icon="🔎",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("Knowledge Base")

    try:
        st.metric("Indexed records", get_record_count())
    except Exception as error:
        st.error(f"Database error: {error}")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents or images",
        type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    website_url = st.text_input(
        "Webpage URL",
        placeholder="https://example.com/article",
    )

    index_clicked = st.button(
        "Index content",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Ollama + ChromaDB")


# --------------------------------------------------
# Indexing
# --------------------------------------------------

if index_clicked:
    if not uploaded_files and not website_url.strip():
        st.warning("Upload a file or enter a webpage URL.")
    else:
        try:
            saved_files = []
            skipped_files = []

            if website_url.strip():
                url_added = save_website_url(website_url)
            else:
                url_added = None

            if uploaded_files:
                saved_files, skipped_files = save_uploaded_files(
                    uploaded_files
                )

            if saved_files:
                st.success(
                    f"Saved {len(saved_files)} new file(s)."
                )
                for filename in saved_files:
                    st.write(f"• {filename}")

            if skipped_files:
                st.info("Files already present or unsupported:")
                for filename in skipped_files:
                    st.write(f"• {filename}")

            if url_added is True:
                st.success("Webpage URL added.")
            elif url_added is False:
                st.info("Webpage URL already exists.")

            with st.spinner("Indexing knowledge sources..."):
                result = run_indexer()

            if result.returncode == 0:
                st.success("Indexing completed successfully!")
                st.metric(
                    "Total indexed records",
                    get_record_count(),
                )

                with st.expander("Indexing details"):
                    st.code(result.stdout, language="text")
            else:
                st.error("Indexing failed.")
                with st.expander("Error details", expanded=True):
                    st.code(
                        result.stdout + "\n" + result.stderr,
                        language="text",
                    )

        except subprocess.TimeoutExpired:
            st.error("Indexing exceeded the time limit.")
        except Exception as error:
            st.error(f"Indexing error: {error}")


# --------------------------------------------------
# Main chat interface
# --------------------------------------------------

st.title("🔎 Local RAG Search")
st.caption("Chat with your documents using locally running AI.")

if not st.session_state.messages:
    st.info(
        "Ask a question about your indexed documents "
        "to begin a conversation."
    )

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander(
                    f"Retrieved sources ({len(sources)})"
                ):
                    for source in sources:
                        st.text(f"• {source}")


# Chat input stays at the bottom of the page
question = st.chat_input(
    "Ask a question about your documents..."
)

if question:
    # Add and display the user's message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        try:
            if get_record_count() == 0:
                answer = (
                    "Your knowledge base is empty. "
                    "Please index some documents first."
                )
                sources = []

            else:
                with st.spinner(
                    "Searching your knowledge base..."
                ):
                    result = run_query(question)

                if result.returncode != 0:
                    answer = (
                        "The search could not be completed. "
                        "Check that Ollama is running."
                    )
                    sources = []

                    with st.expander("Error details"):
                        st.code(
                            result.stdout + "\n" + result.stderr,
                            language="text",
                        )

                else:
                    answer, sources = parse_query_output(
                        result.stdout
                    )

                    if not answer:
                        answer = (
                            "The query completed without "
                            "a readable answer."
                        )

            st.markdown(answer)

            if sources:
                with st.expander(
                    f"Retrieved sources ({len(sources)})"
                ):
                    for source in sources:
                        st.text(f"• {source}")

            # Save the answer for future Streamlit reruns
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except subprocess.TimeoutExpired:
            answer = (
                "Question answering took too long. "
                "Check that Ollama is running."
            )
            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
            })

        except Exception as error:
            answer = f"Search error: {error}"
            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
            })
