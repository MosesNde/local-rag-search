
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import chromadb
import ollama
import pymupdf
import requests
from bs4 import BeautifulSoup


# --------------------------------------------------
# Configuration
# --------------------------------------------------


from app_paths import (
    DOCUMENTS_DIR,
    IMAGES_DIR,
    DB_PATH,
    URLS_FILE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNK_SIZE = 200
OVERLAP = 1
BATCH_SIZE = 16

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


# --------------------------------------------------
# Incremental PDF stage tracking
# --------------------------------------------------
# Stored with the user's persistent app data, never in the bundled source.
# This tracks successfully executed figure/table passes, independently of
# whether the text was already indexed. Older PDFs get one initial pass.
PDF_STATE_FILE = DOCUMENTS_DIR.parent / "pdf_index_stage_state.json"
PDF_STATE_VERSION = 1

# Normal indexing stays fast: PDF text is still indexed, but the optional
# vision-heavy figure and table stages are explicitly enabled when needed.
# To process figures: $env:RAG_PROCESS_PDF_FIGURES = "1"
# To process tables:  $env:RAG_PROCESS_PDF_TABLES = "1"
def env_enabled(name, default="0"):
    return os.getenv(name, default).strip().lower() in (
        "1", "true", "yes", "on"
    )


PROCESS_PDF_FIGURES = env_enabled("RAG_PROCESS_PDF_FIGURES")
PROCESS_PDF_TABLES = env_enabled("RAG_PROCESS_PDF_TABLES")
RETRY_PDF_STAGES = os.getenv("RAG_RETRY_PDF_STAGES", "0").strip().lower() in (
    "1", "true", "yes", "on"
)


def pdf_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf_stage_state():
    if not PDF_STATE_FILE.exists():
        return {"version": PDF_STATE_VERSION, "pdfs": {}}
    try:
        data = json.loads(PDF_STATE_FILE.read_text(encoding="utf-8"))
        if (not isinstance(data, dict)
                or data.get("version") != PDF_STATE_VERSION
                or not isinstance(data.get("pdfs"), dict)):
            raise ValueError("Unexpected PDF state format or version")
        return data
    except (OSError, ValueError) as exc:
        # Fail safely. Do not overwrite an unreadable state or assume that
        # past processing is complete; preserve the existing database.
        raise RuntimeError(
            f"Could not read PDF stage state {PDF_STATE_FILE}: {exc}"
        ) from exc


def save_pdf_stage_state(state):
    PDF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = PDF_STATE_FILE.with_name(
        f"{PDF_STATE_FILE.name}.{os.getpid()}.tmp"
    )
    try:
        temp.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temp, PDF_STATE_FILE)
    finally:
        if temp.exists():
            temp.unlink()


def mark_pdf_stage_complete(state, pdf_path, digest, stage):
    record = state["pdfs"].setdefault(
        pdf_path.name, {"sha256": digest, "stages": {}}
    )
    record["stages"][stage] = True
    save_pdf_stage_state(state)


def pdf_stage_done(state, pdf_path, stage):
    record = state["pdfs"].get(pdf_path.name, {})
    return bool(record.get("stages", {}).get(stage))


def assert_pdf_not_replaced(state, pdf_path, digest, collection):
    record = state["pdfs"].get(pdf_path.name)
    if record and record.get("sha256") != digest:
        raise RuntimeError(
            f"PDF {pdf_path.name!r} has changed since its last processing pass. "
            "Existing ChromaDB records use the filename as their source key. "
            "To preserve them safely, add the revised PDF with a NEW filename "
            "instead of overwriting the existing document."
        )
    # On first migration, the current database has no content hashes. This
    # initializes tracking only; it does not claim previous stages completed.
    if not record:
        state["pdfs"][pdf_path.name] = {
            "sha256": digest, "stages": {}
        }
        save_pdf_stage_state(state)


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def create_chunks(text, chunk_size, overlap):
    sentences = text.replace("\n", " ").split(".")
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if (
            current_chunk
            and current_length + len(sentence) + 2 > chunk_size
        ):
            chunks.append(". ".join(current_chunk) + ".")

            overlap_sentences = (
                current_chunk[-overlap:] if overlap else []
            )

            current_chunk = overlap_sentences.copy()

            current_length = sum(
                len(sentence) + 2
                for sentence in current_chunk
            )

        current_chunk.append(sentence)
        current_length += len(sentence) + 2

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


# --------------------------------------------------
# PDF text extraction
# --------------------------------------------------

def clean_pdf_text(text):
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf_pages(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(
            pdf, start=1
        ):
            blocks = page.get_text("blocks")
            midpoint = page.rect.width / 2

            header = []
            left_column = []
            right_column = []
            spanning_blocks = []

            for block in blocks:
                if len(block) < 7:
                    continue

                x0, y0, x1, y1, text, _, block_type = block

                if block_type != 0:
                    continue

                if x0 < 30 or y0 > page.rect.height - 40:
                    continue

                if page_number == 1 and y0 < 150:
                    header.append((y0, text))

                elif x1 <= midpoint:
                    left_column.append((y0, text))

                elif x0 >= midpoint:
                    right_column.append((y0, text))

                else:
                    spanning_blocks.append((y0, text))

            header.sort()
            left_column.sort()
            right_column.sort()
            spanning_blocks.sort()

            ordered_blocks = (
                header
                + left_column
                + right_column
                + spanning_blocks
            )

            page_text = "\n\n".join(
                block_text.strip()
                for _, block_text in ordered_blocks
            )

            page_text = clean_pdf_text(page_text)

            if page_text:
                pages.append(
                    (page_number, page_text)
                )

    return pages


# --------------------------------------------------
# Webpage extraction
# --------------------------------------------------

def extract_webpage(url):
    response = requests.get(
        url,
        timeout=20,
    )
    response.raise_for_status()

    if (
        not response.encoding
        or response.encoding.lower() == "iso-8859-1"
    ):
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    main_content = (
        soup.select_one('[role="main"]')
        or soup.find("main")
        or soup.find("article")
        or soup
    )

    for element in main_content.select(
        "script, style, nav, footer"
    ):
        element.decompose()

    text = main_content.get_text(
        separator=" ",
        strip=True,
    )

    return re.sub(
        r"\s+", " ", text
    ).strip()


# --------------------------------------------------
# Image descriptions
# --------------------------------------------------

def describe_image(image_path):
    from config import VISION_MODEL

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe this image accurately for "
                    "a document search engine. Identify "
                    "visible text, objects, charts, values, "
                    "and important relationships. "
                    "Do not invent details."
                ),
                "images": [
                    str(image_path)
                ],
            }
        ],
        think=False,
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Record identification
# --------------------------------------------------

def create_chunk_id(
    source,
    page,
    chunk_number,
    file_type,
):
    source_hash = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    if file_type == "pdf":
        return (
            f"pdf_{source_hash}_"
            f"p{page}_chunk{chunk_number}"
        )

    return (
        f"{file_type}_{source_hash}_"
        f"chunk{chunk_number}"
    )


def source_type_already_indexed(
    collection,
    source,
    file_type,
):
    result = collection.get(
        where={
            "$and": [
                {"source": source},
                {"file_type": file_type},
            ]
        },
        limit=1,
        include=["metadatas"],
    )

    return bool(result["ids"])


# --------------------------------------------------
# Main indexing operation
# --------------------------------------------------

def main():
    from config import EMBEDDING_MODEL

    DOCUMENTS_DIR.mkdir(
        parents=True, exist_ok=True
    )
    IMAGES_DIR.mkdir(
        parents=True, exist_ok=True
    )
    DB_PATH.mkdir(
        parents=True, exist_ok=True
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    initial_count = collection.count()

    print(
        f"Database records before indexing: "
        f"{initial_count}"
    )

    all_chunks = []
    all_metadatas = []
    all_ids = []

    pdf_paths = []
    pdf_hashes = {}
    pdf_stage_state = load_pdf_stage_state()

    # --------------------------------------------------
    # 1. TXT and PDF text
    # --------------------------------------------------

    document_paths = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in (
            ".txt", ".pdf"
        )
    )

    print(
        f"\nFound {len(document_paths)} "
        "local documents"
    )

    for document_path in document_paths:
        print(
            f"\nProcessing: {document_path.name}"
        )

        suffix = document_path.suffix.lower()

        if suffix == ".pdf":
            digest = pdf_sha256(document_path)
            assert_pdf_not_replaced(
                pdf_stage_state, document_path, digest, collection
            )
            pdf_hashes[document_path.name] = digest
            pdf_paths.append(document_path)
            file_type = "pdf"
        else:
            file_type = "txt"

        if source_type_already_indexed(
            collection,
            document_path.name,
            file_type,
        ):
            print(
                f"Already indexed as {file_type}. "
                "Skipping text extraction."
            )
            continue

        try:
            if suffix == ".txt":
                text = document_path.read_text(
                    encoding="utf-8"
                )

                chunks = create_chunks(
                    text,
                    CHUNK_SIZE,
                    OVERLAP,
                )

                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)

                    all_ids.append(
                        create_chunk_id(
                            document_path.name,
                            None,
                            i,
                            "txt",
                        )
                    )

                    all_metadatas.append({
                        "source": document_path.name,
                        "chunk_number": i,
                        "file_type": "txt",
                    })

                print(
                    f"Created {len(chunks)} TXT chunks"
                )

            else:
                pages = extract_pdf_pages(
                    document_path
                )

                count = 0

                for page_number, page_text in pages:
                    chunks = create_chunks(
                        page_text,
                        CHUNK_SIZE,
                        OVERLAP,
                    )

                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)

                        all_ids.append(
                            create_chunk_id(
                                document_path.name,
                                page_number,
                                i,
                                "pdf",
                            )
                        )

                        all_metadatas.append({
                            "source": document_path.name,
                            "page": page_number,
                            "chunk_number": i,
                            "file_type": "pdf",
                        })

                        count += 1

                print(
                    f"Created {count} PDF text chunks"
                )

        except Exception as error:
            print(
                f"Text extraction failed for "
                f"{document_path.name}: {error}"
            )

    # --------------------------------------------------
    # 2. Webpages
    # --------------------------------------------------

    if URLS_FILE.exists():
        urls = [
            line.strip()
            for line in URLS_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
            and not line.strip().startswith("#")
        ]
    else:
        urls = []

    print(
        f"\nFound {len(urls)} URLs"
    )

    for url in dict.fromkeys(urls):
        print(
            f"\nProcessing URL: {url}"
        )

        if not url.startswith(
            ("https://", "http://")
        ):
            print("Invalid URL. Skipping.")
            continue

        if source_type_already_indexed(
            collection,
            url,
            "url",
        ):
            print(
                "Already indexed. Skipping."
            )
            continue

        try:
            text = extract_webpage(url)
        except requests.RequestException as error:
            print(
                f"Webpage extraction failed: {error}"
            )
            continue

        if not text:
            print(
                "No readable text. Skipping."
            )
            continue

        chunks = create_chunks(
            text,
            CHUNK_SIZE,
            OVERLAP,
        )

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_ids.append(
                create_chunk_id(
                    url,
                    None,
                    i,
                    "url",
                )
            )

            all_metadatas.append({
                "source": url,
                "url": url,
                "chunk_number": i,
                "file_type": "url",
            })

        print(
            f"Created {len(chunks)} URL chunks"
        )

    # --------------------------------------------------
    # 3. Local images
    # --------------------------------------------------

    image_paths = sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    print(
        f"\nFound {len(image_paths)} images"
    )

    for image_path in image_paths:
        print(
            f"\nProcessing image: "
            f"{image_path.name}"
        )

        source = (
            f"images/{image_path.name}"
        )

        image_id = create_chunk_id(
            source,
            None,
            0,
            "image",
        )

        existing = collection.get(
            ids=[image_id],
            include=["metadatas"],
        )

        if existing["ids"]:
            print(
                "Already indexed. Skipping."
            )
            continue

        try:
            description = describe_image(
                image_path
            )
        except Exception as error:
            print(
                f"Image analysis failed: {error}"
            )
            continue

        if not description:
            print(
                "Empty description. Skipping."
            )
            continue

        all_chunks.append(description)
        all_ids.append(image_id)

        all_metadatas.append({
            "source": source,
            "file_type": "image",
            "image_name": image_path.name,
            "chunk_number": 0,
        })

    # --------------------------------------------------
    # 4. Store text, webpage, and image records
    # --------------------------------------------------

    print(
        f"\nNew text/image records: "
        f"{len(all_chunks)}"
    )

    for start in range(
        0,
        len(all_chunks),
        BATCH_SIZE,
    ):
        end = start + BATCH_SIZE

        batch_chunks = all_chunks[
            start:end
        ]

        batch_ids = all_ids[
            start:end
        ]

        batch_metadatas = all_metadatas[
            start:end
        ]

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch_chunks,
        )

        collection.upsert(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=response["embeddings"],
            metadatas=batch_metadatas,
        )

        print(
            f"Indexed "
            f"{min(end, len(all_chunks))}"
            f"/{len(all_chunks)} records"
        )

    # --------------------------------------------------
    # 5. Process PDF figures independently
    # --------------------------------------------------

    print(
        "\n" + "=" * 60
    )
    print(
        "AUTOMATIC PDF FIGURE PROCESSING"
    )
    print(
        "=" * 60
    )

    figure_failures = []

    if not PROCESS_PDF_FIGURES:
        print("Skipping PDF figure analysis (optional; set RAG_PROCESS_PDF_FIGURES=1 to enable).")
    elif pdf_paths:
        from pdf_pipeline import index_figures

        for pdf_path in pdf_paths:
            if (not RETRY_PDF_STAGES
                    and pdf_stage_done(pdf_stage_state, pdf_path, "figures")):
                print(f"Skipping completed figure stage: {pdf_path.name}")
                continue
            try:
                print(f"Checking figures: {pdf_path.name}")
                index_figures(pdf_path)
                # Successful execution, including PDFs with no detected figures.
                # The pipeline can reject individual figures as non-provisional;
                # force a later recheck with RAG_RETRY_PDF_STAGES=1 if needed.
                mark_pdf_stage_complete(
                    pdf_stage_state, pdf_path,
                    pdf_hashes[pdf_path.name], "figures"
                )
                print(f"Figure pass complete: {pdf_path.name}")

            except Exception as error:
                figure_failures.append(
                    (pdf_path.name, str(error))
                )

                print(
                    f"\nFigure processing failed "
                    f"for {pdf_path.name}: {error}"
                )
                print(
                    "Other indexed records "
                    "have been preserved."
                )

    else:
        print(
            "No PDF files to process."
        )

    # --------------------------------------------------
    # 6. Process PDF tables independently
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUTOMATIC PDF TABLE PROCESSING")
    print("=" * 60)

    # Safe first run: extract and validate, but do not write table records.
    # After checking the combined output in PowerShell, set:
    #   $env:RAG_TABLE_INDEX_DRY_RUN = "0"
    # and rerun the indexer to add eligible table records.
    table_dry_run = os.getenv("RAG_TABLE_INDEX_DRY_RUN", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )
    print(f"Table indexing dry run: {table_dry_run}")
    print(f"Force PDF stages: {RETRY_PDF_STAGES}")

    table_failures = []
    table_prepared = 0
    table_added = 0
    table_skipped = 0

    if not PROCESS_PDF_TABLES:
        print("Skipping PDF table analysis (optional; set RAG_PROCESS_PDF_TABLES=1 to enable).")
    elif pdf_paths:
        try:
            from pdf_table_pipeline import index_tables
        except Exception as error:
            table_failures.append(("table pipeline import", str(error)))
            print(f"Table pipeline could not be imported: {error}")
        else:
            for pdf_path in pdf_paths:
                stage = "tables_dry_run" if table_dry_run else "tables_indexed"
                if (not RETRY_PDF_STAGES
                        and pdf_stage_done(pdf_stage_state, pdf_path, stage)):
                    print(f"Skipping completed table stage ({stage}): {pdf_path.name}")
                    continue
                try:
                    print(f"Checking tables: {pdf_path.name}")
                    table_result = index_tables(
                        pdf_path,
                        dry_run=table_dry_run,
                    )
                    table_prepared += table_result.get("prepared_records", 0)
                    table_added += table_result.get("new_records", 0)
                    table_skipped += table_result.get("already_indexed", 0)
                    print(
                        f"Table result for {pdf_path.name}: "
                        f"prepared={table_result.get('prepared_records', 0)}, "
                        f"added={table_result.get('new_records', 0)}, "
                        f"already indexed={table_result.get('already_indexed', 0)}"
                    )
                    # A dry run never qualifies as actual table indexing.
                    mark_pdf_stage_complete(
                        pdf_stage_state, pdf_path,
                        pdf_hashes[pdf_path.name], stage
                    )
                except Exception as error:
                    table_failures.append((pdf_path.name, str(error)))
                    print(f"Table processing failed for {pdf_path.name}: {error}")
                    print("Previously indexed records have been preserved.")
    else:
        print("No PDF files to process.")

    # --------------------------------------------------
    # 7. Summary
    # --------------------------------------------------

    final_count = collection.count()

    print(
        "\n" + "=" * 60
    )
    print(
        "INDEXING SUMMARY"
    )
    print(
        "=" * 60
    )

    print(
        f"Records before: {initial_count}"
    )
    print(
        f"Text/image records prepared: "
        f"{len(all_chunks)}"
    )
    print(
        f"Records after: {final_count}"
    )
    print(
        f"Figure pipeline failures: "
        f"{len(figure_failures)}"
    )
    print(f"Table indexing dry run: {table_dry_run}")
    print(f"Table records prepared: {table_prepared}")
    print(f"Table records added: {table_added}")
    print(f"Table records already indexed: {table_skipped}")
    print(f"Table pipeline failures: {len(table_failures)}")
    print(f"PDF stage state: {PDF_STATE_FILE}")

    if table_failures:
        for name, error in table_failures:
            print(f"- Table {name}: {error}")

    if figure_failures:
        for name, error in figure_failures:
            print(
                f"- {name}: {error}"
            )

        # Make partial figure-stage failures visible
        # to Streamlit instead of reporting full success.
        return 1

    if table_failures:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
