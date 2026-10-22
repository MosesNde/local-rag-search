
import hashlib
from pathlib import Path

import chromadb
import ollama

from pdf_tables import extract_structured_tables


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
DB_PATH = PROJECT_ROOT / "data" / "chroma"

EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 16


# --------------------------------------------------
# Connect to existing database
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="documents"
)

initial_count = collection.count()

print(f"Existing records: {initial_count}")


# --------------------------------------------------
# Generate stable identifiers
# --------------------------------------------------

def create_table_row_id(
    pdf_name,
    page_number,
    detection_index,
    row_number,
):
    source_hash = hashlib.sha256(
        pdf_name.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"table_{source_hash}_"
        f"p{page_number}_"
        f"t{detection_index}_"
        f"r{row_number}"
    )


# --------------------------------------------------
# Extract PDF tables
# --------------------------------------------------

all_texts = []
all_ids = []
all_metadatas = []

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files")

for pdf_path in pdf_files:
    print(f"\nProcessing: {pdf_path.name}")

    tables = extract_structured_tables(pdf_path)

    print(f"Accepted tables: {len(tables)}")

    for table in tables:
        page = table["page"]
        detection_index = table["detection_index"]

        table_label = (
            table["table_label"]
            or f"Unidentified table on page {page}"
        )

        caption = table["caption"] or ""

        for row in table["rows"]:
            row_number = row["row_number"]

            record_id = create_table_row_id(
                pdf_path.name,
                page,
                detection_index,
                row_number,
            )

            # Never overwrite existing records.
            existing = collection.get(
                ids=[record_id]
            )

            if existing["ids"]:
                continue

            text = (
                f"Source: {pdf_path.name}\n"
                f"Page: {page}\n"
                f"Table: {table_label}\n"
                f"Caption: {caption}\n"
                f"Row {row_number}: {row['text']}"
            )

            all_texts.append(text)
            all_ids.append(record_id)

            all_metadatas.append({
                "source": pdf_path.name,
                "file_type": "pdf_table",
                "page": page,
                "table_label": table_label,
                "table_index": detection_index,
                "row_number": row_number,
                "caption": caption,
            })


# --------------------------------------------------
# Generate embeddings and store rows
# --------------------------------------------------

print(f"\nNew table rows to index: {len(all_texts)}")

for start in range(0, len(all_texts), BATCH_SIZE):
    end = start + BATCH_SIZE

    batch_texts = all_texts[start:end]
    batch_ids = all_ids[start:end]
    batch_metadatas = all_metadatas[start:end]

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=batch_texts,
    )

    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        embeddings=response["embeddings"],
        metadatas=batch_metadatas,
    )

    print(
        f"Indexed "
        f"{min(end, len(all_texts))}/"
        f"{len(all_texts)} table rows"
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

final_count = collection.count()

print("\nTable indexing complete.")
print(f"Previous records: {initial_count}")
print(f"New records added: {final_count - initial_count}")
print(f"Total records: {final_count}")
