
import hashlib
import re
from pathlib import Path

import chromadb
import ollama
import pymupdf
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
DB_PATH = PROJECT_ROOT / "data" / "chroma"
URLS_FILE = PROJECT_ROOT / "data" / "urls.txt"

CHUNK_SIZE = 200
OVERLAP = 1
BATCH_SIZE = 16

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.mkdir(parents=True, exist_ok=True)

def create_chunks(text, chunk_size, overlap):
    sentences = text.replace("\n", " ").split(".")
    chunks = []

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if current_chunk and current_length + len(sentence) + 2 > chunk_size:
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


def clean_pdf_text(text):
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf_pages(pdf_path):
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            blocks = page.get_text("blocks")
            midpoint = page.rect.width / 2

            header = []
            left_column = []
            right_column = []
            spanning_blocks = []

            for block in blocks:
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
                pages.append((page_number, page_text))

    return pages


def extract_webpage(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    if not response.encoding or response.encoding.lower() == "iso-8859-1":
        response.encoding = response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")

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

    return re.sub(r"\s+", " ", text).strip()


def describe_image(image_path):
    response = ollama.chat(
        model="qwen2.5vl:3b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe this image accurately for a "
                    "document search engine. "
                    "Identify visible text, important objects, "
                    "charts, values, and relationships or steps. "
                    "Do not invent details that are not visible."
                ),
                "images": [str(image_path)],
            }
        ],
        think=False,
    )

    return response["message"]["content"].strip()


def create_chunk_id(source, page, chunk_number, file_type):
    source_hash = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    if file_type == "pdf":
        return f"pdf_{source_hash}_p{page}_chunk{chunk_number}"

    return f"{file_type}_{source_hash}_chunk{chunk_number}"


client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="documents"
)

all_chunks = []
all_metadatas = []
all_ids = []


# --------------------------------------------------
# 1. Index local TXT and PDF documents
# --------------------------------------------------

document_paths = sorted(
    path
    for path in DOCUMENTS_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in (".txt", ".pdf")
)

print(f"Found {len(document_paths)} local documents")

for document_path in document_paths:
    print(f"\nProcessing: {document_path.name}")

    existing = collection.get(
        where={"source": document_path.name},
        limit=1,
        include=["metadatas"],
    )

    if existing["ids"]:
        print("Already indexed. Skipping.")
        continue

    if document_path.suffix.lower() == ".txt":
        text = document_path.read_text(encoding="utf-8")
        chunks = create_chunks(text, CHUNK_SIZE, OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)

            all_ids.append(
                create_chunk_id(
                    document_path.name, None, i, "txt"
                )
            )

            all_metadatas.append({
                "source": document_path.name,
                "chunk_number": i,
                "file_type": "txt",
            })

        print(f"Created {len(chunks)} TXT chunks")

    elif document_path.suffix.lower() == ".pdf":
        pages = extract_pdf_pages(document_path)
        pdf_chunk_count = 0

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

                pdf_chunk_count += 1

        print(f"Created {pdf_chunk_count} PDF chunks")


# --------------------------------------------------
# 2. Index webpages from data/urls.txt
# --------------------------------------------------

if URLS_FILE.exists():
    urls = [
        line.strip()
        for line in URLS_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
else:
    urls = []

print(f"\nFound {len(urls)} URLs")

for url in dict.fromkeys(urls):
    print(f"\nProcessing URL: {url}")

    if not url.startswith(("https://", "http://")):
        print("Invalid URL scheme. Skipping.")
        continue

    existing = collection.get(
        where={"source": url},
        limit=1,
        include=["metadatas"],
    )

    if existing["ids"]:
        print("Already indexed. Skipping.")
        continue

    try:
        text = extract_webpage(url)
    except requests.RequestException as error:
        print(f"Could not fetch webpage: {error}")
        continue

    if not text:
        print("No readable text found. Skipping.")
        continue

    chunks = create_chunks(text, CHUNK_SIZE, OVERLAP)

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)

        all_ids.append(
            create_chunk_id(url, None, i, "url")
        )

        all_metadatas.append({
            "source": url,
            "url": url,
            "chunk_number": i,
            "file_type": "url",
        })

    print(f"Characters extracted: {len(text)}")
    print(f"Created {len(chunks)} URL chunks")


# --------------------------------------------------
# 3. Index local images
# --------------------------------------------------

image_paths = sorted(
    path
    for path in IMAGES_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
) if IMAGES_DIR.exists() else []

print(f"\nFound {len(image_paths)} images")

for image_path in image_paths:
    print(f"\nProcessing image: {image_path.name}")

    # Use a path-based source identifier to distinguish
    # images from other document types.
    source = f"images/{image_path.name}"

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
        print("Already indexed. Skipping.")
        continue

    try:
        description = describe_image(image_path)
    except Exception as error:
        print(f"Image analysis failed: {error}")
        continue

    if not description:
        print("Empty image description. Skipping.")
        continue

    all_chunks.append(description)
    all_ids.append(image_id)

    all_metadatas.append({
        "source": source,
        "file_type": "image",
        "image_name": image_path.name,
        "chunk_number": 0,
    })

    print(f"Description generated: {len(description)} characters")


# --------------------------------------------------
# 4. Generate embeddings and store new records
# --------------------------------------------------

print(f"\nNew records to index: {len(all_chunks)}")

for start in range(0, len(all_chunks), BATCH_SIZE):
    end = start + BATCH_SIZE

    batch_chunks = all_chunks[start:end]
    batch_ids = all_ids[start:end]
    batch_metadatas = all_metadatas[start:end]

    response = ollama.embed(
        model="nomic-embed-text",
        input=batch_chunks,
    )

    collection.upsert(
        ids=batch_ids,
        documents=batch_chunks,
        embeddings=response["embeddings"],
        metadatas=batch_metadatas,
    )

    print(
        f"Indexed {min(end, len(all_chunks))}"
        f"/{len(all_chunks)} records"
    )


print("\nIndexing complete.")
print(f"Total records in ChromaDB: {collection.count()}")
