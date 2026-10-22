
import hashlib
import re
from pathlib import Path

import chromadb
import ollama
import pymupdf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "documents"
    / "A_Comprehensive_Software_Vulnerability_Dataset_"
      "Based_on_OWASP_Top_Ten_Standard.pdf"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "figure_tests"
    / "auto_crops"
)

DESCRIPTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "figure_tests"
    / "descriptions"
)

DB_PATH = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "documents"

VISION_MODEL = "qwen2.5vl:3b"
EMBEDDING_MODEL = "nomic-embed-text"

DESCRIPTION_DIR.mkdir(parents=True, exist_ok=True)

CAPTION_PATTERN = re.compile(
    r"^\s*(?:Fig\.?|Figure)\s+(\d+)\.\s+(.+)",
    re.IGNORECASE,
)


# --------------------------------------------------
# Generate stable figure ID
# --------------------------------------------------

def create_figure_id(pdf_name, figure_number):
    pdf_hash = hashlib.sha256(
        pdf_name.encode("utf-8")
    ).hexdigest()[:16]

    return f"figure_{pdf_hash}_f{figure_number}"


# --------------------------------------------------
# Extract figure captions from the PDF
# --------------------------------------------------

def extract_figure_metadata(pdf_path):
    figures = {}

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7 or block[6] != 0:
                    continue

                lines = block[4].splitlines()

                if not lines:
                    continue

                match = CAPTION_PATTERN.match(lines[0])

                if not match:
                    continue

                figure_number = int(match.group(1))

                caption = " ".join(
                    " ".join(
                        [match.group(2)] + lines[1:]
                    ).split()
                )

                figures[figure_number] = {
                    "page": page_number,
                    "caption": caption,
                }

    return figures


# --------------------------------------------------
# Describe the figure with Qwen2.5-VL
# --------------------------------------------------

def describe_figure(image_path, figure_number, caption):
    prompt = f"""
Analyze this figure from a scientific research paper.

Figure number: {figure_number}
Verified caption: {caption}

Describe its visual content for a retrieval-augmented
generation search engine.

Identify, where applicable:
- The type of figure (chart, code example, graph, or diagram).
- Important objects, nodes, labels, and entities.
- Relationships and connections between components.
- The direction of arrows and information flow.
- Differences between subfigures or panels.
- Highlighted code and its significance.
- Axis labels, legends, and categories in charts.
- The main conclusion supported by the visible figure.

IMPORTANT RULES:
- Use the verified caption to establish figure context.
- Do not invent details not supported by the image or caption.
- Do not estimate numerical values from bar heights.
- Report exact numbers only if explicitly printed and legible.
- Do not infer causal relationships without visual evidence.
- If something cannot be determined, say so.
- Keep the description factual and concise.
- Do not repeat these instructions in your response.

Produce a description suitable for semantic search.
"""

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        think=False,
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

if not PDF_PATH.exists():
    raise SystemExit(
        f"PDF not found: {PDF_PATH}"
    )

if not FIGURE_DIR.exists():
    raise SystemExit(
        f"Figure directory not found: {FIGURE_DIR}"
    )

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

initial_count = collection.count()

print("=" * 60)
print("PDF FIGURE INDEXING")
print("=" * 60)

print(f"PDF: {PDF_PATH.name}")
print(f"Database records before indexing: {initial_count}")


# --------------------------------------------------
# Locate figure metadata
# --------------------------------------------------

figure_metadata = extract_figure_metadata(PDF_PATH)

figure_files = sorted(
    FIGURE_DIR.glob("figure_*.png"),
    key=lambda path: (
        int(re.search(r"figure_(\d+)", path.stem).group(1))
        if re.search(r"figure_(\d+)", path.stem)
        else 9999
    ),
)

print(f"Figure images found: {len(figure_files)}")

added = 0
skipped = 0
failed = 0


# --------------------------------------------------
# Index each figure independently
# --------------------------------------------------

for image_path in figure_files:
    match = re.fullmatch(
        r"figure_(\d+)",
        image_path.stem,
    )

    if not match:
        continue

    figure_number = int(match.group(1))

    print("\n" + "-" * 60)
    print(f"Processing Figure {figure_number}")

    if figure_number not in figure_metadata:
        print("SKIPPED: Figure caption not found in PDF.")
        skipped += 1
        continue

    metadata = figure_metadata[figure_number]

    figure_id = create_figure_id(
        PDF_PATH.name,
        figure_number,
    )

    # Avoid duplicate records.
    existing = collection.get(
        ids=[figure_id],
    )

    if existing["ids"]:
        print("SKIPPED: Figure already indexed.")
        skipped += 1
        continue

    page_number = metadata["page"]
    caption = metadata["caption"]

    print(f"Page: {page_number}")
    print(f"Caption: {caption}")

    try:
        # Generate factual visual description.
        description = describe_figure(
            image_path,
            figure_number,
            caption,
        )

        if not description:
            raise ValueError(
                "Vision model returned an empty description."
            )

        # Build searchable document.
        document = (
            f"Source: {PDF_PATH.name}\n"
            f"Page: {page_number}\n"
            f"Figure: Figure {figure_number}\n"
            f"Caption: {caption}\n"
            f"Visual description:\n{description}"
        )

        # Save description for review.
        description_path = (
            DESCRIPTION_DIR
            / f"figure_{figure_number}_description.txt"
        )

        description_path.write_text(
            document,
            encoding="utf-8",
        )

        # Generate text embedding.
        embedding_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=document,
        )

        embedding = embedding_response["embeddings"][0]

        # Add one figure record.
        collection.add(
            ids=[figure_id],
            documents=[document],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": PDF_PATH.name,
                    "file_type": "pdf_figure",
                    "figure_number": figure_number,
                    "page": page_number,
                    "caption": caption,
                    "image_path": str(image_path.relative_to(
                        PROJECT_ROOT
                    )),
                }
            ],
        )

        added += 1

        print("INDEXED successfully.")
        print(
            f"Description saved: "
            f"{description_path.name}"
        )

    except Exception as error:
        failed += 1
        print(f"FAILED: {error}")


# --------------------------------------------------
# Summary
# --------------------------------------------------

final_count = collection.count()

print("\n" + "=" * 60)
print("INDEXING SUMMARY")
print("=" * 60)

print(f"Figures added: {added}")
print(f"Figures skipped: {skipped}")
print(f"Figures failed: {failed}")
print(f"Database records before: {initial_count}")
print(f"Database records after: {final_count}")
