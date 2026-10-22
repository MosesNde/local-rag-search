
import argparse
import hashlib
import json
import re
from pathlib import Path

import chromadb
import ollama

from config import VISION_MODEL, EMBEDDING_MODEL
from pdf_figures import extract_pdf_figures
from pdf_validation import run_validation


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
DB_PATH = PROJECT_ROOT / "data" / "chroma"
VALIDATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "figure_tests"
    / "validation"
)


def figure_id(source, page, number):
    digest = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()[:16]

    return f"figure_{digest}_p{page}_f{number}"


def model_slug():
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        VISION_MODEL,
    )


def already_indexed(collection, source, page, number):
    results = collection.get(
        where={
            "$and": [
                {"source": source},
                {"file_type": "pdf_figure"},
                {"page": page},
                {"figure_number": number},
            ]
        },
        limit=1,
        include=["metadatas"],
    )

    return bool(results["ids"])


def index_figures(pdf_path):
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    initial_count = collection.count()

    print(f"\nProcessing PDF: {pdf_path.name}")
    print(f"Existing records: {initial_count}")

    # Detect and crop figures automatically.
    manifest = extract_pdf_figures(pdf_path)

    figures = manifest["figures"]

    if not figures:
        print("No figure captions detected.")
        return

    # pdf_figures.py stores its manifest alongside
    # the generated images.
    image_path = Path(figures[0]["image_path"])
    manifest_dir = image_path.parent
    manifest_path = manifest_dir / "manifest.json"

    validation_folder = (
        VALIDATION_DIR / manifest_dir.name
    )

    # Determine which figures still need indexing.
    pending = []

    seen = set()

    for figure in figures:
        number = figure["figure_number"]
        page = figure["page"]

        key = (page, number)

        if key in seen:
            print(
                f"Duplicate Figure {number} on "
                f"page {page}; skipping."
            )
            continue

        seen.add(key)

        if already_indexed(
            collection,
            pdf_path.name,
            page,
            number,
        ):
            print(
                f"Figure {number}, page {page}: "
                "already indexed."
            )
            continue

        pending.append(figure)

    if not pending:
        print("\nAll detected figures are already indexed.")
        print(
            f"ChromaDB records: {collection.count()}"
        )
        return

    # The current validation API selects by figure
    # number, so duplicate numbers on different
    # pages are deferred until we support unique
    # page + figure identifiers throughout.
    counts = {}

    for figure in pending:
        n = figure["figure_number"]
        counts[n] = counts.get(n, 0) + 1

    pending = [
        figure
        for figure in pending
        if counts[figure["figure_number"]] == 1
    ]

    if not pending:
        print(
            "No figures can be safely indexed: "
            "figure numbers are ambiguous."
        )
        return

    numbers = sorted({
        figure["figure_number"]
        for figure in pending
    })

    print(
        "\nValidating new figures: "
        + ", ".join(map(str, numbers))
    )

    # Generate descriptions and validation reports.
    # This does not modify ChromaDB.
    run_validation(
        manifest_path=manifest_path,
        figure_numbers=numbers,
    )

    added = 0
    rejected = 0

    for figure in pending:
        number = figure["figure_number"]
        page = figure["page"]

        stem = (
            f"page_{page}_figure_{number}_"
            f"{model_slug()}"
        )

        report_path = (
            validation_folder
            / f"{stem}_report.json"
        )

        if not report_path.is_file():
            print(
                f"Figure {number}: "
                "validation report missing."
            )
            rejected += 1
            continue

        report = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        # Only candidates that passed the current
        # checks are eligible for indexing.
        if report.get("status") != "provisional":
            print(
                f"Figure {number}: "
                "validation incomplete; not indexed."
            )
            rejected += 1
            continue

        description = report.get(
            "final_description", ""
        ).strip()

        if not description:
            print(
                f"Figure {number}: empty description."
            )
            rejected += 1
            continue

        document = (
            f"Source: {pdf_path.name}\n"
            f"Page: {page}\n"
            f"Figure: Figure {number}\n"
            f"Caption: {figure['caption']}\n"
            f"Visual description:\n{description}"
        )

        # Check again before writing.
        if already_indexed(
            collection,
            pdf_path.name,
            page,
            number,
        ):
            print(
                f"Figure {number}: record already exists."
            )
            continue

        try:
            embedding = ollama.embed(
                model=EMBEDDING_MODEL,
                input=document,
            )["embeddings"][0]

            collection.add(
                ids=[
                    figure_id(
                        pdf_path.name,
                        page,
                        number,
                    )
                ],
                documents=[document],
                embeddings=[embedding],
                metadatas=[{
                    "source": pdf_path.name,
                    "page": page,
                    "figure_number": number,
                    "file_type": "pdf_figure",
                    "caption": figure["caption"],
                    "image_path": str(
                        figure["image_path"]
                    ),
                    "validation_status": "provisional",
                    "vision_model": VISION_MODEL,
                }],
            )

            added += 1
            print(
                f"Figure {number}: indexed successfully."
            )

        except Exception as error:
            rejected += 1
            print(
                f"Figure {number}: indexing failed: "
                f"{error}"
            )

    print("\n" + "=" * 50)
    print("PDF figure pipeline completed")
    print(f"New figures indexed: {added}")
    print(f"Figures not indexed: {rejected}")
    print(
        f"ChromaDB records: {collection.count()}"
    )
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically extract, validate, "
            "and index PDF figures."
        )
    )

    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="PDF filename or full path",
    )

    args = parser.parse_args()

    pdf_path = args.pdf

    if not pdf_path.is_file():
        pdf_path = DOCUMENTS_DIR / pdf_path

    index_figures(pdf_path)


if __name__ == "__main__":
    main()
