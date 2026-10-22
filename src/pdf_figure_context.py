
import argparse
import json
import re
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "figure_tests" / "context"
)

MAX_CONTEXT_CHARS = 9000


def normalize_text(text):
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def ordered_page_blocks(page):
    """
    Read conventional two-column pages in
    left-column then right-column order.

    Also supports single-column pages.
    """
    midpoint = page.rect.width / 2
    left = []
    right = []
    spanning = []

    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue

        x0, y0, x1, y1 = block[:4]
        text = normalize_text(block[4])

        if not text:
            continue

        if "authorized licensed use limited to" in text.lower():
            continue

        entry = {
            "y": y0,
            "text": text,
        }

        if x1 <= midpoint:
            left.append(entry)
        elif x0 >= midpoint:
            right.append(entry)
        else:
            spanning.append(entry)

    # Detect whether a meaningful two-column layout exists.
    two_columns = bool(left and right)

    if two_columns:
        ordered = (
            sorted(left, key=lambda x: x["y"])
            + sorted(right, key=lambda x: x["y"])
        )

        # Spanning section headings may need special
        # positioning; retain them as supplementary
        # text rather than inserting them mid-paragraph.
        ordered.extend(
            sorted(spanning, key=lambda x: x["y"])
        )
    else:
        ordered = sorted(
            left + right + spanning,
            key=lambda x: x["y"],
        )

    return [item["text"] for item in ordered]


def figure_reference_pattern(number):
    n = re.escape(str(number))

    return re.compile(
        rf"\b(?:Fig(?:ure)?s?\.?)\s*{n}\b",
        flags=re.IGNORECASE,
    )


def is_section_heading(text):
    return bool(
        re.match(
            r"^(?:[IVX]+\.\s+|"
            r"\d+(?:\.\d+)*\s+)"
            r"[A-Z][A-Z\s/&-]{3,}",
            text.strip(),
        )
    )


def is_figure_caption(text):
    return bool(
        re.match(
            r"^(?:Fig\.?|Figure)\s+\d+\s*[.:]",
            text,
            flags=re.IGNORECASE,
        )
    )


def find_figure_context(
    pdf_path,
    figure_number,
    figure_page=None,
    page_radius=2,
    max_chars=MAX_CONTEXT_CHARS,
):
    """
    Find a figure reference, then collect
    subsequent reading-order text until a
    new major section or the size limit.

    The returned text is supporting evidence,
    not a verified interpretation.
    """
    with pymupdf.open(pdf_path) as pdf:
        if figure_page is None:
            start_page = 0
            end_page = len(pdf)
        else:
            start_page = max(
                0,
                figure_page - 1 - page_radius,
            )
            end_page = min(
                len(pdf),
                figure_page + page_radius,
            )

        blocks = []

        for page_index in range(start_page, end_page):
            for text in ordered_page_blocks(pdf[page_index]):
                blocks.append({
                    "page": page_index + 1,
                    "text": text,
                })

        pattern = figure_reference_pattern(figure_number)

        # Prefer references in discussion text,
        # not the figure caption itself.
        reference_positions = [
            index
            for index, block in enumerate(blocks)
            if pattern.search(block["text"])
            and not is_figure_caption(block["text"])
        ]

        if not reference_positions:
            return []

        # Prefer a reference on the figure's own page.
        if figure_page is not None:
            reference_positions.sort(
                key=lambda index: (
                    abs(blocks[index]["page"] - figure_page),
                    index,
                )
            )

        anchor = reference_positions[0]

        collected = []
        total_chars = 0

        for index in range(anchor, len(blocks)):
            block = blocks[index]
            text = block["text"]

            if index > anchor and is_section_heading(text):
                break

            # Omit chart captions from the prose context.
            if is_figure_caption(text):
                continue

            remaining = max_chars - total_chars

            if remaining <= 0:
                break

            # Preserve the page number of each passage.
            collected.append({
                "page": block["page"],
                "text": text[:remaining],
            })

            total_chars += min(len(text), remaining)

        return collected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract supporting discussion for a "
            "figure from a research PDF."
        )
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--figure",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--page",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    pdf_path = args.pdf

    if not pdf_path.is_file():
        pdf_path = DOCUMENTS_DIR / pdf_path

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    passages = find_figure_context(
        pdf_path=pdf_path,
        figure_number=args.figure,
        figure_page=args.page,
    )

    result = {
        "source": pdf_path.name,
        "figure_number": args.figure,
        "passages": passages,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{pdf_path.stem}_figure_{args.figure}.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Figure {args.figure}: "
        f"{len(passages)} text blocks collected."
    )

    for passage in passages:
        print(
            f"\n[Page {passage['page']}] "
            f"{passage['text']}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
