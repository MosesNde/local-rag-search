
import argparse
import json
import re
from pathlib import Path

import ollama
import pymupdf

from config import (
    VISION_MODEL,
    VISION_CONTEXT_SIZE,
    VISION_MAX_RETRIES,
)

from pdf_figure_context import find_figure_context


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
FIGURES_DIR = PROJECT_ROOT / "data" / "figures"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "figure_tests"
    / "validation"
)

NUM_CTX = VISION_CONTEXT_SIZE
NUM_PREDICT = 2048
MAX_RETRIES = min(max(VISION_MAX_RETRIES, 0), 1)

MAX_FIGURE_TEXT = 3000
MAX_CONTEXT_CHARS = 4000

CHECKED_LABELS = [
    "gets",
    "fgets",
    "scanf",
    "printf",
    "qtdletras",
    "METHOD_RETURN",
    "METHOD",
    "PARAM",
    "BLOCK",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Manifest handling
# --------------------------------------------------

def find_manifest():
    manifests = sorted(
        FIGURES_DIR.rglob("manifest.json")
    )

    if not manifests:
        raise FileNotFoundError(
            "No figure manifest found. "
            "Run pdf_figures.py first."
        )

    if len(manifests) > 1:
        raise RuntimeError(
            "Multiple manifests found. "
            "Use --manifest PATH."
        )

    return manifests[0]


def load_manifest(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def resolve_image_path(image_path, manifest_path):
    path = Path(image_path)

    if path.is_absolute():
        return path

    project_path = PROJECT_ROOT / path

    if project_path.exists():
        return project_path

    return Path(manifest_path).parent / path


# --------------------------------------------------
# Extract figure-specific text
# --------------------------------------------------

def extract_figure_text(pdf, figure):
    page = pdf[figure["page"] - 1]

    crop = pymupdf.Rect(
        figure["crop_bbox"]
    )

    return page.get_text(
        "text",
        clip=crop,
        sort=True,
    ).strip()


# --------------------------------------------------
# Retrieve explanatory context from the paper
# --------------------------------------------------

def retrieve_paper_context(
    pdf_path,
    figure_number,
    figure_page,
):
    """
    Retrieve discussion mentioning the figure.

    This is supporting evidence, not guaranteed
    to be complete or exclusively relevant.
    """

    try:
        passages = find_figure_context(
            pdf_path=pdf_path,
            figure_number=figure_number,
            figure_page=figure_page,
        )

    except Exception as error:
        return {
            "passages": [],
            "text": "",
            "error": str(error),
        }

    selected = []
    used_chars = 0

    for passage in passages:
        text = passage["text"].strip()

        if not text:
            continue

        prefix = f"[Page {passage['page']}] "
        available = (
            MAX_CONTEXT_CHARS
            - used_chars
            - len(prefix)
        )

        if available <= 0:
            break

        shortened = text[:available]

        # Avoid splitting an item unnecessarily.
        if len(shortened) < len(text):
            last_space = shortened.rfind(" ")

            if last_space > len(shortened) // 2:
                shortened = shortened[:last_space]

        selected.append({
            "page": passage["page"],
            "text": shortened,
        })

        used_chars += len(prefix) + len(shortened)

        if used_chars >= MAX_CONTEXT_CHARS:
            break

    combined = "\n\n".join(
        f"[Page {item['page']}] {item['text']}"
        for item in selected
    )

    return {
        "passages": selected,
        "text": combined,
        "error": None,
    }


# --------------------------------------------------
# Figure classification
# --------------------------------------------------

def classify_figure(caption, source_text):
    content = (
        caption + "\n" + source_text
    ).lower()

    if "abstract syntax tree" in content:
        return "ast"

    if any(
        phrase in content
        for phrase in [
            "data flow graph",
            "data-flow graph",
            "data dependency",
            "ddg:",
        ]
    ):
        return "data_flow"

    if any(
        phrase in content
        for phrase in [
            "distribution chart",
            "evaluation results",
            "classifiers",
        ]
    ):
        return "chart"

    if "#include" in content:
        return "code"

    if "pipeline" in content:
        return "pipeline"

    return "general"


# --------------------------------------------------
# Label coverage checks
# --------------------------------------------------

def label_pattern(label):
    return (
        r"(?<![A-Za-z0-9_])"
        + re.escape(label)
        + r"(?![A-Za-z0-9_])"
    )


def check_labels(source_text, description):
    found = []
    missing = []

    for label in CHECKED_LABELS:
        pattern = label_pattern(label)

        if not re.search(
            pattern,
            source_text,
            flags=re.IGNORECASE,
        ):
            continue

        found.append(label)

        if not re.search(
            pattern,
            description,
            flags=re.IGNORECASE,
        ):
            missing.append(label)

    return {
        "source_labels": found,
        "missing_labels": missing,
    }


# --------------------------------------------------
# Suspicious statement checks
# --------------------------------------------------

NEGATIVE_PATTERNS = [
    r"\bdoes not (?:show|represent|establish|indicate|depict)\b",
    r"\bdo not (?:show|represent|establish|indicate)\b",
    r"\bnot (?:shown|represented|established|depicted)\b",
    r"\bcannot (?:infer|determine|establish)\b",
    r"\bnot necessarily\b",
    r"\bunclear\b",
    r"\bunknown\b",
    r"\brather than control flow\b",
    r"\binstead of control flow\b",
    r"\bdata flow, not control flow\b",
    r"\bdata dependencies, not control flow\b",
]


def is_negated_or_uncertain(sentence):
    return any(
        re.search(
            pattern,
            sentence,
            flags=re.IGNORECASE,
        )
        for pattern in NEGATIVE_PATTERNS
    )


def check_semantic_risks(
    figure_type,
    description,
):
    warnings = []

    sentences = re.split(
        r"(?<=[.!?])\s+|\n",
        description,
    )

    for sentence in sentences:
        sentence = sentence.strip()
        lower = sentence.lower()

        if not sentence:
            continue

        if figure_type == "data_flow":

            if re.search(
                r"\b(?:control flow|control-flow|flow of control)\b",
                lower,
            ):
                if not is_negated_or_uncertain(lower):
                    warnings.append(
                        "Potential control-flow claim: "
                        + sentence
                    )

            if re.search(
                r"\ball nodes? (?:link|connect|lead) to\b",
                lower,
            ):
                warnings.append(
                    "Unverified universal edge claim: "
                    + sentence
                )

        elif figure_type == "ast":

            if (
                "execution order" in lower
                and not is_negated_or_uncertain(lower)
            ):
                warnings.append(
                    "Possible execution-order claim: "
                    + sentence
                )

    return list(dict.fromkeys(warnings))


# --------------------------------------------------
# Figure description generation
# --------------------------------------------------

def generate_description(
    image_path,
    figure,
    source_text,
    figure_type,
    context_text,
    retry_feedback=None,
):
    feedback = ""

    if retry_feedback:
        feedback = (
            "\nISSUES FROM PREVIOUS ATTEMPT:\n"
            + "\n".join(
                f"- {item[:150]}"
                for item in retry_feedback[:3]
            )
        )

    prompt = f"""
Analyze the attached scientific figure.

FIGURE TYPE:
{figure_type}

CAPTION:
{figure["caption"]}

TEXT EXTRACTED DIRECTLY FROM THE FIGURE:
{source_text[:MAX_FIGURE_TEXT]}

RELEVANT EXPLANATORY TEXT FROM THE PAPER:
{context_text or "[No supporting passage retrieved]"}

TASK:
Write one concise, evidence-grounded description
of the figure for a local RAG search engine.

Requirements:
- Approximately 100 to 180 words.
- Explain the figure's purpose.
- Identify important elements and relationships.
- Preserve exact labels where useful.
- Describe supported findings.
- Include uncertainty when necessary.

EVIDENCE RULES:
- Use the image for visual structure.
- Use the figure text for exact labels.
- Use the paper discussion for reported
  numerical results and explanations.
- When the paper supplies exact values,
  prefer those over estimates from bar heights.
- State which metric a comparison concerns.
- Do not confuse overall performance with
  being highest on every individual metric.
- Do not invent unsupported values or edges.
- Do not confuse data flow with control flow.
- If the sources seem inconsistent, report
  the inconsistency rather than guessing.
- Do not use unrelated surrounding paragraphs
  as evidence about the figure.
- Do not repeat the same information.
- Finish with a complete sentence.

{feedback}
""".strip()

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
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": 0,
        },
    )

    message = response["message"]

    return {
        "description": (
            message.get("content") or ""
        ).strip(),
        "finish_reason": response.get(
            "done_reason",
            "unknown",
        ),
        "eval_count": response.get("eval_count"),
        "prompt_eval_count": response.get(
            "prompt_eval_count"
        ),
    }


# --------------------------------------------------
# Deterministic numerical evidence checks
# --------------------------------------------------

def check_numerical_evidence(
    description,
    context_text,
    figure_type,
):
    """
    Check numeric percentages in the description
    against retrieved paper context.

    This does not verify associations between
    values, classifiers, and metrics.
    """

    if figure_type != "chart":
        return {
            "unsupported_percentages": [],
            "checked": False,
        }

    description_values = set(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*%",
            description,
        )
    )

    context_values = set(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*%",
            context_text,
        )
    )

    def normalize(value):
        return value.replace(" ", "")

    known = {
        normalize(value)
        for value in context_values
    }

    unsupported = [
        value
        for value in sorted(description_values)
        if normalize(value) not in known
    ]

    return {
        "unsupported_percentages": unsupported,
        "checked": bool(context_text.strip()),
    }


# --------------------------------------------------
# Evaluate description
# --------------------------------------------------

def evaluate_description(
    source_text,
    context_text,
    description,
    figure_type,
    finish_reason,
):
    labels = check_labels(
        source_text,
        description,
    )

    warnings = check_semantic_risks(
        figure_type,
        description,
    )

    numeric = check_numerical_evidence(
        description,
        context_text,
        figure_type,
    )

    empty = not bool(description)

    truncated = finish_reason == "length"

    complete_sentence = bool(
        re.search(
            r'[.!?]["\')\]]?$',
            description,
        )
    )

    needs_retry = bool(
        empty
        or truncated
        or not complete_sentence
        or labels["missing_labels"]
        or warnings
        or numeric["unsupported_percentages"]
    )

    return {
        "empty": empty,
        "truncated": truncated,
        "complete_sentence": complete_sentence,
        "label_checks": labels,
        "semantic_warnings": warnings,
        "numerical_checks": numeric,
        "needs_retry": needs_retry,
    }


# --------------------------------------------------
# Build short retry feedback
# --------------------------------------------------

def build_feedback(evaluation):
    issues = []

    if evaluation["truncated"]:
        issues.append(
            "The response was truncated; shorten it."
        )

    if evaluation["empty"]:
        issues.append(
            "Produce a visible, nonempty description."
        )

    if not evaluation["complete_sentence"]:
        issues.append(
            "Finish with a complete sentence."
        )

    missing = evaluation[
        "label_checks"
    ]["missing_labels"]

    if missing:
        issues.append(
            "Include important visible labels: "
            + ", ".join(missing)
        )

    unsupported = evaluation[
        "numerical_checks"
    ]["unsupported_percentages"]

    if unsupported:
        issues.append(
            "These percentages are not found in "
            "the retrieved supporting text: "
            + ", ".join(unsupported)
        )

    issues.extend(
        evaluation["semantic_warnings"][:2]
    )

    return issues[:3]


# --------------------------------------------------
# Select best attempt
# --------------------------------------------------

def choose_attempt(attempts):
    if not attempts:
        return None, "analysis_failed"

    passed = [
        attempt
        for attempt in attempts
        if not attempt[
            "evaluation"
        ]["needs_retry"]
    ]

    if passed:
        return passed[-1], "provisional"

    def score(attempt):
        evaluation = attempt["evaluation"]

        return (
            int(evaluation["empty"]) * 100
            + int(evaluation["truncated"]) * 50
            + int(
                not evaluation["complete_sentence"]
            ) * 20
            + len(
                evaluation["label_checks"][
                    "missing_labels"
                ]
            ) * 10
            + len(
                evaluation["semantic_warnings"]
            ) * 10
            + len(
                evaluation["numerical_checks"][
                    "unsupported_percentages"
                ]
            ) * 10
        )

    return min(
        attempts,
        key=score,
    ), "review_required"


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def run_validation(
    manifest_path,
    figure_numbers,
):
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    pdf_path = (
        DOCUMENTS_DIR / manifest["source"]
    )

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    output_folder = (
        OUTPUT_DIR / manifest_path.parent.name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_figures = [
        figure
        for figure in manifest["figures"]
        if figure["figure_number"] in figure_numbers
    ]

    with pymupdf.open(pdf_path) as pdf:
        for figure in selected_figures:

            number = figure["figure_number"]
            page_number = figure["page"]

            print("\n" + "=" * 60)
            print(
                f"Figure {number} | Page {page_number}"
            )
            print("=" * 60)

            image_path = resolve_image_path(
                figure["image_path"],
                manifest_path,
            )

            if not image_path.is_file():
                print(f"Missing image: {image_path}")
                continue

            source_text = extract_figure_text(
                pdf,
                figure,
            )

            figure_type = classify_figure(
                figure["caption"],
                source_text,
            )

            context = retrieve_paper_context(
                pdf_path=pdf_path,
                figure_number=number,
                figure_page=page_number,
            )

            context_text = context["text"]

            print(f"Figure type: {figure_type}")
            print(
                "Supporting passages: "
                f"{len(context['passages'])}"
            )
            print(
                "Context characters: "
                f"{len(context_text)}"
            )

            if context["error"]:
                print(
                    "Context retrieval warning: "
                    + context["error"]
                )

            attempts = []
            feedback = None
            last_error = None

            for attempt_number in range(
                1,
                MAX_RETRIES + 2,
            ):
                print(
                    f"\nAnalysis attempt {attempt_number}"
                )

                try:
                    generation = generate_description(
                        image_path=image_path,
                        figure=figure,
                        source_text=source_text,
                        figure_type=figure_type,
                        context_text=context_text,
                        retry_feedback=feedback,
                    )

                except Exception as error:
                    last_error = str(error)
                    print(
                        f"Generation error: {error}"
                    )
                    break

                description = generation[
                    "description"
                ]

                evaluation = evaluate_description(
                    source_text=source_text,
                    context_text=context_text,
                    description=description,
                    figure_type=figure_type,
                    finish_reason=generation[
                        "finish_reason"
                    ],
                )

                attempts.append({
                    "attempt": attempt_number,
                    "description": description,
                    "generation": {
                        "finish_reason": generation[
                            "finish_reason"
                        ],
                        "eval_count": generation[
                            "eval_count"
                        ],
                        "prompt_eval_count": generation[
                            "prompt_eval_count"
                        ],
                    },
                    "evaluation": evaluation,
                })

                print(
                    "Finish reason: "
                    f"{generation['finish_reason']}"
                )
                print(
                    "Description characters: "
                    f"{len(description)}"
                )

                print(
                    "Missing labels: "
                    + (
                        ", ".join(
                            evaluation[
                                "label_checks"
                            ]["missing_labels"]
                        )
                        or "None"
                    )
                )

                print(
                    "Unsupported percentages: "
                    + (
                        ", ".join(
                            evaluation[
                                "numerical_checks"
                            ]["unsupported_percentages"]
                        )
                        or "None"
                    )
                )

                print(
                    "Semantic warnings: "
                    f"{len(evaluation['semantic_warnings'])}"
                )

                print(
                    "Truncated: "
                    f"{evaluation['truncated']}"
                )

                if not evaluation["needs_retry"]:
                    print(
                        "No issues detected "
                        "by current checks."
                    )
                    break

                feedback = build_feedback(
                    evaluation
                )

                if attempt_number <= MAX_RETRIES:
                    print(
                        "Retrying with targeted feedback."
                    )

            chosen, status = choose_attempt(
                attempts
            )

            final_description = (
                chosen["description"]
                if chosen
                else ""
            )

            final_evaluation = (
                chosen["evaluation"]
                if chosen
                else None
            )

            model_slug = re.sub(
                r"[^A-Za-z0-9_-]+",
                "_",
                VISION_MODEL,
            )

            stem = (
                f"page_{page_number}_"
                f"figure_{number}_"
                f"{model_slug}"
            )

            description_path = (
                output_folder
                / f"{stem}_description.txt"
            )

            report_path = (
                output_folder
                / f"{stem}_report.json"
            )

            report = {
                "source": manifest["source"],
                "figure_number": number,
                "page": page_number,
                "caption": figure["caption"],
                "figure_type": figure_type,
                "vision_model": VISION_MODEL,
                "context_size": NUM_CTX,
                "image_path": str(image_path),
                "extraction_status": figure["status"],
                "extracted_figure_text": source_text,
                "paper_context": context["passages"],
                "context_error": context["error"],
                "status": status,
                "attempts": attempts,
                "selected_attempt": (
                    chosen["attempt"]
                    if chosen
                    else None
                ),
                "final_description": final_description,
                "final_evaluation": final_evaluation,
                "last_error": last_error,
                "limitations": [
                    "Passing checks does not prove accuracy.",
                    "Numerical matching checks values only, "
                    "not their association with metrics.",
                    "Figure edge directions are not "
                    "independently verified.",
                    "Context passages may be incomplete "
                    "or only partially relevant.",
                    "Crop completeness is not certified.",
                ],
            }

            report_path.write_text(
                json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            description_path.write_text(
                final_description,
                encoding="utf-8",
            )

            print(f"\nFinal status: {status}")
            print(
                f"Completed attempts: {len(attempts)}"
            )

            if last_error:
                print(f"Last error: {last_error}")

            print(
                f"Description: {description_path}"
            )
            print(f"Report: {report_path}")


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-grounded scientific "
            "figure descriptions."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to figure manifest",
    )

    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        default=[7],
        help="Figure numbers to analyze",
    )

    args = parser.parse_args()

    manifest_path = (
        args.manifest
        if args.manifest
        else find_manifest()
    )

    print(f"Vision model: {VISION_MODEL}")
    print(f"Context size: {NUM_CTX}")
    print(f"Generation limit: {NUM_PREDICT}")

    run_validation(
        manifest_path=manifest_path,
        figure_numbers=args.figures,
    )


if __name__ == "__main__":
    main()
