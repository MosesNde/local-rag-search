"""Query local ChromaDB documents with explicit PDF table/figure references."""
import re
import math
import os
import sys
import time
from pathlib import Path

import chromadb
import ollama

try:
    from config import EMBEDDING_MODEL, ANSWER_MODEL
except ImportError:
    EMBEDDING_MODEL = "nomic-embed-text"
    ANSWER_MODEL = "qwen3:8b"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "chroma"
DISTANCE_THRESHOLD = 1.0
N_RESULTS = 10  # maximum semantic candidates when expansion is needed
INITIAL_SEMANTIC_RESULTS = 3
SECOND_STAGE_RESULTS = 5
MAX_SEMANTIC_RESULTS = 10
SEMANTIC_CONTEXT_CHARS = 3500
MAX_CONTEXT_CHARS = 18000
MAX_TABLE_ROWS = 14
PROFILE_ENABLED = os.getenv("RAG_PROFILE", "0").strip().lower() in {"1", "true", "yes"}


def profile_log(message):
    if PROFILE_ENABLED:
        print(f"[RAG PROFILE] {message}", file=sys.stderr, flush=True)


def seconds(start):
    return f"{time.perf_counter() - start:.3f}s"


INSUFFICIENT_MESSAGE = (
    "I don't have enough information in the documents to answer that question."
)


def roman_to_integer(roman):
    symbols = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    roman = roman.upper()
    if not roman or any(ch not in symbols for ch in roman):
        return None
    total, previous = 0, 0
    for ch in reversed(roman):
        n = symbols[ch]
        total += -n if n < previous else n
        previous = n
    return total if 1 <= total <= 3999 else None


def table_references(question):
    result = []
    # Works for Table 3, TABLE III, Tables 2 and 3, Tables 2, 3.
    pattern = re.compile(r"\btables?\s+((?:\d+|[IVXLCDM]+)(?:\s*(?:,|and|&)\s*(?:\d+|[IVXLCDM]+))*)", re.I)
    for match in pattern.finditer(question):
        for token in re.findall(r"\d+|[IVXLCDM]+", match.group(1), re.I):
            number = int(token) if token.isdigit() else roman_to_integer(token)
            if number is not None and number not in result:
                result.append(number)
    return result


def figure_references(question):
    result = []
    pattern = re.compile(r"\b(?:figures?|figs?\.?)\s+(\d+(?:\s*(?:,|and|&)\s*(?:\d+))*)", re.I)
    for match in pattern.finditer(question):
        for text in re.findall(r"\d+", match.group(1)):
            n = int(text)
            if n not in result:
                result.append(n)
    return result


def explicit_pdf_name(question):
    matches = re.findall(r"(?<![\w/\\])([\w.()+-]+\.pdf)\b", question, flags=re.I)
    return matches[0] if matches else None


def format_source(meta):
    if not meta:
        return "Unknown source"
    source = meta.get("source") or "Unknown source"
    file_type = meta.get("file_type", "")
    extras = []
    if file_type == "pdf_table":
        number = meta.get("table_number")
        # Support both current and old metadata forms.
        label = f"Table {number}" if number is not None else meta.get("table_label", "Unidentified table")
        extras.append(label)
        if meta.get("page") is not None:
            extras.append(f"page {meta['page']}")
        if meta.get("row_number") is not None:
            extras.append(f"row {meta['row_number']}")
    elif file_type == "pdf_figure":
        if meta.get("figure_number") is not None:
            extras.append(f"Figure {meta['figure_number']}")
        if meta.get("page") is not None:
            extras.append(f"page {meta['page']}")
    else:
        if meta.get("page") is not None:
            extras.append(f"page {meta['page']}")
        if meta.get("chunk_number") is not None:
            extras.append(f"chunk {meta['chunk_number']}")
    return f"{source} ({', '.join(extras)})" if extras else source


def get_records(collection, where):
    results = collection.get(where=where, include=["documents", "metadatas"])
    return [(d, m or {}) for d, m in zip(results.get("documents") or [], results.get("metadatas") or []) if d]


def resolve_table_source(collection, question, number, explicit_source=None):
    """Resolve table's PDF using its filename/title, not table number alone."""
    if explicit_source:
        return explicit_source
    table_records = get_records(collection, {"file_type": "pdf_table"})
    sources = {m.get("source") for _, m in table_records
               if m.get("source") and table_number_matches(m, number)}
    if len(sources) == 1:
        return next(iter(sources))
    if not sources:
        return None
    # Match distinctive filename/title words; generic terms do not identify a PDF.
    stop = {"a", "the", "of", "and", "for", "in", "on", "with", "based",
            "comprehensive", "software", "dataset", "datasets", "paper",
            "vulnerability", "vulnerabilities", "table", "standard", "pdf",
            "distribution", "categories", "category", "comparison", "across"}
    qwords = set(re.findall(r"[a-z0-9]+", question.casefold())) - stop
    ranked = []
    for source in sources:
        words = set(re.findall(r"[a-z0-9]+", Path(source).stem.casefold())) - stop
        score = sum(1 for word in qwords & words if len(word) >= 3)
        ranked.append((score, source))
    ranked.sort(reverse=True)
    if ranked and ranked[0][0] > 0 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    return None  # Ambiguous: never silently choose a different paper.


def table_number_matches(meta, number):
    if meta.get("table_number") is not None:
        return str(meta["table_number"]) == str(number)
    label = str(meta.get("table_label", "")).strip().casefold()
    return label in {f"table {number}", f"table {int_to_roman(number).lower()}"}


def table_candidates(collection, number, source):
    """Collect both modern and legacy row formats, scoped to the right PDF."""
    all_rows = get_records(collection, {"file_type": "pdf_table"})
    return [(d, m) for d, m in all_rows
            if (not source or str(m.get("source", "")).casefold() == source.casefold())
            and table_number_matches(m, number)]


def int_to_roman(number):
    result = ""
    for value, symbol in ((1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),(50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")):
        while number >= value:
            number -= value
            result += symbol
    return result


def score_table_row(question, document):
    # Compare entire identifier terms, including hyphenated model/dataset names.
    query = question.casefold()
    doc = document.casefold()
    candidates = re.findall(r"[a-z0-9]+(?:[.+_-][a-z0-9]+)*", query)
    ignore = {"the", "for", "what", "which", "from", "according", "table", "using",
              "with", "and", "that", "does", "how", "was", "were", "in", "on", "is",
              "to", "a", "an", "of", "pdf", "pass", "delta", "value"}
    score = 0
    for term in set(candidates):
        if len(term) < 2 or term in ignore:
            continue
        if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", doc):
            score += 5 if any(c.isdigit() for c in term) or "+" in term else 2
    # Exact field-value matching is essential for HumanEval vs HumanEval+.
    for field in ("model", "system", "dataset"):
        match = re.search(r"(?:^|[|])\s*" + field + r":\s*([^|]+)", doc)
        if not match:
            continue
        value = match.group(1).strip()
        if value and re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w+])", query):
            score += 35
    return score


def select_table_rows(question, records):
    ranked = sorted(records, key=lambda pair: score_table_row(question, pair[0]), reverse=True)
    return ranked[:MAX_TABLE_ROWS]


def collect_explicit(collection, query):
    explicit_source = explicit_pdf_name(query)
    records = []
    for number in table_references(query):
        source = resolve_table_source(collection, query, number, explicit_source)
        if source is None:
            profile_log(f"ambiguous_table_reference=Table {number}")
            continue
        candidates = table_candidates(collection, number, source)
        profile_log(f"table_match=Table {number} source={source} rows={len(candidates)}")
        # 'All/list' requires the entire table in row order, not relevance-ranked snippets.
        is_whole_table = bool(re.search(r"\b(?:all|list|every|entire|complete)\b", query, re.I))
        if is_whole_table:
            candidates.sort(key=lambda item: int(item[1].get("row_number") or 0))
            records.extend(candidates)
        else:
            records.extend(select_table_rows(query, candidates))
    for number in figure_references(query):
        source = explicit_source
        candidates = get_records(collection, {"file_type": "pdf_figure"})
        candidates = [(d, m) for d, m in candidates
                      if (not source or str(m.get("source", "")).lower() == source.lower())
                      and str(m.get("figure_number", "")) == str(number)]
        records.extend(candidates[:4])
    unique = []
    seen = set()
    for d, m in records:
        signature = (m.get("source"), m.get("file_type"), m.get("table_number"),
                     m.get("figure_number"), m.get("row_key"), d)
        if signature not in seen:
            seen.add(signature)
            unique.append((d, m))
    return unique


def direct_table_listing(question, records):
    """Precisely reproduce an explicit table's category/language columns."""
    if len(table_references(question)) != 1 or figure_references(question):
        return None
    if not re.search(r"\b(?:list|all|every)\b", question, re.I):
        return None
    if not re.search(r"categor(?:y|ies)", question, re.I):
        return None
    if not re.search(r"languages?", question, re.I):
        return None
    rows = []
    for doc, meta in records:
        match = re.search(r"OWASP Category:\s*(.*?)\s*\|\s*Most Affected Languages:\s*(.*?)(?:\s*\||$)", doc, re.I | re.S)
        if match:
            rows.append((int(meta.get("row_number") or 0), match.group(1).strip(), match.group(2).strip(), meta))
    if not rows or len(rows) != len(records):
        return None
    rows.sort(key=lambda item: item[0])
    answer = "OWASP categories and most affected programming languages (Table II):\n" + "\n".join(
        f"{i}. {category}: {languages}" for i, (_, category, languages, _) in enumerate(rows, 1)
    )
    sources = list(dict.fromkeys(format_source(meta).rsplit(", row ", 1)[0] + ")" if ", row " in format_source(meta) else format_source(meta)
                                 for _, _, _, meta in rows))
    return answer, sources


def semantic_search(collection, question, source, n_results=INITIAL_SEMANTIC_RESULTS):
    """Fetch a small semantic candidate set; keep distances for expansion.

    This performs one embedding request per call. The caller passes a cached
    query vector when it expands retrieval to avoid redundant embeddings.
    """
    count = collection.count()
    if count == 0:
        return []
    response = ollama.embed(model=EMBEDDING_MODEL, input=question)
    vector = response["embeddings"][0]
    return semantic_search_by_vector(collection, vector, source, n_results)


def semantic_search_by_vector(collection, vector, source, n_results):
    count = collection.count()
    if count == 0:
        return []
    kwargs = {
        "query_embeddings": [vector],
        "n_results": min(n_results, count),
        "include": ["documents", "metadatas", "distances"],
    }
    if source:
        kwargs["where"] = {"source": source}
    try:
        found = collection.query(**kwargs)
    except Exception:
        # Never silently broaden to other PDFs when a source is specified.
        return []
    docs = found["documents"][0]
    metas = found["metadatas"][0]
    distances = found["distances"][0]
    return [(doc, meta or {}) for doc, meta, distance in zip(docs, metas, distances)
            if distance <= DISTANCE_THRESHOLD]



def add_neighboring_text_chunks(collection, records, max_added=8):
    """Expand semantic text hits with local source context, read-only.

    Short TXT files (<= 8 chunks) are included in document order so a
    step-by-step explanation is not truncated after its first sentence.
    For long TXT/PDF sources, add only immediate neighboring chunks.
    Tables, figures, images and URLs are deliberately left unchanged.
    """
    if not records:
        return records

    def key(doc, meta):
        if meta.get("file_type") in {"txt", "pdf"} and meta.get("chunk_number") is not None:
            return ("position", meta.get("source"), meta.get("file_type"),
                    meta.get("page"), meta.get("chunk_number"))
        return ("content", meta.get("source"), meta.get("file_type"), doc)

    unique = {}
    for doc, meta in records:
        unique.setdefault(key(doc, meta), (doc, meta))

    groups = {}
    for doc, meta in records:
        if meta.get("file_type") not in {"txt", "pdf"}:
            continue
        number = meta.get("chunk_number")
        if not isinstance(number, int):
            continue
        group = (meta.get("source"), meta.get("file_type"), meta.get("page"))
        groups.setdefault(group, set()).add(number)

    added = 0
    for (source, file_type, page), anchor_numbers in groups.items():
        if added >= max_added or not source:
            break
        filters = [{"source": source}, {"file_type": file_type}]
        if page is not None:
            filters.append({"page": page})
        # Restrict large PDFs to adjacent chunks; short text sources can
        # provide a complete explanation without importing unrelated pages.
        if file_type == "pdf":
            wanted = sorted({n + offset for n in anchor_numbers
                             for offset in (-1, 0, 1) if n + offset >= 0})
            filters.append({"chunk_number": {"$in": wanted}})
        try:
            result = collection.get(where={"$and": filters},
                                    include=["documents", "metadatas"])
        except Exception as exc:
            profile_log(f"neighbor_lookup_skipped source={source} reason={type(exc).__name__}")
            continue
        candidates = {}
        for doc, meta in zip(result.get("documents") or [], result.get("metadatas") or []):
            meta = meta or {}
            num = meta.get("chunk_number")
            if not isinstance(num, int):
                continue
            candidates.setdefault(num, (doc, meta))
        if file_type == "txt" and len(candidates) <= 8:
            wanted = sorted(candidates)
        elif file_type == "txt":
            wanted = sorted({n + offset for n in anchor_numbers
                             for offset in (-1, 0, 1) if n + offset >= 0})
        for num in wanted:
            item = candidates.get(num)
            if item is None:
                continue
            k = key(*item)
            if k not in unique and added < max_added:
                unique[k] = item
                added += 1
    profile_log(f"neighbor_expansion={added} text_chunks_added")
    # Preserve semantic-hit priority for mixed sources. A source's
    # additional passages appear in their original sequence.
    return list(unique.values())


def remove_repeated_chunk_overlap(text, previous_text):
    """Remove repeated leading sentences from consecutive source chunks."""
    if not previous_text:
        return text
    # The indexer overlaps a whole sentence. Avoid deleting repeated words
    # elsewhere by removing only a long common suffix/prefix.
    max_length = min(len(previous_text), len(text))
    for length in range(max_length, 19, -1):
        if previous_text[-length:].strip() == text[:length].strip():
            return text[length:].lstrip()
    return text


def wants_broader_context(question):
    """Conservative routing: only clearly multi-evidence questions start at 5."""
    return bool(re.search(
        r"\b(?:compare|contrast|differences?|similarities|across|several|multiple|"
        r"summari[sz]e|overview|all (?:the |of the )?(?:ways|reasons|methods)|"
        r"advantages and disadvantages)\b",
        question, flags=re.I,
    ))


def build_context(records, character_limit):
    parts, used = [], []
    length = 0
    previous = {}
    for doc, meta in records:
        group = (meta.get("source"), meta.get("file_type"), meta.get("page"))
        previous_chunk = previous.get(group)
        if (meta.get("file_type") in {"txt", "pdf"} and previous_chunk is not None
                and meta.get("chunk_number") == previous_chunk[0] + 1):
            cleaned = remove_repeated_chunk_overlap(doc, previous_chunk[1])
        else:
            cleaned = doc
        section = f"Source: {format_source(meta)}\nContent: {cleaned}"
        if length + len(section) > character_limit:
            continue
        parts.append(section)
        used.append(meta)
        length += len(section)
        if meta.get("file_type") in {"txt", "pdf"}:
            previous[group] = (meta.get("chunk_number"), doc)
    return "\n\n".join(parts), used


# Conservative extractive citation matching. This checks term overlap and exact
# numeric tokens; it is not an entailment model or a factuality guarantee.
CITATION_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "that", "this",
    "these", "those", "it", "its", "as", "at", "from", "which", "who", "can",
    "then", "also", "into", "using", "use", "uses", "used", "through", "their",
    "them", "they", "such", "each", "when", "how", "does", "what", "will",
}


def normalized_evidence_tokens(text):
    """Normalize frequent grammatical variants for conservative lexical matching."""
    words = re.findall(r"[a-z0-9]+(?:[.+_-][a-z0-9]+)*", text.casefold())
    terms = []
    for word in words:
        if len(word) <= 2 or word in CITATION_STOPWORDS:
            continue
        # Limited inflection rules rather than broad semantic assumptions.
        if word.endswith("ing") and len(word) > 6:
            stem = word[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            word = stem
        elif word.endswith("es") and len(word) > 5 and word[-3] in "shxz":
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
            word = word[:-1]
        terms.append(word)
    return terms


def evidence_terms(text):
    return set(normalized_evidence_tokens(text))


def evidence_match_score(sentence, document):
    """Score a paraphrase only when substantial sentence content matches.

    A short high-overlap clause is useful, but it cannot stand in for an
    otherwise unsupported sentence: whole-sentence coverage is also required.
    """
    tokens = normalized_evidence_tokens(sentence)
    terms = set(tokens)
    dterms = evidence_terms(document)
    shared = terms & dterms
    if len(shared) < 4 or not terms:
        return None
    whole_coverage = len(shared) / len(terms)
    if whole_coverage < 0.48:
        return None
    # Find a meaningful contiguous clause/window in the original answer.
    best_window = 0.0
    for width in range(4, min(10, len(tokens)) + 1):
        for start in range(len(tokens) - width + 1):
            window = set(tokens[start:start + width])
            if len(window) < 4:
                continue
            coverage = len(window & dterms) / len(window)
            if len(window & dterms) >= 4:
                best_window = max(best_window, coverage)
    if whole_coverage >= 0.70:
        return whole_coverage + 0.05 * best_window
    if best_window >= 0.80 and whole_coverage >= 0.48:
        return whole_coverage + 0.05 * best_window
    return None


def multi_evidence_matches(sentence, candidates, max_sources=3):
    """Find complementary passages when no single passage supports a sentence.

    Requires high combined lexical coverage, significant *new* terms from each
    added passage, and consistent numeric evidence. This does not establish
    logical entailment or verify that claims are true.
    """
    terms = evidence_terms(sentence)
    if len(terms) < 7:
        return []
    numeric = set(re.findall(r"(?<![\w])\d+(?:\.\d+)?(?:%|\+)?", sentence))

    # Group by source so a high aggregate score cannot be assembled out of
    # unrelated documents that happen to share common terminology.
    groups = {}
    for doc, meta in candidates:
        doc_numbers = set(re.findall(r"(?<![\w])\d+(?:\.\d+)?(?:%|\+)?", doc))
        dt = evidence_terms(doc)
        shared = terms & dt
        if len(shared) >= 3:
            groups.setdefault(meta.get("source"), []).append((doc, meta, shared, doc_numbers))

    successful = []
    for source, group in groups.items():
        selected, covered, covered_numbers = [], set(), set()
        pool = group[:]
        while pool and len(selected) < max_sources:
            # Choose the passage adding the greatest amount of answer content;
            # break equal scores in favor of earlier original chunk indices.
            pool.sort(key=lambda item: (
                -len(item[2] - covered),
                item[1].get("chunk_number") if isinstance(item[1].get("chunk_number"), int) else 10**9,
            ))
            best = pool.pop(0)
            new_terms = best[2] - covered
            if len(new_terms) < (3 if not selected else 2):
                break
            selected.append(best)
            covered.update(best[2])
            covered_numbers.update(best[3])
            if len(selected) >= 2 and len(covered) / len(terms) >= 0.83 and numeric.issubset(covered_numbers):
                break
        if (2 <= len(selected) <= max_sources
                and len(covered) / len(terms) >= 0.83
                and numeric.issubset(covered_numbers)):
            successful.append((len(covered) / len(terms), len(selected), source, selected))

    if not successful:
        return []
    successful.sort(key=lambda item: (-item[0], item[1]))
    chosen = successful[0][3]
    chosen.sort(key=lambda item: (item[1].get("page") or 0,
        item[1].get("chunk_number") if isinstance(item[1].get("chunk_number"), int) else 10**9))
    return [(doc, meta) for doc, meta, _, _ in chosen]


def annotate_answer_with_evidence(answer, records, used_metadata):
    """Annotate supported sentences using text included in the query context.

    Lexical evidence matching is a useful heuristic, not proof of entailment.
    No citation is added if a supporting passage cannot be identified.
    """
    candidates = []
    for meta in used_metadata:
        matched = next(((doc, m) for doc, m in records if m is meta), None)
        if matched is None:
            matched = next(((doc, m) for doc, m in records if m == meta), None)
        if matched:
            candidates.append(matched)

    # Adjacent indexed chunks intentionally repeat a sentence. For citation
    # attribution, credit that sentence to its FIRST available occurrence,
    # rather than a later chunk that merely contains the overlap.
    # Only trim against passages actually included in the generated context.
    by_position = {}
    for doc, meta in candidates:
        if meta.get("file_type") not in {"txt", "pdf"}:
            continue
        n = meta.get("chunk_number")
        if isinstance(n, int):
            key = (meta.get("source"), meta.get("file_type"), meta.get("page"), n)
            by_position.setdefault(key, doc)

    citation_candidates = []
    for doc, meta in candidates:
        n = meta.get("chunk_number")
        if meta.get("file_type") in {"txt", "pdf"} and isinstance(n, int) and n > 0:
            predecessor = by_position.get((meta.get("source"), meta.get("file_type"),
                                           meta.get("page"), n - 1))
            if predecessor:
                doc = remove_repeated_chunk_overlap(doc, predecessor)
        citation_candidates.append((doc, meta))

    segments = re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer)
    result, citations, ids = [], [], {}
    for sentence in segments:
        original = sentence.strip()
        if not original:
            continue
        numeric = set(re.findall(r"(?<![\w])\d+(?:\.\d+)?(?:%|\+)?", original))
        scored = []
        for doc, meta in citation_candidates:
            nums_doc = set(re.findall(r"(?<![\w])\d+(?:\.\d+)?(?:%|\+)?", doc))
            if numeric and not numeric.issubset(nums_doc):
                continue
            score = evidence_match_score(original, doc)
            if score is not None:
                scored.append((score, doc, meta))
        if scored:
            # When an individual passage covers enough of the answer, prefer
            # that more precise citation to a multi-passage attribution.
            scored.sort(key=lambda item: (-item[0],
                item[2].get("chunk_number") if isinstance(item[2].get("chunk_number"), int) else 10**9))
            selected = [(scored[0][1], scored[0][2])]
        else:
            selected = multi_evidence_matches(original, citation_candidates)
        if not selected:
            result.append(original)
            continue
        numbers = []
        for _, metadata in selected:
            source_key = (metadata.get("source"), metadata.get("file_type"),
                          metadata.get("page"), metadata.get("table_number"),
                          metadata.get("table_label"), metadata.get("figure_number"),
                          metadata.get("chunk_number"), metadata.get("row_number"))
            if source_key not in ids:
                ids[source_key] = len(ids) + 1
                citations.append(f"[{ids[source_key]}] {format_source(metadata)}")
            numbers.append(ids[source_key])
        result.append(original + " " + " ".join(f"[{n}]" for n in dict.fromkeys(numbers)))
    return " ".join(result), citations


def is_multi_part_question(question):
    """Recognize explicitly joined questions without broadening retrieval."""
    return bool(
        len(re.findall(r"\b(?:what|how|why|which|who|when|where)\b", question, re.I)) >= 2
        or len(re.findall(r"[?]", question)) >= 2
        or re.search(r"\b(?:and|also)\s+(?:describe|explain|list|compare|summari[sz]e)\b", question, re.I)
    )


def answer_ends_cleanly(answer):
    """Reject visibly unfinished prose; tables and ordered lists may end differently."""
    answer = answer.strip()
    if not answer or answer == "INSUFFICIENT_INFORMATION":
        return bool(answer)
    if answer[-1] in ".!?":
        return True
    # Preserve structured exact values and complete numbered/bulleted lists.
    if "\n" in answer and re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", answer):
        return True
    return False


def generate_grounded_answer(question, context):
    prompt = (
        "Answer using ONLY the retrieved context. If the evidence is insufficient, "
        "reply exactly INSUFFICIENT_INFORMATION. Never guess. "
        "For questions about how something works, give its essential steps and "
        "outcome in 2-4 sentences, when supported. Answer EVERY part of a "
        "multi-part question; finish each sentence. Be accurate, complete, "
        "and concise; avoid filler and unsupported details. "
        "For tables, preserve exact model, system, dataset and metric associations; "
        "missing values are not zero. "
        "For figures, describe only supported labels and relationships; do not invent "
        "numerical values.\n\n"
        "Context:\n" + context + "\n\nQuestion:\n" + question
    )
    is_complex = (table_references(question) or figure_references(question)
                  or wants_broader_context(question) or is_multi_part_question(question))
    max_output_tokens = 256 if is_complex else 120

    for attempt in range(2):
        generation_started = time.perf_counter()
        response = ollama.chat(
            model=ANSWER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"num_predict": max_output_tokens},
        )
        answer = response["message"]["content"].strip()
        reason = getattr(response, "done_reason", None)
        token_count = getattr(response, "eval_count", None)
        limit_reached = (reason == "length" or
                         (isinstance(token_count, int) and token_count >= max_output_tokens))
        complete = answer_ends_cleanly(answer)
        if PROFILE_ENABLED:
            def ns_to_s(v):
                return f"{(v or 0)/1_000_000_000:.3f}s"
            profile_log(
                "ollama_chat=" + seconds(generation_started)
                + " load=" + ns_to_s(getattr(response, "load_duration", 0))
                + " prompt_eval=" + ns_to_s(getattr(response, "prompt_eval_duration", 0))
                + " token_eval=" + ns_to_s(getattr(response, "eval_duration", 0))
                + f" prompt_tokens={getattr(response, 'prompt_eval_count', None)}"
                + f" generated_tokens={token_count}"
                + f" limit={max_output_tokens} done_reason={reason}"
                + f" complete={complete} attempt={attempt + 1}"
            )
        if answer == "INSUFFICIENT_INFORMATION":
            return answer
        if complete and not limit_reached:
            return answer
        if attempt == 0:
            # Retry the whole question once so a continuation cannot introduce
            # unsupported content disconnected from the retrieved context.
            max_output_tokens = max(384, max_output_tokens * 2)
            profile_log(f"answer_retry=incomplete_or_length new_limit={max_output_tokens}")
    profile_log("answer_rejected=still_incomplete_after_retry")
    return "INSUFFICIENT_INFORMATION"


# --------------------------------------------------
# Deterministic numeric ranking for structured tables
# --------------------------------------------------

def parse_metric_record(document):
    """Parse the explicit key/value format emitted by pdf_table_pipeline.py."""
    fields = {}
    for part in document.split("|"):
        if ":" in part:
            key, value = part.split(":", 1)
            fields[key.strip().casefold()] = value.strip()
    required = ("model", "system", "dataset")
    if any(not fields.get(key) for key in required):
        return None
    return fields


def exact_mention(question, value):
    """Do not confuse datasets like HumanEval and HumanEval+."""
    value = str(value).strip()
    if not value:
        return False
    return bool(re.search(
        r"(?<![\w.+_-])" + re.escape(value) + r"(?![\w.+_-])",
        question, flags=re.IGNORECASE,
    ))


def metric_for_question(question):
    lower = question.casefold()
    # Only perform computation when the requested metric is unambiguous.
    if re.search(r"(?:δ|∆|𝚫|Δ|delta|degradation|drop|reduction).*?(?:pass@?1)?|pass@?1.*?(?:δ|∆|𝚫|Δ|delta|degradation|drop|reduction)", question, re.I):
        return "delta pass@1 (δ)"
    if re.search(r"\bw/o\b|\bwithout fault injection\b", lower):
        return "without fault injection (w/o)"
    if re.search(r"\bw/\b|\bwith fault injection\b", lower):
        return "with fault injection (w/)"
    return None


def parse_numeric_value(text):
    if text is None:
        return None
    raw = str(text).strip().replace("−", "-").replace("–", "-")
    if raw.casefold() in ("", "not reported", "missing", "none", "n/a", "-", "null"):
        return None
    raw = raw.removesuffix("%").strip()
    try:
        result = float(raw)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def deterministic_ranking(collection, question):
    """Return (answer, sources), or None to retain existing RAG fallback.

    Only activate for an explicit single table and unambiguous model/dataset
    pair. Read every table record (not semantic top-k) and sort in Python.
    """
    lower = question.casefold()
    if not re.search(r"\brank(?:ing|ed)?\b|\bsort\b|\border\b", lower):
        return None
    table_numbers = table_references(question)
    if len(table_numbers) != 1 or figure_references(question):
        return None
    source = explicit_pdf_name(question)
    if not source:
        # Avoid accidentally combining identically numbered tables from PDFs.
        return None
    records = table_candidates(collection, table_numbers[0], source)
    parsed = [(parse_metric_record(doc), meta) for doc, meta in records]
    parsed = [(fields, meta) for fields, meta in parsed if fields is not None]
    if not parsed:
        return None

    models = {fields["model"] for fields, _ in parsed}
    datasets = {fields["dataset"] for fields, _ in parsed}
    selected_models = [v for v in models if exact_mention(question, v)]
    selected_datasets = [v for v in datasets if exact_mention(question, v)]
    if len(selected_models) != 1 or len(selected_datasets) != 1:
        return None
    metric = metric_for_question(question)
    if not metric:
        return None

    ascending = bool(re.search(r"lowest\s+to\s+highest|smallest\s+to\s+largest|ascending|increasing\s+order", lower))
    descending = bool(re.search(r"highest\s+to\s+lowest|largest\s+to\s+smallest|descending|decreasing\s+order", lower))
    if ascending == descending:
        return None  # require a single clear ordering

    selected = [(fields, meta) for fields, meta in parsed
                if fields["model"] == selected_models[0]
                and fields["dataset"] == selected_datasets[0]]
    if not selected:
        return None

    values = {}
    absent = set()
    for fields, meta in selected:
        system = fields["system"]
        number = parse_numeric_value(fields.get(metric))
        if number is None:
            absent.add(system)
        else:
            if system in values and values[system] != number:
                return None  # conflicting rows, unsafe to rank
            values[system] = number
            absent.discard(system)
    if not values:
        return None

    ranked = sorted(values.items(), key=lambda item: (item[1], item[0].casefold()), reverse=not ascending)
    lines = [f"{i}. {system}: {value:g}" for i, (system, value) in enumerate(ranked, 1)]
    missing = sorted(absent - values.keys())
    if missing:
        lines.append("Not ranked (not reported): " + ", ".join(missing))
    mode = "ascending" if ascending else "descending"
    answer = (f"{metric} for {selected_models[0]} on {selected_datasets[0]}, "
              f"ranked in {mode} order:\n" + "\n".join(lines))
    # Distinct table/page references, not a separate citation for every row.
    sources = list(dict.fromkeys(
        f"{meta.get('source', source)} (Table {table_numbers[0]}, page {meta.get('page')})"
        for _, meta in selected if meta.get("page") is not None
    ))
    return answer, sources


def answer_question(question):
    total_started = time.perf_counter()
    started = time.perf_counter()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name="documents")
    profile_log("database_connection=" + seconds(started) + f" records={collection.count()}")

    started = time.perf_counter()
    numeric_result = deterministic_ranking(collection, question)
    profile_log("numeric_ranking_check=" + seconds(started))
    if numeric_result is not None:
        profile_log("route=deterministic_ranking total=" + seconds(total_started))
        return numeric_result

    explicit = bool(table_references(question) or figure_references(question))
    source = explicit_pdf_name(question)

    if explicit:
        started = time.perf_counter()
        records = collect_explicit(collection, question)
        profile_log(f"route=explicit retrieval={seconds(started)} records={len(records)}")
        if not records:
            return INSUFFICIENT_MESSAGE, []
        direct_result = direct_table_listing(question, records)
        if direct_result is not None:
            profile_log("route=deterministic_table_listing total=" + seconds(total_started))
            return direct_result
        started = time.perf_counter()
        context, used = build_context(records, MAX_CONTEXT_CHARS)
        profile_log(f"context_build={seconds(started)} chars={len(context)}")
        if not context:
            return INSUFFICIENT_MESSAGE, []
        answer = generate_grounded_answer(question, context)
        profile_log("total=" + seconds(total_started))
        if answer == "INSUFFICIENT_INFORMATION" or not answer:
            return INSUFFICIENT_MESSAGE, []
        cited_answer, matched_sources = annotate_answer_with_evidence(answer, records, used)
        return cited_answer, matched_sources or list(dict.fromkeys(
            format_source(meta) for meta in used if meta.get("source")
        ))

    if collection.count() == 0:
        return INSUFFICIENT_MESSAGE, []
    started = time.perf_counter()
    query_vector = ollama.embed(model=EMBEDDING_MODEL, input=question)["embeddings"][0]
    profile_log("embedding=" + seconds(started))
    first_count = SECOND_STAGE_RESULTS if wants_broader_context(question) else INITIAL_SEMANTIC_RESULTS
    stages = [first_count]
    if first_count < MAX_SEMANTIC_RESULTS:
        stages.append(MAX_SEMANTIC_RESULTS)

    for stage_number, n_results in enumerate(stages, 1):
        started = time.perf_counter()
        records = semantic_search_by_vector(collection, query_vector, source, n_results)
        profile_log(f"stage={stage_number} requested_chunks={n_results} retrieval={seconds(started)} hits={len(records)}")
        if not records:
            continue
        records = add_neighboring_text_chunks(collection, records)
        char_budget = SEMANTIC_CONTEXT_CHARS if n_results == first_count else MAX_CONTEXT_CHARS
        started = time.perf_counter()
        context, used = build_context(records, char_budget)
        profile_log(f"stage={stage_number} context_build={seconds(started)} chars={len(context)} sources={len(used)}")
        if not context:
            continue
        answer = generate_grounded_answer(question, context)
        sufficient = bool(answer and answer != "INSUFFICIENT_INFORMATION")
        profile_log(f"stage={stage_number} sufficient={sufficient} answer_chars={len(answer)}")
        if sufficient:
            profile_log(f"expanded={stage_number > 1} total={seconds(total_started)}")
            cited_answer, matched_sources = annotate_answer_with_evidence(answer, records, used)
            sources = matched_sources or list(dict.fromkeys(
                format_source(meta) for meta in used if meta.get("source")
            ))
            return cited_answer, sources
    profile_log("no_sufficient_answer total=" + seconds(total_started))
    return INSUFFICIENT_MESSAGE, []


def main():
    question = input("Enter your question: ").strip()
    if not question:
        print("\nAnswer:\nPlease enter a question.")
        return
    answer, sources = answer_question(question)
    print("\nAnswer:")
    print(answer)
    if sources:
        print("\nRetrieved sources:")
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
