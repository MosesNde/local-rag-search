"""Query local ChromaDB documents with explicit PDF table/figure references."""
import re
import json
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

from app_paths import DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    # A strong matching clause alone is not enough: a sentence may contain
    # additional claims that require other passages. Require most of the
    # *whole sentence* to be covered before crediting one source.
    if whole_coverage >= 0.83:
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


def wants_concrete_example(question):
    """Identify requests for an illustrative scenario, not mere mentions of examples."""
    return bool(re.search(
        r"\b(?:give|show|provide|create|write|generate|walk\s+through|illustrate)\b"
        r"[^\n?]{0,100}\b(?:example|scenario|case\s+study|illustration)\b"
        r"|\b(?:hypothetical|practical|concrete|real[- ]world)\s+"
        r"(?:example|scenario)\b",
        question, re.I))


def generate_grounded_answer(question, context, answer_mode="direct", conversation_context=""):
    mode_guidance = ""
    if answer_mode == "reason":
        mode_guidance = (
            "The user requests analytical reasoning. Answer using exactly two paragraphs "
            "with the headings 'Documented premise:' and 'Reasoned inference:'. "
            "In Documented premise, state specific facts actually present in the retrieved "
            "source context, without speculation. In Reasoned inference, draw a cautious "
            "conclusion from those facts and explicitly say it is a possibility rather "
            "than an experimentally established result. Do NOT insert citation markers: "
            "the program separately attributes only the documented premise. "
            "If you cannot identify a relevant documented premise, reply exactly "
            "INSUFFICIENT_INFORMATION. "
        )
    # Examples are creative illustrations of supported mechanisms, not source facts.
    # The model must give a situation/action/outcome rather than repeat a definition.
    example_guidance = (
        "The user requests an example, not another explanation. Start with "
        "'Hypothetical example:' and describe ONE specific situation: who/what "
        "is involved, what concrete action happens, and what an observer might "
        "see as a result. For instance, use a named sample document, query, "
        "tool request, or dataset when relevant; explicitly treat invented details "
        "as illustrative rather than facts reported by the source. "
        "Use the retrieved context ONLY for the underlying mechanism; do not "
        "assert the scenario occurred in the document, invent experimental "
        "results, or guarantee an outcome. If behavior is contingent, say 'might' "
        "or 'could'. Keep the example to 3-5 complete sentences. "
        "If the context does not establish the underlying mechanism, answer "
        "exactly INSUFFICIENT_INFORMATION. "
        if wants_concrete_example(question) else ""
    )
    prompt = (
        mode_guidance + example_guidance +
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
        "Previous conversation (for reference resolution, NOT documentary evidence):\n"
        + conversation_context[:2400] + "\n\nContext:\n" + context
        + "\n\nQuestion:\n" + question
    )
    is_complex = (table_references(question) or figure_references(question)
                  or wants_broader_context(question) or is_multi_part_question(question))
    max_output_tokens = 320 if answer_mode == "reason" else (256 if is_complex or example_guidance else 120)

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
        # Some models append the sentinel after an explanatory sentence.
        # Treat any standalone sentinel as a refusal to answer, never render it.
        if re.search(r"\bINSUFFICIENT_INFORMATION\b", answer):
            return "INSUFFICIENT_INFORMATION"
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
    document = re.sub(r"^\s*Row\s+\d+\s*:\s*", "", document, flags=re.I)
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



def _whole_term(text, value):
    """Match an entire literal value, including technical suffixes such as C++."""
    return bool(re.search(r"(?<![A-Za-z0-9_+])" + re.escape(str(value))
                          + r"(?![A-Za-z0-9_+])", text, re.I))


def _table_fields(document):
    """Parse a PDF table row after optional Source/Page/Table/Caption headers.

    Real indexed records start with a multi-line header followed by
    'Row N: Field: value | Field: value'.  Only the row's fields count.
    """
    match = re.search(r"(?m)^\s*Row\s+\d+\s*:\s*", document, re.I)
    if match:
        row_text = document[match.end():]
    else:
        # Keep compatibility with short test fixtures and other row formats.
        row_text = re.sub(r"^\s*Row\s+\d+\s*:\s*", "", document, flags=re.I)
    fields = {}
    for part in row_text.split("|"):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        name = name.strip().casefold()
        if name:
            fields[name] = value.strip()
    return fields


def _query_matched_table_row(question, records):
    """Select one row only when a distinctive literal category/field is named."""
    ranked = []
    for document, meta in records:
        if meta.get("file_type") != "pdf_table":
            continue
        fields = _table_fields(document)
        category = fields.get("owasp category")
        if category and _whole_term(question, category):
            ranked.append((document, meta, fields))
    return ranked[0] if len(ranked) == 1 else None


def _citation_for_explicit(question, answer, records, used_metadata):
    """Ground limited deterministic table/figure citations in retrieved evidence.

    Return None when the record cannot be verified; caller retains conservative
    prose citations and must never fall back to a list of retrieved passages.
    """
    usable = [(doc, m) for doc, m in records if m in used_metadata]
    tables = table_references(question)
    figures = figure_references(question)
    if len(figures) == 1 and not tables:
        figure_matches = [(doc, m) for doc, m in usable
                          if m.get("file_type") == "pdf_figure"
                          and str(m.get("figure_number")) == str(figures[0])]
        if len(figure_matches) != 1:
            return None
        doc, meta = figure_matches[0]
        # Look for nontrivial answer content explicitly present in the figure
        # description, rather than crediting only terms from the question.
        terms = evidence_terms(answer) - evidence_terms(question)
        overlap = terms & evidence_terms(doc)
        if len(overlap) >= 2 and (not terms or len(overlap) / len(terms) >= .40):
            return answer + " [1]", ["[1] " + format_source(meta)]
        return None
    if len(tables) != 1 or figures:
        return None
    table_rows = [(doc, m) for doc, m in usable if m.get("file_type") == "pdf_table"
                  and table_number_matches(m, tables[0])]
    if not table_rows:
        return None
    # Whole-table counts cannot be attributed to an arbitrary single row.
    if re.search(r"\bhow many\b|\bnumber of\b", question, re.I):
        match = re.search(r"\b(\d+)\b", answer)
        row_ids = {str(m.get("row_number")) for _, m in table_rows if m.get("row_number") is not None}
        if (match and len(row_ids) == int(match.group(1))
                and len(row_ids) == len(table_rows) and len(row_ids) >= 2):
            common = table_rows[0][1]
            source = common.get("source", "Unknown source")
            label = (f"Table {common['table_number']}" if common.get("table_number") is not None
                     else common.get("table_label", f"Table {tables[0]}"))
            page = common.get("page")
            loc = f"{source} ({label}, page {page}, rows 1-{len(row_ids)})"
            return answer + " [1]", ["[1] " + loc]
        return None
    hit = _query_matched_table_row(question, table_rows)
    if hit:
        doc, meta, fields = hit
        languages = fields.get("most affected languages")
        # Check every enumerated language, including C / C++ distinction.
        if languages:
            items = [v.strip() for v in languages.split(",") if v.strip()]
            if items and all(_whole_term(answer, item) for item in items):
                return answer + " [1]", ["[1] " + format_source(meta)]
    # Highest/lowest metric questions name the model and dataset but not the
    # winning system. Compute the extremum over all matching reported cells.
    metric = metric_for_question(question)
    if metric and re.search(r"\b(highest|lowest|largest|smallest)\b", question, re.I):
        considered = []
        for doc, meta in table_rows:
            fields = parse_metric_record(doc)
            if not fields or not all(exact_mention(question, fields[k])
                                     for k in ("model", "dataset")):
                continue
            value = parse_numeric_value(fields.get(metric))
            if value is not None:
                considered.append((value, fields["system"], meta))
        if len(considered) >= 2:
            choose_max = bool(re.search(r"\b(highest|largest)\b", question, re.I))
            best_value = (max if choose_max else min)(item[0] for item in considered)
            winners = [item for item in considered if item[0] == best_value]
            if len(winners) == 1:
                number, system, meta = winners[0]
                if _whole_term(answer, system) and _whole_term(answer, f"{number:g}"):
                    return answer + " [1]", ["[1] " + format_source(meta)]
    # Exact metric cell, when the requested model, system and dataset all
    # appear together in the row, and the value matches the answer.
    if metric:
        matches = []
        for doc, meta in table_rows:
            fields = parse_metric_record(doc)
            if not fields:
                continue
            if not all(exact_mention(question, fields[k]) for k in ("model", "system", "dataset")):
                continue
            number = parse_numeric_value(fields.get(metric))
            if number is None:
                continue
            if _whole_term(answer, f"{number:g}"):
                matches.append(meta)
        if len(matches) == 1:
            return answer + " [1]", ["[1] " + format_source(matches[0])]
    return None


def cite_explicit_answer(question, answer, records, used_metadata):
    """Prefer a verified exact structured citation; never claim context as evidence."""
    exact = _citation_for_explicit(question, answer, records, used_metadata)
    if exact:
        return exact
    cited, sources = annotate_answer_with_evidence(answer, records, used_metadata)
    return cited, sources


def retrieved_context_sources(records, used_metadata, question):
    """Return a SMALL provenance list, explicitly NOT verified citations.

    Do not enumerate every retrieved table row as supporting evidence.
    """
    metas = [m for m in used_metadata if m.get("source")]
    if not metas:
        return []
    table_nums = table_references(question)
    if len(table_nums) == 1:
        selected = [m for m in metas if m.get("file_type") == "pdf_table"
                    and table_number_matches(m, table_nums[0])]
        if selected:
            seen = set()
            results = []
            for m in selected:
                label = (f"Table {m['table_number']}" if m.get("table_number") is not None
                         else m.get("table_label", f"Table {table_nums[0]}"))
                page = f", page {m['page']}" if m.get("page") is not None else ""
                key = f"{m['source']} ({label}{page})"
                if key not in seen:
                    seen.add(key)
                    results.append("Retrieved context (citation unverified): " + key)
            return results[:2]
    seen = set()
    results = []
    for m in metas:
        source = m.get("source")
        if source not in seen:
            seen.add(source)
            results.append("Retrieved context (citation unverified): " + format_source(m))
    return results[:3]

def cite_reasoned_answer(answer, records, used_metadata):
    """Cite documented premises only; never mislabel an inference as a reported finding.

    Lexical matching is conservative and not a proof of entailment. Unmatched
    premises are displayed without citation, with the lack of verification explicit.
    """
    premise_match = re.search(
        r"(?is)\bDocumented premise\s*:\s*(.*?)\s*\bReasoned inference\s*:\s*(.*)$",
        answer.strip(),
    )
    if premise_match:
        premise, inference = (part.strip() for part in premise_match.groups())
    else:
        # The model did not follow the two-part format: do not invent a premise.
        return ("**Reasoned analysis (not verified as a document finding):** "
                + re.sub(r"\[\d+\]", "", answer).strip(), [])
    if not premise or not inference:
        return ("**Reasoned analysis (not verified as a document finding):** "
                + re.sub(r"\[\d+\]", "", answer).strip(), [])
    premise = re.sub(r"\[\d+\]", "", premise).strip()
    inference = re.sub(r"\[\d+\]", "", inference).strip()
    cited_premise, sources = annotate_answer_with_evidence(premise, records, used_metadata)
    # All references, if any, belong ONLY to the premise paragraph.
    premise_title = ("**Documented premise (matched to retrieved evidence):** "
                     if sources else "**Documented premise (citation not verified):** ")
    return (premise_title + cited_premise + "\n\n"
            + "**Reasoned inference (not a finding established by the document):** "
            + inference, sources)




# --------------------------------------------------
# Source-balanced multi-document comparison
# --------------------------------------------------
COMPARISON_REQUEST = re.compile(
    r"\b(?:compare|contrast|differences?|similarities|versus|vs\.?|"
    r"how\s+do(?:es)?\s+.+\s+(?:differ|compare))\b", re.I
)
GENERIC_FILENAME_WORDS = {
    'the','and','for','with','from','based','standard','paper','study',
    'research','data','dataset','software','comprehensive','approach',
    'analysis','evaluation','system','systems','pdf','top','ten'
}


def named_comparison_sources(collection, question):
    """Resolve two *explicitly named* indexed PDFs; avoid guessing sources.

    A short distinctive token such as 'AgentChaos' or 'OWASP' must be present
    in the question. A plain word such as 'dataset' cannot resolve a PDF.
    """
    if not COMPARISON_REQUEST.search(question):
        return []
    q = question.casefold()
    meta = collection.get(include=['metadatas']).get('metadatas') or []
    names = sorted({m.get('source') for m in meta
                    if isinstance(m, dict) and str(m.get('source','')).lower().endswith('.pdf')})
    scores = []
    for name in names:
        stem = Path(name).stem.casefold()
        words = [w for w in re.findall(r'[a-z0-9]+', stem)
                 if len(w) >= 4 and w not in GENERIC_FILENAME_WORDS]
        matched = [w for w in set(words) if re.search(r'(?<![a-z0-9])'+re.escape(w)+r'(?![a-z0-9])', q)]
        exact = name.casefold() in q or stem.replace('_',' ') in q
        # Require distinctive names, not generic common terms.
        if exact or matched:
            scores.append((100 if exact else 10*len(matched)+max(map(len,matched)), name))
    scores.sort(key=lambda x:(-x[0],x[1]))
    # The initial version intentionally supports only exactly two clear matches.
    return [name for _, name in scores[:2]] if len(scores) >= 2 else []


def balanced_comparison(collection, question, conversation_context=''):
    """Retrieve a separate contextual passage set for each named PDF.

    This is read-only and does not alter the existing ChromaDB collection.
    """
    names = named_comparison_sources(collection, question)
    if len(names) != 2:
        return None
    groups = []
    for name in names:
        # The same requested comparison aspects are retained, but each
        # embedding is focused on ONE named entity.
        subject = Path(name).stem.replace('_', ' ')
        search_text = (f'{subject}: purpose objectives methods approach '
                       'evaluation vulnerabilities weaknesses limitations. '
                       f'User comparison: {question[:550]}')
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=search_text)['embeddings'][0]
        found = semantic_search_by_vector(collection, embedding, name, 5)
        if not found:
            return (f'I could not find sufficiently relevant evidence in {name} '
                    'for this comparison. No comparison was generated.', [])
        expanded = add_neighboring_text_chunks(collection, found, max_added=2)
        context, used = build_context(expanded, 7000)
        if not context:
            return (f'I could not build usable evidence from {name}. '
                    'No comparison was generated.', [])
        groups.append((name, expanded, used, context))
    # Force balanced input budgets: one PDF cannot crowd out the other.
    joined = '\n\n'.join(f'=== SOURCE {i+1}: {name} ===\n{context}'
                       for i,(name,_,_,context) in enumerate(groups))
    prompt = (
        'Compare the TWO source documents using ONLY the evidence below. '
        'You must cover both separately, then explain similarities and differences '
        'in objectives, methods, and weaknesses addressed. '
        'Do not confuse agent robustness with software vulnerabilities. '
        'Use the exact headings "Source A:", "Source B:", and "Comparison:". '
        'Describe one or two important documented facts about each in its section. '
        'In the Comparison section, distinguish facts from your own interpretations '
        'and avoid unsupported claims. Do not claim that the approaches have no overlap, ''are mutually exclusive, or can be directly ranked without explicit evidence. ''Say their primary focus differs, while possible overlaps need further evaluation. ''Keep the comparison concise, without repeating either source summary. ''Do not include citation markers; the program '
        'will annotate evidence separately. If either source lacks adequate information, '
        'reply INSUFFICIENT_INFORMATION. '
        'Conversation context is for resolving references, never evidence:\n'
        + conversation_context[-1600:] + '\n\nRetrieved documents:\n' + joined
        + '\n\nQuestion: ' + question
    )
    response = ollama.chat(model=ANSWER_MODEL, messages=[{'role':'user','content':prompt}],
                           think=False, options={'num_predict':700})
    answer = response['message']['content'].strip()
    if not answer or answer == 'INSUFFICIENT_INFORMATION':
        return INSUFFICIENT_MESSAGE, []
    pieces = re.search(
        r'(?is)\bSource A:\s*(.*?)\s*\bSource B:\s*(.*?)\s*\bComparison:\s*(.*)',answer)
    if not pieces:
        # Don't offer an unsupported freeform comparison without provenance.
        return ('Could not reliably separate the two documents in the generated '
                'comparison. Please retry the question.', [])
    # The model sometimes emits Markdown heading syntax inside the three
    # captured sections. Normalize it here, before assembling our own headings.
    def clean_comparison_part(raw):
        raw = raw.strip()
        raw = re.sub(r'^\\s*(?:\\*\\*|__)+\\s*', '', raw)
        raw = re.sub(r'\\s*(?:\\*\\*|__)+\\s*$', '', raw)
        raw = re.sub(r'(?m)^\\s*(?:\\*\\*|__)+\\s*$', '', raw)
        raw = re.sub(r'\\n{3,}', '\n\n', raw)
        return raw.strip()

    parts = [clean_comparison_part(p) for p in pieces.groups()]

    # Allocate evidence locators separately for each paper. This guarantees
    # that a source is not silently dropped when sentence-level lexical
    # matching fails. Provenance alone is NOT claim-level verification.
    formatted, numbered_sources = [], []
    for idx in range(2):
        name, records, used, _ = groups[idx]
        cited, matched = annotate_answer_with_evidence(parts[idx], records, used)
        local_ids = {}
        for old_id, locator in enumerate(matched, 1):
            new_id = len(numbered_sources) + 1
            local_ids[str(old_id)] = str(new_id)
            numbered_sources.append(
                f'[{new_id}] ' + re.sub(r'^\[\d+\]\s*', '', locator))
        cited = re.sub(r'\[(\d+)\]',
                       lambda m: '[' + local_ids.get(m.group(1), m.group(1)) + ']',
                       cited)
        if not matched:
            # Preserve the provenance of the actual indexed passage. A source
            # link says where the premise was retrieved, not that every
            # generated statement has been independently verified.
            eligible = [m for m in used if m.get('source', '').casefold() == name.casefold()]
            if eligible:
                best = eligible[0]
                number = len(numbered_sources) + 1
                numbered_sources.append(f'[{number}] ' + format_source(best))
                cited += (f'\n\nSupporting source context [{number}] '
                          '(individual claims not verified).')
        heading = f'**Source {"AB"[idx]} ({name})**'
        formatted.append(heading + '\n' + cited)
    # This paragraph is a model-generated synthesis, not a source quotation.
    # Do not add spurious direct citations to its conclusions.
    formatted.append(
        '**Comparison (evidence-based synthesis)**\n\n'
        + parts[2] + '\n\n'
        + '*The comparison describes differences in primary objectives; '
        'it does not establish that the approaches have no overlapping '
        'security implications or directly comparable effectiveness.*'
    )
    return '\n\n'.join(formatted), numbered_sources



def answer_question(question, answer_mode="direct", conversation_context=""):
    total_started = time.perf_counter()
    started = time.perf_counter()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name="documents")
    profile_log("database_connection=" + seconds(started) + f" records={collection.count()}")

    # Two explicit document names require balanced retrieval, before
    # generic semantic ranking or the single-document explicit route.
    if COMPARISON_REQUEST.search(question):
        comparison = balanced_comparison(collection, question, conversation_context)
        if comparison is not None:
            profile_log("route=balanced_comparison total=" + seconds(total_started))
            return comparison

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
        answer = generate_grounded_answer(question, context, answer_mode, conversation_context)
        profile_log("total=" + seconds(total_started))
        if answer == "INSUFFICIENT_INFORMATION" or not answer:
            return INSUFFICIENT_MESSAGE, []
        if answer_mode == "reason":
            return cite_reasoned_answer(answer, records, used)
        cited, sources = cite_explicit_answer(question, answer, records, used)
        if not sources:
            sources = retrieved_context_sources(records, used, question)
        return cited, sources

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
        answer = generate_grounded_answer(question, context, answer_mode, conversation_context)
        sufficient = bool(answer and answer != "INSUFFICIENT_INFORMATION")
        profile_log(f"stage={stage_number} sufficient={sufficient} answer_chars={len(answer)}")
        if sufficient:
            profile_log(f"expanded={stage_number > 1} total={seconds(total_started)}")
            if answer_mode == "reason":
                return cite_reasoned_answer(answer, records, used)
            cited_answer, matched_sources = annotate_answer_with_evidence(answer, records, used)
            sources = matched_sources or retrieved_context_sources(records, used, question)
            return cited_answer, sources
    profile_log("no_sufficient_answer total=" + seconds(total_started))
    return INSUFFICIENT_MESSAGE, []



# Conversation support is kept in the retrieval engine. Existing standalone
# answer_question() and command-line behavior remain backwards compatible.
EXPLANATION_REQUEST = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:explain(?:\s+(?:it|that|this))?(?:\s+again)?|"
    r"(?:summari[sz]e|simplify|clarify)(?:\s+(?:it|that|this))?|"
    r"(?:give|provide|show)(?:\s+me)?\s+(?:an?\s+)?example(?:\s+of\s+(?:it|that|this))?|"
    r"(?:put|say)\s+(?:it|that|this)\s+in\s+(?:simple|plain)\s+(?:terms|words))"
    r"(?:\s+in\s+(?:simple|plain)\s+(?:terms|words))?\s*[?.!]*\s*$", re.I)
REFERENCE_REQUEST = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|their)\b|^\s*(?:what about|how about)\b", re.I)
REASONING_REQUEST = re.compile(
    r"\b(?:limitations?|drawbacks?|tradeoffs?|trade-offs?|implications?|infer|inference|"
    r"hypothe(?:tical|sis)|what if|why might|why would|could (?:this|it)|"
    r"how (?:could|would) (?:i|we)|apply (?:it|this))\b", re.I)


def _recent_context(history, max_messages=10):
    result = []
    for m in history[-max_messages:]:
        if m.get('role') not in ('user', 'assistant'):
            continue
        content = re.sub(r'\[\d+\]', '', m.get('content', '')).strip()
        if content:
            result.append(f"{m['role']}: {content[:450]}")
    return '\n'.join(result)[-3000:]


def _topic_anchor(history):
    for m in reversed(history[-30:]):
        if m.get('role') != 'user':
            continue
        q = m.get('content', '').strip()
        if q and not EXPLANATION_REQUEST.fullmatch(q) and not REFERENCE_REQUEST.search(q):
            return q[:600]
    return None


def _last_cited_answer(history):
    for msg in reversed(history[-16:]):
        if msg.get('role') != 'assistant':
            continue
        sources = msg.get('sources') or []
        valid = {m.group(1): src for src in sources if (m := re.match(r'^\[(\d+)\]\s+', src))}
        cited = list(dict.fromkeys(re.findall(r'\[(\d+)\]', msg.get('content', ''))))
        numbers = [n for n in cited if n in valid]
        if numbers:
            return msg, numbers, valid
    return None


def _last_grounded_answer(history):
    """Find latest assistant answer with evidence; never steal older topic citations."""
    for message in reversed(history):
        if message.get('role') != 'assistant':
            continue
        content = (message.get('content') or '').strip()
        sources = message.get('sources') or []
        if content and sources:
            return message
        # The latest answer was unsupported: don't jump backward to a
        # different answer and attach the wrong document's citations.
        if content:
            break
    return None


def _retrieve_followup_evidence(previous, history):
    """Recover relevant *actual passages* for an uncited but retrieved answer.

    Do not promote 'Retrieved context (citation unverified)' into a verified
    citation. The returned passages are grounding context, not claim-level proof.
    """
    candidates = set()
    for src in previous.get('sources') or []:
        found = re.search(r'([^/\\\n]+?\.pdf)\s*\(', src, re.I)
        if found:
            candidates.add(found.group(1).strip())
        found = re.search(r'(?<![/\w])([\w.+-]+\.txt)\s*\(', src, re.I)
        if found:
            candidates.add(found.group(1).strip())
    anchor = _topic_anchor(history) or ''
    found = explicit_pdf_name(anchor)
    if found:
        candidates.add(found)
    if not candidates:
        return None
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name='documents')
    question = anchor or (previous.get('content') or '')[:300]
    vector = ollama.embed(model=EMBEDDING_MODEL, input=question)['embeddings'][0]
    for source in sorted(candidates):
        records = semantic_search_by_vector(collection, vector, source, 5)
        if records:
            context, used = build_context(records, 6000)
            if context and used:
                return context, source
    return None


def _reframe_previous(question, history):
    """Transform a cited or retrieved answer, labeling hypothetical evidence honestly."""
    previous = _last_grounded_answer(history)
    if previous is None:
        return ('I could not find supporting document context for the previous '
                'topic. Please ask a document-specific question first.', [])
    is_example = bool(re.search(r'\bexample\b', question, re.I))
    prior_text = re.sub(r'\[\d+\]', '', previous['content']).strip()
    valid = {m.group(1): src for src in previous.get('sources') or []
             if (m := re.match(r'^\[(\d+)\]\s+', src))}
    cited = list(dict.fromkeys(re.findall(r'\[(\d+)\]', previous['content'])))
    numbers = [num for num in cited if num in valid]
    evidence_context = ''
    # A retrieved but unverified source is provenance, not proof of a claim.
    # For an illustrative transformation, the previous answer itself supplies
    # the conceptual premise. Do not make a second search a prerequisite.
    unverified_sources = [
        src for src in previous.get('sources') or []
        if src.startswith('Retrieved context (citation unverified): ')
    ]
    # Keep the original source locator (page/chunk) and status intact.
    unverified_source = unverified_sources[0] if unverified_sources else None
    if not numbers and not unverified_source:
        return ('The previous explanation has no identifiable supporting source; '
                'please ask a document-specific question first.', [])

    instructions = (
        "Write ONE concrete hypothetical scenario that illustrates the mechanism "
        "in the previous explanation. Start with 'Hypothetical example:'. "
        "Name a sample user or actor, a specific document/query/tool/action, "
        "and a possible observable outcome. Example structure: 'Imagine [actor] "
        "does [specific action]. The system [supported mechanism]. The result "
        "might be [plausible outcome].' Do NOT simply explain the mechanism "
        "again. Invent illustrative scenario details only; do not invent facts "
        "about the paper, quantitative results, or guaranteed failures. "
        "Do not claim this scenario happened in the document. "
        if is_example else
        "Explain the previous answer in simple everyday language, without adding facts. "
    )
    prompt = (
        "Continue the SAME subject. " + instructions
        + "Use only the previous answer and supplied supporting passages as factual premises. "
        "Do not output any citation markers; citation status is handled separately. "
        "Keep it to 2-5 complete sentences and avoid new numerical claims.\n"
        + "Recent conversation:\n" + _recent_context(history, 8)
        + "\nPrevious answer:\n" + prior_text[:2200]
        + ("\nRetrieved supporting passages (not independently claim-verified):\n"
           + evidence_context if evidence_context else '')
        + "\nUser request:\n" + question + "\nResponse:"
    )
    response = ollama.chat(model=ANSWER_MODEL,
                           messages=[{'role': 'user', 'content':prompt}],
                           think=False, options={'num_predict':350})
    result = re.sub(r'\[\d+\]', '', response['message']['content']).strip()
    if not result or not re.search(r'[.!?]$', result):
        return ('The generated follow-up was incomplete. Please retry.', [])
    old_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', prior_text+evidence_context))
    new_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', result))
    if not new_numbers.issubset(old_numbers):
        return ('The generated follow-up introduced unsupported numbers. '
                'Please retry with a more specific request.', [])
    sources = [valid[n] for n in numbers]
    if is_example:
        if not result.casefold().startswith('hypothetical example:'):
            result = '**Hypothetical example:** ' + result
        if numbers:
            marker = ' '.join(f'[{num}]' for num in numbers)
            result += ('\n\n*This example is AI-generated, not reported in the document. '
                       'The underlying mechanism is supported by the earlier cited '
                       'evidence ' + marker + '.*')
        else:
            result += ('\n\n*This example is AI-generated, not reported in the document. '
                       'It is based on the earlier explanation, which referenced ' + unverified_source.removeprefix('Retrieved context (citation unverified): ')
                       + '; individual claims have not been citation-verified.*')
            sources = [unverified_source]
    elif numbers:
        result += ' ' + ' '.join(f'[{num}]' for num in numbers)
    else:
        result += ('\n\n*Based on retrieved context from ' + unverified_source
                   + ' (claim-level citation unverified).*')
        sources = [unverified_source]
    return result, sources



# Context condensing keeps the full original chat in SQLite. Summary is ONLY
# conversational context; it is never accepted as source evidence for citations.
RECENT_VERBATIM_MESSAGES = 12
SUMMARY_BATCH_MESSAGES = 8
# This matches the SQLite chat_summaries column's existing size validation.
MAX_SUMMARY_CHARS = 10000
MAX_BATCH_NOTE_CHARS = 900


def condense_older_history(history, previous_summary='', start_message_index=0):
    """Append one immutable, traceable summary segment per eligible message batch.

    Earlier segments are copied without being rewritten by the LLM. The
    source user questions are included verbatim (up to 220 chars each) to
    preserve topic anchors even when the generated batch note is incomplete.
    If the limit is reached or summarization fails, the checkpoint does not
    advance. The full SQLite chat archive is always retained.
    """
    eligible = max(0, len(history) - RECENT_VERBATIM_MESSAGES)
    batch_count = min(SUMMARY_BATCH_MESSAGES, eligible)
    if batch_count == 0:
        return previous_summary, 0
    if not isinstance(start_message_index, int) or start_message_index < 0:
        return previous_summary, 0
    if len(previous_summary) > MAX_SUMMARY_CHARS:
        return previous_summary, 0

    batch = history[:batch_count]
    transcript = []
    verbatim_topics = []
    for offset, message in enumerate(batch):
        role = message.get('role')
        if role not in {'assistant', 'user'}:
            return previous_summary, 0
        content = str(message.get('content') or '').strip()
        content = re.sub(r'\[\d+\]', '', content)
        if role == 'user' and content:
            # Preserve the actual topic/question independently of the LLM summary.
            verbatim_topics.append(f'- Message {start_message_index + offset + 1}: {content[:220]}')
        transcript.append(f'{role}: {content[:800]}')

    prompt = (
        'Write a concise note describing ONLY the following batch of earlier '
        'messages from ONE conversation. Do not rewrite or summarize any older '
        'memory. Preserve significant earlier conclusions, doubts, user goals, '
        'and distinctions between source evidence and hypotheses. Do not invent '
        'facts and never claim hypotheses were proven. Use at most 110 words. '
        'Output only the batch note, no headings.\n\n'
        + '\n'.join(transcript)
    )
    try:
        response = ollama.chat(
            model=ANSWER_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            think=False, options={'num_predict':220, 'temperature':0},
        )
        note = response['message']['content'].strip()
        if not (20 <= len(note) <= MAX_BATCH_NOTE_CHARS):
            return previous_summary, 0
        if note.casefold() in {'insufficient_information', 'none'}:
            return previous_summary, 0
    except Exception as exc:
        profile_log('summary_skip=' + type(exc).__name__)
        return previous_summary, 0

    begin = start_message_index + 1
    finish = start_message_index + batch_count
    segment = (
        f'[Chat messages {begin}-{finish}]\n'
        'User topics (original questions):\n'
        + ('\n'.join(verbatim_topics) if verbatim_topics else '- No user question in this batch')
        + '\nBatch discussion note (generated; not verified document evidence):\n'
        + note + '\n'
    )
    new_summary = previous_summary.rstrip() + ('\n\n' if previous_summary.strip() else '') + segment
    if len(new_summary) > MAX_SUMMARY_CHARS:
        # Never discard old segments or advance the watermark merely to fit.
        profile_log('summary_full; checkpoint left unchanged')
        return previous_summary, 0
    return new_summary, batch_count


def conversation_memory_context(summary):
    if not summary:
        return ''
    return ('Older chat memory (not verified document evidence; never cite it):\n'
            + summary[:MAX_SUMMARY_CHARS] + '\n')


def resolve_conversation_intent(question, history, summary=""):
    """Use the active chat to interpret EVERY incoming question before retrieval.

    Model output is untrusted routing advice: cap its size and select from
    known intent categories. Documentary answers still come from the RAG engine.
    """
    if not history:
        return {'intent': 'research', 'standalone_question': question}

    prompt = (
        "You resolve the meaning of the latest message in one chat. "
        "Return ONLY a JSON object with two string fields: "
        "intent (one of: rewrite, example, research) and standalone_question. "
        "Use earlier messages to resolve pronouns and omitted subjects. "
        "An example request (including 'practical example', 'real-world example', "
        "'show me an example') has intent example when it refers to the prior topic. "
        "Requests to explain or summarize the prior answer have intent rewrite. "
        "Requests for new facts, comparisons or limitations have intent research. "
        "If the latest user message names a new subject, do NOT force it to the old one. "
        "Never answer the question; only restate the latest request in standalone form. "
        "Never insert factual claims, citations or details not established in the chat. "
        "Example: earlier question 'How does a vector database help RAG?', "
        "latest 'Can you give me a practical example?' -> "
        '{"intent":"example","standalone_question":"Give a practical hypothetical '
        'example of how a vector database helps retrieval-augmented generation."}.\n'
        "Chat history:\n" + conversation_memory_context(summary) + _recent_context(history, 12)
        + "\nLatest user message:\n" + question
    )
    try:
        response = ollama.chat(
            model=ANSWER_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            think=False, format='json', options={'num_predict':180, 'temperature':0},
        )
        raw = response['message']['content'].strip()
        data = json.loads(raw)
        intent = data.get('intent')
        standalone = str(data.get('standalone_question', '')).strip()
        if (intent in {'rewrite', 'example', 'research'}
                and 4 <= len(standalone) <= 900
                and not re.search(r'(?i)\b(?:ignore previous instructions|system prompt)\b',standalone)):
            return {'intent': intent, 'standalone_question': standalone}
    except (ValueError, KeyError, TypeError, AttributeError, ollama.ResponseError):
        pass

    # Safe fallback when an offline model cannot produce a valid JSON route.
    # Do not retrieve a generic request like 'give an example' in isolation.
    anchor = _topic_anchor(history)
    if re.search(r'(?i)\bexample\b', question) and anchor and len(question.split()) <= 12:
        return {'intent':'example', 'standalone_question':
                f'Give an illustrative example about: {anchor}'}
    if (re.search(r'(?i)\b(?:explain|simplify|summari[sz]e|plain terms)\b',question)
            and anchor and len(question.split()) <= 12):
        return {'intent':'rewrite', 'standalone_question':f'Explain simply: {anchor}'}
    if anchor and REFERENCE_REQUEST.search(question):
        return {'intent':'research', 'standalone_question':
                question.rstrip(' ?') + '. Regarding: ' + anchor}
    return {'intent': 'research', 'standalone_question': question}


# Explicit requests about the conversation are different from questions about
# documents. Never send these through document retrieval as a substitute for memory.
CHAT_RECALL_RE = re.compile(
    r"\b(?:we|you and i|our)\s+(?:discussed|talked about|concluded|decided|identified|said|mentioned)\b"
    r"|\b(?:earlier|previously|last time)\b.*\b(?:discussed|talked|concluded|decided|identified|mentioned)\b"
    r"|\b(?:what (?:did|was)|remind me)\b.*\b(?:we|our|earlier)\b",
    re.I,
)


def select_recall_messages(question, history, max_pairs=5):
    """Find earlier original chat exchanges by distinctive terms, not recent order.

    Return matching question/answer pairs; never claim that the content has
    been independently verified against documents.
    """
    ignore = {'what','was','were','when','which','where','about','the','that',
              'this','from','with','have','earlier','previously','discussed',
              'talked','concluded','said','our','you','your','we','and','for',
              'could','would','did','does','their','those','they','them','how'}
    terms = {t for t in re.findall(r'[a-z0-9]+',question.casefold())
             if len(t)>=4 and t not in ignore}
    scored=[]
    for i, message in enumerate(history):
        if message.get('role') != 'user':
            continue
        text = str(message.get('content') or '')
        words = set(re.findall(r'[a-z0-9]+',text.casefold()))
        score = sum(3 if len(t)>=8 else 1 for t in terms & words)
        # The assistant's prior answer can hold the details asked about.
        answer = str(history[i+1].get('content') or '') if i+1<len(history) and history[i+1].get('role')=='assistant' else ''
        awords = set(re.findall(r'[a-z0-9]+',answer.casefold()))
        score += sum(1 for t in terms & awords)
        if score:
            scored.append((score,i,text,answer))
    scored.sort(key=lambda r:(-r[0],r[1]))
    chosen=sorted(scored[:max_pairs],key=lambda r:r[1])
    return '\n'.join(f'Earlier user: {user[:550]}\nEarlier assistant: {assistant[:1300]}'
                     for _,_,user,assistant in chosen)[-6500:]


def recall_from_chat(question, history, summary):
    """Answer chat-history recall without inventing document-grounded citations.

    Summaries are helpful memory, but the original chat remains authoritative;
    if no relevant prior discussion is present in this chat, say so clearly.
    """
    transcript = select_recall_messages(question, history)
    memory = conversation_memory_context(summary) + transcript
    if not memory.strip():
        return ('I cannot find an earlier discussion of that topic in this chat. '
                'Try opening the conversation where it was discussed.'), []

    prompt = (
        'You are recalling what was actually discussed in ONE saved chat. '
        'This is a conversation-memory task, not a new research question. '
        'Use ONLY the supplied chat messages and the condensed summary. '
        'Do not infer a fresh answer from general knowledge or document excerpts. '
        'If the requested discussion cannot be identified in this chat, set '
        'found=false. If found=true, summarize the earlier discussion faithfully '
        'and briefly; distinguish a prior hypothesis from a documented finding. '
        'Never claim an unverified prior model answer is a proven fact. '
        'Return ONLY JSON with keys found (boolean) and answer (string); '
        'do not include Markdown formatting or citation markers.\n\n'
        'Relevant chat history (not independently verified evidence):\n'
        + memory[:8500] + '\n\nRecall request:\n' + question
    )
    try:
        response = ollama.chat(
            model=ANSWER_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            think=False, format='json',
            options={'num_predict':240, 'temperature':0},
        )
        data = json.loads(response['message']['content'])
        answer = str(data.get('answer', '')).strip()
        if data.get('found') is True and 15 <= len(answer) <= 1500:
            answer = re.sub(r'\[(?:\d+)\]', '', answer).strip()
            return ('**From our earlier discussion in this chat:** '
                    + answer + '\n\n'
                    '*Conversation recall, not independently verified against the paper.*'), []
    except Exception as exc:
        profile_log('chat_recall_failed=' + type(exc).__name__)
    return ('I could not reliably find that earlier discussion in this chat. '
            'This is a chat-memory limitation, not evidence that the paper '
            'does not discuss the topic.'), []


def answer_conversation(question, history, summary="", original_recall_history=None):
    """Resolve references on every turn, then use citations and RAG as appropriate."""
    history = [m for m in history if m.get('role') in ('user', 'assistant')][-30:]
    if CHAT_RECALL_RE.search(question):
        return recall_from_chat(question, original_recall_history or history, summary)
    route = resolve_conversation_intent(question, history, summary)
    intent = route['intent']
    resolved = route['standalone_question']
    if intent in {'example', 'rewrite'} and history:
        # Base invented illustrations on an existing cited explanation, not
        # a broad semantic search that may discover unrelated Python tutorials.
        prompt_for_rewrite = ('Give an example of it' if intent == 'example'
                              else 'Explain it in simple terms')
        return _reframe_previous(prompt_for_rewrite, history)

    mode = 'reason' if (REASONING_REQUEST.search(resolved) and not wants_concrete_example(question)) else 'direct'
    answer, sources = answer_question(resolved, answer_mode=mode,
                                      conversation_context=conversation_memory_context(summary) + _recent_context(history))
    if mode == 'reason' and answer != INSUFFICIENT_MESSAGE:
        if 'Reasoned inference:' in answer:
            answer = answer.replace('Reasoned inference:',
                '**Reasoned inference (not explicitly stated in the document):**')
    return answer, sources

def main():
    if '--conversation-json' in sys.argv:
        try:
            payload = json.load(sys.stdin)
            question = str(payload.get('question', '')).strip()
            history = payload.get('history', [])
            if not isinstance(history, list):
                raise ValueError('history must be a list')
            earlier_summary = str(payload.get('summary', ''))[:MAX_SUMMARY_CHARS]
            checkpoint = int(payload.get('covered_messages', 0))
            if checkpoint < 0:
                raise ValueError('Invalid memory checkpoint')
            updated_summary, consumed = condense_older_history(history, earlier_summary, checkpoint)
            recent = history[consumed:]
            original_recall_history = payload.get('recall_history')
            if not isinstance(original_recall_history, list):
                original_recall_history = None
            answer, sources = answer_conversation(
                question, recent, updated_summary, original_recall_history)
            print(json.dumps({'answer': answer, 'sources': sources,
                              'summary': updated_summary,
                              'covered_messages': checkpoint + consumed},
                             ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
            sys.exit(1)
        return
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
