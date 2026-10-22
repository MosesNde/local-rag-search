"""Extract, structurally validate, and optionally index PDF table records.

No database writes occur unless --index is supplied or index_tables(..., dry_run=False).
Requires the project's pdf_tables.py, pdf_table_validation.py and config.py.

Run from the project root:
  python src/pdf_table_pipeline.py agentchaos.pdf
  python src/pdf_table_pipeline.py agentchaos.pdf --index

This does not certify that extracted cells match the printed PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "documents"
TEST_DIR = ROOT / "data" / "table_tests"
DB_DIR = ROOT / "data" / "chroma"


def locate_pdf(argument):
    path = Path(argument)
    if not path.is_file():
        path = DOCS_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {argument}")
    return path.resolve()


def compact(value):
    return " ".join(str(value).split()) if value is not None else ""


def get_records(table):
    for field in ("structured", "normalized"):
        block = table.get(field)
        if isinstance(block, dict) and isinstance(block.get("records"), list):
            return block["records"]
    return []


def make_rows(table, allow_taxonomy=False):
    """Return (row identifier, human readable document, confidence label)."""
    rows = get_records(table)
    result = []
    if not rows:
        return result
    is_taxonomy = {'Category', 'Fault type'}.issubset(rows[0])
    if is_taxonomy and not allow_taxonomy:
        return []
    # Long-form metric triples: group exactly three measurements per
    # (model, system, dataset), not 420 unrelated embeddings.
    if {'model', 'system', 'dataset', 'metric', 'value'}.issubset(rows[0]):
        groups = defaultdict(dict)
        for row in rows:
            key = (compact(row['model']), compact(row['system']), compact(row['dataset']))
            groups[key][compact(row['metric'])] = row['value']
        for key, metrics in groups.items():
            doc = (f"Model: {key[0]} | System: {key[1]} | Dataset: {key[2]} | "
                   f"without fault injection (w/o): {compact(metrics.get('w/o')) or 'not reported'} | "
                   f"with fault injection (w/): {compact(metrics.get('w/')) or 'not reported'} | "
                   f"delta pass@1 (Δ): {compact(next((v for k,v in metrics.items() if k in ('𝚫','Δ','∆','delta')), None)) or 'not reported'}")
            result.append(("|".join(key), doc, 'structural_only'))
        return result

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        if is_taxonomy:
            # Do NOT index uncertain checkmarks as facts.
            allowed = ('Category', 'Fault type', 'Real-world scenario')
            filtered = {k:row.get(k) for k in allowed if row.get(k) is not None}
            confidence = 'partial_taxonomy_without_checkmarks'
        else:
            filtered = {k:v for k,v in row.items() if k != 'value_raw'}
            confidence = 'structural_only'
        values = [f"{compact(k)}: {compact(v) if v is not None else 'not reported'}" for k,v in filtered.items()]
        if values:
            signature = json.dumps(filtered, ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]
            result.append((f"row{i:03d}-{digest}", ' | '.join(values), confidence))
    return result


def stable_id(pdf, page, number, row_key):
    # Namespace isolates new table records from legacy chunk/figure/table IDs.
    raw = f"{pdf.resolve()}|{page}|{number}|{row_key}"
    return 'pdf_table_v2_' + hashlib.sha256(raw.encode('utf-8')).hexdigest()


def build_entries(pdf, tables, validation, include_partial_taxonomy=True):
    validation_by_key = {(str(r['number']), int(r['page'])):r for r in validation['tables']}
    entries = []
    counts = {'structurally_consistent':0, 'review_required':0, 'blocked':0}
    for table in tables['tables']:
        number = str(table['number'])
        page = int(table['page'])
        verdict = validation_by_key.get((number, page))
        if not verdict:
            raise ValueError(f"No validation result for Table {number} page {page}")
        status = verdict['validation_status']
        counts[status] = counts.get(status,0) + 1
        eligible = status == 'structurally_consistent'
        partial = (status == 'review_required' and
                   include_partial_taxonomy and
                   any(a.get('code') == 'taxonomy_checkmarks_unverified' for a in verdict.get('issues', [])) and
                   not verdict.get('error_count'))
        if not (eligible or partial):
            continue
        for row_key, content, confidence in make_rows(table, allow_taxonomy=partial):
            doc = (f"Source: {pdf.name} | Table {number} | Page {page} | "
                   f"Caption: {compact(table.get('caption'))} | {content}")
            identifier = stable_id(pdf,page,number,row_key)
            entries.append({'id':identifier, 'document':doc, 'metadata':{
                'source':pdf.name, 'file_type':'pdf_table', 'page':page,
                'table_number':int(number) if number.isdigit() else number,
                'caption':compact(table.get('caption'))[:1000],
                'validation_status':status,'record_confidence':confidence,
                'row_key':row_key[:240]}})
    return entries, counts


def roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = str(s).strip().upper()
    if not s or any(ch not in values for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        value = values[ch]
        total += -value if value < prev else value
        prev = value
    return total


def normalized_table_number(metadata):
    """Recognize both old table_label='Table II' and new table_number=2."""
    number = metadata.get('table_number')
    if number is None:
        label = str(metadata.get('table_label', ''))
        match = re.search(r'\btable\s+([0-9]+|[IVXLCDM]+)\b', label, re.I)
        if not match:
            return None
        number = match.group(1)
    value = str(number).strip()
    return int(value) if value.isdigit() else roman_to_int(value)


def existing_table_inventory(collection, source):
    """Return (page, table number) -> existing IDs, including legacy rows."""
    rows = collection.get(
        where={'$and': [{'source': source}, {'file_type': 'pdf_table'}]},
        include=['metadatas'],
    )
    inventory = defaultdict(set)
    for record_id, metadata in zip(rows['ids'], rows['metadatas']):
        if not metadata:
            continue
        number = normalized_table_number(metadata)
        page = metadata.get('page')
        if number is None or page is None:
            continue
        try:
            inventory[(int(page), number)].add(record_id)
        except (TypeError, ValueError):
            continue
    return inventory


def index_tables(pdf_path, *, dry_run=True, include_partial_taxonomy=True,
                 batch_size=32, check_existing=False):
    pdf = locate_pdf(pdf_path)
    from pdf_tables import extract
    from pdf_table_validation import validate_file
    extraction_path = TEST_DIR / (pdf.stem + '_tables.json')
    # The extractor writes this JSON but does not touch the DB.
    tables = extract(pdf, extraction_path)
    validation = validate_file(extraction_path)
    validation_path = extraction_path.with_name(extraction_path.stem + '_validation.json')
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    entries, counts = build_entries(pdf, tables, validation, include_partial_taxonomy)
    result = {'source':pdf.name,'tables':counts,'prepared_records':len(entries),
              'new_records':0,'already_indexed':0,'dry_run':bool(dry_run),
              'legacy_tables_skipped':[], 'legacy_records_preserved':0,
              'database_checked':False,
              'extraction_report':str(extraction_path), 'validation_report':str(validation_path)}
    print(f"Table records prepared: {len(entries)} | table statuses: {counts}")
    if dry_run and not check_existing:
        print('DRY RUN: ChromaDB has not been accessed or modified.')
        return result
    if not entries:
        print('No eligible records.')
        return result

    import chromadb
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_or_create_collection('documents')
    result['database_checked'] = True
    inventory = existing_table_inventory(collection, pdf.name)
    grouped_entries = defaultdict(list)
    for entry in entries:
        m = entry['metadata']
        grouped_entries[(int(m['page']), int(m['table_number']))].append(entry)

    pending_entries = []
    for (page, number), group in sorted(grouped_entries.items()):
        existing_ids = inventory.get((page, number), set())
        legacy_ids = [id_ for id_ in existing_ids if not id_.startswith('pdf_table_v2_')]
        if legacy_ids:
            # Prefer safety over attempting to reconcile different row granularities.
            # Leave the entire legacy table unchanged, even when its row count differs.
            result['legacy_tables_skipped'].append({
                'page': page, 'table_number': number,
                'existing_legacy_records': len(legacy_ids),
                'candidate_records_skipped': len(group),
            })
            result['legacy_records_preserved'] += len(legacy_ids)
            result['already_indexed'] += len(group)
            print(f'Table {number}, page {page}: {len(legacy_ids)} legacy rows exist; '
                  f'skipping {len(group)} new candidate rows to avoid duplicates.')
        else:
            pending_entries.extend(group)

    # Account for the v2 records already present without replacing their contents.
    unique_ids = [entry['id'] for entry in pending_entries]
    existing_v2 = set()
    for start in range(0, len(unique_ids), max(1, batch_size)):
        portion = unique_ids[start:start + max(1, batch_size)]
        existing_v2.update(collection.get(ids=portion, include=[])['ids'])
    pending_entries = [e for e in pending_entries if e['id'] not in existing_v2]
    result['already_indexed'] += len(existing_v2)
    result['new_records'] = len(pending_entries) if dry_run else 0
    if dry_run:
        print(f'DRY RUN (read-only): {len(pending_entries)} records would be added. '
              'No ChromaDB writes.')
        return result

    if not pending_entries:
        print('No new eligible records.')
        return result
    import ollama
    try:
        from config import EMBEDDING_MODEL
    except ImportError:
        EMBEDDING_MODEL = 'nomic-embed-text'
    for start in range(0, len(pending_entries), batch_size):
        batch = pending_entries[start:start + batch_size]
        response = ollama.embed(model=EMBEDDING_MODEL, input=[e['document'] for e in batch])
        vectors = response['embeddings']
        if len(vectors) != len(batch):
            raise RuntimeError('Embedding count does not match records; batch not written')
        collection.add(ids=[e['id'] for e in batch], documents=[e['document'] for e in batch],
                       metadatas=[e['metadata'] for e in batch], embeddings=vectors)
        result['new_records'] += len(batch)
    print(f"Indexed {result['new_records']} new table records; {result['already_indexed']} already present.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', help='PDF filename in data/documents, or path')
    parser.add_argument('--index',action='store_true',help='Explicitly write eligible records to ChromaDB')
    parser.add_argument('--check-existing',action='store_true',help='Read existing ChromaDB metadata while dry-running; never write')
    parser.add_argument('--exclude-partial-taxonomy',action='store_true',help='Do not index category/fault type data from tables needing review')
    args = parser.parse_args()
    result = index_tables(args.pdf,dry_run=not args.index,
        include_partial_taxonomy=not args.exclude_partial_taxonomy,
        check_existing=args.check_existing)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
