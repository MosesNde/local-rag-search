"""Validate JSON candidates from pdf_tables.py; never writes to ChromaDB.

Usage:
    python src/pdf_table_validation.py data/table_tests/agentchaos_tables.json
    python src/pdf_table_validation.py data/table_tests/agentchaos_tables.json --strict

The checks are structural/arithmetic, NOT ground truth verification against a PDF.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

MISSING = {'', '—', '–', '-', 'n/a', 'na', 'none', 'null'}
NUMBER_RE = re.compile(r'^\s*[+\-−]?\d+(?:\.\d+)?\s*[%×]?\s*$')


def number(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    raw = str(value).strip().replace('−', '-').replace(',', '').rstrip('%×').strip()
    try:
        v = float(raw)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def issue(kind, code, description, record=None):
    result = {'severity': kind, 'code': code, 'message': description}
    if record is not None:
        result['record'] = record
    return result


def validate_table(table):
    number_id = str(table.get('number', '?'))
    struct = table.get('structured') or {}
    normalized = table.get('normalized') or {}
    content = struct if isinstance(struct.get('records'), list) else normalized
    rows = content.get('records', [])
    alerts = []
    extracted = list(content.get('issues') or [])
    for previous_issue in extracted:
        alerts.append(issue('warning', 'extractor_warning', str(previous_issue)))

    if not rows:
        alerts.append(issue('error', 'missing_rows', 'No structured rows available for validation.'))
        return table_result(table, rows, alerts)

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            alerts.append(issue('error', 'invalid_row', 'Row is not an object.', i))
        elif not any(v is not None and str(v).strip() for v in row.values()):
            alerts.append(issue('error', 'empty_row', 'All row fields are empty.', i))
    rows = [r for r in rows if isinstance(r, dict)]

    # Check source-specific *schema features*, not table-number-specific data:
    # long-form triples (model, system, dataset, metric, value)
    if rows and {'model','system','dataset','metric','value'}.issubset(rows[0]):
        groups = defaultdict(dict)
        for i, r in enumerate(rows):
            key = (r.get('model'), r.get('system'), r.get('dataset'))
            metric = str(r.get('metric', '')).strip()
            if metric in groups[key]:
                alerts.append(issue('error', 'duplicate_metric', f'Duplicate metric {metric!r} for {key!r}.', i))
            groups[key][metric] = r.get('value')
            value = r.get('value')
            if value is not None and number(value) is None:
                alerts.append(issue('error', 'invalid_number', f'Non-numeric metric value {value!r}.', i))
        delta_names = {'Δ', '𝚫', '∆', 'delta'}
        for key, metrics in groups.items():
            without = number(metrics.get('w/o'))
            with_fault = number(metrics.get('w/'))
            delta_key = next((k for k in metrics if k in delta_names), None)
            delta = number(metrics.get(delta_key)) if delta_key else None
            if without is not None and with_fault is not None and delta is not None:
                if abs((without - with_fault) - delta) > 0.075:
                    alerts.append(issue('warning', 'delta_mismatch', f'{key}: {without} - {with_fault} != {delta}'))
            elif any(v is not None for v in [without, with_fault, delta]):
                alerts.append(issue('warning', 'incomplete_triple', f'Incomplete w/o, w/, Δ for {key!r}.'))
        if not groups:
            alerts.append(issue('error', 'no_groups', 'No metric groups.'))

    # Column-oriented rows
    if rows and 'system' in rows[0] and any(' / ' in k for k in rows[0]):
        for i, r in enumerate(rows):
            for prefix in ['LLM Calls', 'Tool Calls']:
                a, b, ratio = (number(r.get(prefix+' / '+suffix)) for suffix in ['w/o', 'w/', 'Ratio'])
                if a is None or b is None or ratio is None:
                    alerts.append(issue('warning', 'missing_ratio', f'{prefix} row has missing operands.', i))
                elif a == 0:
                    alerts.append(issue('warning', 'zero_denominator', f'{prefix} baseline is zero.', i))
                # Values can be rounded to 2 decimals; allow resulting ratio rounding uncertainty.
                elif abs(b/a - ratio) > max(0.08, 0.015*abs(ratio)):
                    alerts.append(issue('warning', 'ratio_mismatch', f'{prefix}: {b}/{a} differs from {ratio}.', i))

    if rows and {'Aborted','Reported','Silent'}.issubset(rows[0]):
        for i, r in enumerate(rows):
            vals = [number(r[k]) for k in ['Aborted','Reported','Silent']]
            if any(v is None for v in vals):
                alerts.append(issue('warning', 'non_numeric_percent', 'Missing breakdown percentage.', i))
            elif abs(sum(vals) - 100) > 0.2:
                alerts.append(issue('warning', 'percent_sum', f'Breakdown sums to {sum(vals):.2f}%, not 100%.', i))

    if rows and {'Category','Fault type','Content','Tool call'}.issubset(rows[0]):
        valid_markers = {None, 'observed_tick', 'observed_blank', '✓', '✔', ''}
        for i, r in enumerate(rows):
            if not r.get('Category') or not r.get('Fault type'):
                alerts.append(issue('error','missing_taxonomy_label','Taxonomy row missing category or fault type.',i))
            for key in ('Content','Tool call'):
                if r.get(key) not in valid_markers:
                    alerts.append(issue('warning','uncertain_marker',f'{key}: {r.get(key)!r}',i))
        alerts.append(issue('warning','taxonomy_checkmarks_unverified','Checkmark-to-row associations cannot be certified from JSON alone.'))

    if rows and 'section' in rows[0]:
        for i, r in enumerate(rows):
            if not r.get('section'):
                alerts.append(issue('error','missing_section','Row lacks a section label.',i))

    # General completeness / duplicates for wide records
    for i, r in enumerate(rows):
        if any(not str(k).strip() for k in r):
            alerts.append(issue('warning','blank_header','Column header is blank.',i))
        if any(isinstance(v, str) and '\ufffd' in v for v in r.values()):
            alerts.append(issue('warning','replacement_character','Unicode replacement character suggests decoding loss.',i))

    return table_result(table, rows, alerts)


def table_result(table, rows, alerts):
    errors = sum(a['severity'] == 'error' for a in alerts)
    warnings = sum(a['severity'] == 'warning' for a in alerts)
    return {
        'number': str(table.get('number', '?')),
        'page': table.get('page'),
        'caption': table.get('caption'),
        'source_status': table.get('status'),
        'record_count': len(rows),
        'validation_status': 'blocked' if errors else ('review_required' if warnings else 'structurally_consistent'),
        'error_count': errors,
        'warning_count': warnings,
        'issues': alerts,
    }


def validate_file(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    tables = data.get('tables')
    if not isinstance(tables, list):
        raise ValueError('Expected JSON top-level "tables" array.')
    results = [validate_table(t) for t in tables]
    keys = [(r['number'], r['page']) for r in results]
    if len(keys) != len(set(keys)):
        for result in results:
            if keys.count((result['number'], result['page'])) > 1:
                result['issues'].append(issue('error', 'duplicate_table', 'Same table number and page appears more than once.'))
                result['error_count'] += 1
                result['validation_status'] = 'blocked'
    return {
        'source': data.get('source'),
        'validation_scope': 'JSON structure, arithmetic, completeness heuristics; not source-PDF visual verification',
        'table_count': len(tables),
        'summary': {
            'structurally_consistent': sum(r['validation_status'] == 'structurally_consistent' for r in results),
            'review_required': sum(r['validation_status'] == 'review_required' for r in results),
            'blocked': sum(r['validation_status'] == 'blocked' for r in results),
        },
        'tables': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_path', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--strict', action='store_true', help='Exit nonzero if any table needs review or is blocked.')
    args = parser.parse_args()
    report = validate_file(args.json_path)
    output = args.output or args.json_path.with_name(args.json_path.stem + '_validation.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for r in report['tables']:
        print(f"Table {r['number']} | {r['validation_status']} | rows={r['record_count']} | errors={r['error_count']} | warnings={r['warning_count']}")
        for a in r['issues'][:3]:
            print('   ', a['severity'], a['code'], a['message'])
    print('Saved:', output)
    if args.strict and (report['summary']['blocked'] or report['summary']['review_required']):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
