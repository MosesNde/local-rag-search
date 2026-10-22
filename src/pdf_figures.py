import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / 'data' / 'documents'
FIGURES_DIR = PROJECT_ROOT / 'data' / 'figures'
RENDER_SCALE = 2
MAX_SEARCH_HEIGHT = 360
TOP_PADDING = 2
CAPTION_BOTTOM_PADDING = 0.5

CAPTION_PATTERN = re.compile(
    r'^\s*(?:Fig(?:ure)?s?\.?)\s*(\d+[A-Za-z]?)\s*[.:\-–]\s*(.*)',
    re.IGNORECASE,
)
OTHER_CAPTION_PATTERN = re.compile(
    r'^\s*(?:(?:Fig(?:ure)?s?\.?)|Table)\s*\d+[A-Za-z]?\s*[.:\-–]',
    re.IGNORECASE,
)


def source_identifier(pdf_path):
    digest = hashlib.sha256(pdf_path.name.encode('utf-8')).hexdigest()[:12]
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', pdf_path.stem).strip('_')
    return f'{safe[:48]}_{digest}'


def rect_to_list(rect):
    return [round(float(v), 2) for v in rect]


def horizontal_overlap(a, b):
    return max(0, min(a.x1, b.x1) - max(a.x0, b.x0))


def get_text_lines(page):
    items = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            text = ''.join(span.get('text', '') for span in line['spans']).strip()
            if not text:
                continue
            sizes = [span.get('size', 0) for span in line['spans']]
            items.append({
                'text': text,
                'bbox': pymupdf.Rect(line['bbox']),
                'font_size': max(sizes, default=0),
            })
    return items


def detect_captions(page):
    lines = get_text_lines(page)
    captions = []
    for item in lines:
        match = CAPTION_PATTERN.match(item['text'])
        if not match:
            continue
        number = match.group(1)
        caption = match.group(2).strip()
        bbox = pymupdf.Rect(item['bbox'])
        previous = item
        below = sorted(
            (line for line in lines if line is not item and
             line['bbox'].y0 >= item['bbox'].y0),
            key=lambda line: (line['bbox'].y0, line['bbox'].x0),
        )
        for line in below:
            if len(caption) >= 600 or re.search(r'[.!?]\s*$', caption):
                break
            gap = line['bbox'].y0 - previous['bbox'].y1
            if gap > 6:
                break
            if gap < -2:
                continue
            if abs(line['bbox'].x0 - item['bbox'].x0) > 24:
                continue
            if abs(line['font_size'] - item['font_size']) > 1.5:
                break
            if OTHER_CAPTION_PATTERN.match(line['text']):
                break
            if caption.endswith('-'):
                caption = caption[:-1] + line['text']
            else:
                caption += ' ' + line['text']
            bbox |= line['bbox']
            previous = line
        caption = ' '.join(caption.split())
        if caption and len(caption) <= 650:
            captions.append({'number': number, 'caption': caption, 'bbox': bbox})
    return captions


def get_column(page, caption):
    mid = page.rect.width / 2
    if caption.x0 < mid - 15 and caption.x1 > mid + 15:
        return pymupdf.Rect(30, 0, page.rect.width - 30, page.rect.height)
    if caption.x0 < mid:
        return pymupdf.Rect(30, 0, mid, page.rect.height)
    return pymupdf.Rect(mid, 0, page.rect.width - 30, page.rect.height)


def get_graphic_regions(page, caption, column):
    candidates = []
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for raw in clusters:
        rect = pymupdf.Rect(raw)
        if rect.width < 10 or rect.height < 6:
            continue
        if rect.y1 > caption.y0 or caption.y0 - rect.y1 > MAX_SEARCH_HEIGHT:
            continue
        if horizontal_overlap(rect, column) < rect.width * .65:
            continue
        candidates.append(rect)
    for info in page.get_image_info():
        rect = pymupdf.Rect(info['bbox'])
        if rect.width < 10 or rect.height < 6:
            continue
        if rect.y1 > caption.y0 or caption.y0 - rect.y1 > MAX_SEARCH_HEIGHT:
            continue
        if horizontal_overlap(rect, column) < rect.width * .65:
            continue
        candidates.append(rect)
    return candidates


def prior_caption_bottom(caption, captions, column):
    limit = 50  # exclude normal page header area
    for item in captions:
        other = item['bbox']
        if other.y1 < caption.y0 and horizontal_overlap(other, column) > 0:
            limit = max(limit, other.y1 + 4)
    return limit


def create_candidate(page, info, captions, text_lines):
    caption = info['bbox']
    column = get_column(page, caption)
    lower_limit = prior_caption_bottom(caption, captions, column)
    regions = [r for r in get_graphic_regions(page, caption, column)
               if r.y0 >= lower_limit]
    regions.sort(key=lambda r: r.y1, reverse=True)

    selected = []
    current_top = caption.y0
    for rect in regions:
        gap = max(0, current_top - rect.y1)
        if selected and gap > 90:
            continue
        selected.append(rect)
        current_top = min(current_top, rect.y0)

    if selected:
        top = max(lower_limit, min(r.y0 for r in selected) - TOP_PADDING)
        # Include short labels immediately above graphics, but not prose/table rows.
        for line in text_lines:
            r = line['bbox']
            if (r.y1 <= caption.y0 and r.y0 >= lower_limit and
                top - 5 <= r.y0 < top and len(line['text']) <= 55 and
                horizontal_overlap(r, column) > 0):
                top = max(lower_limit, r.y0 - 1)
        method = 'grouped_graphics'
    else:
        top = max(lower_limit, caption.y0 - MAX_SEARCH_HEIGHT)
        method = 'caption_fallback'

    # Do not expand below the actual caption: the next prose block may start 1pt later.
    crop = pymupdf.Rect(column.x0, top, column.x1,
                        caption.y1 + CAPTION_BOTTOM_PADDING) & page.rect
    return {'crop': crop, 'method': method, 'status': 'review_required',
            'reason': 'Candidate crop; visual completeness not independently verified.',
            'selected_regions': len(selected)}


def extract_pdf_figures(pdf_path, output_dir=None):
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if output_dir is None:
        output_dir = FIGURES_DIR / source_identifier(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'source': pdf_path.name, 'figures': []}
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, 1):
            captions = detect_captions(page)
            lines = get_text_lines(page)
            for info in captions:
                candidate = create_candidate(page, info, captions, lines)
                fig_no = info['number']
                image_path = output_dir / f'page_{number}_figure_{fig_no}.png'
                page.get_pixmap(
                    matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
                    clip=candidate['crop'], alpha=False,
                ).save(str(image_path))
                manifest['figures'].append({
                    'figure_number': fig_no,
                    'page': number,
                    'caption': info['caption'],
                    'image_path': str(image_path),
                    'method': candidate['method'],
                    'status': candidate['status'],
                    'reason': candidate['reason'],
                    'selected_regions': candidate['selected_regions'],
                    'crop_bbox': rect_to_list(candidate['crop']),
                    'caption_bbox': rect_to_list(info['bbox']),
                })
                print(f'Figure {fig_no} | Page {number} | '
                      f'{candidate["method"]} | {candidate["selected_regions"]} regions')
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding='utf-8')
    print(f'\nFigures detected: {len(manifest["figures"])}')
    print(f'Manifest: {manifest_path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Extract figure crops without indexing')
    parser.add_argument('pdf', nargs='?', help='PDF filename or path')
    args = parser.parse_args()
    if args.pdf:
        path = Path(args.pdf)
        if not path.is_file():
            path = DOCUMENTS_DIR / path
        paths = [path]
    else:
        paths = sorted(DOCUMENTS_DIR.glob('*.pdf'))
    for path in paths:
        print(f'\nProcessing: {path.name}')
        try:
            extract_pdf_figures(path)
        except Exception as exc:
            print(f'Extraction failed for {path.name}: {exc}')


if __name__ == '__main__':
    main()
