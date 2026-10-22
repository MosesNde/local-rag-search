"""Read-only, conservative scientific-PDF table extraction.

Uses PyMuPDF table detection when reliable; falls back to caption-anchored
coordinate text reconstruction for borderless or undersized detections.
Outputs diagnostics/JSON only, never touches vector databases.
"""
import argparse
import json
import re
from pathlib import Path
from statistics import median

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
CAPTION_RE = re.compile(r'^\s*TABLE\s+([IVXLCDM]+|\d+)\s*(?:[:.\-–]\s*(.*))?$', re.I)


def canonical_table_number(identifier):
    """Store IEEE Roman numerals in the same integer namespace as Table 1/2."""
    if identifier.isdigit():
        return str(int(identifier))
    roman = identifier.upper()
    mapping = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
               'C': 100, 'D': 500, 'M': 1000}
    total = previous = 0
    for char in reversed(roman):
        value = mapping[char]
        total += -value if value < previous else value
        previous = value
    # Require conventional Roman representation; prevent false matches.
    if total <= 0 or total > 3999:
        return None
    parts = []
    remaining = total
    for value, mark in [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),
                        (100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),
                        (9,'IX'),(5,'V'),(4,'IV'),(1,'I')]:
        while remaining >= value:
            parts.append(mark)
            remaining -= value
    return str(total) if ''.join(parts) == roman else None


NUM_RE = re.compile(r'^(?:[+\-−]?\d+(?:\.\d+)?%?|[–—-])$')


def text_lines(page):
    lines = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            spans = [s for s in line['spans'] if s.get('text', '').strip()]
            if spans:
                lines.append({'text': ''.join(s['text'] for s in spans).strip(),
                              'rect': pymupdf.Rect(line['bbox']),
                              'size': median(s['size'] for s in spans)})
    return sorted(lines, key=lambda z: (round(z['rect'].y0, 1), z['rect'].x0))


def captions(page):
    """Identify numbered and IEEE Roman table labels and wrapped title lines.

    A standalone TABLE II followed by an all-caps title is one caption.
    Excludes narrative references such as 'Table II summarizes ...'.
    """
    lines = text_lines(page)
    result = []
    for line in lines:
        match = CAPTION_RE.fullmatch(line['text'])
        if not match:
            continue
        number = canonical_table_number(match.group(1))
        if number is None:
            continue
        bbox = pymupdf.Rect(line['rect'])
        parts = [line['text']]
        last = line
        standalone = not (match.group(2) or '').strip()
        added_title = False
        for next_line in lines:
            if next_line is line or next_line['rect'].y0 < last['rect'].y1 - 1:
                continue
            gap = next_line['rect'].y0 - last['rect'].y1
            if gap > (8 if standalone and not added_title else 4):
                break
            if gap < -1:
                continue
            if CAPTION_RE.fullmatch(next_line['text']):
                break
            same_font = abs(next_line['size'] - line['size']) <= 1.5
            # IEEE standalone labels are centered; their titles may begin
            # much further left, so don't require matching left edges.
            aligned = (standalone and not added_title) or abs(next_line['rect'].x0 - last['rect'].x0) <= 12
            if not same_font or not aligned:
                continue
            if standalone and not added_title:
                # Standalone IEEE captions normally have an uppercase title.
                title = next_line['text'].strip()
                if not title or title != title.upper():
                    break
                added_title = True
            parts.append(next_line['text'])
            bbox |= next_line['rect']
            last = next_line
            if len(parts) >= 4:
                break
        result.append({'number': number, 'caption': ' '.join(parts),
                       'bbox': list(bbox)})
    result.extend(raw_caption_fallback(page, result))
    return result



def raw_caption_fallback(page, existing):
    """Find standalone IEEE captions in raw text when dict spans omit them.

    Raw text locates a candidate, and search_for recovers bounding boxes.
    Require a nearby uppercase title and a centered label to avoid
    treating body-text citations as table captions.
    """
    raw_lines = [line.strip() for line in page.get_text('text').splitlines()]
    known = {str(cap['number']) for cap in existing}
    found = []
    for index, label in enumerate(raw_lines):
        match = CAPTION_RE.fullmatch(label)
        if not match:
            continue
        number = canonical_table_number(match.group(1))
        if number is None or number in known:
            continue
        title = (match.group(2) or '').strip()
        if not title:
            for next_line in raw_lines[index + 1:index + 4]:
                if not next_line:
                    continue
                if len(next_line) < 8 or next_line != next_line.upper():
                    break
                title = next_line
                break
        if not title:
            continue
        for label_rect in page.search_for(label):
            if label_rect.y0 > page.rect.height * 0.80:
                continue
            mid = (label_rect.x0 + label_rect.x1) / 2
            width = page.rect.width
            possible_centers = (width / 2, width / 4, 3 * width / 4)
            if min(abs(mid - center) for center in possible_centers) > 34:
                continue
            title_rects = [rect for rect in page.search_for(title)
                           if 0 <= rect.y0 - label_rect.y1 <= 18]
            if not title_rects:
                continue
            title_rect = min(title_rects, key=lambda r: r.y0)
            bbox = pymupdf.Rect(label_rect) | title_rect
            found.append({'number': number,
                          'caption': label+' '+title,
                          'bbox': list(bbox)})
            known.add(number)
            break
    return found



def matching_column(page, cap):
    r = pymupdf.Rect(cap['bbox'])
    middle = page.rect.width / 2
    if r.x0 < middle - 10 and r.x1 > middle + 10:
        return pymupdf.Rect(45, 0, page.rect.width - 45, page.rect.height)
    return (pymupdf.Rect(45, 0, middle - 4, page.rect.height)
            if r.x0 < middle else
            pymupdf.Rect(middle + 4, 0, page.rect.width - 45, page.rect.height))


def match_table(page, cap, candidates):
    caption_rect = pymupdf.Rect(cap['bbox'])
    options = []
    for table in candidates:
        rect = pymupdf.Rect(table.bbox)
        gap = rect.y0 - caption_rect.y1
        overlap = max(0, min(rect.x1, caption_rect.x1) - max(rect.x0, caption_rect.x0))
        if -5 <= gap <= 105 and overlap >= min(rect.width, caption_rect.width)*.35:
            options.append((abs(gap), table))
    return min(options, key=lambda a:a[0])[1] if options else None


def relevant_words(page, bbox):
    x0,y0,x1,y1=bbox
    return [w for w in page.get_text('words') if x0 <= (w[0]+w[2])/2 <= x1 and y0 <= (w[1]+w[3])/2 <= y1]


def words_on_row(words, y, tolerance=2.8):
    return sorted((w for w in words if abs(w[1]-y)<=tolerance), key=lambda w:w[0])


def reconstruct_multilevel(page, table):
    raw=table.extract()
    if table.col_count < 8 or len(raw)<3 or len(table.rows)<3:
        return None
    headers, subheads=raw[:2]
    if not any(str(x or '').strip() in ('w/o','w/') for x in subheads):
        return None
    if not table.rows[1].cells or len(table.rows[1].cells)!=table.col_count:
        return None
    rectangles=[pymupdf.Rect(c) if c else None for c in table.rows[1].cells]
    if any(c is None for c in rectangles):
        return None
    words=relevant_words(page,table.bbox)
    model_col,system_col=rectangles[:2]
    buckets={}
    for w in words:
        x=(w[0]+w[2])/2
        if system_col.x0<=x<system_col.x1 and w[1]>=table.rows[2].bbox[1]:
            buckets.setdefault(round(w[1]/2)*2,[]).append(w)
    systems=sorted((sum(w[1] for w in g)/len(g),' '.join(w[4] for w in sorted(g,key=lambda w:w[0]))) for g in buckets.values())
    model_groups={}
    for w in words:
        x=(w[0]+w[2])/2
        if model_col.x0<=x<model_col.x1 and w[1]>=table.rows[2].bbox[1]:
            model_groups.setdefault(round(w[1]/2)*2,[]).append(w)
    models=sorted((sum(w[1] for w in g)/len(g),' '.join(w[4] for w in sorted(g,key=lambda w:w[0]))) for g in model_groups.values())
    if not systems or not models:
        return None
    cols=[];dataset=None
    for i in range(2,table.col_count):
        if headers[i]:dataset=str(headers[i]).strip()
        cols.append({'dataset':dataset,'metric':str(subheads[i] or '').strip(),'rect':rectangles[i]})
    records=[];issues=[]
    for y,system in systems:
        model=min(models,key=lambda m:abs(m[0]-y))[1]
        row_words=words_on_row(words,y)
        values=[]
        for col in cols:
            box=col['rect']
            values.append(' '.join(w[4] for w in row_words if box.x0<=(w[0]+w[2])/2<box.x1).strip())
        if any(not NUM_RE.fullmatch(v) for v in values):
            issues.append({'model':model,'system':system,'values':values,'reason':'Ambiguous or missing cells'})
            continue
        for col,value in zip(cols,values):
            records.append({'model':model,'system':system,'dataset':col['dataset'],
                            'metric':col['metric'],'value':None if value in ('–','—','-') else value,
                            'value_raw':value})
    return {'system_rows_detected':len(systems),'model_groups_detected':len(models),
            'records':records,'issues':issues,'status':'candidate_needs_validation' if not issues else 'review_required'}


def reconstruct_borderless(page, cap, all_caps):
    """Recover caption-following text lines without making unsupported cell claims.

    Detects the data block by line baselines/spacing and outputs positional
    row tokens. This is intentionally an intermediate representation: merged
    headers and category groupings remain marked for validation.
    """
    col=matching_column(page,cap)
    caption_rect=pymupdf.Rect(cap['bbox'])
    other_caps=[pymupdf.Rect(c['bbox']) for c in all_caps if c is not cap]
    stop=min([r.y0 for r in other_caps if r.y0>caption_rect.y1 and r.intersects(col)]
             +[page.rect.height-45])
    # Table content typically starts shortly after the caption; use a narrow
    # initial window to avoid matching a later unrelated paragraph/diagram.
    lines=[l for l in text_lines(page) if l['rect'].y0>=caption_rect.y1+1
           and l['rect'].y0<stop and l['rect'].intersects(col)]
    lines=[l for l in lines if col.x0<=((l['rect'].x0+l['rect'].x1)/2)<=col.x1]
    if not lines:
        return None
    first=min(l['rect'].y0 for l in lines)
    if first-caption_rect.y1>42:
        return None
    # Cluster line elements by their baseline. Break at a pronounced gap,
    # especially when subsequent content is normal body text.
    grouped=[]
    for line in lines:
        cy=(line['rect'].y0+line['rect'].y1)/2
        if grouped and abs(grouped[-1]['y']-cy)<=2.5:
            grouped[-1]['items'].append(line)
        else:
            grouped.append({'y':cy,'items':[line]})
    rows=[];previous=None
    for group in grouped:
        y=group['y']
        if previous is not None and y-previous>15:
            break
        tokens=[]
        for line in sorted(group['items'],key=lambda x:x['rect'].x0):
            tokens.append({'text':line['text'],'x0':round(line['rect'].x0,2),'x1':round(line['rect'].x1,2)})
        rows.append({'y':round(y,2),'segments':tokens})
        previous=y
    if len(rows)<2:
        return None
    return {'status':'coordinate_candidate_needs_validation','row_count':len(rows),
            'rows':rows,'region_bbox':[col.x0,caption_rect.y1,col.x1,previous+7],
            'notes':'Text segments retain their x coordinates; header and merged-cell interpretation is not yet certified.'}



def coordinate_to_records(coordinate):
    """Conservatively turn regularly aligned coordinate rows into named cells.

    Segments must align with stable columns; irregular spanning text remains
    diagnostic-only rather than being forced into invented cells.
    """
    if not coordinate or coordinate['row_count'] < 3:
        return None
    rows=coordinate['rows']
    header=rows[0]['segments']
    if len(header)<2:
        return None
    centers=[(seg['x0']+seg['x1'])/2 for seg in header]
    if any(b-a<3 for a,b in zip(centers,centers[1:])):
        return None
    boundaries=[(a+b)/2 for a,b in zip(centers,centers[1:])]
    matrix=[];issues=[]
    for row in rows[1:]:
        values=[[] for _ in header]
        for seg in row['segments']:
            center=(seg['x0']+seg['x1'])/2
            idx=sum(center>bound for bound in boundaries)
            values[idx].append(seg['text'])
        cells=[' '.join(v) for v in values]
        # A blank major cell or multiple merged columns makes a row ambiguous.
        if not cells[0] or sum(bool(c) for c in cells)<len(header)-1:
            issues.append({'y':row['y'],'cells':cells,'reason':'Missing or merged columns'})
        matrix.append(dict(zip((c['text'] for c in header),cells)))
    return {'status':'coordinate_structured_candidate' if not issues else 'review_required',
            'headers':[c['text'] for c in header], 'records':matrix,'issues':issues}


def reconstruct_multiline_native(table):
    """Recover line-separated systems from merged PyMuPDF table rows.

    Accept only strict same-length groups of 3 metric tokens, retaining the
    existing multi-level headers and leaving irregular cases for review.
    """
    raw=table.extract()
    if len(raw)<3 or table.col_count<5:
        return None
    heads,subheads=raw[:2]
    if not any(str(x or '').strip() in ('w/o','w/','Ratio') for x in subheads):
        return None
    value_rows=raw[2:]
    if len(value_rows)!=1 or not value_rows[0][0]:
        return None
    systems=[s.strip() for s in value_rows[0][0].splitlines() if s.strip()]
    if len(systems)<2:
        return None
    spans=[];current=None
    for k in range(1,len(heads)):
        if heads[k]:current=heads[k]
        if subheads[k]:spans.append((k,current,subheads[k]))
    groups=[]
    for k,group,measure in spans:
        val=value_rows[0][k]
        if val is not None:
            per_line=[line.split() for line in val.splitlines() if line.strip()]
            if len(per_line)!=len(systems):return None
            for vals in per_line:
                if len(vals)!=3:return None
            groups.append((group,per_line))
    if not groups:return None
    records=[]
    for ix,system in enumerate(systems):
        row={'system':system}
        for group,vals in groups:
            for measure,value in zip(('w/o','w/','Ratio'),vals[ix]):
                row[f'{group} / {measure}']=value
        records.append(row)
    return {'status':'multiline_structured_candidate','records':records,'issues':[]}


def reconstruct_sectioned(coordinate):
    """Treat standalone spanning labels as section headings, not data rows."""
    if not coordinate or coordinate['row_count'] < 4:
        return None
    rows = coordinate['rows']
    heads = [x['text'] for x in rows[0]['segments']]
    if len(heads) < 3:
        return None
    centers = [(x['x0'] + x['x1']) / 2 for x in rows[0]['segments']]
    boundaries = [(a+b)/2 for a,b in zip(centers,centers[1:])]
    section = None
    records = []
    issues = []
    sections = []
    for row in rows[1:]:
        segs = row['segments']
        # A single label centered across multiple columns indicates a group.
        if len(segs) == 1 and not NUM_RE.fullmatch(segs[0]['text']):
            section = segs[0]['text'].strip()
            sections.append(section)
            continue
        cells = [[] for _ in heads]
        for seg in segs:
            mid = (seg['x0'] + seg['x1'])/2
            index = sum(mid > bound for bound in boundaries)
            cells[index].append(seg['text'])
        values = [' '.join(cell) for cell in cells]
        # Sectioned tables often legitimately leave the second column blank.
        if not values[0] or sum(bool(v) for v in values[2:]) < len(heads)-3:
            issues.append({'y':row['y'],'cells':values,'reason':'Uncertain row alignment'})
        record = {'section':section}
        record.update({key:(value or None) for key,value in zip(heads,values)})
        records.append(record)
    if len(sections) < 2:
        return None
    return {'status':'sectioned_structured_candidate' if not issues else 'review_required',
            'headers':heads,'sections':sections,'records':records,'issues':issues}


def reconstruct_taxonomy(coordinate):
    """Recover category-centered fault taxonomy without inventing tick values."""
    if not coordinate or coordinate['row_count'] < 5:
        return None
    rows = coordinate['rows'][1:]
    # Require two visibly separated text columns and category/fault alignment.
    left = [s for r in rows for s in r['segments'] if s['x0'] < 75 and s['text'] != '✓']
    faults = [s for r in rows for s in r['segments'] if 76 <= s['x0'] < 115 and s['text'] != '✓']
    if len(left) < 2 or len(faults) < 3:
        return None
    categories = [(r['y'],s['text']) for r in rows for s in r['segments'] if s in left]
    records = []
    for row in rows:
        fault_candidates=[s for s in row['segments'] if s in faults]
        for fault in fault_candidates:
            cat = min(categories, key=lambda pair:abs(pair[0]-row['y']))[1]
            examples=[s['text'] for s in row['segments'] if s['x0']>=150]
            # Tick symbols can be vertically offset on merged rows, so store
            # direct observations only; no unchecked/checked inference.
            ticks=[s for s in row['segments'] if s['text']=='✓']
            records.append({'Category':cat,'Fault type':fault['text'],
                            'Real-world scenario':' '.join(examples) or None,
                            'Content':'observed_tick' if any(112<=t['x0']<130 for t in ticks) else None,
                            'Tool call':'observed_tick' if any(135<=t['x0']<150 for t in ticks) else None})
    return {'status':'taxonomy_partial_candidate','records':records,
            'issues':['Checkmarks and scenario descriptions may be vertically offset; missing values mean unknown, not false.']}


def reconstruct_header_only(page, table, cap, all_caps):
    """Extract rows below a native multilevel header using column centers."""
    native = table.extract()
    if len(native) != 2 or table.col_count < 3:
        return None
    h1,h2=native
    if not any(x for x in h2[1:]):
        return None
    header_end=table.bbox[3]
    col=matching_column(page,cap)
    other_caps=[pymupdf.Rect(c['bbox']) for c in all_caps if c is not cap]
    stop=min([r.y0 for r in other_caps if r.y0>header_end and r.intersects(col)] + [page.rect.height-45])
    rects=[pymupdf.Rect(c) if c else None for c in table.rows[-1].cells]
    if len(rects)!=table.col_count or any(r is None for r in rects):
        return None
    words=[w for w in page.get_text('words') if header_end-1 < (w[1]+w[3])/2 < stop
           and table.bbox[0]-5 < (w[0]+w[2])/2 < table.bbox[2]+5]
    baselines=sorted({round(w[1],1) for w in words})
    groups=[]
    for y in baselines:
        if not groups or y-groups[-1][-1] > 2.5:
            groups.append([y])
        else:
            groups[-1].append(y)
    records=[];last=header_end;issues=[]
    for group in groups:
        y=sum(group)/len(group)
        if y-last>20:
            break
        row_words=[w for w in words if abs(w[1]-y)<=2]
        values=[]
        for rect in rects:
            chunk=sorted([w for w in row_words if rect.x0-1<=(w[0]+w[2])/2<=rect.x1+1],key=lambda w:w[0])
            values.append(' '.join(w[4] for w in chunk).strip())
        if (not values[0] or sum(bool(x) for x in values)<len(values)-1
                or any(not NUM_RE.fullmatch(x) for x in values[1:])):
            # Prose following the table can occupy the same column range.
            # Once valid numeric rows began, stop at the first invalid row.
            if records:
                break
            issues.append({'y':round(y,1),'cells':values})
            continue
        record={'row_label':values[0]}
        groupname=None
        for j in range(1,len(values)):
            if h1[j]: groupname=h1[j]
            record[f'{groupname or "metric"} / {h2[j] or j}']=values[j]
        records.append(record);last=y
    if not records:
        return None
    return {'status':'header_following_rows_candidate' if not issues else 'review_required',
            'records':records,'issues':issues}

def normalize_simple_native(table):
    """Normalize unambiguous single-header native tables into records.

    Keeps complex merged-header tables on their existing specialized paths.
    """
    rows = table.extract()
    if not rows or len(rows) < 3 or table.col_count < 2:
        return None
    headings = rows[0]
    if len(headings) != table.col_count or any(not str(h or '').strip() for h in headings):
        return None
    keys = [' '.join(str(h).split()) for h in headings]
    if len(set(keys)) != len(keys):
        return None
    records = []
    issues = []
    for index, row in enumerate(rows[1:], start=1):
        if len(row) != len(keys):
            issues.append({'row': index, 'reason': 'Column count mismatch'})
            continue
        if not row[0] or not str(row[0]).strip():
            issues.append({'row': index, 'reason': 'Missing primary row label'})
            continue
        records.append({key: (' '.join(str(value).split()) if value else None)
                        for key, value in zip(keys, row)})
    if not records:
        return None
    return {'status': 'native_structured_candidate' if not issues else 'review_required',
            'headers': keys, 'records': records, 'issues': issues}


def extract(pdf_path,output_path):
    result={'source':Path(pdf_path).name,'tables':[]}
    with pymupdf.open(pdf_path) as doc:
        for page_no,page in enumerate(doc,1):
            candidates=list(page.find_tables().tables)
            found=captions(page)
            for cap in found:
                table=match_table(page,cap,candidates)
                item={'number':cap['number'],'page':page_no,'caption':cap['caption'],
                      'caption_bbox':cap['bbox'],'status':'unmatched','native':None,
                      'structured':None,'coordinate':None,'normalized':None}
                if table:
                    native=table.extract()
                    item.update(status='native_candidate',bbox=list(table.bbox),
                                native={'row_count':table.row_count,'col_count':table.col_count,
                                        'header':table.header.names,'rows':native})
                    structured=reconstruct_multilevel(page,table)
                    if not structured:
                        structured=reconstruct_multiline_native(table)
                    if structured:
                        item['structured']=structured
                        item['status']=structured['status']
                if table and item['structured'] is None and table.row_count > 2:
                    normalized_native = normalize_simple_native(table)
                    if normalized_native:
                        item['normalized'] = normalized_native
                        item['status'] = normalized_native['status']
                # Native 1-2 row tables are often header-only detection.
                if not table or (table.row_count<=2 and item['structured'] is None):
                    fallback=reconstruct_borderless(page,cap,found)
                    if fallback:
                        item['coordinate']=fallback
                        item['status']=fallback['status']
                if item['coordinate']:
                    item['normalized']=coordinate_to_records(item['coordinate'])
                    if item['normalized'] and not item['normalized']['issues']:
                        item['status']=item['normalized']['status']
                # Improve grouped captions/section labels without changing
                # prior JSON fields used by the indexing pipeline.
                if item['coordinate'] and item['normalized'] is None:
                    taxonomy=reconstruct_taxonomy(item['coordinate'])
                    if taxonomy:
                        item['normalized']=taxonomy
                        item['status']=taxonomy['status']
                if item['coordinate'] and item['normalized'] and item['normalized'].get('issues'):
                    sectioned=reconstruct_sectioned(item['coordinate'])
                    if sectioned:
                        item['normalized']=sectioned
                        item['status']=sectioned['status']
                if table and table.row_count == 2 and not item['coordinate']:
                    follow=reconstruct_header_only(page,table,cap,found)
                    if follow:
                        item['structured']=follow
                        item['status']=follow['status']
                result['tables'].append(item)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Captions detected:',len(result['tables']))
    for item in result['tables']:
        note=''
        if item['structured']:
            st=item['structured'];note=f" | structured rows={st.get('system_rows_detected',len(st['records']))} | values={len(st['records'])} | issues={len(st['issues'])}"
        if item['coordinate']:
            note=f" | coordinate rows={item['coordinate']['row_count']}"
        if item['normalized']:
            note+=f" | normalized rows={len(item['normalized']['records'])} | warnings={len(item['normalized']['issues'])}"
        print(f"Table {item['number']} | page {item['page']} | {item['status']}{note}")
    print('Saved:',output_path)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('pdf',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    path=args.pdf
    if not path.exists():path=Path('data/documents')/path
    if not path.exists():raise SystemExit(f'PDF not found: {path}')
    out=args.output or Path('data/table_tests')/(path.stem+'_tables.json')
    extract(path,out)


if __name__=='__main__':main()
