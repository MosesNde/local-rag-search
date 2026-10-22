"""Streamlit interface for Local RAG Search with message-scoped citations."""
import re
import html
from difflib import SequenceMatcher
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / 'data' / 'documents'
IMAGES_DIR = ROOT / 'data' / 'images'
DB_PATH = ROOT / 'data' / 'chroma'
URLS_FILE = ROOT / 'data' / 'urls.txt'
INDEX_SCRIPT = ROOT / 'src' / 'index_documents.py'
QUERY_SCRIPT = ROOT / 'src' / 'query_documents.py'
DOCUMENT_TYPES = {'.txt', '.pdf'}
IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.webp'}
for directory in (DOCUMENTS_DIR, IMAGES_DIR, DB_PATH):
    directory.mkdir(parents=True, exist_ok=True)


def get_record_count():
    db = chromadb.PersistentClient(path=str(DB_PATH))
    return db.get_or_create_collection(name='documents').count()


def save_uploaded_files(uploaded_files):
    saved, skipped = [], []
    for item in uploaded_files:
        filename = item.name.replace('\\', '/').split('/')[-1]
        ext = Path(filename).suffix.lower()
        if not filename or filename in ('.', '..') or ext not in DOCUMENT_TYPES | IMAGE_TYPES:
            skipped.append(filename or 'Unnamed file')
            continue
        destination = (DOCUMENTS_DIR if ext in DOCUMENT_TYPES else IMAGES_DIR) / filename
        try:
            with destination.open('xb') as file:
                file.write(item.getbuffer())
            saved.append(filename)
        except FileExistsError:
            skipped.append(filename)
    return saved, skipped


def save_website_url(url):
    url = url.strip()
    if not url:
        return None
    parsed = urlparse(url)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError('Enter a valid HTTP or HTTPS webpage URL.')
    existing = URLS_FILE.read_text(encoding='utf-8').splitlines() if URLS_FILE.exists() else []
    if url in existing:
        return False
    with URLS_FILE.open('a', encoding='utf-8') as file:
        if URLS_FILE.stat().st_size:
            file.write('\n')
        file.write(url + '\n')
    return True


def run_indexer():
    return subprocess.run([sys.executable, str(INDEX_SCRIPT)], cwd=str(ROOT),
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=1800)


def run_query(question):
    return subprocess.run([sys.executable, str(QUERY_SCRIPT)], input=question + '\n',
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=600)


def parse_query_output(output):
    if 'Answer:' not in output:
        return output.strip(), []
    content = output.split('Answer:', 1)[1]
    if 'Retrieved sources:' in content:
        answer, source_text = content.split('Retrieved sources:', 1)
    else:
        answer, source_text = content, ''
    sources = [line.strip()[2:] for line in source_text.splitlines()
               if line.strip().startswith('- ')]
    return answer.strip(), sources


PREFIX = re.compile(r'^\[(\d+)\]\s*')
PDF_LOCATOR = re.compile(r'(?P<name>[^/\\\n]+?\.pdf)\s*\((?P<detail>[^)]*)\)', re.I)


def parse_pdf_source(source):
    source = PREFIX.sub('', source).strip()
    source = re.sub(r'^Retrieved context \(citation unverified\):\s*', '', source, flags=re.I)
    match = PDF_LOCATOR.search(source)
    if not match:
        return None
    name = match.group('name').strip()
    page_match = re.search(r'\bpage\s+(\d+)\b', match.group('detail'), re.I)
    if not page_match or not name or Path(name).name != name:
        return None
    page = int(page_match.group(1))
    return (name, page) if page >= 1 else None


def local_source_file(name):
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        return None
    for base in (DOCUMENTS_DIR, IMAGES_DIR):
        candidate = (base / name).resolve()
        if candidate.parent == base.resolve() and candidate.is_file():
            return candidate
    return None


def local_pdf(name):
    path = local_source_file(name)
    return path if path and path.suffix.lower() == '.pdf' and path.parent == DOCUMENTS_DIR.resolve() else None


def get_source_name(source):
    label = PREFIX.sub('', source).strip()
    label = re.sub(r'^Retrieved context \(citation unverified\):\s*', '', label, flags=re.I)
    for suffix in ('.pdf', '.txt', '.png', '.jpg', '.jpeg', '.webp'):
        match = re.search(re.escape(suffix) + r'(?=\s*\(|$)', label, re.I)
        if match:
            return label[:match.end()].strip()
    return label.split(' (', 1)[0].strip()


def parse_table_row_locator(source):
    match = PDF_LOCATOR.search(PREFIX.sub('', source).strip())
    if not match:
        return None
    details = match.group('detail')
    table = re.search(r'\bTable\s+([IVXLCDM]+|\d+)\b', details, re.I)
    row = re.search(r'\brow\s+(\d+)\b', details, re.I)
    if not table or not row:
        return None
    return 'Table ' + table.group(1), int(row.group(1))


@st.cache_data(show_spinner=False, max_entries=32)
def table_row_category(pdf_name, table_label, page_number, row_number):
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db.get_collection('documents')
    records = collection.get(where={'$and': [
        {'source': pdf_name}, {'file_type': 'pdf_table'},
        {'page': page_number}, {'row_number': row_number},
    ]}, include=['documents', 'metadatas'])
    for document, meta in zip(records.get('documents') or [], records.get('metadatas') or []):
        actual_label = meta.get('table_label') or (
            f"Table {meta['table_number']}" if meta.get('table_number') is not None else None)
        if not actual_label or actual_label.casefold() != table_label.casefold():
            continue
        match = re.search(r'\bRow\s+\d+\s*:\s*OWASP Category\s*:\s*([^|\n]+)',
                          document, re.I)
        if match:
            return match.group(1).strip()
    return None


def parse_pdf_chunk_locator(source):
    """Parse page-local PDF text chunk citation, excluding tables/figures."""
    if parse_table_row_locator(source) is not None:
        return None
    matched = PDF_LOCATOR.search(PREFIX.sub('', source).strip())
    if not matched:
        return None
    detail = matched.group('detail')
    if re.search(r'\b(?:Table|Figure)\b', detail, re.I):
        return None
    chunk = re.search(r'\bchunk\s+(\d+)\b', detail, re.I)
    return int(chunk.group(1)) if chunk else None


@st.cache_data(show_spinner=False, max_entries=64)
def indexed_pdf_chunk(name, page, chunk_number):
    """Read the actual cited chunk; do not silently substitute another chunk."""
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db.get_collection('documents')
    found = collection.get(where={'$and': [
        {'source': name}, {'file_type': 'pdf'},
        {'page': page}, {'chunk_number': chunk_number},
    ]}, include=['documents', 'metadatas'])
    documents = found.get('documents') or []
    return documents[0] if len(documents) == 1 else None


def _word_token(word):
    return re.sub(r'[^a-z0-9]', '', word.casefold())


def highlight_pdf_chunk(page, chunk_text):
    """Highlight a substantial exact word sequence on the specified PDF page.

    Chunk text can contain line-break differences; only a sufficiently long,
    unique contiguous matching word sequence is highlighted. Never approximate
    coordinates based on the chunk number.
    """
    words = page.get_text('words', sort=True)
    page_words = [(item, _word_token(item[4])) for item in words]
    page_words = [(item, tok) for item, tok in page_words if tok]
    chunk_tokens = [_word_token(w) for w in re.findall(r'\S+', chunk_text)]
    chunk_tokens = [tok for tok in chunk_tokens if tok]
    if len(page_words) < 8 or len(chunk_tokens) < 8:
        return False, 0
    a = [tok for _, tok in page_words]
    matcher = SequenceMatcher(None, a, chunk_tokens, autojunk=False)
    matches = [m for m in matcher.get_matching_blocks() if m.size >= 10]
    if not matches:
        return False, 0
    best = max(matches, key=lambda m: m.size)
    # A short common phrase can occur repeatedly. Reject ambiguous matches.
    sequence = a[best.a:best.a + min(best.size, 12)]
    occurrences = sum(a[i:i + len(sequence)] == sequence
                      for i in range(len(a) - len(sequence) + 1))
    if occurrences != 1:
        return False, 0
    count = min(best.size, 65)
    rects = [page_words[i][0][:4] for i in range(best.a, best.a + count)]
    annot = page.add_highlight_annot(rects)
    annot.set_colors(stroke=(1, .75, 0))
    annot.update()
    return True, count


@st.cache_data(show_spinner=False, max_entries=8)
def pdf_page_png(name, page, modified_ns, table_label=None, row_number=None, chunk_text=None):
    """Render the cited page; retain verified PDF table-row highlighting."""
    import fitz
    path = local_pdf(name)
    if path is None:
        raise FileNotFoundError(name)
    category = (table_row_category(name, table_label, page, row_number)
                if table_label and row_number else None)
    with fitz.open(path) as document:
        if page > len(document):
            raise ValueError(f'Page {page} is outside this {len(document)}-page PDF.')
        pdf_page = document.load_page(page - 1)
        highlighted = False
        highlight_type = None
        if category:
            try:
                tables = pdf_page.find_tables()
                for table in tables.tables:
                    for idx, row in enumerate(table.extract()):
                        if not any(cell and category.casefold() in cell.casefold() for cell in row):
                            continue
                        if idx < len(table.rows):
                            shape = pdf_page.new_shape()
                            shape.draw_rect(fitz.Rect(table.rows[idx].bbox))
                            shape.finish(color=(1, 0.48, 0), fill=(1, 0.85, 0.12),
                                         fill_opacity=0.17, width=1.5)
                            shape.commit(overlay=True)
                            highlighted, highlight_type = True, 'row'
                            break
                    if highlighted:
                        break
            except Exception:
                pass
            if not highlighted:
                found = pdf_page.search_for(category)
                left = [r for r in found if r.x0 < pdf_page.rect.width * .55]
                if len(left) == 1:
                    annot = pdf_page.add_highlight_annot(left[0])
                    annot.set_colors(stroke=(1, .75, 0))
                    annot.update()
                    highlighted, highlight_type = True, 'category cell'
        if chunk_text and not highlighted and not table_label:
            highlighted, word_count = highlight_pdf_chunk(pdf_page, chunk_text)
            if highlighted:
                highlight_type = f'PDF text passage ({word_count} matching words)'
        png = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False,
                                 annots=True).tobytes('png')
        return png, len(document), highlighted, highlight_type, category



def parse_txt_chunk_locator(source):
    """Return exact TXT filename and cited chunk, never a guessed chunk."""
    if not PREFIX.match(source):
        return None  # unverified retrieved context must not masquerade as a citation
    name = get_source_name(source)
    if not name.lower().endswith('.txt') or '/' in name or "\\" in name:
        return None
    match = re.search(r'\bchunk\s+(\d+)\b', source, re.I)
    return (name, int(match.group(1))) if match else None


@st.cache_data(show_spinner=False, max_entries=64)
def indexed_txt_chunk(name, number):
    """Read-only lookup of the exact indexed chunk."""
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db.get_collection('documents')
    records = collection.get(where={'$and': [
        {'source': name}, {'file_type': 'txt'}, {'chunk_number': number},
    ]}, include=['documents', 'metadatas'])
    chunks = records.get('documents') or []
    return chunks[0] if len(chunks) == 1 else None


def normalized_with_positions(text):
    """Collapse whitespace while retaining offsets in the original document."""
    chars, offsets, pending_space = [], [], False
    for index, char in enumerate(text):
        if char.isspace():
            pending_space = bool(chars)
            continue
        if pending_space:
            chars.append(' ')
            offsets.append(index)
            pending_space = False
        chars.append(char.casefold())
        offsets.append(index)
    return ''.join(chars), offsets


def match_txt_chunk(document, chunk):
    """Return reliable (start, end) offsets of a chunk in the TXT file.

    Prefer full, unique normalized match; otherwise accept a single substantial
    exact sequence. Never infer a position from the chunk index alone.
    """
    original, positions = normalized_with_positions(document)
    needle, _ = normalized_with_positions(chunk)
    if not needle or not original:
        return None
    index = original.find(needle)
    if index >= 0 and original.find(needle, index + 1) == -1:
        return positions[index], positions[index + len(needle) - 1] + 1
    # Indexer sentence overlaps and text normalization can prevent a full match.
    matcher = SequenceMatcher(None, original, needle, autojunk=False)
    best = matcher.find_longest_match(0, len(original), 0, len(needle))
    if best.size < 80 or best.size < len(needle) * .35:
        return None
    segment = original[best.a:best.a + best.size]
    if original.find(segment, best.a + 1) != -1:
        return None
    return positions[best.a], positions[best.a + best.size - 1] + 1


def show_txt_highlight(source, path):
    """Highlight cited text in the original TXT, or show a clear fallback."""
    locator = parse_txt_chunk_locator(source)
    document = path.read_text(encoding='utf-8', errors='replace')
    st.caption(f'Text document: {path.name}')
    if not locator:
        st.text_area('Document content (preview)', document[:12000],
                     height=380, disabled=True)
        return
    try:
        chunk = indexed_txt_chunk(*locator)
    except Exception as exc:
        st.info(f'Could not retrieve the indexed text chunk ({type(exc).__name__}).')
        chunk = None
    match = match_txt_chunk(document, chunk) if chunk else None
    if match:
        left, right = match
        # Show the surrounding document, without creating an enormous HTML block.
        start, end = max(0, left - 900), min(len(document), right + 900)
        if end - start > 18000:
            end = min(len(document), left + 16000)
        before = html.escape(document[start:left])
        marked = html.escape(document[left:min(right, end)])
        after = html.escape(document[min(right, end):end])
        body = before + '<mark style="background-color:#ffe082;color:#222;">' + marked + '</mark>' + after
        st.success(f'Highlighted cited chunk {locator[1]} in the original document.')
        st.markdown('<div style="white-space:pre-wrap;overflow-wrap:anywhere;'
                    'max-height:520px;overflow-y:auto;line-height:1.6;'
                    'padding:12px;border:1px solid #9995;border-radius:8px;">'
                    + body + '</div>', unsafe_allow_html=True)
    else:
        st.info('The indexed chunk could not be located reliably in the original '
                'TXT file. No highlight was added.')
        if chunk:
            st.text_area(f'Indexed chunk {locator[1]} (reference only)',
                         chunk, height=200, disabled=True)
        st.text_area('Original document preview', document[:12000],
                     height=260, disabled=True)

def show_evidence(selection):
    """Always show the source selected from its original assistant message."""
    if not selection:
        return
    source = selection['source']
    st.subheader('Source evidence')
    st.caption(f"Answer {selection['answer_number']} · Citation [{selection['citation_number']}]")
    st.caption(source)
    pdf = parse_pdf_source(source)
    if pdf:
        name, page = pdf
        path = local_pdf(name)
        if path is None:
            st.info("The original PDF isn't available under data/documents.")
            return
        try:
            locator = parse_table_row_locator(source)
            label, number = locator if locator else (None, None)
            chunk_number = parse_pdf_chunk_locator(source)
            chunk_text = (indexed_pdf_chunk(name, page, chunk_number)
                          if chunk_number is not None else None)
            png, count, highlighted, kind, category = pdf_page_png(
                name, page, path.stat().st_mtime_ns, label, number, chunk_text)
            st.caption(f'Page {page} of {count}')
            if locator:
                if highlighted:
                    st.success(f'Highlighted {kind}: {category} (row {number})')
                else:
                    st.info('Exact row position could not be reliably located; showing the page.')
            if chunk_number is not None:
                if highlighted:
                    st.success(f'Highlighted supporting {kind}.')
                elif chunk_text is None:
                    st.info('Indexed text for this cited PDF chunk was not found. Showing the page without highlighting.')
                else:
                    st.info('The indexed passage could not be matched reliably to the PDF page. No highlight was added.')
            st.image(png, use_container_width=True)
            st.download_button('Download PDF', data=path.read_bytes(),
                               file_name=path.name, mime='application/pdf',
                               key=f"dl_{selection['message_id']}_{selection['citation_number']}")
        except ImportError:
            st.warning('PDF rendering requires PyMuPDF: pip install pymupdf')
        except Exception as exc:
            st.warning(f'Unable to display this PDF page: {exc}')
        return
    name = get_source_name(source)
    path = local_source_file(name)
    if path and path.suffix.lower() == '.txt':
        show_txt_highlight(source, path)
    elif path and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
        st.caption(f'Image: {path.name}')
        st.image(str(path), use_container_width=True)
    elif name.startswith(('https://', 'http://')):
        parsed = urlparse(name)
        if parsed.scheme in ('http', 'https') and parsed.hostname:
            st.link_button('Open original webpage', name)
    else:
        st.info('Source reference available; local preview unavailable for this source.')


def citation_buttons(sources, message_id, answer_number):
    numbered = [(PREFIX.match(source).group(1), source)
                for source in sources if PREFIX.match(source)]
    if not numbered:
        return
    st.caption('View cited evidence:')
    cols = st.columns(min(6, len(numbered)))
    for idx, (number, source) in enumerate(numbered):
        with cols[idx % len(cols)]:
            if st.button(f'[{number}]', key=f'cite_{message_id}_{idx}',
                         help=f'Answer {answer_number}, citation {number}'):
                st.session_state.selected_evidence = {
                    'source': source,
                    'message_id': message_id,
                    'answer_number': answer_number,
                    'citation_number': number,
                }
                st.rerun()


def display_sources(sources):
    if sources:
        with st.expander(f'Retrieved sources ({len(sources)})'):
            for source in sources:
                st.text(f'• {source}')


def render_assistant(answer, sources, message_id, answer_number):
    st.markdown(answer)
    citation_buttons(sources, message_id, answer_number)
    display_sources(sources)


st.set_page_config(page_title='Local RAG Search', page_icon='🔎', layout='wide')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_evidence' not in st.session_state:
    st.session_state.selected_evidence = None

with st.sidebar:
    st.title('Knowledge Base')
    try:
        st.metric('Indexed records', get_record_count())
    except Exception as exc:
        st.error(f'Database error: {exc}')
    st.divider()
    uploaded_files = st.file_uploader('Upload documents or images',
        type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    website_url = st.text_input('Webpage URL', placeholder='https://example.com/article')
    index_clicked = st.button('Index content', type='primary', use_container_width=True)
    st.divider()
    if st.button('Clear chat', use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_evidence = None
        st.rerun()
    st.caption('Powered by Ollama + ChromaDB')

if index_clicked:
    if not uploaded_files and not website_url.strip():
        st.warning('Upload a file or enter a webpage URL.')
    else:
        try:
            added = save_website_url(website_url) if website_url.strip() else None
            saved, skipped = save_uploaded_files(uploaded_files) if uploaded_files else ([], [])
            if saved:
                st.success(f'Saved {len(saved)} new file(s).')
                for name in saved:
                    st.write(f'• {name}')
            if skipped:
                st.info('Files already present or unsupported:')
                for name in skipped:
                    st.write(f'• {name}')
            if added is True:
                st.success('Webpage URL added.')
            elif added is False:
                st.info('Webpage URL already exists.')
            with st.spinner('Indexing knowledge sources...'):
                result = run_indexer()
            if result.returncode == 0:
                st.success('Indexing completed successfully!')
                st.metric('Total indexed records', get_record_count())
                with st.expander('Indexing details'):
                    st.code(result.stdout, language='text')
            else:
                st.error('Indexing failed.')
                with st.expander('Error details', expanded=True):
                    st.code(result.stdout + '\n' + result.stderr, language='text')
        except subprocess.TimeoutExpired:
            st.error('Indexing exceeded the time limit.')
        except Exception as exc:
            st.error(f'Indexing error: {exc}')

st.title('🔎 Local RAG Search')
st.caption('Chat with your documents using locally running AI.')

# Important: consume a newly submitted question BEFORE creating the layout.
# This removes the Close evidence widget from the run that performs a blocking query.
question = st.chat_input('Ask a question about your documents...')
if question:
    st.session_state.selected_evidence = None
    st.session_state.messages.append({'role': 'user', 'content': question})

selected = st.session_state.selected_evidence
if selected:
    chat_col, evidence_col = st.columns([1.15, 1], gap='large')
else:
    chat_col, evidence_col = st.container(), None

with chat_col:
    if not st.session_state.messages:
        st.info('Ask a question about your indexed documents to begin a conversation.')
    answer_number = 0
    for message_id, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            if message['role'] == 'assistant':
                answer_number += 1
                render_assistant(message['content'], message.get('sources', []),
                                 message_id, answer_number)
            else:
                st.markdown(message['content'])

if evidence_col is not None:
    with evidence_col:
        if st.button('Close evidence', key='close_evidence'):
            st.session_state.selected_evidence = None
            st.rerun()
        show_evidence(selected)

if question:
    with chat_col:
        with st.chat_message('assistant'):
            try:
                if get_record_count() == 0:
                    answer, sources = ('Your knowledge base is empty. '
                                       'Please index some documents first.'), []
                else:
                    with st.spinner('Searching your knowledge base...'):
                        result = run_query(question)
                    if result.returncode:
                        answer, sources = ('The search could not be completed. '
                                           'Check that Ollama is running.'), []
                        with st.expander('Error details'):
                            st.code(result.stdout + '\n' + result.stderr, language='text')
                    else:
                        answer, sources = parse_query_output(result.stdout)
                        if not answer:
                            answer = 'The query completed without a readable answer.'
                st.session_state.messages.append({
                    'role': 'assistant', 'content': answer, 'sources': sources})
                render_assistant(answer, sources, len(st.session_state.messages) - 1,
                                 answer_number + 1)
            except subprocess.TimeoutExpired:
                answer = 'Question answering took too long. Check that Ollama is running.'
                st.error(answer)
                st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': []})
            except Exception as exc:
                answer = f'Search error: {exc}'
                st.error(answer)
                st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': []})
