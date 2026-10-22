"""Streamlit UI for the local RAG search engine, including PDF citation viewing."""
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode, urlparse

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
        answer, sources_section = content.split('Retrieved sources:', 1)
    else:
        answer, sources_section = content, ''
    sources = [line.strip()[2:] for line in sources_section.splitlines()
               if line.strip().startswith('- ')]
    return answer.strip(), sources


# Citation labels produced by query_documents.py, e.g.:
# [1] agentchaos.pdf (Figure 3, page 5)
# [1] paper.pdf (Table II, page 6, row 1)
PDF_LOCATOR = re.compile(r'(?P<name>[^/\\\n]+?\.pdf)\s*\((?P<detail>[^)]*)\)', re.I)
PREFIX = re.compile(r'^\[(\d+)\]\s*')


def parse_pdf_source(source):
    # Strip citation number or unverified-context prefix before PDF parsing.
    source = re.sub(r'^\s*\[\d+\]\s*', '', source)
    source = re.sub(r'^\s*Retrieved context \(citation unverified\):\s*', '', source, flags=re.I)
    match = PDF_LOCATOR.search(source)
    if not match:
        return None
    name = match.group('name').strip()
    page_match = re.search(r'\bpage\s+(\d+)\b', match.group('detail'), re.I)
    if not page_match or not name or Path(name).name != name:
        return None
    page = int(page_match.group(1))
    return (name, page) if page >= 1 else None


def local_pdf(name):
    """Never allow a citation to reference paths outside data/documents."""
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        return None
    path = (DOCUMENTS_DIR / name).resolve()
    if path.parent != DOCUMENTS_DIR.resolve() or path.suffix.lower() != '.pdf':
        return None
    return path if path.is_file() else None


def evidence_url(name, page):
    return '?' + urlencode({'evidence_source': name, 'evidence_page': str(page)})


def citation_buttons(sources, message_id):
    """All numbered citations can be selected, regardless of source format."""
    numbered = [(PREFIX.match(source).group(1), source)
                for source in sources if PREFIX.match(source)]
    if not numbered:
        return
    st.caption("View cited evidence:")
    cols = st.columns(min(6, len(numbered)))
    for idx, (number, source) in enumerate(numbered):
        with cols[idx % len(cols)]:
            if st.button(f"[{number}]", key=f"cite_{message_id}_{idx}",
                         help=f"Inspect source {number}"):
                st.session_state.selected_evidence = source
                st.rerun()


def local_source_file(name):
    """Restrict local previews to whitelisted project input directories."""
    if not name or Path(name).name != name or '/' in name or '\\' in name:
        return None
    for base in (DOCUMENTS_DIR, IMAGES_DIR):
        candidate = (base / name).resolve()
        if candidate.parent == base.resolve() and candidate.is_file():
            return candidate
    return None


def get_source_name(source):
    """Extract a source label while preserving spaces and punctuation in names."""
    label = PREFIX.sub('', source).strip()
    for suffix in ('.pdf', '.txt', '.png', '.jpg', '.jpeg', '.webp'):
        match = re.search(re.escape(suffix) + r'(?=\s*\(|$)', label, re.I)
        if match:
            return label[:match.end()].strip()
    return label.split(' (', 1)[0].strip()


def parse_table_row_locator(source):
    """Return (table label, row number) only for explicit row citations."""
    match = PDF_LOCATOR.search(PREFIX.sub('', source).strip())
    if not match:
        return None
    detail = match.group('detail')
    table = re.search(r'\bTable\s+([IVXLCDM]+|\d+)\b', detail, re.I)
    row = re.search(r'\brow\s+(\d+)\b', detail, re.I)
    if not table or not row:
        return None
    return 'Table ' + table.group(1), int(row.group(1))


@st.cache_data(show_spinner=False, max_entries=32)
def table_row_category(pdf_name, table_label, page_number, row_number):
    """Look up the actual indexed row text; read-only against existing ChromaDB."""
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
        # Row text may include Source/Page/Caption lines before Row N.
        record = re.search(r'\bRow\s+\d+\s*:\s*OWASP Category\s*:\s*([^|\n]+)',
                           document, re.I)
        if record:
            return record.group(1).strip()
    return None


@st.cache_data(show_spinner=False, max_entries=8)
def pdf_page_png(name, page, modified_ns, table_label=None, row_number=None):
    """Render one PDF page, highlighting a VERIFIED located table row if possible."""
    import fitz
    path = local_pdf(name)
    if path is None:
        raise FileNotFoundError(name)
    category = None
    if table_label and row_number:
        category = table_row_category(name, table_label, page, row_number)
    with fitz.open(path) as doc:
        if page > len(doc):
            raise ValueError(f'Page {page} is outside this {len(doc)}-page PDF.')
        pdf_page = doc.load_page(page - 1)
        highlighted = False
        highlight_type = None
        if category:
            # Prefer a matching detected table row: never guess based on row height.
            try:
                tables = pdf_page.find_tables()
                for table in tables.tables:
                    extracted = table.extract()
                    for i, row in enumerate(extracted):
                        if not any(cell and category.casefold() in cell.casefold()
                                   for cell in row):
                            continue
                        if i < len(table.rows):
                            rect = fitz.Rect(table.rows[i].bbox)
                            shape = pdf_page.new_shape()
                            shape.draw_rect(rect)
                            shape.finish(color=(1, 0.48, 0), fill=(1, 0.85, 0.12),
                                         fill_opacity=0.17, width=1.5)
                            shape.commit(overlay=True)
                            highlighted = True
                            highlight_type = 'row'
                            break
                    if highlighted:
                        break
            except Exception:
                pass
            if not highlighted:
                # A category label is safer than guessing an entire row boundary.
                found = pdf_page.search_for(category)
                if found:
                    # Category labels reside in the left part of this table.
                    rects = [r for r in found if r.x0 < pdf_page.rect.width * 0.55]
                    if len(rects) == 1:
                        annot = pdf_page.add_highlight_annot(rects[0])
                        annot.set_colors(stroke=(1, 0.75, 0))
                        annot.update()
                        highlighted = True
                        highlight_type = 'category cell'
        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, annots=True)
        return pixmap.tobytes('png'), len(doc), highlighted, highlight_type, category


def show_evidence():
    """Display evidence matching the selected citation, not just PDFs."""
    source = st.session_state.get('selected_evidence')
    if not source:
        return
    st.subheader("Source evidence")
    st.caption(source)
    pdf = parse_pdf_source(source)
    if pdf:
        name, page = pdf
        path = local_pdf(name)
        if path is None:
            st.info("The original PDF isn't available under data/documents.")
            return
        try:
            row_locator = parse_table_row_locator(source)
            label, number = row_locator if row_locator else (None, None)
            png, count, highlighted, highlight_type, category = pdf_page_png(
                name, page, path.stat().st_mtime_ns, label, number)
            st.caption(f"Page {page} of {count}")
            if row_locator:
                if highlighted:
                    st.success(f"Highlighted {highlight_type}: {category} (row {number})")
                else:
                    st.info(f"Table {label}, row {number} is cited, but its position "
                            "could not be reliably located on this page. No highlight was added.")
            st.image(png, use_container_width=True)
            st.download_button("Download PDF", data=path.read_bytes(),
                               file_name=path.name, mime='application/pdf')
        except ImportError:
            st.warning("Install PyMuPDF to view PDF pages: pip install pymupdf")
        except Exception as exc:
            st.warning(f"Unable to display this PDF page: {exc}")
        return

    name = get_source_name(source)
    path = local_source_file(name)
    if path and path.suffix.lower() == '.txt':
        st.caption(f"Text document: {path.name}")
        st.text_area("Document content (preview)",
                     path.read_text(encoding='utf-8', errors='replace')[:12000],
                     height=380, disabled=True)
    elif path and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
        st.caption(f"Image: {path.name}")
        st.image(str(path), use_container_width=True)
    elif name.startswith(('https://', 'http://')):
        parsed = urlparse(name)
        if parsed.scheme in ('http', 'https') and parsed.hostname:
            st.link_button("Open original webpage", name)
        else:
            st.info("Original webpage address unavailable.")
    else:
        st.info("This citation is available as a source reference. A local preview "
                "is not available for this source type or file location.")


def display_sources(sources):
    if not sources:
        return
    with st.expander(f"Retrieved sources ({len(sources)})"):
        for source in sources:
            st.text(f"• {source}")


def render_assistant(answer, sources, message_id):
    st.markdown(answer)
    citation_buttons(sources, message_id)
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
    except Exception as error:
        st.error(f'Database error: {error}')
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
                for filename in saved:
                    st.write(f'• {filename}')
            if skipped:
                st.info('Files already present or unsupported:')
                for filename in skipped:
                    st.write(f'• {filename}')
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
        except Exception as error:
            st.error(f'Indexing error: {error}')

st.title('🔎 Local RAG Search')
st.caption('Chat with your documents using locally running AI.')

# Keep chat full-width until a citation is explicitly selected.
# This also means the evidence pane stays hidden when the app first opens,
# after a new answer, and after the chat is cleared.
selected_evidence = st.session_state.get('selected_evidence')
if selected_evidence:
    chat_col, evidence_col = st.columns([1.15, 1], gap='large')
else:
    chat_col = st.container()
    evidence_col = None

with chat_col:
    if not st.session_state.messages:
        st.info('Ask a question about your indexed documents to begin a conversation.')
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            if message['role'] == 'assistant':
                render_assistant(message['content'], message.get('sources', []), message_index)
            else:
                st.markdown(message['content'])

if evidence_col is not None:
    with evidence_col:
        if st.button('Close evidence', key='close_evidence'):
            st.session_state.selected_evidence = None
            st.rerun()
        show_evidence()

question = st.chat_input('Ask a question about your documents...')
if question:
    st.session_state.selected_evidence = None
    st.session_state.messages.append({'role': 'user', 'content': question})
    with chat_col:
      with st.chat_message('user'):
        st.markdown(question)
    with chat_col:
     with st.chat_message('assistant'):
          try:
              if get_record_count() == 0:
                  answer = 'Your knowledge base is empty. Please index some documents first.'
                  sources = []
              else:
                  with st.spinner('Searching your knowledge base...'):
                      result = run_query(question)
                  if result.returncode != 0:
                      answer = 'The search could not be completed. Check that Ollama is running.'
                      sources = []
                      with st.expander('Error details'):
                          st.code(result.stdout + '\n' + result.stderr, language='text')
                  else:
                      answer, sources = parse_query_output(result.stdout)
                      if not answer:
                          answer = 'The query completed without a readable answer.'
              render_assistant(answer, sources, len(st.session_state.messages))
              st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': sources})
          except subprocess.TimeoutExpired:
              answer = 'Question answering took too long. Check that Ollama is running.'
              st.error(answer)
              st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': []})
          except Exception as error:
              answer = f'Search error: {error}'
              st.error(answer)
              st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': []})
