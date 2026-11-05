"""Streamlit interface for Local RAG Search with message-scoped citations."""
import re
import json
import html
from difflib import SequenceMatcher
import subprocess
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import streamlit as st

from chat_store import (create_chat, list_chats, chat_exists, read_messages,
                        add_message, delete_chat, get_chat_summary, save_chat_summary)


from app_paths import (
    DOCUMENTS_DIR,
    IMAGES_DIR,
    DB_PATH,
    URLS_FILE,
)

ROOT = Path(__file__).resolve().parent.parent
INDEX_SCRIPT = ROOT / 'src' / 'index_documents.py'
QUERY_SCRIPT = ROOT / 'src' / 'langgraph_rag.py'
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


def worker_command(mode):
    """Select the appropriate worker command for development or packaging."""
    if getattr(sys, "frozen", False):
        # In a PyInstaller bundle, sys.executable is LocalRAG.exe.
        return [sys.executable, mode]

    # Preserve the existing development workflow.
    script = QUERY_SCRIPT if mode == "query" else INDEX_SCRIPT

    if mode == "query":
        return [sys.executable, str(script), "--conversation-json"]

    if mode == "index":
        return [sys.executable, str(script)]

    raise ValueError(f"Unsupported worker mode: {mode}")


# One indexing job per Streamlit server, shared across browser sessions.
# This is intentionally not stored in st.session_state (which is per session).
INDEXING_LOG_DIR = DB_PATH.parent / 'logs'
INDEXING_LOG_DIR.mkdir(parents=True, exist_ok=True)
INDEXING_PROGRESS_FILE = INDEXING_LOG_DIR / 'indexing-progress.log'


class IndexingJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.started_at = None
        self.completed_at = None
        self.returncode = None
        self.start_error = None

    def running(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False
            self.process = None
            self.returncode = None
            self.start_error = None
            self.completed_at = None
            self.started_at = time.time()
            INDEXING_PROGRESS_FILE.write_text('', encoding='utf-8')
            env = {**os.environ,
                   'PYTHONUNBUFFERED': '1',
                   'PYTHONIOENCODING': 'utf-8',
                   'LOCALRAG_INDEX_PROGRESS_FILE': str(INDEXING_PROGRESS_FILE)}
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            try:
                # In development, stdout writes directly to the progress file.
                # The frozen GUI launcher also explicitly redirects Python's
                # sys.stdout to this file (windowed PyInstaller sets it to None).
                with INDEXING_PROGRESS_FILE.open('a', encoding='utf-8') as log:
                    self.process = subprocess.Popen(
                        worker_command('index'), cwd=str(ROOT),
                        stdout=log, stderr=subprocess.STDOUT,
                        env=env, creationflags=flags)
            except Exception as exc:
                self.start_error = str(exc)
                raise
            return True

    def snapshot(self):
        with self.lock:
            proc = self.process
            if proc is None:
                state = 'idle'
                code = None
            else:
                code = proc.poll()
                if code is None:
                    state = 'running'
                else:
                    state = 'success' if code == 0 else 'failed'
                    self.returncode = code
                    if self.completed_at is None:
                        self.completed_at = time.time()
            started = self.started_at
        try:
            # Bounded read; avoid rendering enormous logs in Streamlit.
            with INDEXING_PROGRESS_FILE.open('rb') as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(max(0, size - 100_000))
                output = file.read().decode('utf-8', errors='replace')
        except FileNotFoundError:
            output = ''
        return state, code, started, output


@st.cache_resource(show_spinner=False)
def indexing_job():
    return IndexingJob()


def indexing_stage(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith('Analysis attempt'):
            return 'Analyzing a PDF figure with the vision model: ' + line
        if line.startswith('Validating new figures:'):
            return line
        if line.startswith('Processing PDF:'):
            return 'Processing PDF figures: ' + line.removeprefix('Processing PDF:').strip()
        if line.startswith('Processing image:'):
            return line
        if line.startswith('Processing:'):
            return 'Processing document: ' + line.removeprefix('Processing:').strip()
        if line.startswith('Processing URL:'):
            return line
        if line.startswith('Indexed ') and '/' in line:
            return 'Generating and storing embeddings: ' + line
        if line.startswith('AUTOMATIC PDF TABLE PROCESSING'):
            return 'Processing PDF tables'
        if line.startswith('AUTOMATIC PDF FIGURE PROCESSING'):
            return 'Processing PDF figures'
        if line.startswith('INDEXING SUMMARY'):
            return 'Preparing indexing summary'
        if line.startswith('Table result for'):
            return line
    return 'Starting document indexing...'


@st.fragment(run_every='2s')
def show_indexing_progress():
    state, returncode, started, output = indexing_job().snapshot()
    if state == 'idle':
        return
    if state == 'running':
        st.info('Indexing is running. You can continue using the application, '
                'but avoid starting another indexing operation.')
        st.write('**Current stage:** ' + indexing_stage(output))
        if started:
            elapsed = int(time.time() - started)
            st.caption(f'Elapsed: {elapsed // 60} min {elapsed % 60} sec. '
                       'PDF vision analysis may take several minutes per figure.')
            if elapsed > 1800:
                st.warning('Indexing has exceeded 30 minutes. Check the progress '
                           'details; this job will not be terminated automatically.')
    elif state == 'success':
        st.success('Indexing completed successfully!')
        try:
            st.metric('Total indexed records', get_record_count())
        except Exception as exc:
            st.warning(f'Could not refresh record count: {exc}')
    else:
        st.error(f'Indexing finished with an error (exit code {returncode}). '
                 'Previously indexed records have not been intentionally cleared.')
    with st.expander('Indexing progress and details', expanded=state == 'running'):
        st.code(output[-20000:] or '(No progress messages yet.)', language='text')


def run_query(question, history, chat_id):
    """Pass this chat's uncondensed history and its saved summary to Ollama."""
    memory = get_chat_summary(chat_id)
    checkpoint = min(memory['covered_messages'], len(history))
    payload = {'question': question,
               'summary': memory['summary'],
               'covered_messages': checkpoint,
               'chat_id': chat_id,
               'history': [
        {'role': m.get('role'), 'content': m.get('content', ''),
         'sources': m.get('sources', [])}
        for m in history[checkpoint:] if m.get('role') in ('user', 'assistant')]}
    # Original chat messages are authoritative for LangGraph's recall route.
    # Keep them separate from condensed memory; no other chat is included.
    payload['recall_history'] = [
        {'role': m.get('role'), 'content': m.get('content', '')}
        for m in history if m.get('role') in ('user', 'assistant')
    ]
    return subprocess.run(worker_command("query"),
                          input=json.dumps(payload, ensure_ascii=False), cwd=str(ROOT),
                          capture_output=True, text=True, encoding='utf-8',
                          errors='replace', timeout=600,
                          env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})



# Conversation routing: use an active chat's history, never another chat's.
# "Rewrite" turns re-express established evidence; "research" turns retrieve
# new evidence with the topic resolved from earlier messages in this chat.
REWRITE_FOLLOWUP_RE = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)*"
    r"(?:explain(?:\s+(?:it|that|this))?(?:\s+again)?|"
    r"(?:summari[sz]e|simplify|clarify)(?:\s+(?:it|that|this))?|"
    r"(?:put|say)\s+(?:it|that|this)\s+in\s+(?:simple|simpler|plain)\s+(?:terms|words)|"
    r"(?:give|provide|show)(?:\s+me)?\s+(?:an?\s+)?example(?:\s+of\s+(?:it|that|this))?|"
    r"what\s+does\s+(?:that|this|it)\s+mean|"
    r"(?:break|walk)\s+(?:it|that|this)\s+down|"
    r"tell\s+me\s+more\s+about\s+(?:it|that|this))"
    r"(?:\s+in\s+(?:simple|simpler|plain)\s+(?:terms|words))?\s*[?.!]*\s*$",
    re.I,
)
REFERENTIAL_RE = re.compile(
    r"\b(?:it|its|this|that|these|those|they|their|them|above|previous|earlier)\b|"
    r"^\s*(?:what about|how about|and what|what are its)\b",
    re.I,
)


def conversation_topic(history):
    """Get a bounded topic anchor from prior turns in the *active* chat.

    Prefer the most recent standalone user question; skip generic rewrite turns
    and short referential questions. This prevents 'plain English' and 'example'
    from becoming the subject of the next turn, including after app restart.
    """
    turns = history[:-1]  # currently submitted user question is last
    for message in reversed(turns[-24:]):
        if message.get('role') != 'user':
            continue
        candidate = message.get('content', '').strip()
        if candidate and not REWRITE_FOLLOWUP_RE.fullmatch(candidate) and not REFERENTIAL_RE.search(candidate):
            return candidate[:650]
    # The earliest user turn is a useful fallback for an ongoing same-topic chat.
    for message in turns:
        if message.get('role') == 'user' and message.get('content', '').strip():
            return message['content'].strip()[:650]
    return None


def recent_context(history, max_turns=8, max_chars=3500):
    """Bounded context window; never passes entire lifelong chat to the model."""
    result = []
    for msg in history[:-1][-max_turns:]:
        role = msg.get('role')
        if role not in ('user', 'assistant'):
            continue
        body = re.sub(r'\[\d+\]', '', msg.get('content', '')).strip()
        if body:
            result.append(f'{role.upper()}: {body[:550]}')
    return '\n'.join(result)[-max_chars:]


def resolve_conversation_question(question, history):
    """Preserve standalone questions; attach grounded topic to referential ones."""
    topic = conversation_topic(history)
    if not topic:
        return question, False
    if REWRITE_FOLLOWUP_RE.fullmatch(question):
        return f'{question.rstrip(" ?.! ")} about the topic: {topic}', True
    if not REFERENTIAL_RE.search(question):
        return question, False
    # The retrieval engine remains responsible for searching indexed documents
    # and issuing citations. Context is solely for disambiguating the question.
    return (f'{question.rstrip(" ?.! ")}\n'
            f'In this conversation, the subject is: {topic}\n'
            'Answer the latest question about that subject using indexed evidence.'), True


def transform_followup(question, previous_answer, previous_sources, history=None):
    """Rewrite cited facts or give an explicitly *hypothetical* example.

    A rewrite does not silently retrieve unrelated documents, and retains only
    citations present in the original answer. This is not new evidence retrieval.
    """
    import ollama
    from config import ANSWER_MODEL

    plain = re.sub(r'\[\d+\]', '', previous_answer).strip()
    sources = [src for src in previous_sources if PREFIX.match(src)]
    if not plain or not sources:
        raise ValueError('No verifiably cited previous answer to transform.')

    if re.search(r'example', question, re.I):
        instruction = (
            'Give ONE clearly labeled HYPOTHETICAL example illustrating the ' 
            'mechanism in the prior answer. Explicitly say it is an imagined ' 
            'example, not an experiment reported in the paper. Focus on the ' 
            'underlying subject, not the meaning of the word example. If the prior ''answer concerns HTTP/API faults, illustrate an API failure instead ''of an unrelated internal parsing error. '
            'Explain in 2-4 beginner-friendly sentences.')
    elif re.search(r'summari[sz]e|shorter|brief', question, re.I):
        instruction = 'Summarize the prior answer in 1-2 short everyday-language sentences.'
    else:
        instruction = ('Explain the information from the prior answer in 2-4 short '
                       'sentences using everyday words, explaining any jargon.')

    topic = conversation_topic(history) if history else None
    prompt = (
        'You are continuing a conversation about a document. Do NOT answer the '
        'follow-up as a standalone question. Only use the facts in PREVIOUS ANSWER '
        'for statements about the document. An imagined example must be labeled '
        'hypothetical. Do not introduce any factual results, figures, or statistics. '
        'Do not write citation numbers; the app attaches existing citations.\n'
        f'CONVERSATION SUBJECT: {topic or "the previous grounded answer"}\n'
        f'USER FOLLOW-UP: {question}\nINSTRUCTION: {instruction}\n'
        f'PREVIOUS ANSWER:\n{plain[:3200]}\n\nRESPONSE:')
    response = ollama.chat(model=ANSWER_MODEL,
                           messages=[{'role': 'user', 'content': prompt}],
                           think=False, options={'num_predict': 260})
    result = re.sub(r'\[\d+\]', '', response['message']['content']).strip()
    if not result or result == 'INSUFFICIENT_INFORMATION' or not re.search(r'[.!?]$', result):
        raise ValueError('The follow-up answer was incomplete.')
    prior_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', plain))
    next_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', result))
    if not next_numbers.issubset(prior_numbers):
        raise ValueError('The follow-up introduced unsupported numerical details.')
    cited = list(dict.fromkeys(re.findall(r'\[(\d+)\]', previous_answer)))
    valid = {PREFIX.match(src).group(1) for src in sources}
    retained = [n for n in cited if n in valid]
    if not retained:
        raise ValueError('The previous answer had no supported numbered citation.')
    if re.search(r'example', question, re.I):
        # Attach sources to the documented mechanism, never to the invented story.
        return (result + '\n\n*Hypothetical illustration generated by the AI; '
                'not an example reported in the source. Supporting mechanism: '
                + ' '.join(f'[{n}]' for n in retained) + '*'), sources
    return result + ' ' + ' '.join(f'[{n}]' for n in retained), sources


# Only opt into the additional reasoning pass for explicitly analytical requests.
# Ordinary fact questions retain the existing tested RAG pipeline.
REASONING_REQUEST_RE = re.compile(
    r"\b(?:might|could|hypothe(?:sis|ses|tical)|infer|inference|implication|"
    r"why would|why might|what if|trade-?offs?|apply (?:this|it)|"
    r"how (?:could|would) (?:i|we)|what (?:experiment|test) (?:could|would))\b",
    re.I,
)


def is_reasoning_request(question):
    return bool(REASONING_REQUEST_RE.search(question))


def label_reasoned_answer(question, answer, sources):
    """Classify an answer's provenance without treating an inference as a quote.

    The RAG engine has already retrieved and cited the answer. This pass can
    make its epistemic status explicit; on any failure use the original answer.
    """
    import ollama
    from config import ANSWER_MODEL

    numbered = [src for src in sources if PREFIX.match(src)]
    citations = list(dict.fromkeys(re.findall(r"\[(\d+)\]", answer)))
    valid = {PREFIX.match(src).group(1) for src in numbered}
    citations = [num for num in citations if num in valid]
    if not citations:
        return answer  # Never manufacture evidence.
    plain = re.sub(r"\[\d+\]", "", answer).strip()
    if not plain or 'INSUFFICIENT_INFORMATION' in plain:
        return answer
    instructions = (
        'You are a cautious research assistant. The text below is an answer '
        'generated using retrieved sources, NOT a verbatim quote from them. '
        'Rewrite it in concise language, clearly distinguish supported factual '
        'premises from your own reasoning. Use explicit labels "Document-based '
        'premise:" and "Reasoned inference:". Do not state that an inference is '
        'proven by the paper. If the evidence is insufficient, say what cannot '
        'be concluded. Do not add new claims, numbers, paper results, or sources. '
        'Do not output citation numbers; the application handles citations. '
        'Do not claim to have read the source directly beyond the supplied '
        'answer.\n'
        f'USER QUESTION: {question[:650]}\n'
        f'RETRIEVED ANSWER: {plain[:3200]}\n'
        'REVISED ANSWER:'
    )
    try:
        response = ollama.chat(model=ANSWER_MODEL,
                               messages=[{'role': 'user', 'content': instructions}],
                               think=False, options={'num_predict': 330})
        revised = re.sub(r"\[\d+\]", "", response['message']['content']).strip()
        if not revised or not re.search(r'[.!?]$', revised):
            return answer
        old_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', plain))
        new_numbers = set(re.findall(r'(?<!\w)\d+(?:\.\d+)?(?!\w)', revised))
        if not new_numbers.issubset(old_numbers):
            return answer
        if 'Document-based premise:' not in revised or 'Reasoned inference:' not in revised:
            return answer
        # Cite only the supporting premises. The inference remains explicitly labeled.
        before, after = revised.split('Reasoned inference:', 1)
        before = before.rstrip() + ' ' + ' '.join(f'[{n}]' for n in citations)
        return before + '\n\n**Reasoned inference (not explicitly stated in the source):** ' + after.strip()
    except Exception:
        return answer


def is_transform_followup(question):
    return bool(REWRITE_FOLLOWUP_RE.fullmatch(question))


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
def pdf_page_png(name, page, modified_ns, table_label=None, row_number=None, chunk_text=None, figure_number=None):
    """Render the cited page; retain verified PDF table-row highlighting."""
    import pymupdf as fitz
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
        if figure_number is not None and not highlighted:
            highlighted, highlight_type = mark_pdf_figure(pdf_page, figure_number)
        if chunk_text and not highlighted and not table_label and figure_number is None:
            highlighted, word_count = highlight_pdf_chunk(pdf_page, chunk_text)
            if highlighted:
                highlight_type = f'PDF text passage ({word_count} matching words)'
        png = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False,
                                 annots=True).tobytes('png')
        return png, len(document), highlighted, highlight_type, category






def parse_pdf_figure_locator(source):
    """Identify an explicitly cited figure; never infer it from a query."""
    if not PREFIX.match(source) and not source.startswith('Retrieved context (citation unverified): '):
        return None
    matched = PDF_LOCATOR.search(PREFIX.sub('', source).strip())
    if not matched:
        return None
    found = re.search(r'\bFigure\s+(\d+)\b', matched.group('detail'), re.I)
    return int(found.group(1)) if found else None


def mark_pdf_figure(page, figure_number):
    """Outline a figure bounded by unique caption and vector geometry.

    For layouts where the diagram extent is uncertain, highlight ONLY the
    exact caption. This avoids outlining surrounding paragraphs as evidence.
    """
    import pymupdf as fitz
    caption_re = re.compile(r'^\s*(?:Figure|Fig\.)\s*' + re.escape(str(figure_number)) + r'\s*[:.\-]', re.I)
    captions = []
    for block in page.get_text('blocks'):
        if len(block) >= 5 and isinstance(block[4], str) and caption_re.search(block[4]):
            captions.append(fitz.Rect(block[:4]))
    if len(captions) != 1:
        return False, None
    caption = captions[0]
    # Figure must be above its caption. Bound it by the closest preceding
    # other numbered caption or page header, not a guessed figure height.
    previous_captions = [fitz.Rect(b[:4]).y1 for b in page.get_text('blocks')
                         if len(b) >= 5 and isinstance(b[4], str)
                         and re.search(r'^\s*(?:Figure|Fig\.|Table)\s+\d+\s*[:.\-]', b[4], re.I)
                         and fitz.Rect(b[:4]).y1 < caption.y0]
    top = max(previous_captions, default=page.rect.y0 + 45) + 5
    bottom = caption.y0 - 5
    if bottom - top >= 65:
        # Use only dense vector geometry and placed images; avoids PDF text
        # and arbitrary estimates from a caption's row number.
        graphics = []
        for drawing in page.get_drawings():
            r = drawing.get('rect')
            if r is None:
                continue
            r = fitz.Rect(r)
            if r.y0 >= top and r.y1 <= bottom and r.width >= 5 and r.height >= 4:
                graphics.append(r)
        for info in page.get_image_info():
            r = fitz.Rect(info['bbox'])
            if r.y0 >= top and r.y1 <= bottom and r.width >= 35 and r.height >= 30:
                graphics.append(r)
        if len(graphics) >= 12:
            box = fitz.Rect(graphics[0])
            for r in graphics[1:]:
                box |= r
            box += (-6, -6, 6, 6)
            box &= page.rect
            if (box.width >= 160 and box.height >= 70
                    and box.y1 < caption.y0 + 3
                    and box.height < page.rect.height * .50):
                shape = page.new_shape()
                shape.draw_rect(box)
                shape.finish(color=(1, .45, 0), fill=(1, .85, 0),
                             fill_opacity=.09, width=2)
                shape.commit(overlay=True)
                return True, 'figure region'
    annot = page.add_highlight_annot(caption)
    annot.set_colors(stroke=(1, .75, 0))
    annot.update()
    return True, 'figure caption only (region not verified)'


def parse_txt_chunk_locator(source):
    """Return exact TXT filename and cited chunk, never a guessed chunk."""
    if not PREFIX.match(source) and not source.startswith('Retrieved context (citation unverified): '):
        return None
    name = get_source_name(source)
    if not name.lower().endswith('.txt') or '/' in name or "\\" in name:
        return None
    match = re.search(r'\bchunk\s+(\d+)\b', source, re.I)
    return (name, int(match.group(1))) if match else None


@st.cache_data(show_spinner=False, max_entries=64)
def indexed_txt_chunk(name, number):
    db = chromadb.PersistentClient(path=str(DB_PATH))
    collection = db.get_collection('documents')
    records = collection.get(where={'$and': [
        {'source': name}, {'file_type': 'txt'}, {'chunk_number': number},
    ]}, include=['documents', 'metadatas'])
    chunks = records.get('documents') or []
    return chunks[0] if len(chunks) == 1 else None


def normalized_with_positions(text):
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
    original, positions = normalized_with_positions(document)
    needle, _ = normalized_with_positions(chunk)
    if not needle or not original:
        return None
    index = original.find(needle)
    if index >= 0 and original.find(needle, index + 1) == -1:
        return positions[index], positions[index + len(needle) - 1] + 1
    matcher = SequenceMatcher(None, original, needle, autojunk=False)
    best = matcher.find_longest_match(0, len(original), 0, len(needle))
    if best.size < 80 or best.size < len(needle) * .35:
        return None
    segment = original[best.a:best.a + best.size]
    if original.find(segment, best.a + 1) != -1:
        return None
    return positions[best.a], positions[best.a + best.size - 1] + 1


def show_txt_highlight(source, path):
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
    if source.startswith('Retrieved context (citation unverified): '):
        st.info('Source preview only: this passage has not been verified as supporting the answer.')
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
            figure_number = parse_pdf_figure_locator(source)
            chunk_text = (indexed_pdf_chunk(name, page, chunk_number)
                          if chunk_number is not None else None)
            png, count, highlighted, kind, category = pdf_page_png(
                name, page, path.stat().st_mtime_ns, label, number, chunk_text, figure_number)
            st.caption(f'Page {page} of {count}')
            if locator:
                if highlighted:
                    st.success(f'Highlighted {kind}: {category} (row {number})')
                else:
                    st.info('Exact row position could not be reliably located; showing the page.')
            if figure_number is not None:
                if highlighted:
                    st.success(f'Located Figure {figure_number}: highlighted {kind}.')
                else:
                    st.info('The cited figure could not be located reliably; no highlight was added.')
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
    """Show numbered citations AND clearly marked, unverified source previews."""
    if not sources:
        return
    numbered = []
    unverified = []
    for source in sources:
        match = PREFIX.match(source)
        if match:
            numbered.append((match.group(1), source))
        elif source.startswith('Retrieved context (citation unverified): '):
            unverified.append(source)
    if not numbered and not unverified:
        return
    st.caption('View cited evidence:' if numbered else 'View retrieved source (not a verified citation):')
    entries = [(f'[{n}]', src, n) for n, src in numbered]
    entries += [(f'Preview {i}', src, f'preview{i}') for i, src in enumerate(unverified, 1)]
    cols = st.columns(min(5, len(entries)))
    for idx, (label, source, number) in enumerate(entries):
        with cols[idx % len(cols)]:
            if st.button(label, key=f'cite_{message_id}_{idx}',
                         disabled=st.session_state.get('answer_in_progress', False),
                         help=('Source preview only; citation not claim-verified'
                               if label.startswith('Preview') else
                               f'Answer {answer_number}, citation {number}')):
                st.session_state.selected_evidence = {
                    'source': source, 'message_id': message_id,
                    'answer_number': answer_number,
                    'citation_number': number,
                }
                st.rerun()


def display_sources(sources):
    if sources:
        with st.expander(f'Retrieved sources ({len(sources)})'):
            for source in sources:
                st.text(f'• {source}')


def answer_provenance_labels(answer, sources, user_question=''):
    """Classify content type and evidence status independently.

    Use the original user question as well as the generated answer: a model
    can omit the literal phrase 'hypothetical example' even when requested.
    This is presentation metadata, not an assertion that a source proves a claim.
    """
    content = (answer or '').casefold()
    request = (user_question or '').casefold()
    sources = sources or []
    labels = []
    is_recall = content.lstrip().startswith('**from our earlier discussion in this chat:**')
    is_example = (
        bool(re.search(r'\b(hypothetical|illustrative|imaginary)\b', request))
        and bool(re.search(r'\b(example|scenario|illustration)\b', request))
        or bool(re.search(r'\b(example|scenario|illustration)\b', request))
        and bool(re.search(r'\b(give|show|provide|create|write|generate|walk through)\b', request))
        or 'hypothetical example' in content
        or 'this example is ai-generated' in content
    )
    is_inference = (
        not is_example and not is_recall
        and (bool(re.search(r'\b(infer|inference|speculat|deduc|reasoned analysis)\w*\b', request))
             or 'reasoned inference' in content
             or 'reasoned analysis' in content)
    )

    if is_recall:
        labels.append('Conversation recall · Not independently verified against documents')
    elif is_example:
        labels.append('Hypothetical example · AI-generated illustration, not a reported experiment')
    elif is_inference:
        labels.append('Reasoned inference · Not an established document finding')

    unverified = any(
        str(source).startswith('Retrieved context (citation unverified): ')
        for source in sources
    )
    numbered = any(re.match(r'^\[\d+\]\s+', str(source)) for source in sources)
    if unverified:
        labels.append('Evidence: Retrieved context available · Claim-level citation unverified')
    elif numbered:
        labels.append('Evidence: Document citations matched to retrieved passages · Check source evidence')
    elif is_recall:
        labels.append('Evidence: Earlier conversation only · No independent document verification')
    elif is_example or is_inference:
        labels.append('Evidence: No independently verified document citation')
    return labels


def render_assistant(answer, sources, message_id, answer_number, user_question=''):
    st.markdown(answer)
    for label in answer_provenance_labels(answer, sources, user_question):
        st.caption(label)
    citation_buttons(sources, message_id, answer_number)
    display_sources(sources)


st.set_page_config(page_title='LUCY', page_icon='🔎', layout='wide')
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_evidence' not in st.session_state:
    st.session_state.selected_evidence = None
if 'answer_in_progress' not in st.session_state:
    st.session_state.answer_in_progress = False
if 'pending_delete_chat' not in st.session_state:
    st.session_state.pending_delete_chat = None
if 'active_chat_id' not in st.session_state or not chat_exists(st.session_state.active_chat_id):
    previous = st.session_state.get('messages', [])
    existing = list_chats()
    # On upgrade, import the currently displayed unsaved chat once.
    if previous:
        chat_id = create_chat()
        for item in previous:
            add_message(chat_id, item)
    elif existing:
        chat_id = existing[0]['id']
    else:
        chat_id = create_chat()
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = read_messages(chat_id)


with st.sidebar:
    st.title('Chats')
    if st.button('＋ New chat', use_container_width=True,
                 disabled=st.session_state.answer_in_progress):
        st.session_state.active_chat_id = create_chat()
        st.session_state.messages = []
        st.session_state.selected_evidence = None
        st.rerun()
    chats = list_chats()
    pending_delete = st.session_state.get('pending_delete_chat')
    for chat in chats:
        active = chat['id'] == st.session_state.active_chat_id
        title_col, delete_col = st.columns([6, 1], gap='small')
        with title_col:
            if st.button(('● ' if active else '') + chat['title'],
                         key='switch_' + chat['id'], use_container_width=True,
                         disabled=st.session_state.answer_in_progress):
                st.session_state.active_chat_id = chat['id']
                st.session_state.messages = read_messages(chat['id'])
                st.session_state.selected_evidence = None
                st.session_state.pending_delete_chat = None
                st.rerun()
        with delete_col:
            if st.button('🗑', key='delete_button_' + chat['id'],
                         help='Delete this conversation',
                         disabled=st.session_state.answer_in_progress):
                st.session_state.pending_delete_chat = chat['id']
                st.rerun()

    # Deletion is permanent, so confirm before removing the chat AND its messages.
    if pending_delete and any(c['id'] == pending_delete for c in chats):
        doomed = next(c for c in chats if c['id'] == pending_delete)
        st.warning(f"Delete conversation: {doomed['title']}?")
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button('Confirm delete', key='confirm_chat_delete',
                         type='primary', use_container_width=True,
                         disabled=st.session_state.answer_in_progress):
                delete_chat(pending_delete)
                st.session_state.pending_delete_chat = None
                remaining = list_chats()
                if pending_delete == st.session_state.active_chat_id:
                    replacement = remaining[0]['id'] if remaining else create_chat()
                    st.session_state.active_chat_id = replacement
                    st.session_state.messages = read_messages(replacement)
                    st.session_state.selected_evidence = None
                st.rerun()
        with cancel_col:
            if st.button('Cancel', key='cancel_chat_delete',
                         use_container_width=True):
                st.session_state.pending_delete_chat = None
                st.rerun()
    st.divider()
    st.title('Knowledge Base')
    try:
        st.metric('Indexed records', get_record_count())
    except Exception as exc:
        st.error(f'Database error: {exc}')
    st.divider()
    uploaded_files = st.file_uploader('Upload documents or images',
        type=['txt', 'pdf', 'png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)
    website_url = st.text_input('Webpage URL', placeholder='https://example.com/article')
    index_clicked = st.button('Index content', type='primary', use_container_width=True,
                              disabled=indexing_job().running())
    st.divider()
    if st.button('Clear chat', use_container_width=True):
        # Clear only the current conversation; keep all other saved chats.
        delete_chat(st.session_state.active_chat_id)
        st.session_state.active_chat_id = create_chat()
        st.session_state.messages = []
        st.session_state.selected_evidence = None
        st.session_state.pending_delete_chat = None
        st.rerun()
    with st.expander('Conversation memory diagnostics'):
        memory = get_chat_summary(st.session_state.active_chat_id)
        st.caption(f"Saved messages: {len(st.session_state.messages)}")
        st.caption(f"Condensed messages: {memory['covered_messages']}")
        st.caption(f"Summary length: {len(memory['summary'])} characters")
        st.text_area('Saved condensed summary (read-only)',
                     memory['summary'] or '(No summary saved yet)',
                     height=160, disabled=True)
    st.caption('Powered by Ollama + ChromaDB')

if index_clicked:
    job = indexing_job()
    if job.running():
        st.warning('Indexing is already running. Please wait for it to finish.')
    elif not uploaded_files and not website_url.strip():
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
            # A global guard prevents multiple sessions from launching
            # concurrent indexing workers against the same ChromaDB collection.
            if not job.start():
                st.warning('Another indexing job is already running.')
        except Exception as exc:
            st.error(f'Could not start indexing: {exc}')

# Unlike subprocess.run(), this updates while the worker is processing.
show_indexing_progress()

st.title('🔎 LUCY')
st.caption('Local Understanding and Contextual sYnthesis')
st.caption('Chat with your documents using locally running AI.')

# Important: consume a newly submitted question BEFORE creating the layout.
# This removes the Close evidence widget from the run that performs a blocking query.
question = st.chat_input('Ask a question about your documents...')
if not question:
    # A prior run might have been interrupted. No query is running in this run.
    st.session_state.answer_in_progress = False
if question:
    st.session_state.answer_in_progress = True
    st.session_state.selected_evidence = None
    user_message = {'role': 'user', 'content': question}
    st.session_state.messages.append(user_message)
    add_message(st.session_state.active_chat_id, user_message)

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
                prior_question = (st.session_state.messages[message_id - 1].get('content', '')
                                  if message_id > 0 and st.session_state.messages[message_id - 1].get('role') == 'user'
                                  else '')
                render_assistant(message['content'], message.get('sources', []),
                                 message_id, answer_number, prior_question)
            else:
                st.markdown(message['content'])

if evidence_col is not None:
    with evidence_col:
        if st.button('Close evidence', key='close_evidence',
                     disabled=st.session_state.get('answer_in_progress', False)):
            st.session_state.selected_evidence = None
            st.rerun()
        show_evidence(selected)

if question:
    with chat_col:
        with st.chat_message('assistant'):
            try:
                retrieval_question = question
                if get_record_count() == 0:
                    answer, sources = ('Your knowledge base is empty. '
                                       'Please index some documents first.'), []
                else:
                    with st.spinner('Searching your knowledge base...'):
                        # The engine owns conversational resolution, retrieval,
                        # inference, and example generation; the UI only presents it.
                        result = run_query(question, st.session_state.messages[:-1], st.session_state.active_chat_id)
                    if result is not None:
                        if result.returncode:
                            answer, sources = ('The search could not be completed. '
                                               'Check that Ollama is running.'), []
                            with st.expander('Error details'):
                                st.code(result.stdout + '\n' + result.stderr, language='text')
                        else:
                            try:
                                payload = json.loads(result.stdout)
                                answer = payload['answer']
                                sources = payload.get('sources') or []
                                if (isinstance(payload.get('summary'), str)
                                        and isinstance(payload.get('covered_messages'), int)):
                                    save_chat_summary(st.session_state.active_chat_id,
                                                      payload['summary'],
                                                      payload['covered_messages'])
                            except (ValueError, KeyError, TypeError):
                                answer = 'The query engine returned an unreadable response.'
                                sources = []
                            if not answer:
                                answer = 'The query completed without a readable answer.'
                assistant_message = {
                    'role': 'assistant', 'content': answer, 'sources': sources,
                    'retrieval_query': retrieval_question}
                st.session_state.messages.append(assistant_message)
                add_message(st.session_state.active_chat_id, assistant_message)
                render_assistant(answer, sources, len(st.session_state.messages) - 1,
                                 answer_number + 1, question)
            except subprocess.TimeoutExpired:
                answer = 'Question answering took too long. Check that Ollama is running.'
                st.error(answer)
                error_message = {'role': 'assistant', 'content': answer, 'sources': []}
                st.session_state.messages.append(error_message)
                add_message(st.session_state.active_chat_id, error_message)
            except Exception as exc:
                answer = f'Search error: {exc}'
                st.error(answer)
                error_message = {'role': 'assistant', 'content': answer, 'sources': []}
                st.session_state.messages.append(error_message)
                add_message(st.session_state.active_chat_id, error_message)

            finally:
                st.session_state.answer_in_progress = False

# Refresh widgets after generation; buttons created earlier in this run were
# disabled while the query was active and will otherwise stay disabled.
if question:
    st.rerun()
