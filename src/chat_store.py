"""Local SQLite chat history for Local RAG Search. No ChromaDB writes."""
import json
import sqlite3
import uuid
from pathlib import Path

from app_paths import CHAT_DB_PATH

DEFAULT_DB = CHAT_DB_PATH


def _connect(db_path=None):
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15)
    con.execute('PRAGMA foreign_keys = ON')
    con.execute('''CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY, title TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
    con.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK(role IN ('user','assistant')),
        content TEXT NOT NULL,
        sources_json TEXT NOT NULL DEFAULT '[]',
        retrieval_query TEXT)''')
    con.execute('''CREATE TABLE IF NOT EXISTS chat_summaries (
        chat_id TEXT PRIMARY KEY REFERENCES chats(id) ON DELETE CASCADE,
        summary TEXT NOT NULL DEFAULT '',
        covered_messages INTEGER NOT NULL DEFAULT 0 CHECK(covered_messages >= 0),
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
    con.commit()
    return con


def create_chat(db_path=None, title='New chat'):
    chat_id = uuid.uuid4().hex
    with _connect(db_path) as con:
        con.execute('INSERT INTO chats(id,title) VALUES (?,?)', (chat_id, title))
    return chat_id


def list_chats(db_path=None):
    with _connect(db_path) as con:
        rows = con.execute('SELECT id,title FROM chats ORDER BY updated_at DESC, rowid DESC').fetchall()
    return [{'id': row[0], 'title': row[1]} for row in rows]


def chat_exists(chat_id, db_path=None):
    with _connect(db_path) as con:
        return con.execute('SELECT 1 FROM chats WHERE id=?', (chat_id,)).fetchone() is not None


def read_messages(chat_id, db_path=None):
    with _connect(db_path) as con:
        rows = con.execute('''SELECT role,content,sources_json,retrieval_query FROM chat_messages
                              WHERE chat_id=? ORDER BY id''', (chat_id,)).fetchall()
    return [{'role': role, 'content': content, 'sources': json.loads(sources_json),
             'retrieval_query': query} for role,content,sources_json,query in rows]


def add_message(chat_id, message, db_path=None):
    role = message['role']
    if role not in ('user', 'assistant'):
        raise ValueError('Unknown role')
    with _connect(db_path) as con:
        if con.execute('SELECT 1 FROM chats WHERE id=?', (chat_id,)).fetchone() is None:
            raise ValueError('Unknown chat')
        con.execute('''INSERT INTO chat_messages(chat_id,role,content,sources_json,retrieval_query)
                       VALUES (?,?,?,?,?)''', (chat_id, role, message['content'],
              json.dumps(message.get('sources', []), ensure_ascii=False),
              message.get('retrieval_query')))
        if role == 'user':
            old_title, num = con.execute('''SELECT title,(SELECT COUNT(*) FROM chat_messages
                 WHERE chat_id=? AND role='user') FROM chats WHERE id=?''', (chat_id, chat_id)).fetchone()
            if num == 1 and old_title == 'New chat':
                title = ' '.join(message['content'].split())[:65].strip() or 'New chat'
                con.execute('UPDATE chats SET title=? WHERE id=?', (title, chat_id))
        con.execute('UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (chat_id,))


def delete_chat(chat_id, db_path=None):
    with _connect(db_path) as con:
        con.execute('DELETE FROM chats WHERE id=?', (chat_id,))


def get_chat_summary(chat_id, db_path=None):
    """Return a condensed context and its message-count watermark."""
    with _connect(db_path) as con:
        row = con.execute(
            'SELECT summary,covered_messages FROM chat_summaries WHERE chat_id=?',
            (chat_id,)).fetchone()
    return {'summary': row[0], 'covered_messages': row[1]} if row else {
        'summary': '', 'covered_messages': 0}


def save_chat_summary(chat_id, summary, covered_messages, db_path=None):
    """Monotonic, per-chat checkpoint. Never overwrite a newer summary."""
    if not isinstance(summary, str) or not isinstance(covered_messages, int):
        raise ValueError('Invalid summary checkpoint')
    if not (0 <= covered_messages) or len(summary) > 10000:
        raise ValueError('Invalid summary size or checkpoint')
    with _connect(db_path) as con:
        if con.execute('SELECT 1 FROM chats WHERE id=?', (chat_id,)).fetchone() is None:
            raise ValueError('Unknown chat')
        total = con.execute('SELECT COUNT(*) FROM chat_messages WHERE chat_id=?',
                            (chat_id,)).fetchone()[0]
        if covered_messages > total:
            raise ValueError('Summary checkpoint exceeds message count')
        con.execute('''INSERT INTO chat_summaries(chat_id,summary,covered_messages)
                       VALUES (?,?,?) ON CONFLICT(chat_id) DO UPDATE SET
                       summary=excluded.summary,
                       covered_messages=excluded.covered_messages,
                       updated_at=CURRENT_TIMESTAMP
                       WHERE excluded.covered_messages > chat_summaries.covered_messages''',
                    (chat_id, summary, covered_messages))
