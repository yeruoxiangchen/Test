from contextlib import contextmanager
import json
import sqlite3
import time
import uuid


def uid():
    return uuid.uuid4().hex


def encode(value):
    return json.dumps(value, ensure_ascii=False)


class Database:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def one(self, sql, args=()):
        with self.connect() as c:
            row = c.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql, args=()):
        with self.connect() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.connect() as c:
            return c.execute(sql, args).rowcount

    def event(self, run_id, kind, payload):
        self.execute("INSERT INTO events(run_id,kind,payload,created_at) VALUES(?,?,?,?)",
                     (run_id, kind, encode(payload), time.time()))

    def initialize(self):
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript("""
            CREATE TABLE IF NOT EXISTS users(
              id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('admin','member')), created_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(
              token_hash TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
              expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS login_attempts(ip TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS bases(
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
              visibility TEXT NOT NULL CHECK(visibility IN ('all','admin')), created_at REAL);
            CREATE TABLE IF NOT EXISTS documents(
              id TEXT PRIMARY KEY, base_id TEXT REFERENCES bases(id), filename TEXT NOT NULL,
              digest TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL,
              error TEXT, created_at REAL NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
              UNIQUE(base_id,digest));
            CREATE TABLE IF NOT EXISTS chunks(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
              ordinal INTEGER NOT NULL, page INTEGER, text TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(tokens);
            CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
              DELETE FROM chunks_fts WHERE rowid=old.id;
            END;
            CREATE TABLE IF NOT EXISTS vectors(
              chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
              profile TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(chunk_id,profile));
            CREATE TABLE IF NOT EXISTS conversations(
              id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id), title TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS messages(
              id TEXT PRIMARY KEY, conversation_id TEXT REFERENCES conversations(id),
              role TEXT NOT NULL, content TEXT NOT NULL, run_id TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, conversation_id TEXT REFERENCES conversations(id),
              user_id TEXT REFERENCES users(id), question TEXT NOT NULL, base_ids TEXT NOT NULL,
              mode TEXT NOT NULL, status TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}',
              answer TEXT, citations TEXT NOT NULL DEFAULT '[]', error TEXT,
              cancel_requested INTEGER NOT NULL DEFAULT 0, created_at REAL, updated_at REAL);
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT REFERENCES runs(id),
              kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL);
            CREATE TABLE IF NOT EXISTS feedback(
              run_id TEXT REFERENCES runs(id), user_id TEXT REFERENCES users(id),
              rating INTEGER CHECK(rating IN (-1,1)), comment TEXT, created_at REAL,
              PRIMARY KEY(run_id,user_id));
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id,id);
            CREATE INDEX IF NOT EXISTS idx_runs_pending ON runs(status,created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id,created_at);
            """)

    def bases_for(self, user):
        return self.all("SELECT * FROM bases WHERE visibility='all' OR ?='admin' ORDER BY created_at", (user["role"],))

    def check_bases(self, user, ids):
        allowed = {b["id"] for b in self.bases_for(user)}
        if any(i not in allowed for i in ids):
            raise PermissionError("知识库不存在或没有访问权限")
        return list(ids) if ids else sorted(allowed)
