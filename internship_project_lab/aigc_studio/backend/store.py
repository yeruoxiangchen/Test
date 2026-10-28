from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .domain import JobInput, MODES, TERMINAL


class Conflict(Exception):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'assets').mkdir(exist_ok=True)
        self.path = self.root / 'studio.sqlite3'
        with self.db() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, model TEXT NOT NULL, params TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', message TEXT NOT NULL DEFAULT '等待生成服务',
                    created REAL NOT NULL, updated REAL NOT NULL, started REAL,
                    worker_id TEXT, comfy_id TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    retry_of TEXT, request_key TEXT UNIQUE, request_hash TEXT, outputs TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, mime TEXT NOT NULL, kind TEXT NOT NULL,
                    bytes INTEGER NOT NULL, created REAL NOT NULL, job_id TEXT
                );
            ''')
            if 'started' not in {row['name'] for row in db.execute('PRAGMA table_info(jobs)')}:
                db.execute('ALTER TABLE jobs ADD COLUMN started REAL')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def job(row):
        if row is None:
            return None
        result = dict(row)
        result['params'] = json.loads(result['params'])
        result['outputs'] = json.loads(result['outputs'])
        result['cancel_requested'] = bool(result['cancel_requested'])
        result.pop('request_key', None)
        result.pop('request_hash', None)
        return result

    def create(self, spec: JobInput, key: str | None = None, retry_of: str | None = None):
        params = spec.model_dump()
        fingerprint = hashlib.sha256(json.dumps({'params': params, 'retry_of': retry_of}, sort_keys=True).encode()).hexdigest()
        if params['seed'] == -1:
            params['seed'] = secrets.randbelow(2**32)
        ident, now = uuid.uuid4().hex, time.time()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if key:
                old = db.execute('SELECT * FROM jobs WHERE request_key=?', (key,)).fetchone()
                if old:
                    if old['request_hash'] != fingerprint:
                        raise Conflict('重复请求的参数不一致')
                    return self.job(old)
            db.execute('INSERT INTO jobs(id,model,params,created,updated,retry_of,request_key,request_hash) VALUES(?,?,?,?,?,?,?,?)',
                       (ident, MODES[spec.mode], json.dumps(params, ensure_ascii=False), now, now, retry_of, key, fingerprint))
        return self.get(ident)

    def get(self, ident):
        with self.db() as db:
            return self.job(db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone())

    def list_jobs(self, limit=100):
        with self.db() as db:
            return [self.job(r) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC LIMIT ?', (limit,))]

    def active(self, worker_id):
        with self.db() as db:
            return self.job(db.execute("SELECT * FROM jobs WHERE worker_id=? AND status IN ('dispatching','running') ORDER BY created LIMIT 1", (worker_id,)).fetchone())

    def claim(self, worker_id: str, models: list[str]):
        if not models:
            return None
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE worker_id=? AND status IN ('dispatching','running')", (worker_id,)).fetchone():
                return None
            row = db.execute(f"SELECT * FROM jobs WHERE status='queued' AND model IN ({','.join('?' for _ in models)}) ORDER BY created LIMIT 1", models).fetchone()
            if not row:
                return None
            now = time.time()
            db.execute("UPDATE jobs SET status='dispatching',worker_id=?,message='正在提交到推理服务',updated=?,started=? WHERE id=?", (worker_id, now, now, row['id']))
        return self.get(row['id'])

    def update(self, ident, **fields):
        allowed = {'status', 'message', 'comfy_id', 'error', 'outputs', 'cancel_requested'}
        if not fields or set(fields) - allowed:
            raise ValueError('Invalid update fields')
        if 'outputs' in fields:
            fields['outputs'] = json.dumps(fields['outputs'])
        fields['updated'] = time.time()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT status FROM jobs WHERE id=?', (ident,)).fetchone()
            if not old or old['status'] in TERMINAL:
                return
            db.execute(f"UPDATE jobs SET {','.join(k+'=?' for k in fields)} WHERE id=?", (*fields.values(), ident))

    def cancel(self, ident):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?', (ident,)).fetchone()
            if not row:
                return None
            if row['status'] not in TERMINAL:
                queued = row['status'] == 'queued'
                db.execute('UPDATE jobs SET cancel_requested=1,status=?,message=?,updated=? WHERE id=?',
                           ('cancelled' if queued else row['status'], '已取消' if queued else '正在取消', time.time(), ident))
        return self.get(ident)

    def add_asset(self, path: Path, mime: str, kind: str, job_id=None):
        relative = path.resolve().relative_to(self.root / 'assets')
        ident = path.stem
        with self.db() as db:
            db.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?)', (ident, str(relative), mime, kind, path.stat().st_size, time.time(), job_id))
        return self.asset(ident)

    def asset(self, ident):
        with self.db() as db:
            row = db.execute('SELECT * FROM assets WHERE id=?', (ident,)).fetchone()
        return dict(row) if row else None

    def asset_path(self, asset):
        path = (self.root / 'assets' / asset['path']).resolve()
        if not path.is_relative_to(self.root / 'assets'):
            raise ValueError('Invalid asset path')
        return path

