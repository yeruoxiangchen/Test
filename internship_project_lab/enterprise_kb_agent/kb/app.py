from contextlib import asynccontextmanager
import asyncio
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .auth import create_session, create_user, session_user, token_hash, verify_password
from .config import Settings
from .db import Database, uid, encode
from .ingestion import register_document
from .retrieval import Retriever
from .worker import Workers


class Login(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class NewUser(Login):
    role: str = "member"


class NewBase(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    visibility: str = "all"


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    conversation_id: str | None = None
    base_ids: list[str] = Field(default_factory=list, max_length=50)


class Feedback(BaseModel):
    rating: int
    comment: str = Field(default="", max_length=2000)


def create_app(settings=None, start_workers=True):
    cfg = settings or Settings()
    db = Database(cfg.db_path)

    @asynccontextmanager
    async def lifespan(app):
        cfg.prepare()
        lock = (cfg.root / "worker.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            raise RuntimeError("同一数据目录只能运行一个应用进程；请使用 --workers 1") from None
        db.initialize()
        workers = Workers(db, cfg)
        app.state.db, app.state.settings = db, cfg
        if start_workers:
            workers.start()
        try:
            yield
        finally:
            if start_workers:
                workers.stop()
            lock.close()

    app = FastAPI(title="Knowledge Desk", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if request.headers.get("x-kb-client") != "web" or (origin and urlsplit(origin).netloc != request.headers.get("host")):
                return JSONResponse({"detail": "请求来源校验失败"}, status_code=403)
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "非法 Content-Length"}, status_code=400)
            if length > cfg.max_upload_bytes + 1024 * 1024:
                return JSONResponse({"detail": "请求内容过大"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def current_user(request: Request):
        user = session_user(db, request.cookies.get("kb_session"))
        if not user:
            raise HTTPException(401, "请先登录")
        return user

    def admin(user=Depends(current_user)):
        if user["role"] != "admin":
            raise HTTPException(403, "需要管理员权限")
        return user

    def base_access(user, ids):
        try:
            return db.check_bases(user, ids)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from None

    def own_run(user, run_id):
        row = db.one("SELECT * FROM runs WHERE id=? AND user_id=?", (run_id, user["id"]))
        if not row:
            raise HTTPException(404, "任务不存在")
        return row

    def document_access(user, document_id):
        row = db.one("SELECT * FROM documents WHERE id=?", (document_id,))
        if not row:
            raise HTTPException(404, "文档不存在")
        base_access(user, [row["base_id"]])
        return row

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "enterprise-kb-agent", "version": "0.1.0", "pid": os.getpid()}

    @app.post("/api/login")
    def login(payload: Login, request: Request, response: Response):
        ip = request.client.host if request.client else "local"
        now = time.time()
        db.execute("DELETE FROM login_attempts WHERE created_at<?", (now - 600,))
        if db.one("SELECT count(*) AS n FROM login_attempts WHERE ip=?", (ip,))["n"] >= 20:
            raise HTTPException(429, "登录尝试过多，请 10 分钟后重试")
        user = db.one("SELECT * FROM users WHERE username=?", (payload.username,))
        if not user or not verify_password(payload.password, user["password"]):
            db.execute("INSERT INTO login_attempts VALUES(?,?)", (ip, now))
            raise HTTPException(401, "账号或密码错误")
        token = create_session(db, user["id"])
        response.set_cookie("kb_session", token, httponly=True, secure=cfg.secure_cookie, samesite="strict", max_age=8*3600)
        return {k: user[k] for k in ("id", "username", "role")}

    @app.post("/api/logout")
    def logout(request: Request, response: Response, user=Depends(current_user)):
        db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(request.cookies["kb_session"]),))
        response.delete_cookie("kb_session")
        return {"ok": True}

    @app.get("/api/me")
    def me(user=Depends(current_user)):
        return {**user, "mode": cfg.mode, "retrieval": "hybrid" if cfg.embedding_url else "keyword",
                "model": cfg.llm_model if cfg.mode == "agent" else None}

    @app.post("/api/users")
    def add_user(payload: NewUser, user=Depends(admin)):
        try:
            return {"id": create_user(db, payload.username, payload.password, payload.role)}
        except (ValueError, sqlite3.IntegrityError) as exc:
            raise HTTPException(400, "账号已存在或输入不符合要求" if isinstance(exc, sqlite3.IntegrityError) else str(exc)) from None

    @app.get("/api/bases")
    def bases(user=Depends(current_user)):
        rows = db.bases_for(user)
        for row in rows:
            row["document_count"] = db.one("SELECT count(*) AS n FROM documents WHERE base_id=?", (row["id"],))["n"]
        return rows

    @app.post("/api/bases", status_code=201)
    def add_base(payload: NewBase, user=Depends(admin)):
        if payload.visibility not in {"all", "admin"} or not payload.name.strip():
            raise HTTPException(400, "请填写有效名称和可见范围")
        base_id = uid()
        db.execute("INSERT INTO bases VALUES(?,?,?,?,?)", (base_id, payload.name.strip(), payload.description, payload.visibility, time.time()))
        return {"id": base_id}

    @app.get("/api/documents")
    def documents(base_id: str | None = None, user=Depends(current_user)):
        ids = base_access(user, [base_id] if base_id else [])
        if not ids:
            return []
        return db.all("SELECT * FROM documents WHERE base_id IN (" + ",".join("?" for _ in ids) + ") ORDER BY created_at DESC", ids)

    @app.post("/api/documents", status_code=202)
    async def upload(base_id: str = Form(...), file: UploadFile = File(...), user=Depends(admin)):
        base_access(user, [base_id])
        try:
            data = await file.read(cfg.max_upload_bytes + 1)
            doc, duplicate = register_document(db, cfg, base_id, file.filename or "", data)
            return {**doc, "duplicate": duplicate}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        finally:
            await file.close()

    @app.get("/api/documents/{document_id}")
    def detail(document_id: str, user=Depends(current_user)):
        doc = document_access(user, document_id)
        doc["chunks"] = db.all("SELECT id,ordinal,page,text FROM chunks WHERE document_id=? ORDER BY ordinal", (document_id,))
        return doc

    @app.get("/api/documents/{document_id}/download")
    def download(document_id: str, user=Depends(current_user)):
        doc = document_access(user, document_id)
        return FileResponse(cfg.root / "uploads" / document_id, filename=doc["filename"], media_type="application/octet-stream")

    @app.post("/api/documents/{document_id}/reindex")
    def reindex(document_id: str, user=Depends(admin)):
        doc = document_access(user, document_id)
        if doc["status"] == "processing":
            raise HTTPException(409, "文档正在处理中")
        db.execute("UPDATE documents SET status='queued',error=NULL WHERE id=?", (document_id,))
        return {"status": "queued"}

    @app.delete("/api/documents/{document_id}")
    def delete_document(document_id: str, user=Depends(admin)):
        document_access(user, document_id)
        db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        (cfg.root / "uploads" / document_id).unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/search")
    def search(q: str, base_id: str | None = None, user=Depends(current_user)):
        if not 1 <= len(q.strip()) <= 500:
            raise HTTPException(400, "检索词长度需为 1–500 字符")
        ids = base_access(user, [base_id] if base_id else [])
        rows, warnings = Retriever(db, cfg).search(user, q, ids)
        return {"results": rows, "warnings": warnings}

    @app.get("/api/conversations")
    def conversations(user=Depends(current_user)):
        return db.all("SELECT * FROM conversations WHERE user_id=? ORDER BY created_at DESC LIMIT 100", (user["id"],))

    @app.get("/api/conversations/{conversation_id}")
    def conversation(conversation_id: str, user=Depends(current_user)):
        row = db.one("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"]))
        if not row:
            raise HTTPException(404, "会话不存在")
        row["messages"] = db.all("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,))
        row["runs"] = db.all("SELECT id,status,answer,citations,error,question,created_at FROM runs WHERE conversation_id=? ORDER BY created_at", (conversation_id,))
        for run in row["runs"]:
            run["citations"] = json.loads(run["citations"])
        return row

    @app.post("/api/chat", status_code=202)
    def chat(payload: Question, user=Depends(current_user)):
        if not payload.question.strip():
            raise HTTPException(400, "问题不能为空")
        ids = base_access(user, payload.base_ids)
        if not ids:
            raise HTTPException(400, "请先创建可访问的知识库并上传文档")
        conversation_id = payload.conversation_id or uid()
        if payload.conversation_id and not db.one("SELECT id FROM conversations WHERE id=? AND user_id=?", (conversation_id, user["id"])):
            raise HTTPException(404, "会话不存在")
        now, run_id = time.time(), uid()
        try:
            with db.connect() as c:
                c.execute("BEGIN IMMEDIATE")
                if c.execute("SELECT id FROM runs WHERE conversation_id=? AND status IN ('queued','running')", (conversation_id,)).fetchone():
                    raise HTTPException(409, "此会话还有任务正在执行，请等待或取消")
                if not payload.conversation_id:
                    c.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conversation_id, user["id"], payload.question[:45], now))
                c.execute("INSERT INTO messages VALUES(?,?,?,?,?,?)", (uid(), conversation_id, "user", payload.question, run_id, now))
                c.execute("INSERT INTO runs(id,conversation_id,user_id,question,base_ids,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                          (run_id, conversation_id, user["id"], payload.question, encode(ids), cfg.mode, now, now))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "会话状态冲突，请刷新后重试") from None
        return {"run_id": run_id, "conversation_id": conversation_id}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, user=Depends(current_user)):
        run = own_run(user, run_id)
        run.pop("state")
        run["citations"] = json.loads(run["citations"])
        run["events"] = db.all("SELECT id,kind,payload,created_at FROM events WHERE run_id=? ORDER BY id", (run_id,))
        for event in run["events"]:
            event["payload"] = json.loads(event["payload"])
        return run

    @app.get("/api/runs/{run_id}/events")
    def stream_events(run_id: str, request: Request, after: int = 0, user=Depends(current_user)):
        own_run(user, run_id)
        async def events():
            last = after
            for _ in range(600):
                if await request.is_disconnected():
                    break
                rows = db.all("SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, last))
                for row in rows:
                    last = row["id"]
                    yield f"id: {last}\ndata: {encode({'kind':row['kind'], 'payload':json.loads(row['payload'])})}\n\n"
                state = db.one("SELECT status FROM runs WHERE id=?", (run_id,))
                if state["status"] not in {"queued", "running"}:
                    yield f"event: done\ndata: {encode(state)}\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(.5)
        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str, user=Depends(current_user)):
        own_run(user, run_id)
        db.execute("UPDATE runs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=? AND status IN ('queued','running')", (run_id,))
        return {"ok": True}

    @app.post("/api/runs/{run_id}/feedback")
    def feedback(run_id: str, payload: Feedback, user=Depends(current_user)):
        run = own_run(user, run_id)
        if payload.rating not in {-1, 1} or run["status"] not in {"answered", "insufficient", "clarify"}:
            raise HTTPException(400, "请对已完成的回答提交有效评价")
        db.execute("INSERT OR REPLACE INTO feedback VALUES(?,?,?,?,?)", (run_id, user["id"], payload.rating, payload.comment, time.time()))
        return {"ok": True}

    @app.get("/api/feedback")
    def feedback_list(user=Depends(admin)):
        return db.all("""SELECT f.*,r.question,r.answer,u.username FROM feedback f JOIN runs r ON r.id=f.run_id
          JOIN users u ON u.id=f.user_id ORDER BY f.created_at DESC LIMIT 200""")

    web = Path(__file__).parent / "web"
    app.mount("/static", StaticFiles(directory=web, check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    return app


app = create_app()
