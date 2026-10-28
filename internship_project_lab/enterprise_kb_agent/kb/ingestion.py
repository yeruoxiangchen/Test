import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from .db import uid, encode
from .retrieval import tokens, embedding_vectors

ALLOWED = {".pdf", ".docx", ".md", ".txt", ".csv"}


def register_document(db, settings, base_id, filename, content):
    filename = filename.replace("\\", "/").split("/")[-1][:180]
    if not filename or Path(filename).suffix.lower() not in ALLOWED:
        raise ValueError("支持 PDF、DOCX、TXT、Markdown 和 CSV")
    if not content or len(content) > settings.max_upload_bytes:
        raise ValueError("文件为空或超过 20 MiB")
    digest = hashlib.sha256(content).hexdigest()
    duplicate = db.one("SELECT * FROM documents WHERE base_id=? AND digest=?", (base_id, digest))
    if duplicate:
        return duplicate, True
    document_id = uid()
    path = settings.root / "uploads" / document_id
    path.write_bytes(content)
    try:
        db.execute("INSERT INTO documents(id,base_id,filename,digest,size,status,created_at) VALUES(?,?,?,?,?,'queued',?)",
                   (document_id, base_id, filename, digest, len(content), time.time()))
    except Exception:
        path.unlink(missing_ok=True)
        # Concurrent duplicate uploads return the already registered document.
        duplicate = db.one("SELECT * FROM documents WHERE base_id=? AND digest=?", (base_id, digest))
        if duplicate:
            return duplicate, True
        raise
    return db.one("SELECT * FROM documents WHERE id=?", (document_id,)), False


def ingest(db, settings, document_id):
    if not db.execute("UPDATE documents SET status='processing',error=NULL WHERE id=? AND status='queued'", (document_id,)):
        return
    doc = db.one("SELECT * FROM documents WHERE id=?", (document_id,))
    if not doc:
        return
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "kb.extract", str(settings.root / "uploads" / document_id),
                                 Path(doc["filename"]).suffix.lower()], capture_output=True, text=True, timeout=45)
        try:
            output = json.loads(result.stdout)
        except ValueError:
            raise ValueError("文档解析失败或超过资源限制，请检查文件") from None
        if result.returncode:
            raise ValueError(output.get("error", "解析失败"))
        chunks = output["chunks"]
        if not chunks:
            raise ValueError("文档没有可索引文本")
        vectors, warning = [], None
        if settings.embedding_url:
            try:
                for start in range(0, len(chunks), 24):
                    vectors.extend(embedding_vectors(settings, [c["text"] for c in chunks[start:start+24]]))
            except Exception:
                vectors = []
                warning = "向量索引失败；关键词索引可用。配置好向量服务后可重新索引。"
        with db.connect() as c:
            if not c.execute("SELECT id FROM documents WHERE id=?", (document_id,)).fetchone():
                return
            c.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            for i, chunk in enumerate(chunks):
                chunk_id = c.execute("INSERT INTO chunks(document_id,ordinal,page,text) VALUES(?,?,?,?)",
                                     (document_id, i, chunk["page"], chunk["text"])).lastrowid
                c.execute("INSERT INTO chunks_fts(rowid,tokens) VALUES(?,?)", (chunk_id, " ".join(tokens(doc["filename"] + " " + chunk["text"]))))
                if vectors:
                    c.execute("INSERT INTO vectors VALUES(?,?,?)", (chunk_id, settings.embedding_profile, encode(vectors[i])))
            c.execute("UPDATE documents SET status='ready',error=?,chunk_count=? WHERE id=?", (warning, len(chunks), document_id))
    except Exception as exc:
        message = "解析超时，请拆分文件后重试" if isinstance(exc, subprocess.TimeoutExpired) else str(exc)[:500]
        db.execute("UPDATE documents SET status='error',error=? WHERE id=?", (message, document_id))
