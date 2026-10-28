"""Bounded document extraction; invoked in an isolated subprocess by ingestion."""
import json
import re
import sys
import zipfile
from pathlib import Path

MAX_TEXT = 1_000_000
MAX_PAGES = 300


def extract(path, suffix):
    if suffix in {".txt", ".md", ".csv"}:
        raw = Path(path).read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("gb18030")
        parts = [{"page": None, "text": text}]
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("不支持加密 PDF，请先解密后上传")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("PDF 超过 300 页，请拆分后上传")
        parts = []
        size = 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            size += len(text)
            if size > MAX_TEXT:
                raise ValueError("文档提取文本超过 100 万字符")
            parts.append({"page": i + 1, "text": text})
    elif suffix == ".docx":
        from docx import Document
        with zipfile.ZipFile(path) as z:
            if sum(i.file_size for i in z.infolist()) > 80_000_000:
                raise ValueError("DOCX 解压内容过大")
        doc = Document(path)
        blocks = []
        for block in doc.iter_inner_content():
            if hasattr(block, "rows"):
                blocks.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
            else:
                blocks.append(block.text)
        parts = [{"page": None, "text": "\n\n".join(blocks)}]
    else:
        raise ValueError("支持 PDF、DOCX、TXT、Markdown 和 CSV")
    if sum(len(p["text"]) for p in parts) > MAX_TEXT:
        raise ValueError("文档提取文本超过 100 万字符")
    if not any(p["text"].strip() for p in parts):
        raise ValueError("未提取到文字；扫描 PDF 需要先进行 OCR，当前版本不包含 OCR")
    return parts


def chunk_parts(parts, size=900, overlap=120):
    chunks = []
    for part in parts:
        text = re.sub(r"[ \t]+", " ", part["text"]).replace("\x00", "").strip()
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start + size // 2, end), text.rfind("。", start + size // 2, end))
                if boundary > start:
                    end = boundary + 1
            value = text[start:end].strip()
            if value:
                chunks.append({"page": part["page"], "text": value})
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


if __name__ == "__main__":
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    try:
        print(json.dumps({"chunks": chunk_parts(extract(sys.argv[1], sys.argv[2]))}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
