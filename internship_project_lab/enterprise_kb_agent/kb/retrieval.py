import json
import math
import re
import httpx

STOP = set("什么 哪些 如何 可以 我们 你们 多少 有关 请问 怎么 需要 是否 这个 那个 一下 以及 的是 了吗 能否".split())
EN_STOP = set("the a an is are of to for in on and how what which can do does please".split())


def tokens(text):
    result = []
    for word in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            result.extend(word[i:i + 2] for i in range(len(word) - 1) if word[i:i + 2] not in STOP)
            if len(word) == 1:
                result.append(word)
        elif word not in EN_STOP:
            result.append(word)
    return result


def embedding_vectors(settings, texts):
    headers = {"Authorization": f"Bearer {settings.embedding_key}"} if settings.embedding_key else {}
    with httpx.Client(timeout=settings.timeout, trust_env=False) as client:
        response = client.post(settings.embedding_url.rstrip("/") + "/embeddings", headers=headers,
                               json={"model": settings.embedding_model, "input": texts})
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda x: x["index"])
    if len(data) != len(texts) or [r["index"] for r in data] != list(range(len(texts))):
        raise ValueError("向量服务返回数量或索引不匹配")
    vectors = [row["embedding"] for row in data]
    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1 or not next(iter(dimensions)) or any(not math.isfinite(x) for v in vectors for x in v):
        raise ValueError("向量服务返回非法向量")
    return vectors


class Retriever:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    def documents(self, user, base_ids):
        ids = self.db.check_bases(user, base_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        return self.db.all(f"SELECT id,filename,status,chunk_count,base_id FROM documents WHERE base_id IN ({placeholders}) ORDER BY created_at DESC LIMIT 100", ids)

    def chunk(self, user, chunk_id, base_ids):
        ids = self.db.check_bases(user, base_ids)
        row = self.db.one("""SELECT c.*,d.filename,d.base_id,b.name AS base_name FROM chunks c
          JOIN documents d ON d.id=c.document_id JOIN bases b ON b.id=d.base_id
          WHERE c.id=? AND d.status='ready'""", (chunk_id,))
        if not row or row["base_id"] not in ids:
            raise PermissionError("片段不存在或不在当前可访问的知识库中")
        return row

    def search(self, user, query, base_ids, limit=5):
        self.last_rerank = None
        ids = self.db.check_bases(user, base_ids)
        terms = list(dict.fromkeys(tokens(query)))[:80]
        if not ids or not terms:
            return [], []
        placeholders = ",".join("?" for _ in ids)
        match = " OR ".join('"' + t + '"' for t in terms)
        rows = self.db.all(f"""SELECT c.*,d.filename,d.base_id,b.name AS base_name,bm25(chunks_fts) AS lexical_score
          FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
          JOIN documents d ON d.id=c.document_id JOIN bases b ON b.id=d.base_id
          WHERE chunks_fts MATCH ? AND d.status='ready' AND d.base_id IN ({placeholders})
          ORDER BY lexical_score LIMIT 40""", [match, *ids])
        rows = [r for r in rows if len(set(tokens(r["text"] + " " + r["filename"])) & set(terms)) / len(terms) >= .12]
        ranked = {r["id"]: {**r, "score": 1 / (60 + i + 1)} for i, r in enumerate(rows)}
        warnings = []
        if self.settings.embedding_url:
            try:
                query_vector = embedding_vectors(self.settings, [query])[0]
                stored = self.db.all(f"""SELECT c.*,d.filename,d.base_id,b.name AS base_name,v.vector FROM vectors v
                   JOIN chunks c ON c.id=v.chunk_id JOIN documents d ON d.id=c.document_id
                   JOIN bases b ON b.id=d.base_id WHERE v.profile=? AND d.status='ready'
                   AND d.base_id IN ({placeholders})""", [self.settings.embedding_profile, *ids])
                dense = []
                qnorm = math.sqrt(sum(x*x for x in query_vector))
                for row in stored:
                    vector = json.loads(row.pop("vector"))
                    if len(vector) != len(query_vector):
                        continue
                    denom = qnorm * math.sqrt(sum(x*x for x in vector))
                    similarity = sum(a*b for a,b in zip(vector, query_vector)) / denom if denom else 0
                    if similarity >= .25:
                        dense.append((similarity, row))
                for i, (_, row) in enumerate(sorted(dense, key=lambda x: x[0], reverse=True)[:40]):
                    previous = ranked.get(row["id"], {**row, "score": 0})
                    previous["score"] += 1 / (60 + i + 1)
                    ranked[row["id"]] = previous
                if not stored:
                    warnings.append("当前向量模型没有已建立的索引，使用关键词检索；可在文档页重新索引")
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                warnings.append("向量检索暂不可用，本次使用关键词检索")
        if warnings and self.settings.strict_embedding:
            raise RuntimeError("正式混合检索评测禁止静默降级：" + "; ".join(warnings))
        result = sorted(ranked.values(), key=lambda x: (-x["score"], x["id"]))
        if self.settings.reranker_path and result:
            from .reranking import rerank
            result, self.last_rerank = rerank(self.settings, query, result[:self.settings.reranker_candidates])
        result = result[:limit]
        return result, warnings
