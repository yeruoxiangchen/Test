"""Strict CPU cross-encoder bridge. No silent fallback or GPU allocation."""
import atexit
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time

_workers = {}


class Worker:
    def __init__(self, settings):
        self.log = (settings.root / "logs/reranker.log").open("a")
        script = Path(__file__).resolve().parents[1] / "scripts/reranker_worker.py"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        self.process = subprocess.Popen([settings.reranker_python, "-B", str(script), settings.reranker_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        atexit.register(self.close)
        if not self.receive().get("ready"):
            raise RuntimeError("CPU reranker failed to start")

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def receive(self):
        if not select.select([self.process.stdout], [], [], 180)[0]:
            self.close()
            raise RuntimeError("CPU reranker timed out; no BM25 fallback")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("CPU reranker exited; inspect workspace/logs/reranker.log")
        return json.loads(line)

    def score(self, query, documents):
        self.process.stdin.write(json.dumps({"query": query, "documents": documents}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        output = self.receive()
        scores = output.get("scores")
        if not isinstance(scores, list) or len(scores) != len(documents) or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
            raise RuntimeError("Invalid CPU reranker output: " + str(output)[:300])
        return scores


def rerank(settings, query, rows):
    start = time.monotonic()
    key = (settings.reranker_python, settings.reranker_path)
    if key not in _workers:
        _workers[key] = Worker(settings)
    scores = _workers[key].score(query, [r["filename"] + "\n" + r["text"] for r in rows])
    result = sorted([{**r, "rerank_score": s} for r, s in zip(rows, scores)], key=lambda r: (-r["rerank_score"], r["id"]))
    return result, {"device": "cpu", "candidates": len(rows), "seconds": time.monotonic() - start,
                    "ranking": [{"chunk_id": r["id"], "score": r["rerank_score"]} for r in result]}
