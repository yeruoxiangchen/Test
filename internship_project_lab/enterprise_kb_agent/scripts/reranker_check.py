"""Real CPU load/protocol sanity check before consuming an evaluation split."""
import json
from dataclasses import replace
from kb.config import Settings
from kb.reranking import rerank
from kb.benchmark.sources import write_json

cfg = Settings()
cfg = replace(cfg, reranker_path=str(cfg.root / "models/bge-reranker-base"))
cfg.prepare()
rows = [{"id": 1, "filename": "Git 别名", "text": "使用 git config --global alias.co checkout 配置 co 为 checkout 的别名。"},
        {"id": 2, "filename": "天气", "text": "今天晴天，明天可能下雨。"}]
ranked, receipt = rerank(cfg, "怎样配置 Git 的 co 别名？", rows)
if ranked[0]["id"] != 1:
    raise RuntimeError("Reranker sanity check did not rank the clearly relevant document first")
write_json(cfg.root / "models/bge-reranker-base/smoke.json", receipt)
print(json.dumps(receipt, ensure_ascii=False))
