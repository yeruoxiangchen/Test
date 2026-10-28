"""Pin and download a CPU reranker; never changes the shared inference environment."""
import json
import os
import fcntl
import re
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path(os.environ["KB_DATA_DIR"]) / "models/bge-reranker-base"
root.mkdir(parents=True, exist_ok=True)
lock = (root / "setup.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX)
receipt = root / "receipt.json"
if receipt.exists():
    print(receipt.read_text())
else:
    model = "BAAI/bge-reranker-base"
    pin = root / "download_revision.json"
    if pin.exists():
        revision = json.loads(pin.read_text())["revision"]
    else:
        metadata = root / ".cache/huggingface/download/config.json.metadata"
        prior = metadata.read_text().splitlines()[0] if metadata.exists() else ""
        revision = prior if re.fullmatch(r"[0-9a-f]{40}", prior) else HfApi().model_info(model).sha
        pin.write_text(json.dumps({"model": model, "revision": revision}) + "\n")
    for attempt in range(12):
        try:
            path = snapshot_download(model, revision=revision, local_dir=str(root),
                allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                "special_tokens_map.json", "sentencepiece.bpe.model"], max_workers=2)
            break
        except Exception as exc:
            if attempt == 11:
                raise
            # HF resumes its partial file under a file lock; never delete it.
            print(json.dumps({"download_attempt": attempt + 1, "error_type": type(exc).__name__, "next": "resume_same_revision"}), flush=True)
            time.sleep(min(30, 5 * (attempt + 1)))
    if not (root / "model.safetensors").is_file():
        raise RuntimeError("Pinned checkpoint does not contain safetensors")
    receipt.write_text(json.dumps({"model": model, "revision": revision, "path": path,
        "device": "cpu", "dtype": "float32", "max_length": 512, "batch_size": 4,
        "note": "Pair tokenization truncates documents for ranking only; answer evidence stays whole."}, indent=2) + "\n")
    print(receipt.read_text())
