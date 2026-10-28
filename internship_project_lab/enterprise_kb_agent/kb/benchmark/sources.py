"""Small, pinned public corpus. Does not download models or use a GPU."""
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

REVISION = "e5c1da8a64a718f5e53faa4d863f0f5383604ed8"
REPO = "https://github.com/progit/progit2-zh"
RAW = f"https://raw.githubusercontent.com/progit/progit2-zh/{REVISION}/"
DATASET = "git_support_zh_v1"

# Source families are assigned BEFORE running retrieval; paraphrases stay together.
GROUPS = {
    "dev": {
        "01-introduction": ["first-time-setup"],
        "02-git-basics": ["aliases", "tagging"],
        "03-git-branching": ["branch-management"],
        "04-git-server": ["generating-ssh-key"],
        "07-git-tools": ["credentials", "searching", "bundling"],
        "08-customizing-git": ["hooks"],
    },
    "test": {
        "01-introduction": ["what-is-git", "about-version-control", "help"],
        "02-git-basics": ["getting-a-repository", "recording-changes", "remotes", "undoing", "viewing-history"],
        "03-git-branching": ["basic-branching-and-merging", "nutshell", "rebasing", "remote-branches", "workflows"],
        "04-git-server": ["protocols", "git-on-a-server", "setting-up-server"],
        "05-distributed-git": ["distributed-workflows"],
        "07-git-tools": ["interactive-staging", "debugging", "revision-selection", "stashing-cleaning", "reset", "rewriting-history", "submodules", "signing"],
        "08-customizing-git": ["attributes", "config"],
    },
    "distractor": {
        "01-introduction": ["history", "installing", "command-line"],
        "04-git-server": ["git-daemon"],
    },
}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def fetch(path):
    request = urllib.request.Request(RAW + path, headers={"User-Agent": "KnowledgeDesk-evaluation-corpus/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("unexpectedly large source")
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(exist_ok=True)
    (root / "corpus").mkdir(exist_ok=True)
    documents = []
    for split, chapters in GROUPS.items():
        for chapter, names in chapters.items():
            for name in names:
                source_path = f"book/{chapter}/sections/{name}.asc"
                raw_path = root / "raw" / (name + ".asc")
                if not raw_path.exists():
                    raw = fetch(source_path)
                    raw_path.write_bytes(raw)
                raw = raw_path.read_bytes()
                text = raw.decode("utf-8-sig").replace("\r\n", "\n")
                # Text-only corpus: retain headings, examples and table text verbatim.
                # Do not silently resolve external includes/images, or rewrite facts.
                text = "\n".join(line for line in text.splitlines()
                                 if not line.startswith(("image::", "include::", "[[", "indexterm:")))
                text = re.sub(r"[ \t]+", " ", text).replace("\x00", "").strip()
                title = next((line.lstrip("= ") for line in text.splitlines() if line.startswith("==")), name)
                target = root / "corpus" / (name + ".txt")
                if target.exists() and target.read_text(encoding="utf-8") != text:
                    raise ValueError("Existing corpus differs; create a new dataset version: " + name)
                target.write_text(text, encoding="utf-8")
                documents.append({"id": name, "title": title, "path": str(target.relative_to(root)),
                    "source_family": name, "question_split": split, "visibility": "all", "kind": "public_document",
                    "source_url": f"{REPO}/blob/{REVISION}/{source_path}", "source_path": source_path,
                    "revision": REVISION, "license": "CC-BY-NC-SA-3.0", "raw_sha256": digest(raw),
                    "sha256": digest(text.encode()), "characters": len(text),
                    "transformation": "removed image/include/anchor/indexterm directive lines; whitespace normalization; no prose rewrite"})
                print(json.dumps({"source": name, "split": split, "characters": len(text)}, ensure_ascii=False), flush=True)
    license_path = root / "raw" / "LICENSE.asc"
    if not license_path.exists():
        license_path.write_bytes(fetch("LICENSE.asc"))
    write_jsonl(root / "documents.jsonl", documents)
    write_json(root / "source_manifest.json", {"dataset": DATASET, "repository": REPO, "revision": REVISION,
        "documents": len(documents), "license_file": "raw/LICENSE.asc",
        "authors": "Scott Chacon, Ben Straub and the Pro Git Chinese translation contributors",
        "scope": "public technical support proxy, not private company policy or an official QA benchmark",
        "corpus_limitations": "text-only, no diagrams; snapshot facts are not a guarantee of current software behaviour",
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return documents
