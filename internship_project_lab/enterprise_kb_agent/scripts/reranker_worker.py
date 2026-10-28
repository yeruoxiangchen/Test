"""One CPU model per benchmark process, JSONL stdin/stdout, no web service."""
import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
model = AutoModelForSequenceClassification.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False).float().cpu().eval()
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        data = json.loads(line)
        pairs = [[data["query"], d] for d in data["documents"]]
        scores = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 4):
                inputs = tokenizer(pairs[start:start+4], padding=True, truncation=True, max_length=512, return_tensors="pt")
                scores.extend(model(**inputs).logits.view(-1).float().tolist())
        print(json.dumps({"scores": scores}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)[:300]}), flush=True)
