"""Optional localhost-only, single-GPU Transformers smoke endpoint. Not a production serving engine.

Uses the official Qwen3-VL Transformers message interface. Its weights are read-only inputs.
Run in a separate model environment; the main Agent environment does not depend on PyTorch.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_tools(text: str) -> tuple[str, list[dict]]:
    calls = []
    for payload in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.S):
        try:
            call = json.loads(payload)
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict) or not isinstance(call.get("name"), str):
                continue
            calls.append({"id": "call_"+uuid.uuid4().hex[:12], "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}})
        except (ValueError, TypeError):
            continue
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip() if calls else text
    return content, calls


def generation_finish_reason(token_ids, output_limit, eos_token_ids, has_tools):
    """EOS wins even when it occurs at the limit; partial tool output is never executed."""
    eos = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
    ended = bool(token_ids) and token_ids[-1] in eos
    if len(token_ids) >= output_limit and not ended:
        return "length"
    return "tool_calls" if has_tools else "stop"


def normalize_messages(messages: list) -> list:
    from PIL import Image
    normalized = []
    for message in messages:
        row = dict(message)
        row["content"] = row.get("content") or ""
        if row.get("tool_calls"):
            row["tool_calls"] = json.loads(json.dumps(row["tool_calls"]))
            for call in row["tool_calls"]:
                if isinstance(call["function"]["arguments"], str):
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        if isinstance(row["content"], list):
            blocks = []
            for block in row["content"]:
                if block["type"] == "text":
                    blocks.append(block)
                elif block["type"] == "image_url":
                    url = block["image_url"]["url"]
                    if not url.startswith("data:image/") or len(url) > 20_000_000:
                        raise ValueError("Local smoke server accepts bounded image data URLs only")
                    payload = base64.b64decode(url.split(",", 1)[1], validate=True)
                    image = Image.open(io.BytesIO(payload)).convert("RGB")
                    if image.width*image.height > 16_000_000:
                        raise ValueError("Image exceeds local smoke pixel limit")
                    blocks.append({"type": "image", "image": image})
                else:
                    raise ValueError("Unsupported message block")
            row["content"] = blocks
        elif isinstance(row["content"], str):
            # Qwen3-VL's Processor scans every message for image blocks before
            # rendering its template, including system and tool messages.
            row["content"] = [{"type": "text", "text": row["content"]}]
        normalized.append(row)
    return normalized


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-output", type=int, default=1024)
    args = p.parse_args()
    key = os.environ.get("SCA_LOCAL_KEY")
    if not key:
        raise ValueError("Set SCA_LOCAL_KEY for the localhost-only smoke endpoint")
    print(json.dumps({"stage": "importing_model_runtime", "time": time.time()}), flush=True)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(4)
    torch.manual_seed(0)
    print(json.dumps({"stage": "loading_model_weights", "time": time.time(),
                      "model_path": args.model_path,
                      "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    print(json.dumps({"stage": "loading_processor", "time": time.time()}), flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model_lock = threading.Lock()
    model_id = "local-qwen3-vl-8b"

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.send_json(200, {"ready": True, "model": model_id,
                                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()})

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer "+key:
                self.send_json(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0 or length > 24_000_000:
                    raise ValueError("Request exceeds local smoke size limit")
                request = json.loads(self.rfile.read(length))
                messages = normalize_messages(request["messages"])
                tools = request.get("tools")
                requested = request.get("max_tokens", 1024)
                with model_lock, torch.inference_mode():
                    inputs = processor.apply_chat_template(messages, tools=tools, tokenize=True,
                        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
                    prompt_tokens = inputs["input_ids"].shape[-1]
                    if prompt_tokens >= args.max_context:
                        raise ValueError("Local smoke context limit reached")
                    output_limit = min(requested, args.max_output, args.max_context-prompt_tokens)
                    ids = model.generate(**inputs, max_new_tokens=output_limit, do_sample=False)
                    generated = ids[:, prompt_tokens:]
                    text = processor.batch_decode(generated, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]
                    completion_tokens = generated.shape[-1]
                    generated_ids = generated[0].tolist()
                content, calls = parse_tools(text)
                finish_reason = generation_finish_reason(generated_ids, output_limit,
                                                        model.generation_config.eos_token_id, bool(calls))
                if finish_reason == "length":
                    content, calls = text, []
                message = {"role": "assistant", "content": content or None}
                if calls:
                    message["tool_calls"] = calls
                self.send_json(200, {"id": "local_"+uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": model_id,
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                              "total_tokens": prompt_tokens+completion_tokens}})
            except Exception as error:
                traceback.print_exc()
                self.send_json(400, {"error": {"message": str(error), "type": "local_smoke_error"}})

        def log_message(self, fmt, *values):
            print(time.strftime("%H:%M:%S"), fmt % values, flush=True)

    print(json.dumps({"ready": True, "port": args.port, "model": model_id,
                      "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "model_path": args.model_path}), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
