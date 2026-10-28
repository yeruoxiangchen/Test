import argparse
from dataclasses import replace
import getpass
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import httpx
from .auth import create_user
from .config import Settings
from .db import Database, uid, encode
from .demo import QUESTIONS, seed
from .retrieval import Retriever
from .agent import Harness


def write_private(path, value):
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value)
    path.chmod(0o600)


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def server_state(cfg):
    path = cfg.root / "server.json"
    if not path.exists():
        return {}, False
    state = json.loads(path.read_text())
    alive = process_identity(state["pid"]) == state["start_ticks"]
    return state, alive


def start(cfg):
    state, alive = server_state(cfg)
    if alive:
        print(encode({"status": "already_running", **state}))
        return
    project = Path(__file__).resolve().parent.parent
    host, port = os.getenv("KB_HOST", "127.0.0.1"), int(os.getenv("KB_PORT", "8768"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", KB_DATA_DIR=str(cfg.root))
    with (cfg.root / "logs/server.log").open("a") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "kb.app:app", "--app-dir", str(project),
                                    "--host", host, "--port", str(port), "--workers", "1"],
                                   cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state = {"pid": process.pid, "start_ticks": process_identity(process.pid), "host": host, "port": port,
             "url": f"http://127.0.0.1:{port}", "started_at": time.time()}
    for _ in range(60):
        if process.poll() is not None:
            raise SystemExit("启动失败，请查看 " + str(cfg.root / "logs/server.log"))
        try:
            response = httpx.get(state["url"] + "/api/health", timeout=1, trust_env=False)
            if response.status_code == 200 and response.json().get("service") == "enterprise-kb-agent" and response.json().get("pid") == process.pid:
                # Only save ownership after our process survives startup.
                time.sleep(.2)
                if process.poll() is None:
                    write_private(cfg.root / "server.json", encode(state))
                    print(encode({"status": "running", **state}))
                    return
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(.2)
    os.killpg(process.pid, signal.SIGTERM)
    raise SystemExit("启动未就绪，已停止本次进程，请查看日志")


def stop(cfg):
    state, alive = server_state(cfg)
    if not alive:
        print("项目服务未运行")
        return
    command = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
    if b"kb.app:app" not in command or str(Path(__file__).resolve().parent.parent).encode() not in command:
        raise SystemExit("进程身份不匹配，未发送停止信号")
    os.killpg(state["pid"], signal.SIGTERM)
    for _ in range(60):
        if process_identity(state["pid"]) != state["start_ticks"]:
            print("项目服务已停止")
            return
        time.sleep(.2)
    print("已请求停止；等待当前工具调用退出，请稍后查看 status")


def evaluate(cfg, use_agent=False):
    # Isolated data prevents changing the live application's documents or conversations.
    target = cfg.root / "artifacts" / ("evaluation-" + time.strftime("%Y%m%d-%H%M%S"))
    evaluation = replace(cfg, root=target, mode="agent" if use_agent else "extractive")
    if not use_agent:
        evaluation.embedding_url = evaluation.embedding_model = evaluation.embedding_key = ""
    evaluation.prepare()
    db = Database(evaluation.db_path)
    db.initialize()
    user_id = create_user(db, "eval", secrets.token_urlsafe(24))
    user = {"id": user_id, "username": "eval", "role": "member"}
    base_id, _ = seed(db, evaluation)
    results = []
    for question, expected_file, expected_text in QUESTIONS:
        matches, _ = Retriever(db, evaluation).search(user, question, [base_id], limit=3)
        now, conv, run = time.time(), uid(), uid()
        db.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conv, user_id, question, now))
        db.execute("INSERT INTO runs(id,conversation_id,user_id,question,base_ids,mode,status,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                   (run, conv, user_id, question, encode([base_id]), evaluation.mode, now, now))
        Harness(db, evaluation).execute(run)
        row = db.one("SELECT * FROM runs WHERE id=?", (run,))
        results.append({"question": question, "retrieval_hit_at_1": bool(matches) and expected_file in matches[0]["filename"],
                        "retrieval_hit_at_3": any(expected_file in m["filename"] for m in matches),
                        "status": row["status"], "contains_expected_text": expected_text in (row["answer"] or ""),
                        "citation_count": len(json.loads(row["citations"])), "error": row["error"]})
        print(encode(results[-1]), flush=True)
    summary = {"scope": "authored_demo_smoke_cases_not_business_benchmark", "mode": evaluation.mode,
               "questions": len(results), "retrieval_hits_at_1": sum(r["retrieval_hit_at_1"] for r in results),
               "retrieval_hits_at_3": sum(r["retrieval_hit_at_3"] for r in results),
               "completed": sum(r["status"] == "answered" for r in results),
               "contains_expected_text": sum(r["contains_expected_text"] for r in results), "results": results}
    (target / "report.json").write_text(encode(summary))
    print("报告：" + str(target / "report.json"))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "demo", "user", "doctor", "start", "stop", "status", "evaluate"])
    parser.add_argument("--username")
    parser.add_argument("--role", default="member", choices=["admin", "member"])
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    cfg = Settings()
    cfg.prepare()
    db = Database(cfg.db_path)
    db.initialize()
    if args.action == "init":
        if db.one("SELECT id FROM users WHERE role='admin' LIMIT 1"):
            print("管理员已存在，未更改账号和密码。")
        else:
            password = secrets.token_urlsafe(18)
            create_user(db, "admin", password, "admin")
            credentials = cfg.root / "credentials.txt"
            write_private(credentials, "账号：admin\n密码：" + password + "\n")
            print("管理员已创建；账号密码保存在 " + str(credentials) + "（仅当前用户可读）")
    elif args.action == "user":
        username = args.username or input("账号：").strip()
        password = getpass.getpass("密码（至少 10 字符）：")
        create_user(db, username, password, args.role)
        print("账号已创建")
    elif args.action == "demo":
        base_id, ids = seed(db, cfg)
        print(encode({"base_id": base_id, "documents": len(ids), "note": "虚构演示资料，非真实企业数据"}))
    elif args.action == "evaluate":
        result = evaluate(cfg, args.agent)
        if result["completed"] != result["questions"] or result["retrieval_hits_at_3"] != result["questions"]:
            raise SystemExit(1)
    elif args.action == "doctor":
        output = {"root": str(cfg.root), "mode": cfg.mode, "model": cfg.llm_model or None,
                  "retrieval": "hybrid" if cfg.embedding_url else "keyword", "fts5": True,
                  "documents": db.one("SELECT count(*) AS n FROM documents")["n"]}
        if args.live and cfg.llm_url:
            headers = {"Authorization": f"Bearer {cfg.llm_key}"} if cfg.llm_key else {}
            try:
                response = httpx.get(cfg.llm_url.rstrip("/") + "/models", headers=headers, timeout=10, trust_env=False)
                output["model_endpoint_status"] = response.status_code
                output["note"] = "仅检查接口连通性；工具调用质量需 evaluate --agent 验证"
            except httpx.HTTPError:
                output["model_endpoint_status"] = "unreachable"
        print(json.dumps(output, ensure_ascii=False, indent=2))
    elif args.action == "start":
        start(cfg)
    elif args.action == "stop":
        stop(cfg)
    elif args.action == "status":
        state, alive = server_state(cfg)
        print(json.dumps({"alive": alive, **state,
                          "documents": db.all("SELECT status,count(*) AS count FROM documents GROUP BY status"),
                          "runs": db.all("SELECT status,count(*) AS count FROM runs GROUP BY status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
