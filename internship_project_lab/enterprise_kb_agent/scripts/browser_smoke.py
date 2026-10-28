"""Browser E2E against a separate app/database; never changes the live workspace."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
from playwright.sync_api import sync_playwright, expect
from kb.auth import create_user
from kb.config import Settings
from kb.db import Database, uid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['extractive','agent'],default='extractive')
    parser.add_argument('--policy',default='guarded_v4')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    font_config=Path(os.environ["KB_DATA_DIR"])/"cache/fonts/fonts.conf"
    if font_config.exists():
        os.environ['FONTCONFIG_FILE']=str(font_config)
    root=args.output or Path(os.environ["KB_DATA_DIR"])/"artifacts"/("browser-"+time.strftime("%Y%m%d-%H%M%S"))
    if (root/'report.json').exists():
        print((root/'report.json').read_text());return
    cfg=Settings(root=root,mode=args.mode,agent_policy=args.policy,embedding_url="",embedding_model="")
    cfg.prepare();db=Database(cfg.db_path);db.initialize()
    # Ephemeral test-only credential, generated and never printed.
    password=uid()
    create_user(db,"browser-admin",password,"admin")
    with socket.socket() as s:
        s.bind(("127.0.0.1",0));port=s.getsockname()[1]
    env=dict(os.environ,KB_DATA_DIR=str(root),KB_MODE=args.mode,KB_AGENT_POLICY=args.policy,KB_EMBED_BASE_URL="",KB_EMBED_MODEL="")
    with (root/"server.log").open("w") as log:
        server=subprocess.Popen([sys.executable,"-m","uvicorn","kb.app:app","--host","127.0.0.1","--port",str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
    errors=[]
    try:
        for _ in range(80):
            if server.poll() is not None:
                raise RuntimeError("E2E server exited; see server.log")
            try:
                if httpx.get(f"http://127.0.0.1:{port}/api/health",timeout=1,trust_env=False).status_code==200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.2)
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,args=["--no-sandbox"])
            page=browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            page.locator('#login-form input[name="username"]').fill("browser-admin")
            page.locator('#login-form input[name="password"]').fill(password)
            page.locator('#login-form button').click()
            expect(page.locator('#workspace')).to_be_visible()
            page.get_by_role('button',name='▤ 文档资料').click()
            page.locator('#new-base').click()
            page.locator('#base-form input[name="name"]').fill('浏览器验证知识库')
            page.locator('#base-form textarea').fill('自动化测试资料')
            page.locator('#base-form button').click()
            expect(page.locator('#document-base')).not_to_have_value('')
            page.locator('#file-input').set_input_files({"name":"住宿标准.md","mimeType":"text/markdown","buffer":"# 住宿报销标准\n出差住宿标准为一线城市每晚 600 元，其他城市每晚 400 元。".encode()})
            expect(page.locator('#documents')).to_contain_text('已就绪',timeout=20000)
            page.screenshot(path=str(root/'documents.png'),full_page=True)
            page.locator('#new-chat').click()
            page.locator('#question').fill('出差住宿报销标准是多少？')
            page.locator('#send').click()
            expect(page.locator('.answer-body')).to_contain_text('600',timeout=120000)
            expect(page.locator('.source-card')).to_have_count(1)
            page.locator('#toast').evaluate("node => node.classList.add('hidden')")
            page.wait_for_timeout(300)
            page.screenshot(path=str(root/'chat.png'),full_page=True)
            page.locator('.source-card').click()
            expect(page.locator('#source-dialog')).to_be_visible()
            expect(page.locator('#source-content')).to_contain_text('400')
            page.locator('[data-close="source-dialog"]').click()
            page.locator('[data-feedback]').click()
            page.locator('#feedback-form textarea').fill('浏览器流程验证通过')
            page.locator('#feedback-form button').click()
            page.get_by_role('button',name='◇ 用户反馈').click()
            expect(page.locator('#feedback-list')).to_contain_text('浏览器流程验证通过')
            page.reload()
            expect(page.locator('#workspace')).to_be_visible()
            page.locator('[data-conversation]').first.click()
            expect(page.locator('.answer-body')).to_contain_text('600')
            page.set_viewport_size({"width":390,"height":844})
            page.screenshot(path=str(root/'mobile.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), 'mobile overflow'
            browser.close()
        assert not errors,errors
        report={"status":"passed","mode":args.mode,"policy":args.policy,"isolated_workspace":True,"checks":["login","create_knowledge_base","upload_and_ingest","ask_and_answer","citations","source_preview","feedback","session_reload","history","mobile_layout"],"page_errors":errors}
        (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({"status":"passed","report":str(root/'report.json')},ensure_ascii=False))
    finally:
        server.terminate()
        try:server.wait(timeout=15)
        except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':main()
