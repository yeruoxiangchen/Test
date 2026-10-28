import pytest
from fastapi.testclient import TestClient
from kb.app import create_app
from kb.auth import create_user
from kb.config import Settings
from kb.db import Database


@pytest.fixture
def system(tmp_path):
    cfg = Settings(root=tmp_path, mode="extractive", embedding_url="", embedding_model="", tokenize_url="")
    cfg.prepare()
    db = Database(cfg.db_path)
    db.initialize()
    admin_id = create_user(db, "admin", "test-admin-123", "admin")
    member_id = create_user(db, "member", "test-member-123", "member")
    second_id = create_user(db, "second", "test-second-123", "member")
    app = create_app(cfg, start_workers=False)
    with TestClient(app, headers={"X-KB-Client": "web"}) as client:
        client.post("/api/login", json={"username":"admin", "password":"test-admin-123"})
        public = client.post("/api/bases", json={"name":"公开资料"}).json()["id"]
        private = client.post("/api/bases", json={"name":"内部财务", "visibility":"admin"}).json()["id"]
        yield cfg, db, client, public, private, {"id":admin_id,"role":"admin","username":"admin"}, {"id":member_id,"role":"member","username":"member"}
