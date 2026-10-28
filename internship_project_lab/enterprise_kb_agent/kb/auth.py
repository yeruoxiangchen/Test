import hashlib
import hmac
import secrets
import time
from .db import uid


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 210_000).hex()
    return f"pbkdf2_sha256${salt}${derived}"


def verify_password(password, stored):
    try:
        salt = stored.split("$")[1]
        return hmac.compare_digest(password_hash(password, salt), stored)
    except (ValueError, IndexError):
        return False


def create_user(db, username, password, role="member"):
    if not 2 <= len(username) <= 64 or len(password) < 10 or role not in {"admin", "member"}:
        raise ValueError("账号需 2–64 字符，密码至少 10 字符，角色必须是 admin 或 member")
    user_id = uid()
    db.execute("INSERT INTO users VALUES(?,?,?,?,?)", (user_id, username, password_hash(password), role, time.time()))
    return user_id


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db, user_id):
    token = secrets.token_urlsafe(32)
    db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
    db.execute("INSERT INTO sessions VALUES(?,?,?)", (token_hash(token), user_id, time.time() + 8 * 3600))
    return token


def session_user(db, token):
    if not token:
        return None
    return db.one("SELECT u.id,u.username,u.role FROM users u JOIN sessions s ON s.user_id=u.id WHERE s.token_hash=? AND s.expires>?",
                  (token_hash(token), time.time()))
