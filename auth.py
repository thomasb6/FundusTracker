import os
import sqlite3
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "fundus_users.db"


class User(UserMixin):
    def __init__(self, id, username, is_admin=False):
        self.id = str(id)
        self.username = username
        self.is_admin = bool(is_admin)


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS shares (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id      INTEGER NOT NULL,
                patient_key   TEXT    NOT NULL,
                shared_with_id INTEGER NOT NULL,
                UNIQUE(owner_id, patient_key, shared_with_id)
            );
        """)
        # Create default admin on first run
        if not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,1)",
                ("admin", generate_password_hash("admin123")),
            )
        conn.commit()


def get_user_by_id(user_id):
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return User(row["id"], row["username"], row["is_admin"]) if row else None


def get_user_by_username(username):
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return User(row["id"], row["username"], row["is_admin"]) if row else None


def verify_user(username, password):
    with _get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return User(row["id"], row["username"], row["is_admin"])
    return None


def create_user(username, password, is_admin=False):
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,?)",
                (username, generate_password_hash(password), int(is_admin)),
            )
            conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def delete_user(user_id):
    with _get_db() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()


def update_password(user_id, new_password):
    with _get_db() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (generate_password_hash(new_password), user_id),
        )
        conn.commit()


def get_all_users():
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, is_admin, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def get_userdata_dir(user_id):
    path = os.path.join("userdata", str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def load_patient_data(user_id):
    path = os.path.join(get_userdata_dir(user_id), "patients.json")
    if os.path.exists(path):
        with open(path) as f:
            return json_load(f)
    return {}


def save_patient_data(user_id, data):
    path = os.path.join(get_userdata_dir(user_id), "patients.json")
    with open(path, "w") as f:
        json_dump(data, f)


# Import json lazily to avoid circular imports
import json as _json
json_load = _json.load
json_dump = _json.dump
