import base64
import io as _io
import json as _json
import os
import re as _re
import secrets as _secrets
import sqlite3
import uuid as _uuid
from datetime import datetime as _datetime

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from PIL import Image as _PILImage

# All persistent state lives under DATA_DIR so a single mounted volume
# (-v host_dir:/app/userdata) survives container rebuilds/redeploys.
DATA_DIR = os.environ.get("FUNDUS_DATA_DIR", "userdata")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "fundus_users.db")

# ── Helpers ───────────────────────────────────────────────────────────────────
json_load = _json.load
json_dump = _json.dump


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── User model ─────────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username, is_admin=False):
        self.id = str(id)
        self.username = username
        self.is_admin = bool(is_admin)


# ── DB init ────────────────────────────────────────────────────────────────────
def init_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                is_admin      INTEGER DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS shares (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id       INTEGER NOT NULL,
                patient_key    TEXT    NOT NULL,
                shared_with_id INTEGER NOT NULL,
                UNIQUE(owner_id, patient_key, shared_with_id)
            );
        """)
        if not conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            # Pas de mot de passe par défaut codé en dur (repo public) :
            # FUNDUS_ADMIN_PASSWORD si fourni, sinon généré et affiché une fois.
            password = os.environ.get("FUNDUS_ADMIN_PASSWORD")
            if not password:
                password = _secrets.token_urlsafe(12)
                print(
                    f"[FundusTracker] Compte admin initial créé — "
                    f"identifiant : admin / mot de passe : {password}",
                    flush=True,
                )
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?,?,1)",
                ("admin", generate_password_hash(password)),
            )
        conn.commit()


# ── User CRUD ──────────────────────────────────────────────────────────────────
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


# Politique de mot de passe (comptes multi-utilisateurs). Volontairement
# simple et explicite : longueur minimale + au moins une lettre et un chiffre.
PASSWORD_MIN_LENGTH = 8


def password_problem(password):
    """Retourne un message d'erreur si le mot de passe est trop faible, sinon None."""
    if not password or len(password) < PASSWORD_MIN_LENGTH:
        return f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not _re.search(r"[A-Za-z]", password) or not _re.search(r"\d", password):
        return "Password must contain at least one letter and one digit."
    return None


def create_user(username, password, is_admin=False):
    username = (username or "").strip()
    if not username:
        return False, "Username is required."
    problem = password_problem(password)
    if problem:
        return False, problem
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


# ── File-based storage helpers ─────────────────────────────────────────────────
def get_userdata_dir(user_id):
    path = os.path.join(DATA_DIR, str(user_id))
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


_GLOBAL_PATIENTS_PATH = os.path.join(DATA_DIR, "patients.json")


def load_global_patient_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(_GLOBAL_PATIENTS_PATH):
        with open(_GLOBAL_PATIENTS_PATH) as f:
            return json_load(f)
    return {}


def save_global_patient_data(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(_GLOBAL_PATIENTS_PATH, "w") as f:
        json_dump(data, f)


def count_all_patients():
    """Total number of patients across every user's store (admin view only)."""
    total = 0
    base = DATA_DIR
    if not os.path.isdir(base):
        return 0
    for entry in os.listdir(base):
        ppath = os.path.join(base, entry, "patients.json")
        if os.path.isfile(ppath):
            try:
                with open(ppath) as f:
                    total += len(json_load(f))
            except Exception:
                continue
    return total


def migrate_global_patient_data_to_admin():
    """One-time migration of the legacy shared store into the first admin account.

    Before isolation, all patients lived in a single userdata/patients.json shared
    by every account. Move that data into the first admin user's per-user store so
    nothing is lost, and rename the legacy file to .bak so this runs only once.
    """
    if not os.path.exists(_GLOBAL_PATIENTS_PATH):
        return
    try:
        with open(_GLOBAL_PATIENTS_PATH) as f:
            legacy = json_load(f)
    except Exception:
        return
    if not legacy:
        os.replace(_GLOBAL_PATIENTS_PATH, _GLOBAL_PATIENTS_PATH + ".bak")
        return
    with _get_db() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE is_admin=1 ORDER BY id LIMIT 1"
        ).fetchone()
    if not row:
        return  # no admin yet; retry on a later startup
    admin_id = row["id"]
    dest = os.path.join(get_userdata_dir(admin_id), "patients.json")
    if os.path.exists(dest):
        return  # admin already has a store; don't clobber it
    save_patient_data(admin_id, legacy)
    os.replace(_GLOBAL_PATIENTS_PATH, _GLOBAL_PATIENTS_PATH + ".bak")


# ── Patient analytics helpers ──────────────────────────────────────────────────
def get_recent_patients(user_id, n=5):
    """Return up to n recently modified patients from this user's patient store."""
    patients = load_patient_data(user_id)
    summaries = []
    for patient_id, records in patients.items():
        if not records:
            continue
        last_date = max(
            (r.get("saved_at") or r.get("exam_date") or "" for r in records),
            default="",
        )
        parts = patient_id.split("_")
        label = " ".join(parts[:2]) if len(parts) >= 2 else patient_id
        summaries.append({
            "id": patient_id,
            "label": label,
            "last_modified": last_date,
            "n_exams": len(records),
        })
    summaries.sort(key=lambda x: x["last_modified"], reverse=True)
    return summaries[:n]


def get_user_stats(user_id):
    """Return aggregate stats: this user's patient counts + dossier counts."""
    patients = load_patient_data(user_id)
    n_patients = len(patients)
    n_exams = sum(len(v) for v in patients.values())
    n_annotations = sum(
        len(r.get("shapes", [])) for records in patients.values() for r in records
    )
    dossiers = _load_index(user_id)
    n_dossiers = len(dossiers)
    n_dossier_annotations = sum(d.get("annotation_count", 0) for d in dossiers)
    tag_counts: dict = {}
    for d in dossiers:
        for t in d.get("pathology_tags") or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    return {
        "n_patients": n_patients,
        "n_exams": n_exams,
        "n_annotations": n_annotations,
        "n_dossiers": n_dossiers,
        "n_dossier_annotations": n_dossier_annotations,
        "tag_counts": tag_counts,
    }


# ── Dossier storage ────────────────────────────────────────────────────────────
def _dossiers_dir(user_id):
    path = os.path.join(get_userdata_dir(user_id), "dossiers")
    os.makedirs(path, exist_ok=True)
    return path


def _index_path(user_id):
    return os.path.join(get_userdata_dir(user_id), "dossiers_index.json")


# Les dossier_id arrivent depuis des ids de composants côté client : on ne
# construit jamais un chemin avec une valeur non validée (traversée de répertoire).
_DOSSIER_ID_RE = _re.compile(r"^[0-9a-fA-F-]{1,64}$")


def _dossier_path(user_id, dossier_id):
    if not _DOSSIER_ID_RE.match(str(dossier_id)):
        return None
    return os.path.join(_dossiers_dir(user_id), f"{dossier_id}.json")


def _load_index(user_id):
    p = _index_path(user_id)
    if os.path.exists(p):
        with open(p) as f:
            return json_load(f)
    return []


def _save_index(user_id, index):
    with open(_index_path(user_id), "w") as f:
        json_dump(index, f)


def _make_thumbnail(image_b64, width=160):
    """Return a small JPEG thumbnail as base64 data-URL, or None on failure."""
    try:
        data = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        img = _PILImage.open(_io.BytesIO(base64.b64decode(data))).convert("RGB")
        ratio = width / img.width
        img = img.resize((width, max(1, int(img.height * ratio))), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=55)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def save_dossier(user_id, *, dossier_id=None, name="", eye="NA", date_exam="",
                 pathology_tags=None, notes="", image_b64="", image_filename="",
                 annotations=None, sift_applied=False, sift_homography=None,
                 patient_id=None, patient_name=None,
                 subject_nom=None, subject_prenom=None, subject_ddn=None):
    """Create or update a dossier. Returns dossier_id."""
    now = _datetime.now().isoformat()
    is_new = dossier_id is None
    if is_new:
        dossier_id = str(_uuid.uuid4())

    dossier_path = _dossier_path(user_id, dossier_id)
    if dossier_path is None:
        return None
    existing = {}
    if os.path.exists(dossier_path):
        with open(dossier_path) as f:
            existing = json_load(f)

    data = {
        "id": dossier_id,
        "name": name or "Unnamed",
        "created_at": existing.get("created_at", now),
        "modified_at": now,
        "eye": eye or "NA",
        "date_exam": date_exam or "",
        "pathology_tags": pathology_tags or [],
        "notes": notes or "",
        "image_b64": image_b64 or existing.get("image_b64", ""),
        "image_filename": image_filename or existing.get("image_filename", ""),
        "annotations": annotations if annotations is not None else existing.get("annotations", []),
        "sift_applied": sift_applied,
        "sift_homography": sift_homography,
        "patient_id": patient_id or existing.get("patient_id") or None,
        "patient_name": patient_name or existing.get("patient_name") or None,
        "subject_nom": subject_nom or existing.get("subject_nom") or None,
        "subject_prenom": subject_prenom or existing.get("subject_prenom") or None,
        "subject_ddn": subject_ddn or existing.get("subject_ddn") or None,
    }

    with open(dossier_path, "w") as f:
        json_dump(data, f)

    # Update lightweight index
    thumb = _make_thumbnail(data["image_b64"]) if data["image_b64"] else None
    meta = {
        "id": dossier_id,
        "name": data["name"],
        "created_at": data["created_at"],
        "modified_at": data["modified_at"],
        "eye": data["eye"],
        "date_exam": data["date_exam"],
        "pathology_tags": data["pathology_tags"],
        "annotation_count": len(data["annotations"]),
        "image_filename": data["image_filename"],
        "thumbnail": thumb,
        "patient_id": data["patient_id"],
        "patient_name": data["patient_name"],
        "subject_nom": data["subject_nom"],
        "subject_prenom": data["subject_prenom"],
        "subject_ddn": data["subject_ddn"],
    }
    index = [m for m in _load_index(user_id) if m["id"] != dossier_id]
    index.insert(0, meta)
    index.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
    _save_index(user_id, index)

    return dossier_id


def list_dossiers(user_id):
    """Return lightweight dossier index (no image_b64)."""
    return _load_index(user_id)


def list_dossiers_for_patient(user_id, patient_id):
    """Return dossiers linked to a specific patient_id."""
    return [d for d in _load_index(user_id) if d.get("patient_id") == patient_id]


def get_dossier(user_id, dossier_id):
    """Return full dossier dict including image_b64."""
    path = _dossier_path(user_id, dossier_id)
    if path and os.path.exists(path):
        with open(path) as f:
            return json_load(f)
    return None


def delete_dossier(user_id, dossier_id):
    path = _dossier_path(user_id, dossier_id)
    if path and os.path.exists(path):
        os.remove(path)
    _save_index(user_id, [m for m in _load_index(user_id) if m["id"] != dossier_id])


def update_dossier_meta(user_id, dossier_id, **kwargs):
    """Update metadata fields only (name, tags, notes, eye, date_exam, patient_id, patient_name)."""
    path = _dossier_path(user_id, dossier_id)
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        data = json_load(f)
    allowed = {"name", "eye", "date_exam", "pathology_tags", "notes", "patient_id", "patient_name",
               "subject_nom", "subject_prenom", "subject_ddn"}
    for k, v in kwargs.items():
        if k in allowed:
            data[k] = v
    data["modified_at"] = _datetime.now().isoformat()
    with open(path, "w") as f:
        json_dump(data, f)
    index = _load_index(user_id)
    for m in index:
        if m["id"] == dossier_id:
            for k, v in kwargs.items():
                if k in m:
                    m[k] = v
            m["modified_at"] = data["modified_at"]
            break
    _save_index(user_id, index)
