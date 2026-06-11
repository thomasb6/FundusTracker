"""Tests du stockage par compte et des protections de sécurité (auth.py)."""
import os

import pytest

import auth


@pytest.fixture(autouse=True)
def fresh_db():
    auth.init_db()
    yield


# ── Comptes ────────────────────────────────────────────────────────────────────
class TestAccounts:
    def test_initial_admin_uses_env_password(self):
        # FUNDUS_ADMIN_PASSWORD est fixé par conftest.
        assert auth.verify_user("admin", "test-admin-pw") is not None
        assert auth.verify_user("admin", "admin123") is None  # plus de défaut public

    def test_create_and_verify_user(self):
        ok, err = auth.create_user("grader_cm", "s3cret")
        assert ok and err is None
        user = auth.verify_user("grader_cm", "s3cret")
        assert user is not None and user.username == "grader_cm"
        assert user.is_admin is False

    def test_duplicate_username_rejected(self):
        auth.create_user("dup", "pw")
        ok, err = auth.create_user("dup", "pw2")
        assert ok is False and "exists" in err.lower()

    def test_wrong_password_fails(self):
        auth.create_user("bob", "rightpw")
        assert auth.verify_user("bob", "wrongpw") is None


# ── Dossiers : CRUD ────────────────────────────────────────────────────────────
class TestDossierCrud:
    def test_save_and_get(self):
        did = auth.save_dossier(42, name="Birdshot OD", eye="OD")
        d = auth.get_dossier(42, did)
        assert d["name"] == "Birdshot OD"
        assert d["eye"] == "OD"

    def test_update_meta(self):
        did = auth.save_dossier(42, name="x")
        auth.update_dossier_meta(42, did, name="renamed", notes="suivi")
        d = auth.get_dossier(42, did)
        assert d["name"] == "renamed" and d["notes"] == "suivi"

    def test_delete(self):
        did = auth.save_dossier(42, name="temp")
        auth.delete_dossier(42, did)
        assert auth.get_dossier(42, did) is None

    def test_isolation_between_users(self):
        did = auth.save_dossier(1, name="patient de user1")
        # Un autre compte ne voit pas le dossier via son propre espace.
        assert auth.get_dossier(2, did) is None


# ── Sécurité : traversée de répertoire ─────────────────────────────────────────
class TestPathTraversal:
    @pytest.mark.parametrize("bad_id", [
        "../../etc/passwd",
        "../evil",
        "..%2f..%2f",
        "a/b/c",
        "id with spaces",
    ])
    def test_get_rejects_malicious_id(self, bad_id):
        assert auth.get_dossier(1, bad_id) is None

    def test_save_rejects_malicious_id(self):
        assert auth.save_dossier(1, dossier_id="../escape", name="x") is None

    def test_delete_and_update_are_noops_on_bad_id(self, tmp_path):
        # Ne doivent rien créer ni lever.
        auth.delete_dossier(1, "../../something")
        auth.update_dossier_meta(1, "../../something", name="x")

    def test_legitimate_uuid_accepted(self):
        did = auth.save_dossier(1, name="ok")
        # save_dossier renvoie un UUID valide, qui doit être relisible.
        assert auth.get_dossier(1, did) is not None
