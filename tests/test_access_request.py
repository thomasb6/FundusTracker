"""Tests des demandes d'accès (modèle sur invitation)."""
import auth


def test_access_request_flow():
    auth.init_db()
    ok, err = auth.create_access_request("Dr Test", "dr@example.com", "CHU", "pilote")
    assert ok and err is None
    reqs = auth.list_access_requests()
    match = [r for r in reqs if r["email"] == "dr@example.com"]
    assert match and match[0]["institution"] == "CHU"
    rid = match[0]["id"]
    auth.delete_access_request(rid)
    assert all(r["id"] != rid for r in auth.list_access_requests())


def test_access_request_rejects_bad_email():
    auth.init_db()
    ok, err = auth.create_access_request("x", "not-an-email")
    assert ok is False and "email" in err.lower()


def test_access_request_requires_email():
    auth.init_db()
    ok, err = auth.create_access_request("x", "")
    assert ok is False
