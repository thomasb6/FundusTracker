"""Tests du durcissement web (en-têtes de sécurité, cookies, proxy)."""
from werkzeug.middleware.proxy_fix import ProxyFix

import main


def test_security_headers_present():
    client = main.server.test_client()
    resp = client.get("/")
    h = resp.headers
    assert h.get("X-Content-Type-Options") == "nosniff"
    assert h.get("X-Frame-Options") == "DENY"
    assert "frame-ancestors 'none'" in h.get("Content-Security-Policy", "")
    assert h.get("Referrer-Policy")
    assert h.get("Permissions-Policy")
    assert "max-age=" in h.get("Strict-Transport-Security", "")


def test_session_cookie_hardening():
    cfg = main.server.config
    assert cfg["SESSION_COOKIE_HTTPONLY"] is True
    assert cfg["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert cfg["REMEMBER_COOKIE_HTTPONLY"] is True


def test_proxyfix_applied():
    # L'app doit interpréter les en-têtes X-Forwarded-* de nginx.
    assert isinstance(main.server.wsgi_app, ProxyFix)


def test_secure_cookie_follows_env():
    # conftest fixe FUNDUS_SECURE_COOKIES=0 → flag Secure désactivé pour les tests.
    assert main.server.config["SESSION_COOKIE_SECURE"] is False
