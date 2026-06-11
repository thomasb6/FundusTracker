"""Tests du limiteur anti-brute-force du login (main.py)."""
import main


def _reset():
    main._LOGIN_FAILURES.clear()


def test_no_delay_before_threshold():
    _reset()
    key = ("1.2.3.4", "admin")
    for _ in range(main._LOGIN_FREE_ATTEMPTS - 1):
        main._login_record_failure(key)
    assert main._login_retry_delay(key) == 0


def test_lock_after_threshold():
    _reset()
    key = ("1.2.3.4", "admin")
    for _ in range(main._LOGIN_FREE_ATTEMPTS):
        main._login_record_failure(key)
    delay = main._login_retry_delay(key)
    assert main._LOGIN_LOCK_BASE_S * 0.9 < delay <= main._LOGIN_LOCK_BASE_S


def test_backoff_grows():
    _reset()
    key = ("1.2.3.4", "admin")
    for _ in range(main._LOGIN_FREE_ATTEMPTS + 1):
        main._login_record_failure(key)
    # Un échec de plus → délai (presque) doublé.
    assert main._login_retry_delay(key) > main._LOGIN_LOCK_BASE_S


def test_backoff_capped():
    _reset()
    key = ("1.2.3.4", "admin")
    for _ in range(50):
        main._login_record_failure(key)
    assert main._login_retry_delay(key) <= main._LOGIN_LOCK_MAX_S


def test_keys_are_isolated_per_ip():
    _reset()
    for _ in range(main._LOGIN_FREE_ATTEMPTS):
        main._login_record_failure(("1.1.1.1", "admin"))
    assert main._login_retry_delay(("2.2.2.2", "admin")) == 0


def test_client_ip_uses_last_forwarded_for():
    # Derrière nginx : on garde la dernière valeur (ajoutée par le proxy),
    # pas celle que le client peut falsifier en tête de chaîne.
    with main.server.test_request_context(
        headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.5"}
    ):
        assert main._login_client_ip() == "10.0.0.5"


def test_login_blocks_then_allows_other_ip():
    _reset()
    with main.server.test_request_context(headers={"X-Forwarded-For": "5.5.5.5"}):
        for _ in range(main._LOGIN_FREE_ATTEMPTS + 1):
            out = main.handle_login(1, "admin", "wrong")
        assert "Too many failed attempts" in str(out[1].children)
    # Une autre IP n'est pas affectée par le verrou.
    with main.server.test_request_context(headers={"X-Forwarded-For": "6.6.6.6"}):
        out = main.handle_login(1, "admin", "test-admin-pw")
        assert out[0]["username"] == "admin"
