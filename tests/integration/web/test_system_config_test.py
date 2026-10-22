"""Интеграционные тесты проверки соединения с LLM (POST /system/config/test)."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


@contextmanager
def mock_server(*, status=200, response_body=b'{"choices":[{"message":{"content":"pong"}}]}'):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                self.rfile.read(length)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_config_test_success(configured_client):
    with mock_server(status=200) as base_url:
        response = configured_client.post(
            "/system/config/test",
            json={
                "base_url": f"{base_url}/v1",
                "model": "test-model",
                "api_key": "test-key",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["error"] is None
        assert "успешно" in data["message"].lower()


def test_config_test_invalid_api_key_sanitized(configured_client):
    secret_key = "sk-secret-invalid-key-12345"
    with mock_server(
        status=401,
        response_body=b'{"error": "Unauthorized: invalid api key sk-secret-invalid-key-12345"}',
    ) as base_url:
        response = configured_client.post(
            "/system/config/test",
            json={
                "base_url": f"{base_url}/v1",
                "model": "test-model",
                "api_key": secret_key,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert data["error"] is not None
        assert secret_key not in response.text


def test_config_test_unreachable_host(configured_client):
    # Порт 59999 не слушает соединений
    response = configured_client.post(
        "/system/config/test",
        json={
            "base_url": "http://127.0.0.1:59999/v1",
            "model": "test-model",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["error"] is not None


def test_config_test_missing_model_or_key_validation(client):
    response = client.post(
        "/system/config/test",
        json={
            "model": "",
            "analysis_model": "",
            "processing_model": "",
            "api_key": "",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["error"] is not None
    assert "модел" in data["error"].lower() or "ключ" in data["error"].lower()


def test_config_test_never_leaks_secrets(configured_client):
    secret_key = "super-secret-key-xyz-987"
    secret_proxy_pass = "top-secret-password-456"
    proxy_url = f"http://alice:{secret_proxy_pass}@127.0.0.1:59999"

    with mock_server(
        status=500,
        response_body=f'{{"error": "Internal error with {secret_key} and {secret_proxy_pass}"}}'.encode(),
    ) as base_url:
        response = configured_client.post(
            "/system/config/test",
            json={
                "base_url": f"{base_url}/v1",
                "model": "test-model",
                "api_key": secret_key,
                "proxy_mode": "custom",
                "proxy_url": proxy_url,
            },
        )
        assert response.status_code == 200
        assert secret_key not in response.text
        assert secret_proxy_pass not in response.text


def test_config_test_does_not_leak_current_key_to_new_base_url(configured_client):
    received_requests = []

    class TrackingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            received_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), TrackingHandler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{http.server_port}"
        response = configured_client.post(
            "/system/config/test",
            json={
                "base_url": f"{base_url}/v1",
                "model": "test-model",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert "ключ" in data["message"].lower() or "ключ" in (data["error"] or "").lower()
        assert len(received_requests) == 0
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_config_test_does_not_leak_current_key_to_new_proxy(configured_client):
    response = configured_client.post(
        "/system/config/test",
        json={
            "proxy_mode": "custom",
            "proxy_url": "http://127.0.0.1:59998",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert "ключ" in data["message"].lower() or "ключ" in (data["error"] or "").lower()


def test_config_test_rejects_form_data(configured_client):
    response = configured_client.post(
        "/system/config/test",
        data={"model": "test-model"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "bad_url",
    [
        "ftp://example.com/v1",
        "http://user:pass@example.com/v1",
        "http://example.com/v1?token=123",
        "http://example.com/v1#section",
        "http://169.254.169.254/v1",
        "http://metadata.google.internal/computeMetadata/v1",
    ],
)
def test_config_test_rejects_invalid_or_metadata_url(configured_client, bad_url):
    response = configured_client.post(
        "/system/config/test",
        json={
            "base_url": bad_url,
            "model": "test-model",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["error"] is not None


def test_config_test_does_not_reveal_response_body_on_error(configured_client):
    with mock_server(
        status=403,
        response_body=b'{"secret_internal_detail": "internal-db-password-12345"}',
    ) as base_url:
        response = configured_client.post(
            "/system/config/test",
            json={
                "base_url": f"{base_url}/v1",
                "model": "test-model",
                "api_key": "test-key",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert "secret_internal_detail" not in response.text
        assert "internal-db-password-12345" not in response.text


def test_config_test_rejects_invalid_proxy_url(configured_client):
    response = configured_client.post(
        "/system/config/test",
        json={
            "proxy_mode": "custom",
            "proxy_url": "http://[invalid]",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["message"] == "Ошибка конфигурации подключения"
    assert data["error"] == "Некорректный адрес прокси"


def test_config_test_rejects_unsupported_provider(configured_client):
    response = configured_client.post(
        "/system/config/test",
        json={
            "provider": "unsupported_provider",
            "model": "test-model",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is False
    assert data["message"] == "Ошибка конфигурации подключения"
    assert data["error"] == "Провайдер не поддерживается"
