from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib.parse import urlencode

from pipeline.web_full import make_server


def _request(
    server,
    method: str,
    path: str,
    body: str | None = None,
    *,
    headers: dict[str, str] | None = None,
):
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    request_headers = dict(headers or {})
    payload = None
    if body is not None:
        payload = body.encode("utf-8")
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
        request_headers["Content-Length"] = str(len(payload))
    connection.request(method, path, body=payload, headers=request_headers)
    response = connection.getresponse()
    data = response.read()
    response_headers = dict(response.getheaders())
    connection.close()
    return response.status, response_headers, data


def test_web_access_token_guards_pages(tmp_path: Path) -> None:
    server = make_server("127.0.0.1", 0, root=tmp_path, auth_token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, _ = _request(server, "GET", "/")
        assert status == 303
        assert headers["Location"] == "/login"

        status, _, data = _request(
            server,
            "POST",
            "/login",
            urlencode({"token": "wrong"}),
        )
        assert status == 200
        assert "Access token".encode() in data

        status, headers, _ = _request(
            server,
            "POST",
            "/login",
            urlencode({"token": "secret-token"}),
        )
        assert status == 303
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert cookie.startswith("lora_web_token=")

        status, _, data = _request(server, "GET", "/", headers={"Cookie": cookie})
        assert status == 200
        assert "LoRA".encode() in data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_healthz_remains_available_without_login(tmp_path: Path) -> None:
    server = make_server("127.0.0.1", 0, root=tmp_path, auth_token="secret-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, data = _request(server, "GET", "/healthz")
        assert status == 200
        assert data == b"ok\n"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
