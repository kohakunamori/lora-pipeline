from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.web_full import make_server
from pipeline.web_jobs import active_gpu_jobs, create_job, update_job


def _serve(tmp_path: Path, *, auth_token: str | None = None):
    server = make_server("127.0.0.1", 0, root=tmp_path, auth_token=auth_token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(
    server,
    method: str,
    path: str,
    *,
    form: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
):
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    outgoing = dict(headers or {})
    payload = None
    if form is not None:
        payload = urlencode(form).encode("utf-8")
        outgoing["Content-Type"] = "application/x-www-form-urlencoded"
        outgoing["Content-Length"] = str(len(payload))
    connection.request(method, path, body=payload, headers=outgoing)
    response = connection.getresponse()
    data = response.read()
    response_headers = dict(response.getheaders())
    status = response.status
    connection.close()
    return status, response_headers, data


def test_token_protects_web_pages_and_login_cookie_unlocks_them(tmp_path: Path) -> None:
    DatasetWorkspace.create("demo", root=tmp_path)
    server, thread = _serve(tmp_path, auth_token="secret token")
    try:
        status, headers, _ = _request(server, "GET", "/datasets")
        assert status == 303
        assert headers["Location"] == "/login"

        status, headers, _ = _request(
            server,
            "POST",
            "/login",
            form={"token": "secret token"},
        )
        assert status == 303
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert cookie.startswith("lora_web_token=")

        status, _, data = _request(
            server,
            "GET",
            "/datasets",
            headers={"Cookie": cookie},
        )
        assert status == 200
        assert b"demo" in data
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_source_delete_requires_typed_confirmation(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGB", (64, 64), "white").save(incoming / "one.png")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(
        incoming,
        kind="image_directory",
        label="official",
    )
    source_id = str(record["id"])

    server, thread = _serve(tmp_path)
    try:
        csrf = server.RequestHandlerClass.app.csrf
        status, _, data = _request(
            server,
            "POST",
            "/datasets/demo/source-action",
            form={
                "_csrf": csrf,
                "source_id": source_id,
                "action": "delete",
            },
        )
        assert status == 400
        assert "先停用".encode("utf-8") in data
        assert source_id in DatasetWorkspace.load("demo", root=tmp_path).sources

        workspace = DatasetWorkspace.load("demo", root=tmp_path)
        workspace.set_source_enabled(source_id, False)

        status, _, data = _request(
            server,
            "POST",
            "/datasets/demo/source-action",
            form={
                "_csrf": csrf,
                "source_id": source_id,
                "action": "delete",
            },
        )
        assert status == 200
        assert source_id.encode() in data
        assert source_id in DatasetWorkspace.load("demo", root=tmp_path).sources

        status, _, _ = _request(
            server,
            "POST",
            "/dataset-tools/demo/delete-source",
            form={
                "_csrf": csrf,
                "source_id": source_id,
                "confirm": "wrong",
            },
        )
        assert status == 400
        assert source_id in DatasetWorkspace.load("demo", root=tmp_path).sources

        status, headers, _ = _request(
            server,
            "POST",
            "/dataset-tools/demo/delete-source",
            form={
                "_csrf": csrf,
                "source_id": source_id,
                "confirm": source_id,
            },
        )
        assert status == 303
        assert headers["Location"] == "/datasets/demo"
        assert source_id not in DatasetWorkspace.load("demo", root=tmp_path).sources
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_starting_gpu_job_is_reserved(tmp_path: Path) -> None:
    job = create_job("train", {"project": "demo"}, root=tmp_path)
    update_job(str(job["id"]), root=tmp_path, status="starting")

    active = active_gpu_jobs(root=tmp_path)
    assert [item["id"] for item in active] == [job["id"]]
