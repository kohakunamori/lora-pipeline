from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.web_full import make_server


def _request(server, method: str, path: str, body: str | None = None):
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    headers = {}
    payload = None
    if body is not None:
        payload = body.encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        headers["Content-Length"] = str(len(payload))
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    data = response.read()
    headers_out = dict(response.getheaders())
    connection.close()
    return response.status, headers_out, data


def test_source_delete_requires_typed_source_id(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    Image.new("RGB", (48, 48), "white").save(incoming / "one.png")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(incoming, kind="image_directory", label="images")
    source_id = str(record["id"])

    server = make_server("127.0.0.1", 0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        csrf = server.RequestHandlerClass.app.csrf
        request_delete = urlencode(
            {"_csrf": csrf, "source_id": source_id, "action": "delete"}
        )
        status, _, data = _request(server, "POST", "/datasets/demo/source-action", request_delete)
        assert status == 200
        assert source_id.encode() in data
        assert source_id in DatasetWorkspace.load("demo", root=tmp_path).sources

        wrong = urlencode(
            {"_csrf": csrf, "source_id": source_id, "confirm": "wrong"}
        )
        status, _, _ = _request(server, "POST", "/dataset-tools/demo/delete-source", wrong)
        assert status == 400
        assert source_id in DatasetWorkspace.load("demo", root=tmp_path).sources

        correct = urlencode(
            {"_csrf": csrf, "source_id": source_id, "confirm": source_id}
        )
        status, headers, _ = _request(server, "POST", "/dataset-tools/demo/delete-source", correct)
        assert status == 303
        assert headers["Location"] == "/datasets/demo"
        assert source_id not in DatasetWorkspace.load("demo", root=tmp_path).sources
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
