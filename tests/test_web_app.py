from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib.parse import urlencode

from PIL import Image

from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.state import ProjectState
from pipeline.web_app import _safe_child, _training_command, make_server


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


def test_web_dataset_grid_and_exclusion(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (64, 96), "white").save(source / "one.png")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(source, kind="image_directory", label="images")
    source_id = str(record["id"])

    server = make_server("127.0.0.1", 0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, data = _request(server, "GET", f"/datasets/demo/source/{source_id}")
        assert status == 200
        assert b"one.png" in data

        csrf = server.RequestHandlerClass.app.csrf
        key = f"{source_id}/one.png"
        form = urlencode(
            {
                "_csrf": csrf,
                "source_id": source_id,
                "action": "exclude",
                "keys": key,
            }
        )
        status, headers, _ = _request(server, "POST", "/datasets/demo/bulk", form)
        assert status == 303
        assert headers["Location"].endswith(f"/source/{source_id}")
        assert DatasetWorkspace.load("demo", root=tmp_path).items(source_id=source_id)[0].excluded is True

        media = f"/media/dataset/demo/{source_id}/one.png"
        status, headers, data = _request(server, "GET", media)
        assert status == 200
        assert headers["Content-Type"].startswith("image/png")
        assert data.startswith(b"\x89PNG")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_web_rejects_bad_csrf(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (32, 32), "white").save(source / "one.png")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(source, kind="image_directory")
    source_id = str(record["id"])

    server = make_server("127.0.0.1", 0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        form = urlencode(
            {
                "_csrf": "wrong",
                "source_id": source_id,
                "action": "exclude",
                "keys": f"{source_id}/one.png",
            }
        )
        status, _, data = _request(server, "POST", "/datasets/demo/bulk", form)
        assert status == 400
        assert b"CSRF" in data
        assert DatasetWorkspace.load("demo", root=tmp_path).items(source_id=source_id)[0].excluded is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_training_command_uses_frozen_workflow(tmp_path: Path) -> None:
    state = ProjectState.create(
        tmp_path / "projects" / "run-demo",
        name="run-demo",
        concept_type="character",
        base="base",
        trigger="zz_demo",
        strategy="quality",
    )
    state.payload["project"]["training_config_snapshot"] = {"images_seen": 2400}
    state.payload["project"]["interactive_preferences"] = {
        "run_dedup": False,
        "run_identity": True,
        "caption_mode": "existing_taglist_clean",
        "run_review": False,
        "allow_trigger_only": True,
    }
    state.save()

    command = _training_command(state)
    assert command[-1] == "--allow-trigger-only"
    assert "--skip-dedup" in command
    assert "--skip-review" in command
    assert "--skip-identity" not in command
    assert command[command.index("--images-seen") + 1] == "2400"
    assert command[command.index("--caption-mode") + 1] == "existing_taglist_clean"
    assert "--skip-evaluate" in command


def test_safe_child_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "image.png"
    inside.write_bytes(b"x")
    assert _safe_child(root, "image.png") == inside.resolve()

    try:
        _safe_child(root, "../secret")
    except Exception as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("path traversal was accepted")
