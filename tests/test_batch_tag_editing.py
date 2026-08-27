from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest
from PIL import Image

from pipeline.dataset_tag_editor import batch_edit_tags
from pipeline.dataset_workspace import DatasetWorkspace
from pipeline.models import PipelineError
from pipeline.web_full import make_server


def _image(path: Path, color: str = "white") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 128), color).save(path)


def _workspace(tmp_path: Path) -> tuple[DatasetWorkspace, str, dict[str, str]]:
    source = tmp_path / "source"
    _image(source / "a.png", "red")
    _image(source / "b.png", "blue")
    _image(source / "c.png", "green")
    (source / "a.txt").write_text("1girl, blue_hair, smile\n", encoding="utf-8")
    (source / "b.txt").write_text("solo, Blue Hair, outdoors\n", encoding="utf-8")
    (source / "c.txt").write_text("portrait\n", encoding="utf-8")
    workspace = DatasetWorkspace.create("demo", root=tmp_path)
    record = workspace.add_source_from_directory(source, kind="image_directory", label="images")
    items = {item.relative.name: item.key for item in workspace.items()}
    return workspace, str(record["id"]), items


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


def _serve(tmp_path: Path):
    server = make_server("127.0.0.1", 0, root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_batch_prepend_moves_existing_semantic_duplicates_to_front(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    result = batch_edit_tags(
        workspace,
        [items["a.png"], items["b.png"]],
        ["blue hair", "trigger", "Trigger"],
        action="prepend",
    )
    assert result["changed"] == 2
    assert result["tags"] == ["blue hair", "trigger"]
    assert workspace.caption_text(items["a.png"]) == "blue hair, trigger, 1girl, smile"
    assert workspace.caption_text(items["b.png"]) == "blue hair, trigger, solo, outdoors"


def test_batch_append_moves_existing_tags_to_tail_and_remove_normalizes_names(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    batch_edit_tags(
        workspace,
        [items["a.png"], items["c.png"]],
        ["smile", "masterpiece"],
        action="append",
    )
    assert workspace.caption_text(items["a.png"]) == "1girl, blue_hair, smile, masterpiece"
    assert workspace.caption_text(items["c.png"]) == "portrait, smile, masterpiece"
    result = batch_edit_tags(
        workspace,
        [items["a.png"], items["b.png"]],
        ["BLUE HAIR", "smile"],
        action="remove",
    )
    assert result["changed"] == 2
    assert workspace.caption_text(items["a.png"]) == "1girl, masterpiece"
    assert workspace.caption_text(items["b.png"]) == "solo, outdoors"


def test_batch_edit_validates_every_key_before_writing(tmp_path: Path) -> None:
    workspace, _, items = _workspace(tmp_path)
    before = workspace.caption_text(items["a.png"])
    with pytest.raises(PipelineError, match="Unknown dataset item"):
        batch_edit_tags(
            workspace,
            [items["a.png"], "image-directory-999/missing.png"],
            ["trigger"],
            action="prepend",
        )
    assert workspace.caption_text(items["a.png"]) == before


def test_web_source_grid_exposes_and_applies_batch_tag_actions(tmp_path: Path) -> None:
    workspace, source_id, items = _workspace(tmp_path)
    del workspace
    server, thread = _serve(tmp_path)
    try:
        status, _, data = _request(server, "GET", f"/datasets/demo/source/{source_id}")
        assert status == 200
        assert "Tag 添加到首部".encode() in data
        assert "Tag 添加到尾部".encode() in data
        assert "删除指定 Tag".encode() in data

        csrf = server.RequestHandlerClass.app.csrf
        form = urlencode(
            [
                ("_csrf", csrf),
                ("source_id", source_id),
                ("action", "tag-prepend"),
                ("tags", "trigger, blue hair"),
                ("keys", items["a.png"]),
                ("keys", items["b.png"]),
            ]
        )
        status, headers, _ = _request(server, "POST", "/datasets/demo/bulk", form)
        assert status == 303
        assert headers["Location"].endswith(f"/source/{source_id}")
        reloaded = DatasetWorkspace.load("demo", root=tmp_path)
        assert reloaded.caption_text(items["a.png"]) == "trigger, blue hair, 1girl, smile"
        assert reloaded.caption_text(items["b.png"]) == "trigger, blue hair, solo, outdoors"

        remove_form = urlencode(
            [
                ("_csrf", csrf),
                ("source_id", source_id),
                ("action", "tag-remove"),
                ("tags", "BLUE_HAIR"),
                ("keys", items["a.png"]),
                ("keys", items["b.png"]),
            ]
        )
        status, _, _ = _request(server, "POST", "/datasets/demo/bulk", remove_form)
        assert status == 303
        reloaded = DatasetWorkspace.load("demo", root=tmp_path)
        assert reloaded.caption_text(items["a.png"]) == "trigger, 1girl, smile"
        assert reloaded.caption_text(items["b.png"]) == "trigger, solo, outdoors"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_entrypoints_combine_batch_tags_and_composition_metadata() -> None:
    from pipeline.interactive import InteractiveWizard
    from pipeline.interactive_batch_tags import InteractiveWizard as BatchTagWizard
    from pipeline.interactive_composition import InteractiveWizard as CompositionWizard
    from pipeline.web_full import FullHandler
    from pipeline.web_metadata_batch import MetadataBatchHandler

    assert issubclass(InteractiveWizard, BatchTagWizard)
    assert issubclass(InteractiveWizard, CompositionWizard)
    assert issubclass(FullHandler, MetadataBatchHandler)
