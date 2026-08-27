from __future__ import annotations

import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from .dataset_metadata import (
    analyze_workspace_compositions,
    import_composition_records,
    seed_source_defaults,
)
from .dataset_workspace import DatasetWorkspace
from .models import PipelineError
from .video_composition import build_enriched_character_dataset
from .video_source import is_url
from .web_jobs import job_data_dir, read_job, update_job
from .web_worker import _load_subject_report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m pipeline.web_worker_enriched JOB_ID")
    job_id = sys.argv[1]
    job = read_job(job_id)
    kind = str(job.get("kind") or "")
    if kind not in {"dataset_analyze", "video_finalize"}:
        from .web_worker import main as legacy_main

        legacy_main()
        return

    try:
        update_job(job_id, status="running", pid=os.getpid(), error=None)
        payload = dict(job.get("payload") or {})
        print(f"[web] enriched job={job_id} kind={kind}", flush=True)
        if kind == "dataset_analyze":
            result = _dataset_analyze(payload)
        else:
            result = _video_finalize(job_id, payload)
        update_job(
            job_id,
            status="completed",
            pid=None,
            result=result,
            finished_at=_now(),
        )
    except BaseException as exc:
        traceback.print_exc()
        update_job(
            job_id,
            status="failed",
            pid=None,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_now(),
        )
        raise


def _dataset_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = DatasetWorkspace.load(str(payload["dataset"]))
    source_id = str(payload.get("source_id") or "") or None
    proxy = int(payload.get("detection_proxy_long_edge", 1280))
    print(
        f"[web] composition analysis dataset={workspace.name} source={source_id or 'all'} proxy={proxy}",
        flush=True,
    )
    return analyze_workspace_compositions(
        workspace,
        source_id=source_id,
        detection_proxy_long_edge=proxy,
    )


def _video_finalize(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    workspace = DatasetWorkspace.load(str(payload["dataset"]))
    selected_cluster = int(payload["selected_cluster"])
    source = str(payload["source"])
    label = str(payload.get("label") or "video").strip() or "video"
    data_dir = job_data_dir(job_id)

    import json

    cluster_payload = json.loads((data_dir / "clusters.json").read_text(encoding="utf-8"))
    cluster = next(
        (
            value
            for value in cluster_payload.get("clusters", [])
            if int(value.get("cluster_id")) == selected_cluster
        ),
        None,
    )
    if cluster is None:
        raise PipelineError(f"Unknown video identity cluster: {selected_cluster}")

    training_dir = data_dir / "training"
    mode = str(cluster_payload.get("mode") or "full_frame")
    processing: dict[str, Any] = {
        "source_kind": "remote_url" if is_url(source) else "local_video",
        "identity_preselection": {
            "method": cluster_payload.get("method"),
            "selected_cluster": selected_cluster,
            "selected_candidates": int(cluster.get("size", 0)),
            "mode": mode,
        },
    }
    previous = read_job(job_id)
    if isinstance(previous.get("result"), dict) and isinstance(previous["result"].get("video"), dict):
        processing.update(previous["result"]["video"])

    composition = None
    if mode == "character_crop":
        report = _load_subject_report(data_dir / "character", data_dir / "frames")
        selected = [report.identity_dir / str(name) for name in cluster.get("frames", [])]
        composition = build_enriched_character_dataset(report, selected, training_dir)
        processing["character_detection"] = report.as_dict()
        processing["composition"] = composition.as_dict()
    else:
        training_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = data_dir / "frames"
        for index, name in enumerate(cluster.get("frames", []), start=1):
            source_image = frames_dir / str(name)
            if not source_image.is_file():
                continue
            shutil.copy2(
                source_image,
                training_dir / f"train-{index:05d}{source_image.suffix.lower()}",
            )
        if not any(training_dir.iterdir()):
            raise PipelineError("Selected full-frame identity cluster contains no available frames")

    record = workspace.add_source_from_directory(
        training_dir,
        kind="remote_video" if is_url(source) else "local_video",
        label=label,
        origin=source,
        processing=processing,
    )
    source_id = str(record["id"])
    if composition is not None:
        import_composition_records(
            workspace,
            source_id,
            [item.as_dict(root=composition.output_dir) for item in composition.records],
            selected_cluster=selected_cluster,
        )
    else:
        seed_source_defaults(workspace, source_id)

    result = {
        "dataset": workspace.name,
        "source_id": source_id,
        "label": str(record["label"]),
        "selected_cluster": selected_cluster,
        "mode": mode,
        "composition": composition.as_dict() if composition is not None else None,
    }
    shutil.rmtree(data_dir, ignore_errors=True)
    return result


def _now() -> str:
    from .web_jobs import utc_now

    return utc_now()


if __name__ == "__main__":
    main()
