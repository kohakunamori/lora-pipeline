from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from .dataset_workspace import DatasetWorkspace
from .models import OptionalBackendUnavailable, PipelineError
from .service import load_project, run_remaining, run_single_step
from .state import ProjectState
from .video_character import (
    VideoSubject,
    VideoSubjectReport,
    build_balanced_character_dataset,
    detect_video_subjects,
)
from .video_identity import cluster_video_identity
from .video_source import (
    VideoAuth,
    VideoProxy,
    detect_cookies_file,
    extract_video_frames,
    is_url,
    is_youtube_url,
)
from .web_jobs import job_data_dir, read_job, update_job


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m pipeline.web_worker JOB_ID")
    job_id = sys.argv[1]
    job = read_job(job_id)
    try:
        update_job(job_id, status="running", error=None)
        kind = str(job.get("kind") or "")
        payload = dict(job.get("payload") or {})
        print(f"[web] job={job_id} kind={kind}", flush=True)
        if kind == "train":
            result = _train(payload)
            update_job(job_id, status="completed", result=result, finished_at=_now())
        elif kind == "evaluate":
            result = _evaluate(payload)
            update_job(job_id, status="completed", result=result, finished_at=_now())
        elif kind == "dataset_tag":
            result = _dataset_tag(payload)
            update_job(job_id, status="completed", result=result, finished_at=_now())
        elif kind == "video_prepare":
            _video_prepare(job_id, payload)
        elif kind == "video_finalize":
            result = _video_finalize(job_id, payload)
            update_job(job_id, status="completed", result=result, finished_at=_now())
        else:
            raise PipelineError(f"Unknown web worker kind: {kind}")
    except BaseException as exc:
        traceback.print_exc()
        update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_now(),
        )
        raise


def _train(payload: dict[str, Any]) -> dict[str, Any]:
    project = str(payload["project"])
    state = load_project(project)
    preferences = dict(state.payload["project"].get("interactive_preferences") or {})
    skip = {"evaluate"}
    if not bool(preferences.get("run_dedup", True)):
        skip.add("dedup")
    if state.concept_type != "character" or not bool(preferences.get("run_identity", True)):
        skip.add("identity")
    caption_mode = str(preferences.get("caption_mode", "generate"))
    if caption_mode == "skip":
        skip.add("caption")
    if not bool(preferences.get("run_review", True)):
        skip.add("review")

    interrupted = [
        record
        for record in state.payload.get("runs", [])
        if str(record.get("status")) == "interrupted"
    ]
    resume_run = str(interrupted[-1]["id"]) if interrupted else None
    print(
        f"[web] guided plan project={project} skip={sorted(skip)} resume={resume_run or '-'}",
        flush=True,
    )

    def on_step(step: str) -> None:
        fresh = ProjectState.load(state.project_dir)
        print(f"[web] step={step} current={fresh.status(step).value}", flush=True)

    results = run_remaining(
        state,
        skip=skip,
        caption_mode=caption_mode,
        exclude_exact=bool(preferences.get("exclude_exact_duplicates", False)),
        allow_trigger_only=bool(preferences.get("allow_trigger_only", False)),
        resume_run=resume_run,
        verbose=1,
        on_step=on_step,
    )
    fresh = load_project(project)
    run = fresh.payload.get("runs", [])[-1] if fresh.payload.get("runs") else None
    return {
        "project": project,
        "steps": [name for name, _result in results],
        "next_step": fresh.next_actionable_step(),
        "run_id": str(run.get("id")) if isinstance(run, dict) and run.get("id") else None,
        "run_status": str(run.get("status")) if isinstance(run, dict) else None,
    }


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    project = str(payload["project"])
    stage = str(payload.get("stage") or "screening")
    run_id = str(payload["run_id"])
    checkpoints = [str(value) for value in payload.get("checkpoints", [])]
    print(
        f"[web] evaluate project={project} run={run_id} stage={stage} checkpoints={checkpoints or 'profile selection'}",
        flush=True,
    )
    result = run_single_step(
        load_project(project),
        "evaluate",
        force=bool(payload.get("force", False)),
        verbose=1,
        evaluation_stage=stage,
        evaluation_run=run_id,
        evaluation_checkpoints=checkpoints if stage == "full" else [],
    )
    return {"project": project, "run_id": run_id, "stage": stage, "details": dict(result.details)}


def _dataset_tag(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = DatasetWorkspace.load(str(payload["dataset"]))
    source_id = str(payload.get("source_id") or "") or None
    threshold = float(payload.get("threshold", 0.35))
    overwrite = bool(payload.get("overwrite", False))
    print(
        f"[web] tag dataset={workspace.name} source={source_id or 'all'} threshold={threshold} overwrite={overwrite}",
        flush=True,
    )
    return workspace.auto_tag(
        source_id=source_id,
        threshold=threshold,
        overwrite=overwrite,
    )


def _video_prepare(job_id: str, payload: dict[str, Any]) -> None:
    workspace = DatasetWorkspace.load(str(payload["dataset"]))
    source = str(payload["source"])
    interval = int(payload.get("interval_seconds", 2))
    max_frames = int(payload.get("max_frames", 250))
    data_dir = job_data_dir(job_id)
    frames = data_dir / "frames"
    character_dir = data_dir / "character"

    proxy_mode = str(payload.get("proxy_mode") or "environment")
    proxy_url = str(payload.get("proxy_url") or "") or None
    proxy = VideoProxy(mode=proxy_mode, url=proxy_url)
    cookies_path = str(payload.get("cookies_path") or "").strip()
    if not cookies_path and is_url(source) and is_youtube_url(source):
        _origin, detected = detect_cookies_file()
        if detected is not None:
            cookies_path = str(detected)
    auth = VideoAuth(mode="cookies", cookies_path=cookies_path) if cookies_path else VideoAuth()

    print(f"[web] extracting video source={source} interval={interval}s max={max_frames}", flush=True)
    report = extract_video_frames(
        source,
        frames,
        interval_seconds=interval,
        max_frames=max_frames,
        proxy=proxy,
        auth=auth,
    )
    report_payload = report.as_dict()
    report_payload.pop("downloaded_video", None)

    mode = "character_crop"
    subject_payload: dict[str, Any] | None = None
    try:
        print("[web] detecting anime characters on high-resolution frames", flush=True)
        subjects = detect_video_subjects(
            frames,
            character_dir,
            interval_seconds=interval,
        )
        identity_dir = subjects.identity_dir
        subject_payload = subjects.as_dict(include_records=True)
    except OptionalBackendUnavailable as exc:
        print(f"[web] DeepGHS detector unavailable; falling back to full-frame CCIP: {exc}", flush=True)
        mode = "full_frame"
        identity_dir = frames

    print(f"[web] clustering identity candidates mode={mode}", flush=True)
    identity = cluster_video_identity(identity_dir)
    clusters = []
    for cluster in identity.clusters:
        clusters.append(
            {
                "cluster_id": cluster.cluster_id,
                "size": cluster.size,
                "frames": [path.name for path in cluster.frames],
                "representatives": [
                    str(path.relative_to(data_dir)) if path.is_relative_to(data_dir) else path.name
                    for path in cluster.representatives
                ],
            }
        )
    cluster_payload = {
        "method": identity.method,
        "mode": mode,
        "total": identity.total_frames,
        "outliers": len(identity.outliers),
        "clusters": clusters,
    }
    (data_dir / "clusters.json").write_text(
        json.dumps(cluster_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "dataset": workspace.name,
        "video": report_payload,
        "mode": mode,
        "subjects": subject_payload,
        "identity": cluster_payload,
    }
    update_job(
        job_id,
        status="awaiting_identity",
        result=result,
        pid=None,
        finished_at=_now(),
    )
    print("[web] waiting for target identity selection in browser", flush=True)


def _video_finalize(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    workspace = DatasetWorkspace.load(str(payload["dataset"]))
    selected_cluster = int(payload["selected_cluster"])
    source = str(payload["source"])
    label = str(payload.get("label") or "video").strip() or "video"
    data_dir = job_data_dir(job_id)
    cluster_payload = json.loads((data_dir / "clusters.json").read_text(encoding="utf-8"))
    cluster = next(
        (value for value in cluster_payload.get("clusters", []) if int(value.get("cluster_id")) == selected_cluster),
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

    if mode == "character_crop":
        report = _load_subject_report(data_dir / "character", data_dir / "frames")
        selected = [report.identity_dir / str(name) for name in cluster.get("frames", [])]
        composition = build_balanced_character_dataset(report, selected, training_dir)
        processing["character_detection"] = report.as_dict()
        processing["composition"] = composition.as_dict()
    else:
        training_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = data_dir / "frames"
        for index, name in enumerate(cluster.get("frames", []), start=1):
            source_image = frames_dir / str(name)
            if not source_image.is_file():
                continue
            shutil.copy2(source_image, training_dir / f"train-{index:05d}{source_image.suffix.lower()}")
        if not any(training_dir.iterdir()):
            raise PipelineError("Selected full-frame identity cluster contains no available frames")

    record = workspace.add_source_from_directory(
        training_dir,
        kind="remote_video" if is_url(source) else "local_video",
        label=label,
        origin=source,
        processing=processing,
    )
    result = {
        "dataset": workspace.name,
        "source_id": str(record["id"]),
        "label": str(record["label"]),
        "selected_cluster": selected_cluster,
        "mode": mode,
    }
    # The Dataset now owns copies of the final images. Remove large temporary video
    # frames so a web import does not silently double NAS storage usage.
    shutil.rmtree(data_dir, ignore_errors=True)
    return result


def _load_subject_report(character_dir: Path, frames_dir: Path) -> VideoSubjectReport:
    path = character_dir / "subjects.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity_dir = character_dir / "identity"
    subjects: list[VideoSubject] = []
    for item in payload.get("subjects", []):
        subjects.append(
            VideoSubject(
                subject_id=str(item["subject_id"]),
                identity_path=identity_dir / Path(str(item["identity_image"])).name,
                source_frame=frames_dir / str(item["source_frame"]),
                source_timestamp_seconds=(
                    float(item["source_timestamp_seconds"])
                    if item.get("source_timestamp_seconds") is not None
                    else None
                ),
                source_resolution=tuple(int(value) for value in item["source_resolution"]),
                person_bbox=tuple(int(value) for value in item["person_bbox"]),
                head_bbox=(
                    tuple(int(value) for value in item["head_bbox"])
                    if item.get("head_bbox") is not None
                    else None
                ),
                halfbody_bbox=(
                    tuple(int(value) for value in item["halfbody_bbox"])
                    if item.get("halfbody_bbox") is not None
                    else None
                ),
                person_score=float(item["person_score"]) if item.get("person_score") is not None else None,
                head_score=float(item["head_score"]) if item.get("head_score") is not None else None,
                halfbody_score=float(item["halfbody_score"]) if item.get("halfbody_score") is not None else None,
                detection_kind=str(item["detection_kind"]),
                quality_tier=str(item["quality_tier"]),
                frame_subject_count=int(item["frame_subject_count"]),
                native_identity_resolution=tuple(int(value) for value in item["native_identity_resolution"]),
                saved_identity_resolution=tuple(int(value) for value in item["saved_identity_resolution"]),
            )
        )
    return VideoSubjectReport(
        identity_dir=identity_dir,
        subjects=tuple(subjects),
        total_frames=int(payload["total_frames"]),
        frames_with_subjects=int(payload["frames_with_subjects"]),
        detected_persons=int(payload["detected_persons"]),
        head_fallbacks=int(payload["head_fallbacks"]),
        rejected_low_resolution=int(payload["rejected_low_resolution"]),
        detection_proxy_long_edge=int(payload["detection_proxy_long_edge"]),
        minimum_person_height=int(payload["minimum_person_height"]),
        minimum_head_size=int(payload["minimum_head_size"]),
        maximum_saved_long_edge=int(payload["maximum_saved_long_edge"]),
        maximum_saved_pixels=int(payload["maximum_saved_pixels"]),
    )


def _now() -> str:
    from .web_jobs import utc_now

    return utc_now()


if __name__ == "__main__":
    main()
