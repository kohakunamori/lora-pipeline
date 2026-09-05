from __future__ import annotations

import json

from ..config import stable_hash, write_json_atomic
from ..models import PipelineError, StepResult
from ..target_preflight import _augment_preflight_report
from .preflight_core import run as _run_core


def run(state, *, minimum_free_gib: float = 10.0) -> StepResult:
    """Run core preflight, then explicitly compose target-aware diagnostics."""

    original_error: PipelineError | None = None
    result: StepResult | None = None
    try:
        result = _run_core(state, minimum_free_gib=minimum_free_gib)
    except PipelineError as exc:
        original_error = exc

    report_path = state.project_dir / "preflight.json"
    if not report_path.is_file():
        if original_error is not None:
            raise original_error
        assert result is not None
        return result

    report = json.loads(report_path.read_text(encoding="utf-8"))
    _augment_preflight_report(state, report)
    report["status"] = "READY" if not report.get("blocking") else "BLOCKED"
    report["input_hash"] = stable_hash(report.get("checks", {}))
    write_json_atomic(report_path, report)

    if report.get("blocking"):
        raise PipelineError("Preflight BLOCKED: " + "; ".join(report["blocking"])) from original_error

    assert result is not None
    details = dict(result.details)
    details.update(
        {
            "status": "READY",
            "warnings": list(report.get("warnings", [])),
            "checks": dict(report.get("checks", {})),
        }
    )
    return StepResult(
        status=result.status,
        input_hash=report["input_hash"],
        output_manifest=str(report_path),
        details=details,
    )
