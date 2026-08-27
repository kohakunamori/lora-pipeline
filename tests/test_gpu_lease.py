from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.models import PipelineError
from pipeline.trainer import sd_scripts


def test_command_gpu_lease_acquires_and_releases(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sd_scripts.subprocess, "run", fake_run)
    lease = sd_scripts.gpu_lease_from_info(
        {
            "gpu_lease": {
                "acquire_command": ["gpu-lease", "acquire"],
                "release_command": ["gpu-lease", "release"],
            }
        }
    )
    with lease:
        assert calls == [["gpu-lease", "acquire"]]
    assert calls == [["gpu-lease", "acquire"], ["gpu-lease", "release"]]


def test_command_gpu_lease_releases_inprocess_resources_before_acquire(monkeypatch) -> None:
    events: list[str] = []

    def fake_release_resources() -> tuple[str, ...]:
        events.append("release-resources")
        return ()

    def fake_run(command, **kwargs):
        events.append("command:" + command[1])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        sd_scripts.gpu_resources,
        "release_inprocess_gpu_resources",
        fake_release_resources,
    )
    monkeypatch.setattr(sd_scripts.subprocess, "run", fake_run)
    lease = sd_scripts.CommandGpuLease(["gpu-lease", "acquire"], ["gpu-lease", "release"])

    with lease:
        assert events == ["release-resources", "command:acquire"]

    assert events == ["release-resources", "command:acquire", "command:release"]


def test_null_gpu_lease_also_releases_inprocess_resources(monkeypatch) -> None:
    releases: list[bool] = []
    monkeypatch.setattr(
        sd_scripts.gpu_resources,
        "release_inprocess_gpu_resources",
        lambda: releases.append(True),
    )

    with sd_scripts.NullGpuLease():
        pass

    assert releases == [True]


def test_command_gpu_lease_reports_acquire_failure(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=75, stdout="", stderr="GPU is busy")

    monkeypatch.setattr(sd_scripts.subprocess, "run", fake_run)
    lease = sd_scripts.CommandGpuLease(["gpu-lease", "acquire"], ["gpu-lease", "release"])
    with pytest.raises(PipelineError, match="GPU is busy"):
        lease.__enter__()


def test_gpu_lease_requires_both_commands() -> None:
    with pytest.raises(PipelineError, match="requires both"):
        sd_scripts.gpu_lease_from_info(
            {"gpu_lease": {"acquire_command": ["gpu-lease", "acquire"]}}
        )
