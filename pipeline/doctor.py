from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .config import load_base_registry, repository_root


def run_doctor(*, root: Path | None = None) -> dict[str, Any]:
    root = root or repository_root()
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any, *, required: bool = True) -> None:
        checks.append({"name": name, "status": "PASS" if ok else ("FAIL" if required else "WARN"), "required": required, "detail": detail})

    record("Python", sys.version_info >= (3, 11), {"version": sys.version.split()[0], "executable": sys.executable})
    torch = None
    try:
        import torch as torch_module

        torch = torch_module
        record("PyTorch", True, torch.__version__)
    except BaseException as exc:
        record("PyTorch", False, f"{type(exc).__name__}: {exc}")
    if torch is not None:
        cuda = bool(torch.cuda.is_available())
        record("CUDA available", cuda, cuda)
        if cuda:
            capability = torch.cuda.get_device_capability()
            arches = list(torch.cuda.get_arch_list())
            properties = torch.cuda.get_device_properties(0)
            record(
                "GPU",
                "V100" in torch.cuda.get_device_name(0),
                {"name": torch.cuda.get_device_name(0), "vram_gib": round(properties.total_memory / 1024**3, 3)},
            )
            record("Compute capability", capability[0] == 7, f"{capability[0]}.{capability[1]}")
            record("sm_70 build", "sm_70" in arches, arches)
            try:
                left = torch.randn((1, 2, 8, 64), device="cuda", dtype=torch.float16, requires_grad=True)
                right = torch.randn((1, 2, 8, 64), device="cuda", dtype=torch.float16, requires_grad=True)
                value = torch.nn.functional.scaled_dot_product_attention(left, right, right).mean()
                value.backward()
                record("FP16 SDPA forward/backward", True, "ok")
                del left, right, value
                torch.cuda.empty_cache()
            except BaseException as exc:
                record("FP16 SDPA forward/backward", False, f"{type(exc).__name__}: {exc}")
    info_path = root / "environment" / "environment-info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        sd_scripts = Path(info["sd_scripts_path"])
        entrypoint = sd_scripts / "sdxl_train_network.py"
        record("sd-scripts entrypoint", entrypoint.is_file(), str(entrypoint))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=sd_scripts, check=False, capture_output=True, text=True
        )
        actual_commit = commit.stdout.strip() if commit.returncode == 0 else None
        record(
            "sd-scripts pinned commit",
            actual_commit == info.get("sd_scripts_commit"),
            {"expected": info.get("sd_scripts_commit"), "actual": actual_commit},
        )
    except BaseException as exc:
        record("Validated environment record", False, f"{type(exc).__name__}: {exc}")
        info = {}
    lease = info.get("gpu_lease")
    if lease:
        commands = (
            lease.get("acquire_command") if isinstance(lease, dict) else None,
            lease.get("release_command") if isinstance(lease, dict) else None,
        )
        valid = all(
            isinstance(command, list)
            and bool(command)
            and all(isinstance(part, str) and bool(part) for part in command)
            for command in commands
        )
        record(
            "Optional GPU lease hook",
            valid,
            "configured with shell-free command arrays" if valid else "invalid command configuration",
            required=False,
        )
    else:
        record(
            "Optional GPU lease hook",
            False,
            "not configured; ensure exclusive GPU access before training",
            required=False,
        )
    _import_check("bitsandbytes", record)
    try:
        ort = importlib.import_module("onnxruntime")
        providers = ort.get_available_providers()
        record("ONNX Runtime CUDA provider", "CUDAExecutionProvider" in providers, providers)
    except BaseException as exc:
        record("ONNX Runtime CUDA provider", False, f"{type(exc).__name__}: {exc}")
    try:
        importlib.import_module("imgutils")
        importlib.import_module("imgutils.tagging")
        record("imgutils/tagger backend", True, "importable")
    except BaseException as exc:
        record("imgutils/tagger backend", False, f"{type(exc).__name__}: {exc}")
    cache_root = root / ".cache" / "huggingface" / "hub"
    tagger_cached = any(cache_root.glob("models--SmilingWolf--wd-eva02-large-tagger-v3*")) if cache_root.exists() else False
    record(
        "WD EVA02-Large Tagger v3 cache",
        tagger_cached,
        "cached" if tagger_cached else "not cached yet; first caption run may download it",
        required=False,
    )
    try:
        bases = load_base_registry(root)
        missing = [f"{base_id}: {base.path}" for base_id, base in bases.items() if base.enabled and not base.path.is_file()]
        record(
            "Registered base paths",
            bool(bases) and not missing,
            missing or (f"{len(bases)} registered" if bases else "no base models registered"),
        )
    except BaseException as exc:
        record("Registered base paths", False, f"{type(exc).__name__}: {exc}")
    projects = root / "projects"
    try:
        projects.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=projects, prefix=".doctor-", delete=True):
            pass
        record("Projects path writable", True, str(projects))
    except OSError as exc:
        record("Projects path writable", False, f"{type(exc).__name__}: {exc}")
    usage = shutil.disk_usage(root)
    free_gib = round(usage.free / 1024**3, 3)
    record("Free disk", free_gib >= 10, f"{free_gib} GiB")
    required_failures = [check for check in checks if check["required"] and check["status"] == "FAIL"]
    return {"status": "PASS" if not required_failures else "FAIL", "checks": checks, "required_failures": len(required_failures)}


def _import_check(name: str, record: Any) -> None:
    try:
        module = importlib.import_module(name)
        record(name, True, getattr(module, "__version__", "importable"))
    except BaseException as exc:
        record(name, False, f"{type(exc).__name__}: {exc}")
