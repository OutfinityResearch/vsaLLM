from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil
import torch

from .common import PROJECT_ROOT, environment_snapshot


def _locked_dependencies() -> tuple[dict[str, str], dict[str, str | None]]:
    expected: dict[str, str] = {}
    lock_path = PROJECT_ROOT / "requirements-dgx.txt"
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, pinned = line.partition("==")
        if not separator or not name or not pinned:
            raise RuntimeError(f"DGX dependency is not exactly pinned: {line}")
        expected[name] = pinned
    observed: dict[str, str | None] = {}
    for name in expected:
        try:
            observed[name] = version(name)
        except PackageNotFoundError:
            observed[name] = None
    return expected, observed


def _node_version() -> str | None:
    try:
        return subprocess.check_output(
            ["node", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dgx_spark_doctor(cfg: dict[str, Any]) -> dict[str, Any]:
    machine = platform.machine().lower()
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
    memory_bytes = int(psutil.virtual_memory().total)
    disk_free_bytes = int(shutil.disk_usage(PROJECT_ROOT).free)
    environment = environment_snapshot()
    digest = os.environ.get("VSA_CONTAINER_IMAGE_DIGEST", "")
    derived_image_id = os.environ.get("VSA_DERIVED_IMAGE_ID", "")
    dependency_expected, dependency_observed = _locked_dependencies()
    node_version = _node_version()
    dependency_mismatches = {
        name: {"expected": expected, "observed": dependency_observed[name]}
        for name, expected in dependency_expected.items()
        if dependency_observed[name] != expected
    }
    source_revisions = {
        "dataset": cfg["sources"].get("dataset_revision"),
        "hellaswag": cfg["sources"].get("hellaswag_revision"),
        **{
            f"official:{item['id']}": item.get("revision")
            for item in cfg["sources"].get("official_models", [])
        },
    }
    floating_revisions = {
        name: revision
        for name, revision in source_revisions.items()
        if not re.fullmatch(r"[0-9a-f]{40,64}", str(revision or ""))
    }

    checks = {
        "arm64_host": {
            "passed": machine in {"aarch64", "arm64"},
            "observed": machine,
            "required": "aarch64/arm64",
        },
        "cuda_available": {
            "passed": cuda_available,
            "observed": cuda_available,
            "required": True,
        },
        "bf16_supported": {
            "passed": bf16_supported,
            "observed": bf16_supported,
            "required": True,
        },
        "gb10_gpu": {
            "passed": bool(gpu_name and "GB10" in gpu_name.upper()),
            "observed": gpu_name,
            "required": "NVIDIA GB10",
        },
        "blackwell_compute_capability": {
            "passed": bool(
                environment["cuda_devices"]
                and environment["cuda_devices"][0]["compute_capability"] == [12, 1]
            ),
            "observed": (
                environment["cuda_devices"][0]["compute_capability"]
                if environment["cuda_devices"]
                else None
            ),
            "required": [12, 1],
        },
        "system_memory": {
            "passed": memory_bytes >= 110_000_000_000,
            "observed_bytes": memory_bytes,
            "required_minimum_bytes": 110_000_000_000,
        },
        "disk_free": {
            "passed": disk_free_bytes >= 20_000_000_000,
            "observed_bytes": disk_free_bytes,
            "required_minimum_bytes": 20_000_000_000,
        },
        "clean_git_worktree": {
            "passed": environment["git_worktree_dirty"] is False,
            "observed": environment["git_worktree_dirty"],
            "required": False,
        },
        "containerized": {
            "passed": environment["containerized"] is True,
            "observed": environment["containerized"],
            "required": True,
        },
        "ngc_container_tag": {
            "passed": environment["container_image"] == "nvcr.io/nvidia/pytorch:25.11-py3",
            "observed": environment["container_image"],
            "required": "nvcr.io/nvidia/pytorch:25.11-py3",
        },
        "pinned_container_digest": {
            "passed": bool(re.search(r"(?:^|@)sha256:[0-9a-f]{64}$", digest)),
            "observed": digest or None,
            "required": "sha256:<64 hex characters>",
        },
        "derived_image_id": {
            "passed": bool(re.fullmatch(r"sha256:[0-9a-f]{64}", derived_image_id)),
            "observed": derived_image_id or None,
            "required": "sha256:<64 hex characters>",
        },
        "locked_python_dependencies": {
            "passed": not dependency_mismatches,
            "observed": dependency_observed,
            "required": dependency_expected,
            "mismatches": dependency_mismatches,
        },
        "immutable_source_revisions": {
            "passed": not floating_revisions,
            "observed": source_revisions,
            "required": "40-64 hexadecimal commit SHA for every dataset/model",
            "mismatches": floating_revisions,
        },
        "node_runtime": {
            "passed": bool(
                node_version
                and re.fullmatch(r"v(2[2-9]|[3-9][0-9])(?:\.[0-9]+){2}", node_version)
            ),
            "observed": node_version,
            "required": "Node.js >=22",
        },
        "profile_runtime": {
            "passed": (
                str(cfg["runtime"]["device"]).lower() == "cuda"
                and str(cfg["runtime"]["dtype"]).lower() in {"bf16", "bfloat16"}
            ),
            "observed": dict(cfg["runtime"]),
            "required": {"device": "cuda", "dtype": "bf16"},
        },
    }
    return {
        "profile": cfg["_profile"],
        "ready": all(item["passed"] for item in checks.values()),
        "gpu_name": gpu_name,
        "checks": checks,
        "environment": environment,
        "notes": [
            "Unified-memory capacity is checked through system RAM, not nvidia-smi VRAM.",
            "The official run must use a recorded ARM64 NGC image digest.",
        ],
    }
