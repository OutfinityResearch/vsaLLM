from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from .common import environment_snapshot, resolve_path
from .model import ModelConfig, VSAPathMoE, parameter_accounting


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def doctor(cfg: dict[str, Any]) -> dict[str, Any]:
    model = VSAPathMoE(ModelConfig.from_mapping(cfg["model"]))
    accounting = parameter_accounting(model)
    assets = resolve_path(cfg["paths"]["assets_manifest"])
    prepared = resolve_path(cfg["data"]["prepared_dir"])
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    return {
        "profile": cfg["_profile"],
        "environment": environment_snapshot(),
        "dependencies": {
            name: _version(name)
            for name in [
                "torch",
                "numpy",
                "sentencepiece",
                "transformers",
                "huggingface_hub",
                "datasets",
                "psutil",
                "PyYAML",
            ]
        },
        "paths": {
            "assets_manifest": {"path": str(assets), "exists": assets.exists()},
            "prepared": {"path": str(prepared), "exists": (prepared / "metadata.json").exists()},
            "run_dir": {"path": str(run_dir), "exists": run_dir.exists()},
        },
        "model": {
            "config": model.config.__dict__,
            "accounting": accounting,
            "near_10m": 9_500_000 <= accounting["total_parameters"] <= 10_500_000,
            "active_fraction_below_20_percent": accounting["active_fraction"] < 0.20,
        },
    }
