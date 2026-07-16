from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .common import atomic_json, resolve_path


def validated_official_model_asset(
    cfg: dict[str, Any], manifest: dict[str, Any], model_index: int
) -> dict[str, Any]:
    """Return one official asset only after validating the complete manifest list.

    Model indexes are part of the CLI protocol, so validating only the selected
    row would not detect a stale or reordered manifest.  In particular, the
    trained-context override must come from the frozen configuration rather
    than being silently inherited from an older download manifest.
    """
    configured = cfg.get("sources", {}).get("official_models")
    downloaded = manifest.get("official_models")
    if not isinstance(configured, list) or not isinstance(downloaded, list):
        raise RuntimeError(
            "Official model configuration and asset manifest must contain model lists"
        )
    if len(configured) != len(downloaded):
        raise RuntimeError(
            "Official model asset manifest length does not match the resolved configuration: "
            f"{len(downloaded)} != {len(configured)}"
        )

    for index, (expected, actual) in enumerate(zip(configured, downloaded)):
        expected_revision = str(expected.get("revision", "main"))
        actual_revision = actual.get("requested_revision")
        expected_context = expected.get("trained_context_length")
        actual_context = actual.get("trained_context_length")
        expected_context = None if expected_context is None else int(expected_context)
        actual_context = None if actual_context is None else int(actual_context)
        mismatches = []
        if actual.get("id") != expected.get("id"):
            mismatches.append(f"id={actual.get('id')!r} != {expected.get('id')!r}")
        if actual_revision != expected_revision:
            mismatches.append(
                f"requested_revision={actual_revision!r} != {expected_revision!r}"
            )
        if actual_context != expected_context:
            mismatches.append(
                f"trained_context_length={actual_context!r} != {expected_context!r}"
            )
        if mismatches:
            raise RuntimeError(
                f"Official model asset manifest row {index} does not match the resolved "
                f"configuration: {', '.join(mismatches)}"
            )

    model_index = int(model_index)
    if not 0 <= model_index < len(configured):
        raise IndexError(
            f"Official model index {model_index} outside [0, {len(configured)})"
        )
    selected = dict(downloaded[model_index])
    # This value is protocol metadata, not a property inferred from the
    # downloaded artifact's (often larger) positional-embedding allocation.
    selected["trained_context_length"] = configured[model_index].get(
        "trained_context_length"
    )
    return selected


def _repo_sha(repo_id: str, repo_type: str, revision: str) -> str:
    """Resolve floating revisions or fail rather than silently benchmarking `main`."""
    api = HfApi()
    if repo_type == "dataset":
        sha = api.dataset_info(repo_id, revision=revision).sha
    else:
        sha = api.model_info(repo_id, revision=revision).sha
    if not sha:
        raise RuntimeError(f"Could not resolve immutable revision for {repo_type} {repo_id}")
    return str(sha)


def download_assets(cfg: dict[str, Any], *, mode: str = "auto") -> dict[str, Any]:
    source = cfg["sources"]
    data_cfg = cfg["data"]
    manifest_path = resolve_path(cfg["paths"]["assets_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_repo = source["dataset_repo"]
    revision = source.get("dataset_revision", "main")
    dataset_sha = _repo_sha(dataset_repo, "dataset", revision)
    dataset_download_revision = dataset_sha
    use_valid_as_train = bool(data_cfg.get("use_valid_as_train", False))
    full = mode == "full" or (mode == "auto" and not use_valid_as_train)

    valid_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        revision=dataset_download_revision,
        filename=source["valid_file"],
    )
    prompts_path = hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        revision=dataset_download_revision,
        filename=source["prompts_file"],
    )
    train_path = valid_path
    if full:
        train_path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=dataset_download_revision,
            filename=source["train_file"],
        )

    models: list[dict[str, Any]] = []
    for item in source["official_models"]:
        model_id = item["id"]
        model_revision = item.get("revision", "main")
        model_sha = _repo_sha(model_id, "model", model_revision)
        model_download_revision = model_sha
        local_path = snapshot_download(
            repo_id=model_id,
            revision=model_download_revision,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.bin",
                "*.safetensors",
                "*.md",
            ],
        )
        models.append(
            {
                "id": model_id,
                "requested_revision": model_revision,
                "resolved_sha": model_sha,
                "snapshot_path": local_path,
                "trained_context_length": item.get("trained_context_length"),
            }
        )

    hellaswag: dict[str, Any] | None = None
    if (
        mode != "deployment"
        and int(cfg.get("evaluation", {}).get("hellaswag_examples", -1)) >= 0
    ):
        from datasets import load_dataset, load_from_disk

        hellaswag_id = source.get("hellaswag_dataset", "Rowan/hellaswag")
        hellaswag_revision = source.get("hellaswag_revision", "main")
        hellaswag_split = source.get("hellaswag_split", "validation")
        hellaswag_sha = _repo_sha(hellaswag_id, "dataset", hellaswag_revision)
        hellaswag_download_revision = hellaswag_sha
        hellaswag_tag = hellaswag_sha.replace("/", "_")[:16]
        hellaswag_path = manifest_path.parent / f"hellaswag_validation_{hellaswag_tag}"
        if (hellaswag_path / "dataset_info.json").exists():
            hs_dataset = load_from_disk(str(hellaswag_path))
        else:
            hs_dataset = load_dataset(
                hellaswag_id,
                split=hellaswag_split,
                revision=hellaswag_download_revision,
            )
            hellaswag_path.parent.mkdir(parents=True, exist_ok=True)
            hs_dataset.save_to_disk(str(hellaswag_path))
        hellaswag = {
            "id": hellaswag_id,
            "requested_revision": hellaswag_revision,
            "resolved_sha": hellaswag_sha,
            "split": hellaswag_split,
            "path": str(hellaswag_path.resolve()),
            "rows": int(len(hs_dataset)),
            "fingerprint": getattr(hs_dataset, "_fingerprint", None),
        }

    manifest = {
        "dataset": {
            "id": dataset_repo,
            "requested_revision": revision,
            "resolved_sha": dataset_sha,
            "train_path": str(Path(train_path).resolve()),
            "valid_path": str(Path(valid_path).resolve()),
            "prompts_path": str(Path(prompts_path).resolve()),
            "full_train_downloaded": full,
        },
        "official_models": models,
        "hellaswag": hellaswag,
        "environment": {
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest
