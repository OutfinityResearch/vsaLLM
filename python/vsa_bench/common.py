from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(profile: str) -> dict[str, Any]:
    base_path = PROJECT_ROOT / "config" / "base.json"
    profile_path = PROJECT_ROOT / "config" / f"{profile}.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Unknown profile {profile!r}: {profile_path}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    override = json.loads(profile_path.read_text(encoding="utf-8"))
    cfg = deep_merge(base, override)
    cfg["_profile"] = profile
    cfg["model"]["vocab_size"] = int(cfg["data"]["vocab_size"])
    data_routes = int(cfg["data"]["router_product_k"]) ** 2
    routing_mode = str(cfg["model"].get("routing_mode", "vsa")).lower()
    if routing_mode not in {"vsa", "fixed_dense"}:
        raise ValueError(f"Unknown model.routing_mode: {routing_mode}")
    cfg["model"]["num_routes"] = 1 if routing_mode == "fixed_dense" else data_routes
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail early when a configuration cannot be represented by on-disk formats.

    Prepared token ids are uint16 and route ids are uint8.  Previously, larger
    experimental configurations failed much later (or silently wrapped while
    being written), which could invalidate a long training run.
    """
    vocab_size = int(cfg["data"]["vocab_size"])
    routes = int(cfg["data"]["router_product_k"]) ** 2
    if not 4 <= vocab_size <= np.iinfo(np.uint16).max:
        raise ValueError(f"vocab_size must fit uint16 and include special tokens: {vocab_size}")
    if not 1 <= routes <= np.iinfo(np.uint8).max + 1:
        raise ValueError(f"num_routes must fit uint8: {routes}")
    if int(cfg["model"]["d_model"]) % int(cfg["model"]["n_head"]):
        raise ValueError("model.d_model must be divisible by model.n_head")
    for field in (
        "block_size",
        "d_model",
        "n_layer",
        "n_head",
        "shared_hidden",
        "expert_hidden",
    ):
        if int(cfg["model"][field]) < 1:
            raise ValueError(f"model.{field} must be positive")
    dropout = float(cfg["model"].get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    for field in (
        "conditioning_prefix_characters",
        "minimum_continuation_chars",
        "tokenizer_sample_stories",
        "router_sample_stories",
        "vsa_dimension",
        "router_prefix_bytes",
        "role_period",
        "router_product_k",
        "router_kmeans_iterations",
    ):
        if int(cfg["data"][field]) < 1:
            raise ValueError(f"data.{field} must be positive")
    if int(cfg["data"]["vsa_dimension"]) % 2:
        raise ValueError("data.vsa_dimension must be even")
    validation_fraction = float(cfg["data"]["official_valid_validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("official_valid_validation_fraction must be between 0 and 1")
    if int(cfg["runtime"]["threads"]) < 1:
        raise ValueError("runtime.threads must be positive")
    training = cfg["training"]
    for field in (
        "batch_size",
        "gradient_accumulation",
        "log_every_steps",
        "checkpoint_every_target_tokens",
        "eval_every_target_tokens",
        "eval_batches",
        "eval_batch_size",
    ):
        if int(training[field]) < 1:
            raise ValueError(f"training.{field} must be positive")
    token_budgets = (
        int(training["joint_target_tokens"]),
        int(training["expert_specialization_target_tokens"]),
    )
    if any(value < 0 for value in token_budgets) or sum(token_budgets) <= 0:
        raise ValueError("training token budgets must be non-negative and not both zero")
    for peak_name, minimum_name in (
        ("learning_rate", "minimum_learning_rate"),
        ("expert_learning_rate", "expert_minimum_learning_rate"),
    ):
        peak = float(training[peak_name])
        minimum = float(training[minimum_name])
        if not 0.0 < minimum <= peak:
            raise ValueError(f"training.{minimum_name} must be in (0, {peak_name}]")
    if not 0.0 <= float(training["warmup_fraction"]) <= 0.5:
        raise ValueError("training.warmup_fraction must be in [0, 0.5]")
    if float(training["weight_decay"]) < 0.0:
        raise ValueError("training.weight_decay must be non-negative")
    if float(training["gradient_clip"]) <= 0.0:
        raise ValueError("training.gradient_clip must be positive")
    if float(training["route_sampling_exponent"]) < 0.0:
        raise ValueError("training.route_sampling_exponent must be non-negative")

    evaluation = cfg["evaluation"]
    for field in (
        "test_stories",
        "common_context_length",
        "scoring_stride",
        "prompt_count",
        "ablation_batches",
        "ablation_batch_size",
    ):
        if int(evaluation[field]) < 1:
            raise ValueError(f"evaluation.{field} must be positive")
    if int(evaluation["hellaswag_examples"]) < -1:
        raise ValueError("evaluation.hellaswag_examples must be -1, 0, or positive")
    if str(evaluation["primary_quality_context"]).lower() not in {"native", "common"}:
        raise ValueError("evaluation.primary_quality_context must be native or common")
    generation = evaluation["generation"]
    for field in ("samples_per_prompt", "max_new_tokens"):
        if int(generation[field]) < 1:
            raise ValueError(f"evaluation.generation.{field} must be positive")
    blind_index = int(evaluation["blind_sample_index"])
    if not 0 <= blind_index < int(generation["samples_per_prompt"]):
        raise ValueError("evaluation.blind_sample_index is outside generated samples")
    benchmark = cfg["benchmark"]
    targets = [int(value) for value in benchmark["prompt_character_targets"]]
    if not targets or any(value < 1 for value in targets) or len(set(targets)) != len(targets):
        raise ValueError("benchmark prompt targets must be unique positive integers")
    for field in ("decode_tokens", "repeats", "prompt_samples_per_target"):
        if int(benchmark[field]) < 1:
            raise ValueError(f"benchmark.{field} must be positive")
    if int(benchmark["repeats"]) < 2:
        raise ValueError("benchmark.repeats must be at least 2 for bootstrap reporting")
    if int(benchmark["warmup_repeats"]) < 0:
        raise ValueError("benchmark.warmup_repeats must be non-negative")
    if int(benchmark["common_context_length"]) <= int(benchmark["decode_tokens"]):
        raise ValueError("benchmark common context must exceed decode tokens")
    primary_target = int(benchmark.get("primary_prompt_character_target", max(targets)))
    if primary_target not in targets:
        raise ValueError("benchmark.primary_prompt_character_target must be benchmarked")
    if int(benchmark.get("primary_threads", 1)) < 1:
        raise ValueError("benchmark.primary_threads must be positive")
    if str(benchmark.get("primary_device", "cpu")).lower() != "cpu":
        raise ValueError("The primary CPU benchmark device must be 'cpu'")
    if str(benchmark.get("primary_dtype", "fp32")).lower() not in {
        "fp32",
        "float32",
    }:
        raise ValueError("The primary CPU benchmark dtype must be FP32")
    if str(benchmark.get("primary_machine_architecture", "x86_64")).lower() not in {
        "x86_64",
        "amd64",
    }:
        raise ValueError("The primary CPU benchmark must target x86_64/amd64")
    required_affinity = int(
        benchmark.get("required_cpu_affinity_logical_cpus", 0) or 0
    )
    if required_affinity < 0:
        raise ValueError(
            "benchmark.required_cpu_affinity_logical_cpus must be non-negative"
        )
    if not isinstance(
        benchmark.get("require_distinct_physical_cores", False), bool
    ):
        raise ValueError(
            "benchmark.require_distinct_physical_cores must be a boolean"
        )
    configured_ids = [item["id"] for item in cfg["sources"]["official_models"]]
    if len(set(configured_ids)) != len(configured_ids):
        raise ValueError("sources.official_models contains duplicate IDs")
    primary_id = cfg.get("comparison", {}).get("primary_official_model_id")
    if primary_id and primary_id not in configured_ids:
        raise ValueError(f"Primary official model is not configured: {primary_id}")
    controls = list(
        cfg.get("comparison", {}).get("matched_control_profiles", {}).values()
    )
    if len(set(controls)) != len(controls):
        raise ValueError("comparison matched-control profile names must be unique")


def apply_overrides(
    cfg: dict[str, Any],
    *,
    device: str | None = None,
    dtype: str | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    if device:
        cfg["runtime"]["device"] = device
    if dtype:
        cfg["runtime"]["dtype"] = dtype
    if threads is not None:
        cfg["runtime"]["threads"] = int(threads)
    return cfg


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_prepared_compatibility(
    checkpoint_metadata: dict[str, Any],
    prepared_metadata: dict[str, Any],
    *,
    routing_mode: str,
) -> None:
    """Reject checkpoints whose token ids or route ids have different semantics.

    Provenance is deliberately fail-closed: a legacy checkpoint without the
    preparation signature cannot establish that its token and route ids mean
    the same thing as the current prepared store.
    """
    expected_prepared = checkpoint_metadata.get("prepared_metadata")
    if not isinstance(expected_prepared, dict):
        raise RuntimeError(
            "Checkpoint lacks prepared-data provenance; retrain it with the current harness"
        )
    comparisons = {
        "preparation signature": (
            expected_prepared.get("preparation_signature"),
            prepared_metadata.get("preparation_signature"),
        ),
        "tokenizer SHA-256": (
            expected_prepared.get("tokenizer", {}).get("model_sha256"),
            prepared_metadata.get("tokenizer", {}).get("model_sha256"),
        ),
        "router SHA-256": (
            expected_prepared.get("router", {}).get("path_sha256"),
            prepared_metadata.get("router", {}).get("path_sha256"),
        ),
    }
    for label, (expected, actual) in comparisons.items():
        if not expected or not actual:
            raise RuntimeError(
                f"Cannot verify checkpoint/prepared compatibility: missing {label}"
            )
        if expected != actual:
            raise RuntimeError(f"Checkpoint and prepared data have different {label}")
    if "routing_mode" not in checkpoint_metadata:
        raise RuntimeError(
            "Checkpoint lacks routing-mode provenance; retrain it with the current harness"
        )
    expected_mode = str(checkpoint_metadata["routing_mode"])
    if expected_mode != str(routing_mode):
        raise RuntimeError(
            f"Checkpoint routing mode {expected_mode!r} != requested {routing_mode!r}"
        )


def stable_hash64(text: str, seed: int) -> int:
    key = int(seed).to_bytes(16, "little", signed=False)
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8", errors="replace"), key=key, digest_size=8).digest(),
        "little",
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_threads(threads: int) -> None:
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def select_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return device


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    name = name.lower()
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {name}")
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("FP16 CPU is intentionally disabled; use FP32 or BF16")
    return dtype


@contextlib.contextmanager
def autocast_context(device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    enabled = dtype in {torch.float16, torch.bfloat16}
    if enabled:
        with torch.autocast(device_type=device.type, dtype=dtype):
            yield
    else:
        yield


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def entropy_bits(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() <= 0:
        return 0.0
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def git_worktree_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(output.strip())
    except Exception:
        return None


def runtime_source_files() -> list[Path]:
    """Return every repository file that defines the executable protocol."""
    files: set[Path] = set()
    for root, pattern in (
        (PROJECT_ROOT / "python" / "vsa_bench", "*.py"),
        (PROJECT_ROOT / "scripts", "*.mjs"),
        (PROJECT_ROOT / "config", "*.json"),
    ):
        if root.exists():
            files.update(path for path in root.rglob(pattern) if path.is_file())
    for name in (
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dgx.txt",
        ".dockerignore",
        "containers/dgx-spark/Dockerfile",
    ):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)

    return sorted(
        files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()
    )


def runtime_source_sha256() -> str:
    """Hash the executable project sources independently of Git state."""
    files = runtime_source_files()

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _parse_linux_cpu_list(value: str) -> list[int]:
    """Parse Linux sysfs CPU-list syntax such as ``0-3,8,10-11``."""

    cpus: set[int] = set()
    for part in value.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"Invalid Linux CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"Invalid Linux CPU ID: {part}")
            cpus.add(cpu)
    return sorted(cpus)


def _cpu_topology_snapshot(
    affinity: list[int] | None,
    *,
    sysfs_cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> dict[str, Any]:
    """Read Linux topology only for logical CPUs available to this process."""

    affinity_cpus = sorted(set(int(cpu) for cpu in (affinity or [])))
    rows: list[dict[str, Any]] = []
    complete = bool(affinity_cpus)
    for logical_cpu in affinity_cpus:
        topology_root = sysfs_cpu_root / f"cpu{logical_cpu}" / "topology"
        physical_package_id: int | None = None
        core_id: int | None = None
        thread_siblings_list: list[int] | None = None
        try:
            physical_package_id = int(
                (topology_root / "physical_package_id").read_text(
                    encoding="utf-8"
                ).strip()
            )
            core_id = int(
                (topology_root / "core_id").read_text(encoding="utf-8").strip()
            )
            thread_siblings_list = _parse_linux_cpu_list(
                (topology_root / "thread_siblings_list").read_text(
                    encoding="utf-8"
                )
            )
            if logical_cpu not in thread_siblings_list:
                raise ValueError("thread_siblings_list omits its logical CPU")
        except (OSError, ValueError):
            complete = False
        rows.append(
            {
                "logical_cpu": logical_cpu,
                "physical_package_id": physical_package_id,
                "core_id": core_id,
                "thread_siblings_list": thread_siblings_list,
            }
        )

    known_pairs = {
        (int(row["physical_package_id"]), int(row["core_id"]))
        for row in rows
        if row["physical_package_id"] is not None and row["core_id"] is not None
    }
    selected = set(affinity_cpus)
    contains_smt_siblings = len(known_pairs) < len(rows)
    if complete:
        contains_smt_siblings = any(
            len(selected.intersection(row["thread_siblings_list"] or [])) > 1
            for row in rows
        )
    return {
        "logical_cpus": rows,
        "complete": complete,
        "affinity_logical_cpu_count": len(affinity_cpus),
        "affinity_physical_core_count": len(known_pairs),
        "affinity_contains_smt_siblings": contains_smt_siblings,
    }


def validate_benchmark_cpu_affinity(
    cfg: dict[str, Any],
    environment: dict[str, Any],
    *,
    label: str = "benchmark",
) -> None:
    """Fail closed on an explicitly configured physical-core CPUSET contract."""

    benchmark = cfg.get("benchmark", {})
    required = int(benchmark.get("required_cpu_affinity_logical_cpus", 0) or 0)
    require_distinct = benchmark.get("require_distinct_physical_cores", False)
    if not isinstance(require_distinct, bool):
        raise RuntimeError(
            "benchmark.require_distinct_physical_cores must be a boolean"
        )
    if required <= 0 and not require_distinct:
        return

    affinity = environment.get("cpu_affinity")
    topology = environment.get("cpu_topology")
    if not isinstance(affinity, list) or not all(
        isinstance(cpu, int) and cpu >= 0 for cpu in affinity
    ):
        raise RuntimeError(f"{label} CPU affinity is missing or malformed")
    if len(set(affinity)) != len(affinity):
        raise RuntimeError(f"{label} CPU affinity contains duplicate logical CPUs")
    if required > 0 and len(affinity) != required:
        raise RuntimeError(
            f"{label} requires exactly {required} affinity logical CPUs; "
            f"found {len(affinity)}"
        )
    if not isinstance(topology, dict) or topology.get("complete") is not True:
        raise RuntimeError(f"{label} CPU topology is incomplete")
    rows = topology.get("logical_cpus")
    if not isinstance(rows, list) or len(rows) != len(affinity):
        raise RuntimeError(f"{label} CPU topology rows do not match affinity")
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("logical_cpu"), int)
        and row["logical_cpu"] >= 0
        for row in rows
    ):
        raise RuntimeError(f"{label} CPU topology logical CPUs are malformed")
    row_ids = [row["logical_cpu"] for row in rows]
    if sorted(row_ids) != sorted(affinity) or len(set(row_ids)) != len(row_ids):
        raise RuntimeError(f"{label} CPU topology logical CPUs do not match affinity")
    pairs: set[tuple[int, int]] = set()
    selected = set(affinity)
    derived_smt = False
    for row in rows:
        package = row.get("physical_package_id")
        core = row.get("core_id")
        siblings = row.get("thread_siblings_list")
        if not isinstance(package, int) or not isinstance(core, int):
            raise RuntimeError(f"{label} CPU topology has an incomplete core identity")
        if not isinstance(siblings, list) or not all(
            isinstance(cpu, int) and cpu >= 0 for cpu in siblings
        ) or row["logical_cpu"] not in siblings:
            raise RuntimeError(f"{label} CPU topology has malformed SMT siblings")
        pair = (package, core)
        if pair in pairs:
            derived_smt = True
        pairs.add(pair)
        if len(selected.intersection(siblings)) > 1:
            derived_smt = True
    if topology.get("affinity_logical_cpu_count") != len(affinity):
        raise RuntimeError(f"{label} CPU topology logical count is inconsistent")
    if topology.get("affinity_physical_core_count") != len(pairs):
        raise RuntimeError(f"{label} CPU topology physical-core count is inconsistent")
    if topology.get("affinity_contains_smt_siblings") is not derived_smt:
        raise RuntimeError(f"{label} CPU topology SMT summary is inconsistent")
    if require_distinct and len(pairs) != len(affinity):
        raise RuntimeError(
            f"{label} requires distinct physical cores; found {len(pairs)} cores "
            f"for {len(affinity)} logical CPUs"
        )
    if require_distinct and derived_smt:
        raise RuntimeError(f"{label} affinity contains SMT siblings")


def environment_snapshot() -> dict[str, Any]:
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cpu_model = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu_model and cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    cpu_topology = _cpu_topology_snapshot(affinity)
    governors: list[str] = []
    for governor_path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    ):
        try:
            governors.append(governor_path.read_text(encoding="utf-8").strip())
        except OSError:
            pass
    dependency_versions: dict[str, str | None] = {}
    for name in (
        "sentencepiece",
        "PyYAML",
        "psutil",
        "huggingface_hub",
        "transformers",
        "datasets",
        "safetensors",
    ):
        try:
            dependency_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[name] = None
    cuda_device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    cuda_devices = []
    for index in range(cuda_device_count):
        properties = torch.cuda.get_device_properties(index)
        cuda_devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": [int(properties.major), int(properties.minor)],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    nvidia_driver_version = None
    try:
        nvidia_driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()[0].strip()
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_topology": cpu_topology,
        "cpu_frequency_governors": sorted(set(governors)),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "runtime_dependencies": dependency_versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": cuda_devices,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": nvidia_driver_version,
        "mps_available": mps,
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
        "runtime_source_sha256": runtime_source_sha256(),
        "containerized": Path("/.dockerenv").exists(),
        "container_image": os.environ.get("VSA_CONTAINER_IMAGE"),
        "container_image_digest": os.environ.get("VSA_CONTAINER_IMAGE_DIGEST"),
        "container_derived_image_id": os.environ.get("VSA_DERIVED_IMAGE_ID"),
    }


def cosine_lr(progress: float, peak: float, minimum: float, warmup_fraction: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    warmup = max(0.0, min(float(warmup_fraction), 0.5))
    if warmup and progress < warmup:
        return peak * (progress / warmup)
    q = (progress - warmup) / max(1e-12, 1.0 - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * q))
