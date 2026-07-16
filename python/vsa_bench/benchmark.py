from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .adapters import CustomAdapter, OfficialAdapter
from .assets import validated_official_model_asset
from .common import (
    atomic_json,
    autocast_context,
    configure_threads,
    environment_snapshot,
    resolve_path,
    seed_all,
    synchronize,
    validate_benchmark_cpu_affinity,
)
from .evaluate import load_prompts


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(np.mean(array)),
        "p10_ms": float(np.percentile(array, 10)),
        "p90_ms": float(np.percentile(array, 90)),
        "min_ms": float(np.min(array)),
        "max_ms": float(np.max(array)),
    }


def _make_prompt(base: str, target_chars: int) -> str:
    base = base.strip() or "Once upon a time,"
    pieces: list[str] = []
    length = 0
    while length < target_chars:
        pieces.append(base)
        length += len(base) + (1 if len(pieces) > 1 else 0)
    return " ".join(pieces)[:target_chars]


def _prompt_workload(
    prompts: list[str], targets: list[int], samples_per_target: int
) -> list[dict[str, Any]]:
    """Build a deterministic workload with distinct rendered prompts per target.

    Target zero starts with source prompt zero, target one with source prompt one,
    and so on.  Scanning forward preserves the historical one-sample mapping and
    makes increasing ``samples_per_target`` append cases instead of replacing the
    first case.  Distinctness is checked after rendering because different source
    strings can become identical once truncated to a short target.
    """

    if not prompts:
        raise ValueError("At least one benchmark prompt is required")
    samples_per_target = int(samples_per_target)
    if samples_per_target < 1:
        raise ValueError("benchmark.prompt_samples_per_target must be positive")

    workload: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        target = int(target)
        if target < 1:
            raise ValueError("Benchmark prompt character targets must be positive")
        seen: set[str] = set()
        selected: list[tuple[int, str]] = []
        start = target_index % len(prompts)
        for offset in range(len(prompts)):
            source_index = (start + offset) % len(prompts)
            prompt = _make_prompt(prompts[source_index], target)
            if prompt in seen:
                continue
            seen.add(prompt)
            selected.append((source_index, prompt))
            if len(selected) == samples_per_target:
                break
        if len(selected) != samples_per_target:
            raise ValueError(
                f"Target {target} yields only {len(selected)} distinct benchmark "
                f"prompts, fewer than prompt_samples_per_target={samples_per_target}"
            )
        for sample_index, (source_index, prompt) in enumerate(selected):
            workload.append(
                {
                    "prompt": prompt,
                    "prompt_target_characters": target,
                    "prompt_characters": len(prompt),
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "prompt_sample_index": sample_index,
                    "prompt_source_index": source_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
    return workload


def _peak_rss_hwm(
    process: psutil.Process, memory_info: Any | None = None
) -> tuple[int | None, str | None]:
    """Return the process-lifetime resident-set high-water mark when available."""

    info = memory_info if memory_info is not None else process.memory_info()
    peak_wset = getattr(info, "peak_wset", None)
    if peak_wset is not None:
        return int(peak_wset), "psutil.memory_info.peak_wset"

    # Linux exposes VmHWM without requiring a sampling thread.  Reading it only
    # at stage boundaries cannot perturb the timed inference region.
    status_path = Path(f"/proc/{process.pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("VmHWM:"):
                continue
            parts = line.split()
            value = int(parts[1])
            unit = parts[2].lower() if len(parts) > 2 else "b"
            multiplier = {"b": 1, "kb": 1024, "kib": 1024}.get(unit)
            if multiplier is not None:
                return value * multiplier, "proc_status.VmHWM"
    except (OSError, ValueError, IndexError):
        pass

    # getrusage is limited to the current process.  macOS reports bytes while
    # Linux and the BSDs report KiB.
    if process.pid == os.getpid():
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if value > 0:
                multiplier = 1 if sys.platform == "darwin" else 1024
                return value * multiplier, "resource.getrusage.ru_maxrss"
        except (ImportError, OSError, ValueError):
            pass
    return None, None


def _memory_snapshot(process: psutil.Process, stage: str) -> dict[str, Any]:
    """Capture memory at a stage boundary, without background polling."""

    info = process.memory_info()
    peak, peak_source = _peak_rss_hwm(process, info)
    uss: int | None = None
    pss: int | None = None
    try:
        full = process.memory_full_info()
        if getattr(full, "uss", None) is not None:
            uss = int(full.uss)
        if getattr(full, "pss", None) is not None:
            pss = int(full.pss)
    except (AttributeError, NotImplementedError, OSError, psutil.Error):
        pass
    return {
        "stage": stage,
        "rss_bytes": int(info.rss),
        "uss_bytes": uss,
        "pss_bytes": pss,
        "peak_rss_hwm_bytes": peak,
        "peak_rss_hwm_source": peak_source,
    }


def _memory_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for field in ("rss_bytes", "uss_bytes", "pss_bytes", "peak_rss_hwm_bytes"):
        left = before.get(field)
        right = after.get(field)
        name = field.removesuffix("_bytes") + "_delta_bytes"
        result[name] = int(right) - int(left) if left is not None and right is not None else None
    return result


def _memory_report(
    before_load: dict[str, Any],
    after_load: dict[str, Any],
    after_benchmark: dict[str, Any],
) -> dict[str, Any]:
    return {
        "measurement_method": (
            "boundary snapshots only; no polling thread; peak RSS is the OS process-lifetime "
            "high-water mark when available"
        ),
        "stage_semantics": {
            "before_load": "after garbage collection, immediately before adapter load",
            "after_load": (
                "immediately after adapter load, before prompt loading or warmup; loaded "
                "steady-state snapshot"
            ),
            "after_benchmark": "after all warm, measured, and cold benchmark requests",
        },
        "snapshots": {
            "before_load": before_load,
            "after_load": after_load,
            "after_benchmark": after_benchmark,
        },
        "deltas": {
            "load": _memory_delta(before_load, after_load),
            "benchmark": _memory_delta(after_load, after_benchmark),
            "total": _memory_delta(before_load, after_benchmark),
        },
    }


def _validate_prompt(
    adapter: CustomAdapter | OfficialAdapter,
    prompt: str,
    decode_tokens: int,
    common_context: int,
) -> tuple[list[int], int]:
    ids = adapter.encode(prompt)
    effective_context = min(int(common_context), adapter.context_length)
    maximum_prompt_tokens = effective_context - int(decode_tokens)
    sequence = [adapter.bos_id(), *ids]
    if len(sequence) > maximum_prompt_tokens:
        raise RuntimeError(
            f"Benchmark prompt has {len(sequence)} tokens for {adapter.name}, but only "
            f"{maximum_prompt_tokens} fit before {decode_tokens} decode steps. Reduce "
            "benchmark.prompt_character_targets. Inputs are never silently truncated."
        )
    return sequence, effective_context


@torch.inference_mode()
def _custom_request(
    adapter: CustomAdapter,
    prompt: str,
    decode_tokens: int,
    common_context: int,
    *,
    cold_route: bool,
) -> dict[str, Any]:
    synchronize(adapter.device)
    t0 = time.perf_counter_ns()
    ids = adapter.encode(prompt)
    t1 = time.perf_counter_ns()
    route = adapter.route(prompt)
    t2 = time.perf_counter_ns()
    if cold_route:
        adapter.model.clear_fused()
    adapter.model.prepare_route(route)
    t3 = time.perf_counter_ns()
    effective_context = min(int(common_context), adapter.context_length)
    sequence = [adapter.bos_id(), *ids]
    maximum = effective_context - int(decode_tokens)
    if len(sequence) > maximum:
        raise RuntimeError(
            f"Prompt does not fit custom common context: {len(sequence)} > {maximum}"
        )
    input_ids = torch.tensor([sequence], dtype=torch.long, device=adapter.device)
    with autocast_context(adapter.device, adapter.dtype):
        logits, _, cache = adapter.model(
            input_ids, route, use_cache=True, kernel_mode=adapter.kernel_mode
        )
    synchronize(adapter.device)
    t4 = time.perf_counter_ns()
    generated: list[int] = []
    for step in range(int(decode_tokens)):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_id.item()))
        if step + 1 >= int(decode_tokens):
            break
        with autocast_context(adapter.device, adapter.dtype):
            logits, _, cache = adapter.model(
                next_id,
                route,
                past_key_values=cache,
                use_cache=True,
                kernel_mode=adapter.kernel_mode,
            )
    synchronize(adapter.device)
    t5 = time.perf_counter_ns()
    text = adapter.decode(generated)
    t6 = time.perf_counter_ns()
    return {
        "prompt_tokens": len(sequence),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "generated_tokens": len(generated),
        "generated_utf8_bytes": len(text.encode("utf-8", errors="replace")),
        "route": route,
        "common_context": effective_context,
        "input_truncated": False,
        "tokenize_ms": (t1 - t0) / 1e6,
        "route_ms": (t2 - t1) / 1e6,
        "prepare_route_ms": (t3 - t2) / 1e6,
        "prefill_ms": (t4 - t3) / 1e6,
        "decode_ms": (t5 - t4) / 1e6,
        "detokenize_ms": (t6 - t5) / 1e6,
        "end_to_end_ms": (t6 - t0) / 1e6,
    }


@torch.inference_mode()
def _official_request(
    adapter: OfficialAdapter,
    prompt: str,
    decode_tokens: int,
    common_context: int,
) -> dict[str, Any]:
    synchronize(adapter.device)
    t0 = time.perf_counter_ns()
    ids = adapter.encode(prompt)
    t1 = time.perf_counter_ns()
    effective_context = min(int(common_context), adapter.context_length)
    sequence = [adapter.bos_id(), *ids]
    maximum = effective_context - int(decode_tokens)
    if len(sequence) > maximum:
        raise RuntimeError(
            f"Prompt does not fit official common context: {len(sequence)} > {maximum}"
        )
    input_ids = torch.tensor([sequence], dtype=torch.long, device=adapter.device)
    with autocast_context(adapter.device, adapter.dtype):
        output = adapter.model(input_ids=input_ids, use_cache=True)
    synchronize(adapter.device)
    logits, cache = output.logits, output.past_key_values
    t2 = time.perf_counter_ns()
    generated: list[int] = []
    for step in range(int(decode_tokens)):
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_id.item()))
        if step + 1 >= int(decode_tokens):
            break
        with autocast_context(adapter.device, adapter.dtype):
            output = adapter.model(
                input_ids=next_id, past_key_values=cache, use_cache=True
            )
        logits, cache = output.logits, output.past_key_values
    synchronize(adapter.device)
    t3 = time.perf_counter_ns()
    text = adapter.decode(generated)
    t4 = time.perf_counter_ns()
    return {
        "prompt_tokens": len(sequence),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "generated_tokens": len(generated),
        "generated_utf8_bytes": len(text.encode("utf-8", errors="replace")),
        "route": None,
        "common_context": effective_context,
        "input_truncated": False,
        "tokenize_ms": (t1 - t0) / 1e6,
        "route_ms": 0.0,
        "prepare_route_ms": 0.0,
        "prefill_ms": (t2 - t1) / 1e6,
        "decode_ms": (t3 - t2) / 1e6,
        "detokenize_ms": (t4 - t3) / 1e6,
        "end_to_end_ms": (t4 - t0) / 1e6,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "tokenize_ms",
        "route_ms",
        "prepare_route_ms",
        "prefill_ms",
        "decode_ms",
        "detokenize_ms",
        "end_to_end_ms",
    ]
    result = {field: _summary([float(row[field]) for row in rows]) for field in fields}
    decode_ms = result["decode_ms"]["median_ms"]
    e2e_ms = result["end_to_end_ms"]["median_ms"]
    generated_tokens = float(np.median([row["generated_tokens"] for row in rows]))
    generated_bytes = float(np.median([row["generated_utf8_bytes"] for row in rows]))
    result["derived"] = {
        "decode_tokens_per_second": generated_tokens / max(decode_ms / 1000.0, 1e-12),
        "end_to_end_generated_tokens_per_second": generated_tokens
        / max(e2e_ms / 1000.0, 1e-12),
        "end_to_end_generated_bytes_per_second": generated_bytes
        / max(e2e_ms / 1000.0, 1e-12),
        "median_prompt_tokens": float(np.median([row["prompt_tokens"] for row in rows])),
        "median_prompt_utf8_bytes": float(
            np.median([row["prompt_utf8_bytes"] for row in rows])
        ),
        "median_generated_utf8_bytes": generated_bytes,
    }
    return result


def benchmark_adapter(
    adapter: CustomAdapter | OfficialAdapter,
    prompts: list[str],
    cfg: dict[str, Any],
    *,
    load_seconds: float,
    memory_before_load: dict[str, Any],
    memory_after_load: dict[str, Any],
    benchmark_environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bench = cfg["benchmark"]
    warmup = int(bench["warmup_repeats"])
    repeats = int(bench["repeats"])
    decode_tokens = int(bench["decode_tokens"])
    common_context = int(bench["common_context_length"])
    targets = [int(value) for value in bench["prompt_character_targets"]]
    samples_per_target = int(bench.get("prompt_samples_per_target", 1))
    workload = _prompt_workload(prompts, targets, samples_per_target)
    process = psutil.Process()
    cases: list[dict[str, Any]] = []
    for workload_case in workload:
        prompt = str(workload_case["prompt"])
        _validate_prompt(adapter, prompt, decode_tokens, common_context)
        if isinstance(adapter, CustomAdapter):
            for _ in range(warmup):
                _custom_request(
                    adapter, prompt, decode_tokens, common_context, cold_route=False
                )
            warm_rows = [
                _custom_request(
                    adapter, prompt, decode_tokens, common_context, cold_route=False
                )
                for _ in range(repeats)
            ]
            cold_rows = [
                _custom_request(
                    adapter, prompt, decode_tokens, common_context, cold_route=True
                )
                for _ in range(max(3, repeats // 5))
            ]
            cases.append(
                {
                    **{key: value for key, value in workload_case.items() if key != "prompt"},
                    "warm_route_cache": _aggregate(warm_rows),
                    "cold_route_cache": _aggregate(cold_rows),
                    "raw_warm_rows": warm_rows,
                    "raw_cold_rows": cold_rows,
                }
            )
        else:
            for _ in range(warmup):
                _official_request(adapter, prompt, decode_tokens, common_context)
            rows = [
                _official_request(adapter, prompt, decode_tokens, common_context)
                for _ in range(repeats)
            ]
            cases.append(
                {
                    **{key: value for key, value in workload_case.items() if key != "prompt"},
                    "warm_route_cache": _aggregate(rows),
                    "cold_route_cache": None,
                    "raw_warm_rows": rows,
                    "raw_cold_rows": None,
                }
            )
    memory_after_benchmark = _memory_snapshot(process, "after_benchmark")
    memory = _memory_report(
        memory_before_load, memory_after_load, memory_after_benchmark
    )
    workload_manifest = [
        {key: value for key, value in item.items() if key != "prompt"} for item in workload
    ]
    workload_sha256 = hashlib.sha256(
        json.dumps(workload_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    resolved_benchmark_config = dict(bench)
    resolved_benchmark_config["prompt_samples_per_target"] = samples_per_target
    return {
        "schema_version": 3,
        "profile": cfg["_profile"],
        "environment": benchmark_environment or environment_snapshot(),
        "threads": int(cfg["runtime"]["threads"]),
        "model": adapter.metadata(),
        "load_seconds": load_seconds,
        "rss_before_load_bytes": memory_before_load["rss_bytes"],
        "rss_after_load_bytes": memory_after_load["rss_bytes"],
        "rss_after_benchmark_bytes": memory_after_benchmark["rss_bytes"],
        "rss_load_delta_bytes": memory["deltas"]["load"]["rss_delta_bytes"],
        "rss_benchmark_delta_bytes": memory["deltas"]["benchmark"]["rss_delta_bytes"],
        "rss_total_delta_bytes": memory["deltas"]["total"]["rss_delta_bytes"],
        "peak_rss_hwm_before_load_bytes": memory_before_load["peak_rss_hwm_bytes"],
        "peak_rss_hwm_after_load_bytes": memory_after_load["peak_rss_hwm_bytes"],
        "peak_rss_hwm_after_benchmark_bytes": memory_after_benchmark[
            "peak_rss_hwm_bytes"
        ],
        "memory": memory,
        "benchmark_config": resolved_benchmark_config,
        "workload": {
            "prompt_samples_per_target": samples_per_target,
            "case_count": len(workload_manifest),
            "manifest_sha256": workload_sha256,
            "cases": workload_manifest,
        },
        "cases": cases,
    }


def _assets(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads(resolve_path(cfg["paths"]["assets_manifest"]).read_text(encoding="utf-8"))


def _runtime_slug(adapter: CustomAdapter | OfficialAdapter) -> str:
    device = adapter.device.type.lower()
    dtype = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
    }.get(adapter.dtype, str(adapter.dtype).removeprefix("torch.").lower())
    return f"{device}_{dtype}"


def benchmark_custom(cfg: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    configure_threads(int(cfg["runtime"]["threads"]))
    seed_all(int(cfg["project"]["seed"]))
    benchmark_environment = environment_snapshot()
    validate_benchmark_cpu_affinity(
        cfg,
        benchmark_environment,
        label=f"custom benchmark profile={cfg['_profile']}",
    )
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    checkpoint_path = Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "best.pt"
    process = psutil.Process()
    gc.collect()
    memory_before = _memory_snapshot(process, "before_load")
    start = time.perf_counter()
    adapter = CustomAdapter.load(
        checkpoint_path,
        resolve_path(cfg["data"]["prepared_dir"]),
        device_name=cfg["runtime"]["device"],
        dtype_name=cfg["runtime"]["dtype"],
        kernel_mode=cfg["benchmark"]["custom_kernel_mode"],
        expected_routing_mode=cfg["model"].get("routing_mode", "vsa"),
    )
    load_seconds = time.perf_counter() - start
    memory_after = _memory_snapshot(process, "after_load")
    prompts = load_prompts(Path(_assets(cfg)["dataset"]["prompts_path"]))
    result = benchmark_adapter(
        adapter,
        prompts,
        cfg,
        load_seconds=load_seconds,
        memory_before_load=memory_before,
        memory_after_load=memory_after,
        benchmark_environment=benchmark_environment,
    )
    atomic_json(
        run_dir
        / "benchmark"
        / f"custom_{_runtime_slug(adapter)}_threads_{cfg['runtime']['threads']}.json",
        result,
    )
    return result


def benchmark_official(cfg: dict[str, Any], model_index: int = 0) -> dict[str, Any]:
    configure_threads(int(cfg["runtime"]["threads"]))
    seed_all(int(cfg["project"]["seed"]))
    benchmark_environment = environment_snapshot()
    validate_benchmark_cpu_affinity(
        cfg,
        benchmark_environment,
        label=f"official benchmark profile={cfg['_profile']}",
    )
    assets = _assets(cfg)
    item = validated_official_model_asset(cfg, assets, model_index)
    process = psutil.Process()
    gc.collect()
    memory_before = _memory_snapshot(process, "before_load")
    start = time.perf_counter()
    adapter = OfficialAdapter.load(
        item["id"],
        item["snapshot_path"],
        revision=item.get("resolved_sha") or item.get("requested_revision"),
        device_name=cfg["runtime"]["device"],
        dtype_name=cfg["runtime"]["dtype"],
        trained_context_length=item.get("trained_context_length"),
    )
    load_seconds = time.perf_counter() - start
    memory_after = _memory_snapshot(process, "after_load")
    prompts = load_prompts(Path(assets["dataset"]["prompts_path"]))
    result = benchmark_adapter(
        adapter,
        prompts,
        cfg,
        load_seconds=load_seconds,
        memory_before_load=memory_before,
        memory_after_load=memory_after,
        benchmark_environment=benchmark_environment,
    )
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    safe = item["id"].replace("/", "__")
    atomic_json(
        run_dir
        / "benchmark"
        / (
            f"official_{safe}_{_runtime_slug(adapter)}_threads_"
            f"{cfg['runtime']['threads']}.json"
        ),
        result,
    )
    return result
