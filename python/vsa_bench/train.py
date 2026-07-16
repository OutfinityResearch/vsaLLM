from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .common import (
    atomic_json,
    autocast_context,
    configure_threads,
    cosine_lr,
    dtype_from_name,
    environment_snapshot,
    resolve_path,
    seed_all,
    select_device,
    sha256_json,
    validate_prepared_compatibility,
)
from .data import IndexedStoryStore
from .model import (
    ModelConfig,
    VSAPathMoE,
    load_checkpoint,
    parameter_accounting,
    save_checkpoint,
    set_trainable_phase,
)


def _optimizer(model: VSAPathMoE, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    decay = [parameter for parameter in trainable if parameter.ndim >= 2]
    no_decay = [parameter for parameter in trainable if parameter.ndim < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=(0.9, 0.95),
    )


def _target_weighted_loss(
    mean_loss: torch.Tensor, valid_targets: int, total_targets: int
) -> torch.Tensor:
    """Scale a microbatch mean so accumulated gradients equal one token mean."""
    valid_targets = int(valid_targets)
    total_targets = int(total_targets)
    if valid_targets < 0 or total_targets <= 0 or valid_targets > total_targets:
        raise ValueError(
            f"Invalid target counts: valid={valid_targets}, total={total_targets}"
        )
    return mean_loss * (valid_targets / total_targets)


def _training_rng(
    seed: int,
    global_step: int,
    saved_state: dict[str, Any] | None = None,
) -> np.random.Generator:
    rng = np.random.default_rng(int(seed) + 101 + int(global_step))
    if saved_state is not None:
        rng.bit_generator.state = saved_state
    return rng


def _restore_grad_scaler(
    scaler: torch.amp.GradScaler, saved_state: dict[str, Any] | None
) -> None:
    if saved_state and scaler.is_enabled():
        scaler.load_state_dict(saved_state)


def _session_throughput(
    total_seen: int, session_start_total: int, elapsed_seconds: float
) -> float:
    session_tokens = max(0, int(total_seen) - int(session_start_total))
    return session_tokens / max(1e-9, float(elapsed_seconds))


def _runtime_training_state(
    rng: np.random.Generator,
    scaler: torch.amp.GradScaler,
    next_eval_total: int,
    next_checkpoint_total: int,
) -> dict[str, Any]:
    return {
        "numpy_rng_state": rng.bit_generator.state,
        "grad_scaler_state": scaler.state_dict(),
        "next_eval_total": int(next_eval_total),
        "next_checkpoint_total": int(next_checkpoint_total),
    }


def build_eval_batches(
    store: IndexedStoryStore,
    split: str,
    count: int,
    batch_size: int,
    block_size: int,
    seed: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    rng = np.random.default_rng(seed)
    probabilities = store.route_probabilities(split, exponent=1.0)
    routes = rng.choice(store.num_routes, size=int(count), p=probabilities)
    return [
        (
            int(route),
            *store.batch(split, int(route), int(batch_size), int(block_size), rng),
        )
        for route in routes
    ]


@torch.inference_mode()
def evaluate_batches(
    model: VSAPathMoE,
    batches: list[tuple[int, torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    *,
    route_transform: Callable[[int], int] | None = None,
    shared_scale: float = 1.0,
    expert_scale: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_targets = 0
    start = time.perf_counter()
    for route, x, y in batches:
        selected = int(route if route_transform is None else route_transform(route))
        if model.config.num_routes == 1:
            selected = 0
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        valid = int((y >= 0).sum())
        with autocast_context(device, dtype):
            _, loss, _ = model(
                x,
                selected,
                y,
                shared_scale=shared_scale,
                expert_scale=expert_scale,
                kernel_mode="fused",
            )
        if loss is None:
            raise RuntimeError("Evaluation loss missing")
        total_nll += float(loss) * valid
        total_targets += valid
    mean = total_nll / max(1, total_targets)
    return {
        "loss": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "target_tokens": total_targets,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False) + "\n")


def _validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    resolved_config_sha256: str,
    training_environment: dict[str, Any],
) -> None:
    """Fail closed instead of silently mixing two training protocols."""
    if checkpoint.get("format_version") != 2:
        raise RuntimeError("Resume requires a current format-version 2 checkpoint")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("Resume checkpoint lacks training provenance metadata")
    if metadata.get("resolved_config_sha256") != resolved_config_sha256:
        raise RuntimeError(
            "Resume checkpoint was created from a different resolved configuration"
        )
    previous_environment = metadata.get("environment")
    if not isinstance(previous_environment, dict):
        raise RuntimeError("Resume checkpoint lacks its training environment")
    for field in (
        "runtime_source_sha256",
        "machine",
        "python",
        "torch",
        "numpy",
        "runtime_dependencies",
        "cuda_version",
        "cuda_devices",
        "cudnn_version",
        "nvidia_driver_version",
        "container_image",
        "container_image_digest",
        "container_derived_image_id",
    ):
        if previous_environment.get(field) != training_environment.get(field):
            raise RuntimeError(
                f"Resume checkpoint has a different training environment field: {field}"
            )
    state = checkpoint.get("training_state")
    if not isinstance(state, dict):
        raise RuntimeError("Resume checkpoint lacks training state")
    required_state = {
        "phase_index",
        "phase_seen_target_tokens",
        "total_seen_target_tokens",
        "global_step",
        "numpy_rng_state",
        "next_eval_total",
        "next_checkpoint_total",
    }
    missing = sorted(required_state.difference(state))
    if missing:
        raise RuntimeError(
            "Resume checkpoint is incomplete; missing training state: "
            + ", ".join(missing)
        )
    if checkpoint.get("optimizer_state") is None:
        raise RuntimeError("Resume requires last.pt with optimizer state, not best.pt")


def _validate_dgx_preparation_environment(
    prepared_metadata: dict[str, Any], training_environment: dict[str, Any]
) -> None:
    prepared_environment = prepared_metadata.get("environment")
    if not isinstance(prepared_environment, dict):
        raise RuntimeError("DGX prepared data lacks environment provenance")
    for field in (
        "runtime_source_sha256",
        "git_commit",
        "git_worktree_dirty",
        "python",
        "torch",
        "numpy",
        "runtime_dependencies",
        "cuda_version",
        "cuda_devices",
        "cudnn_version",
        "nvidia_driver_version",
        "container_image",
        "container_image_digest",
        "container_derived_image_id",
    ):
        if prepared_environment.get(field) != training_environment.get(field):
            raise RuntimeError(
                f"DGX preparation and training environment differ in {field}"
            )


def train_model(cfg: dict[str, Any], *, resume: str | None = None) -> dict[str, Any]:
    runtime = cfg["runtime"]
    train_cfg = cfg["training"]
    if str(cfg["_profile"]).startswith("dgx_spark"):
        from .dgx import dgx_spark_doctor

        preflight = dgx_spark_doctor(cfg)
        if not preflight["ready"]:
            failed = [
                name
                for name, item in preflight["checks"].items()
                if not item["passed"]
            ]
            raise RuntimeError(
                "DGX Spark training preflight failed: " + ", ".join(failed)
            )
    seed = int(cfg["project"]["seed"])
    configure_threads(int(runtime["threads"]))
    seed_all(seed)
    device = select_device(runtime["device"])
    dtype = dtype_from_name(runtime["dtype"], device)
    prepared_dir = resolve_path(cfg["data"]["prepared_dir"])
    store = IndexedStoryStore(prepared_dir)
    model_cfg = ModelConfig.from_mapping(cfg["model"])
    if model_cfg.vocab_size != store.tokenizer.vocab_size():
        raise ValueError("Configured vocabulary does not match prepared tokenizer")
    model = VSAPathMoE(model_cfg).to(device)
    accounting = parameter_accounting(model)
    resolved_config_sha256 = sha256_json(cfg)
    training_environment = environment_snapshot()
    if str(cfg["_profile"]).startswith("dgx_spark"):
        _validate_dgx_preparation_environment(store.metadata, training_environment)

    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    checkpoint_dir = run_dir / "checkpoints"
    artifact_dir = run_dir / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "train_log.jsonl"
    if not resume:
        log_path.unlink(missing_ok=True)

    phases = [
        {
            "name": "joint",
            "target_tokens": int(train_cfg["joint_target_tokens"]),
            "peak_lr": float(train_cfg["learning_rate"]),
            "minimum_lr": float(train_cfg["minimum_learning_rate"]),
            "sampling_exponent": float(train_cfg["route_sampling_exponent"]),
        },
        {
            "name": "experts",
            "target_tokens": int(train_cfg["expert_specialization_target_tokens"]),
            "peak_lr": float(train_cfg["expert_learning_rate"]),
            "minimum_lr": float(train_cfg["expert_minimum_learning_rate"]),
            "sampling_exponent": 0.0,
        },
    ]

    resume_state: dict[str, Any] = {}
    resume_optimizer: dict[str, Any] | None = None
    if resume:
        loaded, checkpoint = load_checkpoint(Path(resume), device="cpu")
        _validate_resume_checkpoint(
            checkpoint,
            resolved_config_sha256=resolved_config_sha256,
            training_environment=training_environment,
        )
        validate_prepared_compatibility(
            checkpoint.get("metadata", {}),
            store.metadata,
            routing_mode=str(cfg["model"].get("routing_mode", "vsa")),
        )
        model.load_state_dict(loaded.state_dict())
        model.to(device)
        resume_state = dict(checkpoint.get("training_state", {}))
        resume_optimizer = checkpoint.get("optimizer_state")
        del loaded
        del checkpoint
        gc.collect()

    global_step = int(resume_state.get("global_step", 0))
    total_seen = int(resume_state.get("total_seen_target_tokens", 0))
    best_loss = float(resume_state.get("best_validation_loss", float("inf")))
    resume_phase_index = int(resume_state.get("phase_index", 0))
    resume_phase_seen = int(resume_state.get("phase_seen_target_tokens", 0))
    route_usage = np.zeros(store.num_routes, dtype=np.int64)
    if "route_usage" in resume_state:
        old = np.asarray(resume_state["route_usage"], dtype=np.int64)
        route_usage[: min(len(old), len(route_usage))] = old[: len(route_usage)]

    eval_batches = build_eval_batches(
        store,
        "validation",
        int(train_cfg["eval_batches"]),
        int(train_cfg["eval_batch_size"]),
        model_cfg.block_size,
        seed + 501,
    )
    rng = _training_rng(seed, global_step, resume_state.get("numpy_rng_state"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and dtype == torch.float16)
    )
    _restore_grad_scaler(scaler, resume_state.get("grad_scaler_state"))
    start_time = time.perf_counter()
    session_start_total = total_seen
    eval_interval = int(train_cfg["eval_every_target_tokens"])
    if eval_interval <= 0:
        raise ValueError("eval_every_target_tokens must be positive")
    next_eval_total = int(
        resume_state.get("next_eval_total", total_seen + eval_interval)
    )
    checkpoint_interval = int(
        train_cfg.get("checkpoint_every_target_tokens", eval_interval)
    )
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_every_target_tokens must be positive")
    next_checkpoint_total = int(
        resume_state.get(
            "next_checkpoint_total", total_seen + checkpoint_interval
        )
    )

    metadata = {
        "profile": cfg["_profile"],
        "routing_mode": cfg["model"].get("routing_mode", "vsa"),
        "resolved_config_sha256": resolved_config_sha256,
        "prepared_metadata": store.metadata,
        "environment": training_environment,
        "parameter_accounting": accounting,
    }

    last_optimizer: torch.optim.Optimizer | None = None
    for phase_index, phase in enumerate(phases):
        if phase["target_tokens"] <= 0:
            continue
        if phase_index < resume_phase_index:
            continue
        phase_name = str(phase["name"])
        set_trainable_phase(model, phase_name)
        model.clear_fused()
        optimizer = _optimizer(model, phase["peak_lr"], float(train_cfg["weight_decay"]))
        last_optimizer = optimizer
        phase_seen = resume_phase_seen if phase_index == resume_phase_index else 0
        if phase_index == resume_phase_index and resume_optimizer:
            optimizer.load_state_dict(resume_optimizer)
            resume_optimizer = None
            gc.collect()
        probabilities = store.route_probabilities(
            "train", exponent=float(phase["sampling_exponent"])
        )
        if phase_name == "experts":
            active = store.active_routes("train")
            probabilities[:] = 0.0
            probabilities[active] = 1.0 / len(active)

        while phase_seen < int(phase["target_tokens"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            microbatches: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
            accumulated_targets = 0
            step_loss_numerator = 0.0
            for _ in range(int(train_cfg["gradient_accumulation"])):
                route = int(rng.choice(store.num_routes, p=probabilities))
                x, y = store.batch(
                    "train",
                    route,
                    int(train_cfg["batch_size"]),
                    model_cfg.block_size,
                    rng,
                )
                valid = int((y >= 0).sum())
                route_usage[route] += valid
                microbatches.append((route, x, y, valid))
                accumulated_targets += valid

            if accumulated_targets <= 0:
                raise RuntimeError("Training step has no causal continuation targets")

            for route, x, y, valid in microbatches:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast_context(device, dtype):
                    model_route = 0 if model.config.num_routes == 1 else route
                    _, loss, _ = model(x, model_route, y, kernel_mode="fused")
                    if loss is None:
                        raise RuntimeError("Training loss missing")
                    scaled_loss = _target_weighted_loss(
                        loss, valid, accumulated_targets
                    )
                scaler.scale(scaled_loss).backward()
                step_loss_numerator += float(loss.detach()) * valid

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(train_cfg["gradient_clip"]),
            )
            phase_seen += accumulated_targets
            total_seen += accumulated_targets
            progress = phase_seen / max(1, int(phase["target_tokens"]))
            lr = cosine_lr(
                progress,
                float(phase["peak_lr"]),
                float(phase["minimum_lr"]),
                float(train_cfg["warmup_fraction"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if global_step % int(train_cfg["log_every_steps"]) == 0:
                row = {
                    "phase": phase_name,
                    "phase_index": phase_index,
                    "global_step": global_step,
                    "phase_seen_target_tokens": phase_seen,
                    "total_seen_target_tokens": total_seen,
                    "train_loss": step_loss_numerator / max(1, accumulated_targets),
                    "learning_rate": lr,
                    "target_tokens_per_second": _session_throughput(
                        total_seen,
                        session_start_total,
                        time.perf_counter() - start_time,
                    ),
                }
                print(json.dumps(row), flush=True)
                _append_jsonl(log_path, row)

            phase_complete = phase_seen >= int(phase["target_tokens"])
            should_eval = total_seen >= next_eval_total or phase_complete
            should_checkpoint = total_seen >= next_checkpoint_total or phase_complete
            if should_eval or should_checkpoint:
                validation = (
                    evaluate_batches(model, eval_batches, device, dtype)
                    if should_eval
                    else None
                )
                while next_eval_total <= total_seen:
                    next_eval_total += eval_interval
                while next_checkpoint_total <= total_seen:
                    next_checkpoint_total += checkpoint_interval
                candidate_best = (
                    min(best_loss, float(validation["loss"]))
                    if validation is not None
                    else best_loss
                )
                state = {
                    "phase_index": phase_index,
                    "phase_name": phase_name,
                    "phase_seen_target_tokens": phase_seen,
                    "phase_target_tokens": int(phase["target_tokens"]),
                    "total_seen_target_tokens": total_seen,
                    "global_step": global_step,
                    "best_validation_loss": candidate_best,
                    "route_usage": route_usage.tolist(),
                    **_runtime_training_state(
                        rng,
                        scaler,
                        next_eval_total,
                        next_checkpoint_total,
                    ),
                }
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer_state=optimizer.state_dict(),
                    training_state=state,
                    metadata=metadata,
                )
                if validation is not None:
                    improved = validation["loss"] < best_loss
                    if improved:
                        best_loss = float(validation["loss"])
                        state["best_validation_loss"] = best_loss
                        save_checkpoint(
                            checkpoint_dir / "best.pt",
                            model,
                            optimizer_state=None,
                            training_state=state,
                            metadata=metadata,
                        )
                    event_row = {
                        **state,
                        "event": "validation",
                        "validation": validation,
                        "improved": improved,
                    }
                else:
                    event_row = {**state, "event": "checkpoint"}
                print(json.dumps(event_row), flush=True)
                _append_jsonl(log_path, event_row)

        # Start specialization from the best joint checkpoint, not merely the
        # last joint update. The best file can later be replaced by a better
        # specialization checkpoint.
        if phase_name == "joint" and (checkpoint_dir / "best.pt").exists():
            best_model, best_checkpoint = load_checkpoint(
                checkpoint_dir / "best.pt", device="cpu"
            )
            model.load_state_dict(best_model.state_dict())
            model.to(device)
            del best_model
            del best_checkpoint
            gc.collect()
        resume_optimizer = None
        resume_phase_seen = 0

    final_state = {
        "phase_index": len(phases),
        "phase_name": "complete",
        "phase_seen_target_tokens": 0,
        "total_seen_target_tokens": total_seen,
        "global_step": global_step,
        "best_validation_loss": best_loss,
        "route_usage": route_usage.tolist(),
        **_runtime_training_state(
            rng,
            scaler,
            next_eval_total,
            next_checkpoint_total,
        ),
    }
    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer_state=(
            last_optimizer.state_dict()
            if last_optimizer is not None
            else resume_optimizer
        ),
        training_state=final_state,
        metadata=metadata,
    )
    result = {
        "profile": cfg["_profile"],
        "environment": environment_snapshot(),
        "provenance": {
            "resolved_config_sha256": resolved_config_sha256,
            "preparation_signature": store.metadata["preparation_signature"],
            "training_runtime_source_sha256": training_environment[
                "runtime_source_sha256"
            ],
            "training_git_commit": training_environment["git_commit"],
            "training_git_worktree_dirty": training_environment[
                "git_worktree_dirty"
            ],
            "training_machine": training_environment["machine"],
            "training_python": training_environment["python"],
            "training_torch_version": training_environment["torch"],
            "training_numpy_version": training_environment["numpy"],
            "training_runtime_dependencies": training_environment[
                "runtime_dependencies"
            ],
            "training_cuda_version": training_environment["cuda_version"],
            "training_cuda_devices": training_environment.get("cuda_devices"),
            "training_cudnn_version": training_environment.get("cudnn_version"),
            "training_nvidia_driver_version": training_environment.get(
                "nvidia_driver_version"
            ),
            "training_containerized": training_environment["containerized"],
            "training_container_image": training_environment["container_image"],
            "training_container_image_digest": training_environment[
                "container_image_digest"
            ],
            "training_container_derived_image_id": training_environment[
                "container_derived_image_id"
            ],
        },
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "parameter_accounting": accounting,
        "model_config": model_cfg.__dict__,
        "training": {
            **final_state,
            "elapsed_seconds": time.perf_counter() - start_time,
            "active_routes": int((route_usage > 0).sum()),
        },
        "paths": {
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "last_checkpoint": str(checkpoint_dir / "last.pt"),
            "log": str(log_path),
        },
    }
    atomic_json(artifact_dir / "training_summary.json", result)
    return result
