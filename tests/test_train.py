from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import vsa_bench.train as train_module
from vsa_bench.train import (
    _validate_dgx_preparation_environment,
    _restore_grad_scaler,
    _session_throughput,
    _target_weighted_loss,
    _training_rng,
    _validate_resume_checkpoint,
)


def test_accumulated_loss_is_weighted_by_valid_targets():
    value = torch.tensor(0.25, requires_grad=True)
    short_mean = (value - 2.0).square()
    long_mean = (value + 1.0).square()
    accumulated = _target_weighted_loss(short_mean, 1, 4)
    accumulated = accumulated + _target_weighted_loss(long_mean, 3, 4)
    accumulated.backward()
    actual_gradient = value.grad.detach().clone()

    reference = torch.tensor(0.25, requires_grad=True)
    expected = (
        (reference - 2.0).square() + 3 * (reference + 1.0).square()
    ) / 4
    expected.backward()
    torch.testing.assert_close(actual_gradient, reference.grad)


def test_training_rng_restores_exact_next_values():
    original = _training_rng(seed=31, global_step=7)
    original.integers(0, 10_000, size=23)
    saved_state = original.bit_generator.state
    expected = original.integers(0, 10_000, size=16)

    restored = _training_rng(seed=999, global_step=999, saved_state=saved_state)
    actual = restored.integers(0, 10_000, size=16)
    np.testing.assert_array_equal(actual, expected)


def test_scaler_restore_and_resume_throughput_helpers():
    class FakeScaler:
        def __init__(self) -> None:
            self.loaded = None

        def is_enabled(self) -> bool:
            return True

        def load_state_dict(self, state) -> None:
            self.loaded = state

    scaler = FakeScaler()
    state = {"scale": 1024.0, "growth_tracker": 5}
    _restore_grad_scaler(scaler, state)
    assert scaler.loaded == state
    assert _session_throughput(1_250, 1_000, 2.0) == 125.0


def test_resume_checkpoint_rejects_protocol_drift():
    environment = {
        "runtime_source_sha256": "a" * 64,
        "torch": "2.10.0",
        "cuda_version": "13.0",
        "container_image_digest": "sha256:" + "b" * 64,
    }
    checkpoint = {
        "format_version": 2,
        "metadata": {
            "resolved_config_sha256": "c" * 64,
            "environment": dict(environment),
        },
        "optimizer_state": {"state": {}},
        "training_state": {
            "phase_index": 0,
            "phase_seen_target_tokens": 10,
            "total_seen_target_tokens": 10,
            "global_step": 1,
            "numpy_rng_state": {},
            "next_eval_total": 20,
            "next_checkpoint_total": 20,
        },
    }
    _validate_resume_checkpoint(
        checkpoint,
        resolved_config_sha256="c" * 64,
        training_environment=environment,
    )

    drifted = dict(environment)
    drifted["runtime_source_sha256"] = "d" * 64
    try:
        _validate_resume_checkpoint(
            checkpoint,
            resolved_config_sha256="c" * 64,
            training_environment=drifted,
        )
    except RuntimeError as error:
        assert "runtime_source_sha256" in str(error)
    else:
        raise AssertionError("runtime source drift must reject resume")


def test_dgx_preparation_and_training_environment_must_match():
    environment = {
        "runtime_source_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "git_worktree_dirty": False,
        "torch": "2.10.0",
        "cuda_version": "13.0",
        "container_image": "nvcr.io/nvidia/pytorch:25.11-py3",
        "container_image_digest": "sha256:" + "c" * 64,
        "container_derived_image_id": "sha256:" + "d" * 64,
    }
    _validate_dgx_preparation_environment(
        {"environment": dict(environment)}, environment
    )
    changed = dict(environment)
    changed["cuda_version"] = "13.1"
    try:
        _validate_dgx_preparation_environment(
            {"environment": dict(environment)}, changed
        )
    except RuntimeError as error:
        assert "cuda_version" in str(error)
    else:
        raise AssertionError("DGX preparation/runtime drift must be rejected")


def test_training_writes_weights_only_best_and_resumable_last(tmp_path, monkeypatch):
    class FakeTokenizer:
        @staticmethod
        def vocab_size() -> int:
            return 16

    class FakeStore:
        def __init__(self, _prepared_dir: Path) -> None:
            self.tokenizer = FakeTokenizer()
            self.metadata = {
                "fixture": True,
                "preparation_signature": "f" * 64,
                "tokenizer": {"model_sha256": "a" * 64},
                "router": {"path_sha256": "b" * 64},
            }
            self.num_routes = 1
            self.train_batches = 0

        @staticmethod
        def route_probabilities(_split: str, exponent: float = 1.0) -> np.ndarray:
            return np.asarray([1.0], dtype=np.float64)

        @staticmethod
        def active_routes(_split: str) -> np.ndarray:
            return np.asarray([0], dtype=np.int64)

        def batch(
            self,
            split: str,
            _route: int,
            _batch_size: int,
            block_size: int,
            _rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert block_size == 4
            x = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
            if split == "train":
                self.train_batches += 1
                if self.train_batches % 2:
                    y = torch.tensor([[3, -1, -1, -1]], dtype=torch.long)
                else:
                    y = torch.tensor([[3, 4, 5, -1]], dtype=torch.long)
            else:
                y = torch.tensor([[3, 4, -1, -1]], dtype=torch.long)
            return x, y

    monkeypatch.setattr(train_module, "IndexedStoryStore", FakeStore)
    monkeypatch.setattr(train_module, "configure_threads", lambda _threads: None)
    monkeypatch.setattr(
        train_module,
        "environment_snapshot",
        lambda: {
            "runtime_source_sha256": "c" * 64,
            "git_commit": "d" * 40,
            "git_worktree_dirty": False,
            "machine": "x86_64",
            "python": "3.14.6",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "runtime_dependencies": {},
            "cuda_version": None,
            "containerized": False,
            "container_image": None,
            "container_image_digest": None,
            "container_derived_image_id": None,
        },
    )

    cfg = {
        "_profile": "unit",
        "project": {"seed": 17},
        "runtime": {"device": "cpu", "dtype": "fp32", "threads": 1},
        "paths": {"run_dir": str(tmp_path / "runs")},
        "data": {"prepared_dir": str(tmp_path / "prepared")},
        "model": {
            "vocab_size": 16,
            "block_size": 4,
            "d_model": 8,
            "n_layer": 1,
            "n_head": 2,
            "shared_hidden": 6,
            "expert_hidden": 2,
            "num_routes": 1,
        },
        "training": {
            "batch_size": 1,
            "gradient_accumulation": 2,
            "joint_target_tokens": 4,
            "expert_specialization_target_tokens": 0,
            "learning_rate": 0.001,
            "minimum_learning_rate": 0.0001,
            "expert_learning_rate": 0.001,
            "expert_minimum_learning_rate": 0.0001,
            "warmup_fraction": 0.0,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "route_sampling_exponent": 1.0,
            "log_every_steps": 1,
            "checkpoint_every_target_tokens": 2,
            "eval_every_target_tokens": 4,
            "eval_batches": 1,
            "eval_batch_size": 1,
        },
    }

    result = train_module.train_model(cfg)
    best_path = Path(result["paths"]["best_checkpoint"])
    last_path = Path(result["paths"]["last_checkpoint"])
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)

    assert best["optimizer_state"] is None
    assert last["optimizer_state"] is not None
    assert last["optimizer_state"]["state"]
    assert "numpy_rng_state" in last["training_state"]
    assert "grad_scaler_state" in last["training_state"]
    assert last["training_state"]["next_eval_total"] == 8
    assert last["training_state"]["next_checkpoint_total"] == 6

    rows = [
        json.loads(line)
        for line in Path(result["paths"]["log"]).read_text(encoding="utf-8").splitlines()
    ]
    training_rows = [row for row in rows if "target_tokens_per_second" in row]
    assert len(training_rows) == 1
    assert training_rows[0]["total_seen_target_tokens"] == 4
    assert training_rows[0]["target_tokens_per_second"] > 0
