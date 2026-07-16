from __future__ import annotations

import copy

import pytest

from vsa_bench import common as common_module
from vsa_bench.common import (
    load_config,
    validate_benchmark_cpu_affinity,
    validate_config,
    validate_prepared_compatibility,
)


def _prepared(signature: str = "prep") -> dict:
    return {
        "preparation_signature": signature,
        "tokenizer": {"model_sha256": "tokenizer"},
        "router": {"path_sha256": "router"},
    }


def test_prepared_compatibility_accepts_exact_provenance():
    validate_prepared_compatibility(
        {"routing_mode": "vsa", "prepared_metadata": _prepared()},
        _prepared(),
        routing_mode="vsa",
    )


@pytest.mark.parametrize(
    ("metadata", "prepared", "mode"),
    [
        ({"routing_mode": "vsa"}, _prepared(), "vsa"),
        ({"routing_mode": "vsa", "prepared_metadata": {}}, _prepared(), "vsa"),
        (
            {"routing_mode": "vsa", "prepared_metadata": _prepared("old")},
            _prepared("new"),
            "vsa",
        ),
        ({"prepared_metadata": _prepared()}, _prepared(), "vsa"),
        (
            {"routing_mode": "fixed_dense", "prepared_metadata": _prepared()},
            _prepared(),
            "vsa",
        ),
    ],
)
def test_prepared_compatibility_fails_closed(metadata, prepared, mode):
    with pytest.raises(RuntimeError):
        validate_prepared_compatibility(metadata, prepared, routing_mode=mode)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "n_layer", 0),
        ("data", "vsa_dimension", 511),
        ("training", "checkpoint_every_target_tokens", 0),
        ("training", "minimum_learning_rate", 1.0),
        ("evaluation", "blind_sample_index", 99),
        ("benchmark", "repeats", 1),
        ("benchmark", "required_cpu_affinity_logical_cpus", -1),
        ("benchmark", "require_distinct_physical_cores", "yes"),
    ],
)
def test_invalid_long_run_configuration_fails_early(section, field, value):
    cfg = load_config("full_cpu")
    cfg[section][field] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def _write_cpu_topology(root, logical_cpu, package, core, siblings):
    topology = root / f"cpu{logical_cpu}" / "topology"
    topology.mkdir(parents=True)
    (topology / "physical_package_id").write_text(str(package), encoding="utf-8")
    (topology / "core_id").write_text(str(core), encoding="utf-8")
    (topology / "thread_siblings_list").write_text(siblings, encoding="utf-8")


def test_linux_cpu_topology_snapshot_covers_affinity_and_detects_smt(tmp_path):
    _write_cpu_topology(tmp_path, 0, 0, 0, "0-1")
    _write_cpu_topology(tmp_path, 2, 0, 1, "2-3")
    distinct = common_module._cpu_topology_snapshot(
        [0, 2], sysfs_cpu_root=tmp_path
    )
    assert distinct == {
        "logical_cpus": [
            {
                "logical_cpu": 0,
                "physical_package_id": 0,
                "core_id": 0,
                "thread_siblings_list": [0, 1],
            },
            {
                "logical_cpu": 2,
                "physical_package_id": 0,
                "core_id": 1,
                "thread_siblings_list": [2, 3],
            },
        ],
        "complete": True,
        "affinity_logical_cpu_count": 2,
        "affinity_physical_core_count": 2,
        "affinity_contains_smt_siblings": False,
    }

    smt = common_module._cpu_topology_snapshot([0, 1], sysfs_cpu_root=tmp_path)
    assert smt["complete"] is False  # CPU 1 has no independent sysfs row in the fixture.

    _write_cpu_topology(tmp_path, 1, 0, 0, "0-1")
    smt = common_module._cpu_topology_snapshot([0, 1], sysfs_cpu_root=tmp_path)
    assert smt["complete"] is True
    assert smt["affinity_physical_core_count"] == 1
    assert smt["affinity_contains_smt_siblings"] is True


def test_environment_snapshot_embeds_cpu_topology(monkeypatch):
    topology = {
        "logical_cpus": [],
        "complete": False,
        "affinity_logical_cpu_count": 0,
        "affinity_physical_core_count": 0,
        "affinity_contains_smt_siblings": False,
    }
    monkeypatch.setattr(common_module, "_cpu_topology_snapshot", lambda _affinity: topology)
    monkeypatch.setattr(common_module, "runtime_source_sha256", lambda: "a" * 64)
    monkeypatch.setattr(common_module, "git_commit", lambda: "b" * 40)
    monkeypatch.setattr(common_module, "git_worktree_dirty", lambda: False)
    snapshot = common_module.environment_snapshot()
    assert snapshot["cpu_topology"] is topology


def _guard_environment():
    return {
        "cpu_affinity": [0, 2],
        "cpu_topology": {
            "logical_cpus": [
                {
                    "logical_cpu": 0,
                    "physical_package_id": 0,
                    "core_id": 0,
                    "thread_siblings_list": [0, 1],
                },
                {
                    "logical_cpu": 2,
                    "physical_package_id": 0,
                    "core_id": 1,
                    "thread_siblings_list": [2, 3],
                },
            ],
            "complete": True,
            "affinity_logical_cpu_count": 2,
            "affinity_physical_core_count": 2,
            "affinity_contains_smt_siblings": False,
        },
    }


def test_benchmark_cpu_affinity_guard_accepts_distinct_cores_and_rejects_drift():
    cfg = {
        "benchmark": {
            "required_cpu_affinity_logical_cpus": 2,
            "require_distinct_physical_cores": True,
        }
    }
    validate_benchmark_cpu_affinity(cfg, _guard_environment())

    wrong_count = _guard_environment()
    wrong_count["cpu_affinity"].pop()
    with pytest.raises(RuntimeError, match="requires exactly 2"):
        validate_benchmark_cpu_affinity(cfg, wrong_count)

    incomplete = _guard_environment()
    incomplete["cpu_topology"]["complete"] = False
    with pytest.raises(RuntimeError, match="topology is incomplete"):
        validate_benchmark_cpu_affinity(cfg, incomplete)

    smt = copy.deepcopy(_guard_environment())
    smt["cpu_affinity"] = [0, 1]
    smt["cpu_topology"]["logical_cpus"][1] = {
        "logical_cpu": 1,
        "physical_package_id": 0,
        "core_id": 0,
        "thread_siblings_list": [0, 1],
    }
    smt["cpu_topology"]["logical_cpus"][0]["thread_siblings_list"] = [0, 1]
    smt["cpu_topology"]["affinity_physical_core_count"] = 1
    smt["cpu_topology"]["affinity_contains_smt_siblings"] = True
    with pytest.raises(RuntimeError, match="requires distinct physical cores"):
        validate_benchmark_cpu_affinity(cfg, smt)
