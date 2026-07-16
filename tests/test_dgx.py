from __future__ import annotations

from vsa_bench.common import load_config
from vsa_bench.dgx import dgx_spark_doctor
from vsa_bench.model import ModelConfig, VSAPathMoE, parameter_accounting


def test_dgx_profile_and_preflight_contract():
    cfg = load_config("dgx_spark")
    assert cfg["runtime"] == {"device": "cuda", "dtype": "bf16", "threads": 4}
    assert cfg["benchmark"]["primary_device"] == "cpu"
    assert cfg["benchmark"]["primary_dtype"] == "fp32"
    assert cfg["benchmark"]["required_cpu_affinity_logical_cpus"] == 8
    assert cfg["benchmark"]["require_distinct_physical_cores"] is True
    result = dgx_spark_doctor(cfg)
    assert set(result["checks"]) >= {
        "arm64_host",
        "cuda_available",
        "pinned_container_digest",
        "clean_git_worktree",
    }
    assert result["ready"] is all(
        item["passed"] for item in result["checks"].values()
    )


def test_dgx_controls_share_protocol_and_match_parameter_budgets():
    sparse_cfg = load_config("dgx_spark")
    active_cfg = load_config("dgx_spark_dense_active")
    total_cfg = load_config("dgx_spark_dense_total")
    sparse = parameter_accounting(
        VSAPathMoE(ModelConfig.from_mapping(sparse_cfg["model"]))
    )
    active = parameter_accounting(
        VSAPathMoE(ModelConfig.from_mapping(active_cfg["model"]))
    )
    total = parameter_accounting(
        VSAPathMoE(ModelConfig.from_mapping(total_cfg["model"]))
    )

    assert active["total_parameters"] == sparse["active_parameters_per_request"]
    assert total["total_parameters"] == sparse["total_parameters"]
    for control in (active_cfg, total_cfg):
        assert control["runtime"] == sparse_cfg["runtime"]
        assert control["data"] == sparse_cfg["data"]
        assert control["sources"] == sparse_cfg["sources"]
        assert control["training"] == sparse_cfg["training"]
        assert control["benchmark"] == sparse_cfg["benchmark"]
        assert control["model"]["routing_mode"] == "fixed_dense"
        assert control["model"]["num_routes"] == 1
