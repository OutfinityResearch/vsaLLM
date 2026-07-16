from __future__ import annotations

from types import SimpleNamespace

import psutil
import pytest

from vsa_bench import benchmark as benchmark_module


def _snapshot(stage: str, rss: int, uss: int, pss: int, peak: int) -> dict:
    return {
        "stage": stage,
        "rss_bytes": rss,
        "uss_bytes": uss,
        "pss_bytes": pss,
        "peak_rss_hwm_bytes": peak,
        "peak_rss_hwm_source": "test",
    }


def _timing_row(prompt: str, *, cold: bool = False) -> dict:
    value = 2.0 if cold else 1.0
    return {
        "prompt_tokens": len(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "generated_tokens": 2,
        "generated_utf8_bytes": 4,
        "route": 3,
        "common_context": 64,
        "input_truncated": False,
        "tokenize_ms": value,
        "route_ms": value,
        "prepare_route_ms": value,
        "prefill_ms": value,
        "decode_ms": value,
        "detokenize_ms": value,
        "end_to_end_ms": value * 6,
    }


def test_prompt_workload_is_deterministic_and_distinct_per_target():
    prompts = ["Alpha story begins", "Beta story begins", "Gamma story begins"]
    first = benchmark_module._prompt_workload(prompts, [8, 12], 2)
    second = benchmark_module._prompt_workload(prompts, [8, 12], 2)

    assert first == second
    assert [(row["prompt_target_characters"], row["prompt_source_index"]) for row in first] == [
        (8, 0),
        (8, 1),
        (12, 1),
        (12, 2),
    ]
    for target in (8, 12):
        rows = [row for row in first if row["prompt_target_characters"] == target]
        assert len({row["prompt"] for row in rows}) == 2
        assert len({row["prompt_sha256"] for row in rows}) == 2


def test_prompt_workload_rejects_impossible_distinct_sample_count():
    with pytest.raises(ValueError, match="distinct benchmark prompts"):
        benchmark_module._prompt_workload(["same", "same"], [4], 2)


def test_memory_snapshot_and_delta_include_optional_uss_pss(monkeypatch):
    class FakeProcess:
        pid = -1

        def memory_info(self):
            return SimpleNamespace(rss=1000)

        def memory_full_info(self):
            return SimpleNamespace(uss=700, pss=850)

    monkeypatch.setattr(
        benchmark_module, "_peak_rss_hwm", lambda _process, _info: (1200, "test_hwm")
    )
    snapshot = benchmark_module._memory_snapshot(FakeProcess(), "after_load")
    assert snapshot == {
        "stage": "after_load",
        "rss_bytes": 1000,
        "uss_bytes": 700,
        "pss_bytes": 850,
        "peak_rss_hwm_bytes": 1200,
        "peak_rss_hwm_source": "test_hwm",
    }

    delta = benchmark_module._memory_delta(
        _snapshot("before", 100, 80, 90, 120),
        _snapshot("after", 250, 180, 210, 300),
    )
    assert delta == {
        "rss_delta_bytes": 150,
        "uss_delta_bytes": 100,
        "pss_delta_bytes": 120,
        "peak_rss_hwm_delta_bytes": 180,
    }


def test_memory_snapshot_tolerates_unavailable_full_info(monkeypatch):
    class FakeProcess:
        pid = -1

        def memory_info(self):
            return SimpleNamespace(rss=1000)

        def memory_full_info(self):
            raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(
        benchmark_module, "_peak_rss_hwm", lambda _process, _info: (None, None)
    )
    snapshot = benchmark_module._memory_snapshot(FakeProcess(), "before_load")
    assert snapshot["uss_bytes"] is None
    assert snapshot["pss_bytes"] is None
    assert snapshot["peak_rss_hwm_bytes"] is None


def test_benchmark_saves_cold_rows_memory_stages_and_shared_prompt_manifest(monkeypatch):
    class FakeCustomAdapter:
        def metadata(self):
            return {"name": "custom"}

    class FakeOfficialAdapter:
        def metadata(self):
            return {"name": "official"}

    cfg = {
        "_profile": "test",
        "runtime": {"threads": 1},
        "benchmark": {
            "warmup_repeats": 1,
            "repeats": 2,
            "decode_tokens": 2,
            "common_context_length": 64,
            "prompt_character_targets": [8],
            "prompt_samples_per_target": 2,
        },
    }
    prompts = ["Alpha story", "Beta story", "Gamma story"]
    before = _snapshot("before_load", 100, 80, 90, 120)
    after_load = _snapshot("after_load", 200, 150, 170, 240)
    after_benchmark = _snapshot("after_benchmark", 250, 180, 200, 300)

    monkeypatch.setattr(benchmark_module, "CustomAdapter", FakeCustomAdapter)
    monkeypatch.setattr(benchmark_module, "_validate_prompt", lambda *_args: None)
    monkeypatch.setattr(benchmark_module, "environment_snapshot", lambda: {})
    monkeypatch.setattr(
        benchmark_module,
        "_memory_snapshot",
        lambda _process, stage: {**after_benchmark, "stage": stage},
    )
    monkeypatch.setattr(
        benchmark_module,
        "_custom_request",
        lambda _adapter, prompt, *_args, cold_route: _timing_row(
            prompt, cold=cold_route
        ),
    )
    custom = benchmark_module.benchmark_adapter(
        FakeCustomAdapter(),
        prompts,
        cfg,
        load_seconds=0.25,
        memory_before_load=before,
        memory_after_load=after_load,
    )

    assert custom["schema_version"] == 3
    assert custom["rss_before_load_bytes"] == 100
    assert custom["rss_after_load_bytes"] == 200
    assert custom["rss_after_benchmark_bytes"] == 250
    assert custom["rss_load_delta_bytes"] == 100
    assert custom["rss_benchmark_delta_bytes"] == 50
    assert custom["rss_total_delta_bytes"] == 150
    assert custom["memory"]["deltas"]["load"]["uss_delta_bytes"] == 70
    assert custom["memory"]["deltas"]["benchmark"]["pss_delta_bytes"] == 30
    assert len(custom["cases"]) == 2
    assert len({case["prompt_sha256"] for case in custom["cases"]}) == 2
    assert all(len(case["raw_warm_rows"]) == 2 for case in custom["cases"])
    assert all(len(case["raw_cold_rows"]) == 3 for case in custom["cases"])

    monkeypatch.setattr(
        benchmark_module,
        "_official_request",
        lambda _adapter, prompt, *_args: _timing_row(prompt),
    )
    official = benchmark_module.benchmark_adapter(
        FakeOfficialAdapter(),
        prompts,
        cfg,
        load_seconds=0.5,
        memory_before_load=before,
        memory_after_load=after_load,
    )
    assert official["workload"]["manifest_sha256"] == custom["workload"][
        "manifest_sha256"
    ]
    assert [case["prompt_sha256"] for case in official["cases"]] == [
        case["prompt_sha256"] for case in custom["cases"]
    ]
    assert all(case["raw_cold_rows"] is None for case in official["cases"])

    default_cfg = {**cfg, "benchmark": dict(cfg["benchmark"])}
    default_cfg["benchmark"].pop("prompt_samples_per_target")
    default_result = benchmark_module.benchmark_adapter(
        FakeOfficialAdapter(),
        prompts,
        default_cfg,
        load_seconds=0.5,
        memory_before_load=before,
        memory_after_load=after_load,
    )
    assert default_result["benchmark_config"]["prompt_samples_per_target"] == 1
    assert default_result["workload"]["case_count"] == 1


@pytest.mark.parametrize("entrypoint", ["custom", "official"])
def test_cpu_affinity_guard_fails_before_adapter_load(monkeypatch, entrypoint):
    cfg = {
        "_profile": "guarded",
        "project": {"seed": 1},
        "runtime": {"threads": 1},
        "benchmark": {
            "required_cpu_affinity_logical_cpus": 2,
            "require_distinct_physical_cores": True,
        },
    }
    invalid_environment = {
        "cpu_affinity": [0],
        "cpu_topology": {
            "logical_cpus": [],
            "complete": False,
            "affinity_logical_cpu_count": 1,
            "affinity_physical_core_count": 0,
            "affinity_contains_smt_siblings": False,
        },
    }
    loaded = False

    def unexpected_load(*_args, **_kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("adapter load must not be reached")

    monkeypatch.setattr(
        benchmark_module, "environment_snapshot", lambda: invalid_environment
    )
    monkeypatch.setattr(benchmark_module.CustomAdapter, "load", unexpected_load)
    monkeypatch.setattr(benchmark_module.OfficialAdapter, "load", unexpected_load)
    function = (
        benchmark_module.benchmark_custom
        if entrypoint == "custom"
        else benchmark_module.benchmark_official
    )
    with pytest.raises(RuntimeError, match="requires exactly 2"):
        function(cfg)
    assert loaded is False
