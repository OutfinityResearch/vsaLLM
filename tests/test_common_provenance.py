from __future__ import annotations

from vsa_bench.common import PROJECT_ROOT, runtime_source_files, runtime_source_sha256


def test_runtime_source_hash_is_deterministic_and_hex():
    first = runtime_source_sha256()
    second = runtime_source_sha256()
    assert first == second
    assert len(first) == 64
    int(first, 16)


def test_runtime_source_hash_covers_core_runtime_files():
    relative = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in runtime_source_files()
    }
    assert {
        "python/vsa_bench/train.py",
        "scripts/experiment.mjs",
        "config/base.json",
        "requirements-dgx.txt",
        "containers/dgx-spark/Dockerfile",
    } <= relative
