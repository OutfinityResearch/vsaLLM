from __future__ import annotations

import pytest

from vsa_bench.assets import validated_official_model_asset


def _config() -> dict:
    return {
        "sources": {
            "official_models": [
                {
                    "id": "example/Tiny-8M",
                    "revision": "main",
                    "trained_context_length": 512,
                },
                {
                    "id": "example/Tiny-33M",
                    "revision": "release-v1",
                    "trained_context_length": 512,
                },
            ]
        }
    }


def _manifest() -> dict:
    return {
        "official_models": [
            {
                "id": "example/Tiny-8M",
                "requested_revision": "main",
                "resolved_sha": "sha-8m",
                "snapshot_path": "/cache/8m",
                "trained_context_length": 512,
            },
            {
                "id": "example/Tiny-33M",
                "requested_revision": "release-v1",
                "resolved_sha": "sha-33m",
                "snapshot_path": "/cache/33m",
                "trained_context_length": 512,
            },
        ]
    }


def test_validated_official_asset_returns_selected_row_with_config_context():
    selected = validated_official_model_asset(_config(), _manifest(), 1)

    assert selected["id"] == "example/Tiny-33M"
    assert selected["resolved_sha"] == "sha-33m"
    assert selected["trained_context_length"] == 512


def test_validated_official_asset_rejects_stale_revision_in_unselected_row():
    manifest = _manifest()
    manifest["official_models"][1]["requested_revision"] = "old-release"

    with pytest.raises(RuntimeError, match="row 1.*requested_revision"):
        validated_official_model_asset(_config(), manifest, 0)


def test_validated_official_asset_rejects_reordered_manifest():
    manifest = _manifest()
    manifest["official_models"].reverse()

    with pytest.raises(RuntimeError, match="row 0.*id="):
        validated_official_model_asset(_config(), manifest, 0)


def test_validated_official_asset_rejects_stale_context_override():
    manifest = _manifest()
    manifest["official_models"][0]["trained_context_length"] = 2048

    with pytest.raises(RuntimeError, match="trained_context_length=2048 != 512"):
        validated_official_model_asset(_config(), manifest, 0)


def test_validated_official_asset_rejects_length_and_index_mismatches():
    manifest = _manifest()
    manifest["official_models"].pop()
    with pytest.raises(RuntimeError, match="length"):
        validated_official_model_asset(_config(), manifest, 0)

    with pytest.raises(IndexError, match="outside"):
        validated_official_model_asset(_config(), _manifest(), -1)
