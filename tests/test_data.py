from pathlib import Path

from vsa_bench.data import (
    IndexedStoryStore,
    iter_stories,
    preparation_signature,
    split_conditioning_prefix,
    validate_assets_for_preparation,
)


def test_story_parser_reads_fixture():
    path = Path("tests/fixtures/tiny_train.txt")
    stories = list(iter_stories(path, min_chars=20, chunk_chars=37))
    assert len(stories) == 24
    assert stories[0].startswith("Once upon a time")


def test_conditioning_split_is_exact():
    story = "abcdefghi"
    prefix, continuation = split_conditioning_prefix(story, 4)
    assert prefix == "abcd"
    assert continuation == "efghi"
    assert prefix + continuation == story


def test_conditioning_split_prefers_boundary_before_whitespace():
    story = "A small rabbit walked through the green forest and found a bright red kite."
    prefix, continuation = split_conditioning_prefix(story, 32)
    assert prefix + continuation == story
    assert continuation[0].isspace()
    assert 1 <= abs(len(prefix) - 32) <= 32


def test_inference_routing_reuses_training_prefix_boundary():
    store = IndexedStoryStore.__new__(IndexedStoryStore)
    store.routing_prefix_characters = 24

    class RecordingRouter:
        seen = None

        def route(self, text):
            self.seen = text
            return 3

    store.router = RecordingRouter()
    prompt = "Once upon a time a rabbit found a key beside the old garden gate."
    expected, _ = split_conditioning_prefix(prompt, 24)
    assert store.route_prompt(prompt) == 3
    assert store.router.seen == expected


def test_prepared_prefix_is_not_canonicalized_twice():
    store = IndexedStoryStore.__new__(IndexedStoryStore)
    store.routing_prefix_characters = 96

    class RecordingRouter:
        seen = None

        def route(self, text):
            self.seen = text
            return 1

    store.router = RecordingRouter()
    prepared_prefix = "a" * 100 + " "
    assert store.route_conditioning_prefix(prepared_prefix) == 1
    assert store.router.seen == prepared_prefix


def test_preparation_signature_is_independent_of_cache_paths():
    cfg = {
        "project": {"seed": 7},
        "data": {"use_valid_as_train": False, "vocab_size": 320},
        "sources": {"train_file": "train.txt", "valid_file": "valid.txt"},
    }
    first = {
        "dataset": {
            "id": "owner/data",
            "resolved_sha": "abc123",
            "train_path": "/machine-a/cache/train.txt",
            "valid_path": "/machine-a/cache/valid.txt",
        }
    }
    second = {
        "dataset": {
            "id": "owner/data",
            "resolved_sha": "abc123",
            "train_path": "/machine-b/cache/train.txt",
            "valid_path": "/machine-b/cache/valid.txt",
        }
    }
    assert preparation_signature(cfg, first) == preparation_signature(cfg, second)


def test_deployment_only_assets_cannot_prepare_training_data():
    assets = {"dataset": {"full_train_downloaded": False}}
    try:
        validate_assets_for_preparation({"use_valid_as_train": False}, assets)
    except RuntimeError as error:
        assert "deployment-only" in str(error)
    else:
        raise AssertionError("deployment assets must not masquerade as training assets")

    validate_assets_for_preparation({"use_valid_as_train": True}, assets)
