from __future__ import annotations

import json
import os
import shutil
from array import array
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import sentencepiece as spm
import torch

from .common import (
    atomic_json,
    entropy_bits,
    environment_snapshot,
    resolve_path,
    sha256_file,
    sha256_json,
    stable_hash64,
)
from .router import RouterConfig, VSAProductRouter, fit_router

STORY_DELIMITER = "<|endoftext|>"
PREPARATION_FORMAT_VERSION = 5


def preparation_signature(cfg: dict[str, Any], assets: dict[str, Any]) -> str:
    """Identify token/route semantics without machine-local cache paths."""
    data_cfg = cfg["data"]
    source = cfg["sources"]
    use_valid_as_train = bool(data_cfg.get("use_valid_as_train", False))
    dataset = assets.get("dataset", {})
    return sha256_json(
        {
            "format_version": PREPARATION_FORMAT_VERSION,
            "project_seed": int(cfg["project"]["seed"]),
            "data_config": data_cfg,
            "dataset": {
                "id": dataset.get("id"),
                "resolved_sha": dataset.get("resolved_sha"),
                "train_file": (
                    source["valid_file"]
                    if use_valid_as_train
                    else source["train_file"]
                ),
                "valid_file": source["valid_file"],
            },
        }
    )


def validate_assets_for_preparation(
    data_cfg: dict[str, Any], assets: dict[str, Any]
) -> None:
    if (
        not bool(data_cfg.get("use_valid_as_train", False))
        and not bool(assets.get("dataset", {}).get("full_train_downloaded", False))
    ):
        raise RuntimeError(
            "The assets manifest is deployment-only and has no TinyStories train corpus; "
            "run download-full before preparing data"
        )


def iter_stories(
    path: Path, *, min_chars: int = 20, chunk_chars: int = 1 << 20
) -> Iterator[str]:
    """Stream TinyStories records without loading the multi-GB file into RAM."""
    buffer = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while chunk := fh.read(chunk_chars):
            buffer += chunk
            parts = buffer.split(STORY_DELIMITER)
            buffer = parts.pop()
            for part in parts:
                story = part.strip()
                if len(story) >= min_chars:
                    yield story
    story = buffer.strip()
    if len(story) >= min_chars:
        yield story


def split_conditioning_prefix(story: str, characters: int) -> tuple[str, str]:
    """Choose a causal raw-text boundary that is stable across tokenizers.

    The preferred split is immediately *before* whitespace near the requested
    character position. This leaves the whitespace at the beginning of the
    continuation, preventing GPT-style leading-space tokens (and SentencePiece
    whitespace pieces) from straddling the prefix/continuation boundary.
    When no nearby whitespace exists, the exact requested position is kept and
    later tokenization additivity checks fail fast if the boundary is unsafe.
    """
    requested = min(len(story), max(0, int(characters)))
    if requested <= 0 or requested >= len(story):
        return story[:requested], story[requested:]

    search_radius = min(96, max(16, requested // 2))
    lower = max(1, requested - search_radius)
    boundary = requested
    for index in range(requested, lower - 1, -1):
        if index < len(story) and story[index].isspace():
            boundary = index
            break
    else:
        upper = min(len(story) - 1, requested + search_radius)
        for index in range(requested + 1, upper + 1):
            if story[index].isspace():
                boundary = index
                break
    return story[:boundary], story[boundary:]


def reservoir_sample(
    path: Path,
    count: int,
    seed: int,
    *,
    max_stories: int = 0,
    min_chars: int = 20,
    prefix_characters: int = 96,
    minimum_continuation_chars: int = 24,
) -> tuple[list[str], int]:
    rng = np.random.default_rng(seed)
    sample: list[str] = []
    observed = 0
    for story in iter_stories(path, min_chars=min_chars):
        prefix, continuation = split_conditioning_prefix(story, prefix_characters)
        if len(continuation) < minimum_continuation_chars:
            continue
        observed += 1
        if len(sample) < count:
            sample.append(story)
        else:
            index = int(rng.integers(observed))
            if index < count:
                sample[index] = story
        if max_stories and observed >= max_stories:
            break
    return sample, observed


def _write_tokenizer_sample(stories: list[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for story in stories:
            fh.write(story.replace("\n", " ").strip() + "\n")


def train_tokenizer(stories: list[str], out_dir: Path, vocab_size: int, threads: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "tokenizer_sample.txt"
    _write_tokenizer_sample(stories, sample_path)
    prefix = out_dir / "tinystories_bpe"
    spm.SentencePieceTrainer.train(
        input=str(sample_path),
        model_prefix=str(prefix),
        vocab_size=int(vocab_size),
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        input_sentence_size=0,
        shuffle_input_sentence=True,
        num_threads=max(1, min(int(threads), os.cpu_count() or 1)),
        hard_vocab_limit=True,
    )
    sample_path.unlink(missing_ok=True)
    model_path = prefix.with_suffix(".model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))
    # BPB is only meaningful when the custom tokenizer is lossless.
    for story in stories[: min(256, len(stories))]:
        reconstructed = tokenizer.decode(tokenizer.encode(story, out_type=int))
        if reconstructed != story:
            raise RuntimeError(
                "SentencePiece round-trip failed; conditional bits-per-byte would be invalid"
            )
    return model_path


class _IndexedSplitWriter:
    def __init__(self, root: Path, split: str, num_routes: int) -> None:
        self.root = root
        self.split = split
        self.num_routes = num_routes
        split_dir = root / "indexed" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_path = split_dir / "tokens.u16"
        self.handle: BinaryIO = self.tokens_path.open("wb")
        self.offsets = array("Q", [0])
        self.prefix_lengths = array("H")
        self.routes = array("B")
        self.target_tokens = array("I")
        self.route_story_counts = np.zeros(num_routes, dtype=np.int64)
        self.route_target_counts = np.zeros(num_routes, dtype=np.int64)

    def append(self, sequence: np.ndarray, prefix_length: int, route: int) -> None:
        if sequence.dtype != np.uint16:
            sequence = sequence.astype(np.uint16)
        if not 1 <= prefix_length < len(sequence):
            raise ValueError("Each record needs at least one continuation target")
        self.handle.write(sequence.tobytes(order="C"))
        self.offsets.append(self.offsets[-1] + len(sequence))
        self.prefix_lengths.append(int(prefix_length))
        self.routes.append(int(route))
        targets = len(sequence) - int(prefix_length)
        self.target_tokens.append(targets)
        self.route_story_counts[route] += 1
        self.route_target_counts[route] += targets

    def close(self) -> dict[str, Any]:
        self.handle.close()
        split_dir = self.tokens_path.parent
        offsets = np.frombuffer(self.offsets, dtype=np.uint64).copy()
        prefix_lengths = np.frombuffer(self.prefix_lengths, dtype=np.uint16).copy()
        routes = np.frombuffer(self.routes, dtype=np.uint8).copy()
        target_tokens = np.frombuffer(self.target_tokens, dtype=np.uint32).copy()
        np.save(split_dir / "offsets.npy", offsets)
        np.save(split_dir / "prefix_lengths.npy", prefix_lengths)
        np.save(split_dir / "routes.npy", routes)
        np.save(split_dir / "target_tokens.npy", target_tokens)
        order = np.argsort(routes, kind="stable").astype(np.uint32)
        counts = np.bincount(routes, minlength=self.num_routes).astype(np.uint64)
        bounds = np.concatenate((np.asarray([0], dtype=np.uint64), np.cumsum(counts)))
        np.save(split_dir / "route_order.npy", order)
        np.save(split_dir / "route_bounds.npy", bounds)
        return {
            "stories": int(len(routes)),
            "tokens_with_bos_eos": int(offsets[-1]) if len(offsets) else 0,
            "target_tokens": int(target_tokens.astype(np.uint64).sum()),
            "active_routes": int((self.route_story_counts > 0).sum()),
            "route_story_counts": self.route_story_counts.tolist(),
            "route_target_token_counts": self.route_target_counts.tolist(),
            "route_entropy_bits": entropy_bits(self.route_story_counts),
            "route_entropy_max_bits": float(np.log2(self.num_routes)),
        }


def _encode_record(
    tokenizer: spm.SentencePieceProcessor,
    prefix: str,
    continuation: str,
) -> tuple[np.ndarray, int]:
    prefix_ids = tokenizer.encode(prefix, out_type=int)
    continuation_ids = tokenizer.encode(continuation, out_type=int)
    combined_ids = tokenizer.encode(prefix + continuation, out_type=int)
    if [*prefix_ids, *continuation_ids] != list(combined_ids):
        raise RuntimeError(
            "Unsafe prefix boundary: SentencePiece tokenization is not additive. "
            "The continuation must begin at a tokenizer-stable raw-text boundary."
        )
    sequence = np.asarray(
        [tokenizer.bos_id(), *prefix_ids, *continuation_ids, tokenizer.eos_id()],
        dtype=np.uint16,
    )
    prefix_length = 1 + len(prefix_ids)
    return sequence, prefix_length


def _write_eval_json(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_data(cfg: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    data_cfg = cfg["data"]
    seed = int(cfg["project"]["seed"])
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
                "DGX Spark preparation preflight failed: " + ", ".join(failed)
            )
    assets_path = resolve_path(cfg["paths"]["assets_manifest"])
    if not assets_path.exists():
        raise FileNotFoundError(
            f"Missing {assets_path}. Run `npm run download -- --profile {cfg['_profile']}` first."
        )
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    validate_assets_for_preparation(data_cfg, assets)
    train_path = Path(assets["dataset"]["train_path"])
    valid_path = Path(assets["dataset"]["valid_path"])
    if data_cfg.get("use_valid_as_train", False):
        train_path = valid_path

    out_dir = resolve_path(data_cfg["prepared_dir"])
    done_path = out_dir / "PREPARED.json"
    preparation_signature_value = preparation_signature(cfg, assets)
    if done_path.exists() and not force:
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        if existing.get("preparation_signature") != preparation_signature_value:
            raise RuntimeError(
                f"Prepared data in {out_dir} does not match the current configuration "
                f"or dataset revision. Re-run prepare with --force."
            )
        return existing
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix_chars = int(data_cfg["conditioning_prefix_characters"])
    min_cont = int(data_cfg["minimum_continuation_chars"])
    sample_count = max(
        int(data_cfg["tokenizer_sample_stories"]),
        int(data_cfg["router_sample_stories"]),
    )
    sample, observed_train = reservoir_sample(
        train_path,
        sample_count,
        seed + 1,
        max_stories=int(data_cfg.get("max_train_stories", 0)),
        min_chars=int(data_cfg.get("min_story_chars", 20)),
        prefix_characters=prefix_chars,
        minimum_continuation_chars=min_cont,
    )
    if len(sample) < max(8, int(data_cfg["router_product_k"])):
        raise RuntimeError(f"Not enough usable stories: {len(sample)}")

    tokenizer_model = train_tokenizer(
        sample[: int(data_cfg["tokenizer_sample_stories"])],
        out_dir,
        int(data_cfg["vocab_size"]),
        int(cfg["runtime"]["threads"]),
    )
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    prefixes = [
        split_conditioning_prefix(story, prefix_chars)[0]
        for story in sample[: int(data_cfg["router_sample_stories"])]
    ]
    router_cfg = RouterConfig(
        dimension=int(data_cfg["vsa_dimension"]),
        prefix_bytes=int(data_cfg["router_prefix_bytes"]),
        role_period=int(data_cfg["role_period"]),
        product_k=int(data_cfg["router_product_k"]),
        kmeans_iterations=int(data_cfg["router_kmeans_iterations"]),
        seed=seed + 11,
    )
    router, router_stats = fit_router(prefixes, router_cfg)
    router_path = out_dir / "router.npz"
    router.save(router_path)

    writers = {
        split: _IndexedSplitWriter(out_dir, split, router.num_routes)
        for split in ("train", "validation", "test")
    }
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    validation_json = (eval_dir / "validation.jsonl").open("w", encoding="utf-8")
    test_json = (eval_dir / "test.jsonl").open("w", encoding="utf-8")

    max_train = int(data_cfg.get("max_train_stories", 0))
    accepted = 0
    for story in iter_stories(train_path, min_chars=int(data_cfg.get("min_story_chars", 20))):
        prefix, continuation = split_conditioning_prefix(story, prefix_chars)
        if len(continuation) < min_cont:
            continue
        route = router.route(prefix)
        sequence, prefix_length = _encode_record(tokenizer, prefix, continuation)
        writers["train"].append(sequence, prefix_length, route)
        accepted += 1
        if accepted % 100_000 == 0:
            print(f"prepared train stories: {accepted:,}", flush=True)
        if max_train and accepted >= max_train:
            break

    max_eval = int(data_cfg.get("max_eval_stories", 0))
    validation_fraction = float(data_cfg["official_valid_validation_fraction"])
    accepted_eval = 0
    for source_id, story in enumerate(
        iter_stories(valid_path, min_chars=int(data_cfg.get("min_story_chars", 20)))
    ):
        prefix, continuation = split_conditioning_prefix(story, prefix_chars)
        if len(continuation) < min_cont:
            continue
        ratio = stable_hash64(story, seed + 21) / float(2**64)
        split = "validation" if ratio < validation_fraction else "test"
        route = router.route(prefix)
        sequence, prefix_length = _encode_record(tokenizer, prefix, continuation)
        writers[split].append(sequence, prefix_length, route)
        record = {
            "id": source_id,
            "route": route,
            "prefix": prefix,
            "continuation": continuation,
            "continuation_utf8_bytes": len(continuation.encode("utf-8")),
        }
        _write_eval_json(validation_json if split == "validation" else test_json, record)
        accepted_eval += 1
        if max_eval and accepted_eval >= max_eval:
            break

    validation_json.close()
    test_json.close()
    split_stats = {split: writer.close() for split, writer in writers.items()}

    metadata = {
        "format_version": PREPARATION_FORMAT_VERSION,
        "preparation_signature": preparation_signature_value,
        "profile": cfg["_profile"],
        "seed": seed,
        "environment": environment_snapshot(),
        "protocol": {
            "conditioning_prefix_characters": prefix_chars,
            "conditioning_prefix_characters_requested": prefix_chars,
            "conditioning_boundary_rule": (
                "nearest preceding whitespace within the search radius, otherwise "
                "nearest following whitespace; continuation retains separator"
            ),
            "minimum_continuation_characters": min_cont,
            "prefix_and_continuation_tokenization_additive": True,
            "route_uses_only_conditioning_prefix": True,
            "loss_masks_conditioning_prefix": True,
            "tokenizer_roundtrip_audited": True,
        },
        "source": {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "train_sha256": sha256_file(train_path),
            "valid_sha256": sha256_file(valid_path),
            "observed_train_stories_first_pass": observed_train,
            "use_valid_as_train": bool(data_cfg.get("use_valid_as_train", False)),
        },
        "tokenizer": {
            "model_path": str(tokenizer_model),
            "model_sha256": sha256_file(tokenizer_model),
            "vocab_size": tokenizer.vocab_size(),
            "bos_id": tokenizer.bos_id(),
            "eos_id": tokenizer.eos_id(),
            "pad_id": tokenizer.pad_id(),
            "unk_id": tokenizer.unk_id(),
            "byte_fallback": True,
            "normalization": "identity",
        },
        "router": {
            **router_stats,
            "path": str(router_path),
            "path_sha256": sha256_file(router_path),
        },
        "splits": split_stats,
        "stream_dtype": "uint16",
    }
    atomic_json(out_dir / "metadata.json", metadata)
    atomic_json(done_path, metadata)
    return metadata


class IndexedStoryStore:
    def __init__(self, prepared_dir: Path) -> None:
        self.root = Path(prepared_dir)
        self.metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=str(self.root / "tinystories_bpe.model")
        )
        self.router = VSAProductRouter.load(self.root / "router.npz")
        self.num_routes = self.router.num_routes
        self.routing_prefix_characters = int(
            self.metadata["protocol"]["conditioning_prefix_characters"]
        )
        self._arrays: dict[tuple[str, str], np.ndarray] = {}
        self.pad_id = int(self.tokenizer.pad_id())

    def routing_prefix(self, text: str) -> str:
        """Apply the same raw-text boundary rule used when experts were trained."""
        prefix, _ = split_conditioning_prefix(text, self.routing_prefix_characters)
        return prefix

    def route_prompt(self, text: str) -> int:
        """Route an arbitrary request after applying the training boundary rule."""
        return self.router.route(self.routing_prefix(text))

    def route_conditioning_prefix(self, prefix: str) -> int:
        """Route an already-canonical prefix without applying the boundary twice."""
        return self.router.route(prefix)

    def _path(self, split: str, name: str) -> Path:
        return self.root / "indexed" / split / name

    def _array(self, split: str, name: str) -> np.ndarray:
        key = (split, name)
        if key not in self._arrays:
            if name == "tokens.u16":
                path = self._path(split, name)
                self._arrays[key] = np.memmap(path, dtype=np.uint16, mode="r")
            else:
                self._arrays[key] = np.load(self._path(split, name), mmap_mode="r")
        return self._arrays[key]

    def active_routes(self, split: str) -> np.ndarray:
        bounds = self._array(split, "route_bounds.npy")
        return np.flatnonzero(np.diff(bounds) > 0)

    def route_probabilities(self, split: str, exponent: float = 1.0) -> np.ndarray:
        counts = np.asarray(
            self.metadata["splits"][split]["route_target_token_counts"], dtype=np.float64
        )
        positive = counts > 0
        probs = np.zeros_like(counts)
        probs[positive] = np.power(counts[positive], float(exponent))
        probs /= probs.sum()
        return probs

    def _story_index_for_route(
        self, split: str, route: int, rng: np.random.Generator
    ) -> int:
        order = self._array(split, "route_order.npy")
        bounds = self._array(split, "route_bounds.npy")
        lo, hi = int(bounds[route]), int(bounds[route + 1])
        if hi <= lo:
            raise ValueError(f"Route {route} has no stories in split {split}")
        return int(order[int(rng.integers(lo, hi))])

    def batch(
        self,
        split: str,
        route: int,
        batch_size: int,
        block_size: int,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._array(split, "tokens.u16")
        offsets = self._array(split, "offsets.npy")
        prefix_lengths = self._array(split, "prefix_lengths.npy")
        x = np.full((batch_size, block_size), self.pad_id, dtype=np.int64)
        y = np.full((batch_size, block_size), -1, dtype=np.int64)
        maximum_usable = 0
        for row in range(batch_size):
            story = self._story_index_for_route(split, int(route), rng)
            begin, end = int(offsets[story]), int(offsets[story + 1])
            sequence = np.asarray(tokens[begin:end], dtype=np.int64)
            prefix_length = int(prefix_lengths[story])
            maximum_start = max(0, len(sequence) - (block_size + 1))
            minimum_start = max(0, prefix_length - block_size)
            minimum_start = min(minimum_start, maximum_start)
            start = int(rng.integers(minimum_start, maximum_start + 1))
            window = sequence[start : start + block_size + 1]
            usable = max(0, len(window) - 1)
            maximum_usable = max(maximum_usable, usable)
            if usable:
                x[row, :usable] = window[:-1]
                targets = window[1:].copy()
                absolute_targets = start + 1 + np.arange(usable)
                targets[absolute_targets < prefix_length] = -1
                y[row, :usable] = targets
            if not np.any(y[row] >= 0):
                raise RuntimeError("Batch sampler produced no causal continuation targets")
        # Avoid running attention, FFNs and the vocabulary projection over a
        # tail that is padding for every story in the batch.
        return (
            torch.from_numpy(x[:, :maximum_usable]),
            torch.from_numpy(y[:, :maximum_usable]),
        )

    def eval_records(self, split: str, limit: int = 0) -> list[dict[str, Any]]:
        path = self.root / "eval" / f"{split}.jsonl"
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
        return rows
