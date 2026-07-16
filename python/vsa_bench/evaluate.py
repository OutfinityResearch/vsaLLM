from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .adapters import CustomAdapter, LanguageModelAdapter, OfficialAdapter
from .assets import validated_official_model_asset
from .common import (
    atomic_json,
    autocast_context,
    configure_threads,
    environment_snapshot,
    resolve_path,
    seed_all,
    stable_hash64,
)
from .train import evaluate_batches


def load_prompts(path: Path, limit: int = 0) -> list[str]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"Expected YAML list in {path}")
    prompts = [str(item) for item in values]
    return prompts[:limit] if limit else prompts


def _read_eval_records(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda row: stable_hash64(row["prefix"] + row["continuation"], seed))
    return records[:limit] if limit else records


def score_conditional_bpb(
    adapter: LanguageModelAdapter,
    records: list[dict[str, Any]],
    *,
    context_limit: int,
    stride: int,
) -> dict[str, Any]:
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    per_story: list[dict[str, Any]] = []
    start = time.perf_counter()
    for index, record in enumerate(records, 1):
        result = adapter.score_conditional(
            record["prefix"], record["continuation"], context_limit, stride
        )
        total_nll += float(result["nll"])
        total_tokens += int(result["tokens"])
        total_bytes += int(result["bytes"])
        per_story.append(
            {
                "id": record.get("id"),
                "nll": float(result["nll"]),
                "tokens": int(result["tokens"]),
                "bytes": int(result["bytes"]),
                "route": result.get("route"),
                "bits_per_byte": float(result["nll"])
                / math.log(2.0)
                / max(1, int(result["bytes"])),
            }
        )
        if index % 250 == 0:
            print(f"conditional BPB: {index}/{len(records)}", flush=True)
    nats_per_token = total_nll / max(1, total_tokens)
    return {
        "protocol": "fixed raw-text prefix; score continuation only",
        "stories": len(records),
        "continuation_tokens": total_tokens,
        "continuation_utf8_bytes": total_bytes,
        "total_nll_nats": total_nll,
        "nats_per_token_internal_only": nats_per_token,
        "token_perplexity_internal_only": math.exp(min(nats_per_token, 20.0)),
        "conditional_bits_per_byte": total_nll / math.log(2.0) / max(1, total_bytes),
        "context_limit": context_limit,
        "stride": stride,
        "elapsed_seconds": time.perf_counter() - start,
        "per_story": per_story,
    }


def evaluate_quality_contexts(
    adapter: LanguageModelAdapter,
    records: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate both native deployment context and a matched-context diagnostic.

    The native result is the conservative primary comparison: the official
    checkpoint is allowed to use its full published context. The common-context
    result isolates behavior when both continuous tensor windows are capped at
    the custom model's block size.
    """
    stride = int(eval_cfg["scoring_stride"])
    common_limit = min(int(eval_cfg["common_context_length"]), adapter.context_length)
    native_limit = int(adapter.context_length)
    native = score_conditional_bpb(
        adapter, records, context_limit=native_limit, stride=stride
    )
    native["context_mode"] = "native"
    if bool(eval_cfg.get("run_common_context_diagnostic", True)) and common_limit != native_limit:
        common = score_conditional_bpb(
            adapter, records, context_limit=common_limit, stride=stride
        )
        common["context_mode"] = "common"
    else:
        common = dict(native)
        common["context_mode"] = "common_equals_native"
    primary_name = str(eval_cfg.get("primary_quality_context", "native")).lower()
    if primary_name not in {"native", "common"}:
        raise ValueError(f"Unknown primary_quality_context: {primary_name}")
    primary = native if primary_name == "native" else common
    return {
        "primary_context_mode": primary_name,
        "heldout": primary,
        "heldout_native": native,
        "heldout_common_context": common,
    }


def text_diagnostics(texts: list[str]) -> dict[str, Any]:
    word_counts: list[int] = []
    repeated_trigram: list[float] = []
    distinct_1: list[float] = []
    distinct_2: list[float] = []
    sentence_end: list[bool] = []
    quote_balance: list[bool] = []
    for text in texts:
        words = re.findall(r"[A-Za-z']+", text.lower())
        word_counts.append(len(words))
        distinct_1.append(len(set(words)) / max(1, len(words)))
        bigrams = list(zip(words, words[1:]))
        distinct_2.append(len(set(bigrams)) / max(1, len(bigrams)))
        trigrams = list(zip(words, words[1:], words[2:]))
        repeated_trigram.append(1.0 - len(set(trigrams)) / max(1, len(trigrams)))
        sentence_end.append(bool(re.search(r"[.!?][\"']?\s*$", text.strip())))
        quote_balance.append(text.count('"') % 2 == 0)
    return {
        "count": len(texts),
        "mean_words": float(np.mean(word_counts)) if texts else 0.0,
        "median_words": float(np.median(word_counts)) if texts else 0.0,
        "mean_repeated_trigram_fraction": float(np.mean(repeated_trigram)) if texts else 0.0,
        "mean_distinct_1": float(np.mean(distinct_1)) if texts else 0.0,
        "mean_distinct_2": float(np.mean(distinct_2)) if texts else 0.0,
        "sentence_end_fraction": float(np.mean(sentence_end)) if texts else 0.0,
        "balanced_double_quotes_fraction": float(np.mean(quote_balance)) if texts else 0.0,
        "empty_fraction": float(np.mean([not text.strip() for text in texts])) if texts else 0.0,
    }


def generate_suite(
    adapter: LanguageModelAdapter,
    prompts: list[str],
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    samples_per_prompt = int(cfg["samples_per_prompt"])
    for prompt_index, prompt in enumerate(prompts):
        greedy = adapter.generate(
            prompt,
            int(cfg["max_new_tokens"]),
            1.0,
            0,
            1.0,
            seed + prompt_index,
            True,
        )
        samples = [
            adapter.generate(
                prompt,
                int(cfg["max_new_tokens"]),
                float(cfg["temperature"]),
                int(cfg["top_k"]),
                float(cfg["top_p"]),
                seed + 100_000 + prompt_index * 100 + sample_index,
                False,
            )
            for sample_index in range(samples_per_prompt)
        ]
        rows.append({"id": prompt_index, "prompt": prompt, "greedy": greedy, "samples": samples})
        if (prompt_index + 1) % 10 == 0:
            print(f"generation: {prompt_index + 1}/{len(prompts)}", flush=True)
    sampled = [item["text"] for row in rows for item in row["samples"]]
    greedy = [row["greedy"]["text"] for row in rows]
    return {
        "prompt_count": len(prompts),
        "samples_per_prompt": samples_per_prompt,
        "greedy_diagnostics": text_diagnostics(greedy),
        "sampled_diagnostics": text_diagnostics(sampled),
        "items": rows,
    }


def _hellaswag_dataset(max_examples: int, seed: int, local_path: str | None = None):
    from datasets import load_dataset, load_from_disk

    if local_path and Path(local_path).exists():
        dataset = load_from_disk(local_path)
    else:
        dataset = load_dataset("Rowan/hellaswag", split="validation")
    if max_examples > 0 and max_examples < len(dataset):
        indices = np.random.default_rng(seed).choice(len(dataset), max_examples, replace=False)
        dataset = dataset.select(sorted(int(index) for index in indices))
    return dataset


@torch.inference_mode()
def score_options(
    adapter: LanguageModelAdapter,
    context: str,
    options: list[str],
    context_limit: int,
) -> tuple[list[float], list[float], bool]:
    """Score HellaSwag endings with one option-independent visible context.

    The route is computed from exactly the context tokens visible to the custom
    Transformer, never from text that was truncated away.  We reserve space for
    the longest ending first, so all four options receive the same context and
    no route can depend on an option's token length.
    """
    context_ids_full = adapter.encode(context)
    option_texts = [" " + option for option in options]
    option_ids_all: list[list[int]] = []
    for option_text in option_texts:
        option_ids = adapter.encode(option_text)
        if [*context_ids_full, *option_ids] != adapter.encode(context + option_text):
            raise RuntimeError(
                f"Unsafe HellaSwag boundary for {adapter.name}; context and option "
                "tokenization are not additive."
            )
        option_ids_all.append(option_ids)

    longest_option = max((len(ids) for ids in option_ids_all), default=0)
    if longest_option > int(context_limit):
        raise RuntimeError(
            f"A HellaSwag ending needs {longest_option} tokens for {adapter.name}, "
            f"exceeding context limit {context_limit}."
        )
    maximum_context = max(0, int(context_limit) - longest_option)
    truncated = len(context_ids_full) > maximum_context
    context_ids = (
        context_ids_full[-maximum_context:] if truncated and maximum_context else
        ([] if truncated else context_ids_full)
    )
    visible_context = adapter.decode(context_ids)
    route = adapter.route(visible_context)
    if isinstance(adapter, CustomAdapter):
        adapter.model.prepare_route(int(route))

    raw_scores: list[float] = []
    normalized: list[float] = []
    for option_ids in option_ids_all:
        sequence = [adapter.bos_id(), *context_ids, *option_ids]
        x = torch.tensor([sequence[:-1]], dtype=torch.long, device=adapter.device)
        y = torch.tensor(sequence[1:], dtype=torch.long, device=adapter.device)
        start = len(context_ids)
        with autocast_context(adapter.device, adapter.dtype):
            if isinstance(adapter, CustomAdapter):
                logits, _, _ = adapter.model(
                    x, int(route), kernel_mode=adapter.kernel_mode
                )
            elif isinstance(adapter, OfficialAdapter):
                logits = adapter.model(input_ids=x, use_cache=False).logits
            else:
                raise TypeError(type(adapter))
        logp = F.log_softmax(logits[0, start:], dim=-1)
        target = y[start:]
        score = float(logp.gather(1, target[:, None]).sum())
        raw_scores.append(score)
        normalized.append(score / max(1, len(target)))
    return raw_scores, normalized, truncated


def evaluate_hellaswag(
    adapter: LanguageModelAdapter,
    *,
    max_examples: int,
    seed: int,
    common_context: int,
    local_path: str | None = None,
) -> dict[str, Any]:
    if max_examples < 0:
        return {"skipped": True}
    dataset = _hellaswag_dataset(max_examples, seed, local_path)
    raw_correct = 0
    normalized_correct = 0
    truncated = 0
    start = time.perf_counter()
    for index, row in enumerate(dataset, 1):
        raw, norm, was_truncated = score_options(
            adapter,
            str(row["ctx"]),
            list(row["endings"]),
            min(common_context, adapter.context_length),
        )
        label = int(row["label"])
        raw_correct += int(np.argmax(raw) == label)
        normalized_correct += int(np.argmax(norm) == label)
        truncated += int(was_truncated)
        if index % 250 == 0:
            print(f"HellaSwag: {index}/{len(dataset)}", flush=True)
    count = len(dataset)
    return {
        "examples": count,
        "accuracy_raw": raw_correct / max(1, count),
        "accuracy_length_normalized": normalized_correct / max(1, count),
        "chance": 0.25,
        "context_limit": min(common_context, adapter.context_length),
        "example_truncation_fraction": truncated / max(1, count),
        "elapsed_seconds": time.perf_counter() - start,
    }


def custom_ablations(adapter: CustomAdapter, cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 404)
    probabilities = adapter.store.route_probabilities("test", exponent=1.0)
    batches = []
    for route in rng.choice(
        adapter.store.num_routes, size=int(cfg["ablation_batches"]), p=probabilities
    ):
        x, y = adapter.store.batch(
            "test",
            int(route),
            int(cfg["ablation_batch_size"]),
            adapter.model.config.block_size,
            rng,
        )
        batches.append((int(route), x, y))
    if adapter.model.config.num_routes == 1:
        return {
            "full": evaluate_batches(
                adapter.model, batches, adapter.device, adapter.dtype
            ),
            "skipped": True,
            "reason": "Routing ablations do not apply to the fixed-route dense control",
        }
    wrong = lambda route: (int(route) * 17 + 13) % adapter.store.num_routes
    return {
        "full": evaluate_batches(adapter.model, batches, adapter.device, adapter.dtype),
        "shared_only": evaluate_batches(
            adapter.model, batches, adapter.device, adapter.dtype, expert_scale=0.0
        ),
        "expert_only": evaluate_batches(
            adapter.model, batches, adapter.device, adapter.dtype, shared_scale=0.0
        ),
        "permuted_route": evaluate_batches(
            adapter.model, batches, adapter.device, adapter.dtype, route_transform=wrong
        ),
        "fixed_route_zero": evaluate_batches(
            adapter.model,
            batches,
            adapter.device,
            adapter.dtype,
            route_transform=lambda route: 0,
        ),
    }


def _assets(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads(resolve_path(cfg["paths"]["assets_manifest"]).read_text(encoding="utf-8"))


def _common_inputs(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    prepared = resolve_path(cfg["data"]["prepared_dir"])
    eval_cfg = cfg["evaluation"]
    records = _read_eval_records(
        prepared / "eval" / "test.jsonl",
        int(eval_cfg["test_stories"]),
        int(cfg["project"]["seed"]) + 50,
    )
    prompts = load_prompts(
        Path(_assets(cfg)["dataset"]["prompts_path"]), int(eval_cfg["prompt_count"])
    )
    return records, prompts


def evaluate_custom(cfg: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    seed = int(cfg["project"]["seed"])
    seed_all(seed)
    configure_threads(int(cfg["runtime"]["threads"]))
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    checkpoint_path = Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "best.pt"
    adapter = CustomAdapter.load(
        checkpoint_path,
        resolve_path(cfg["data"]["prepared_dir"]),
        device_name=cfg["runtime"]["device"],
        dtype_name=cfg["runtime"]["dtype"],
        kernel_mode=cfg["benchmark"]["custom_kernel_mode"],
        expected_routing_mode=cfg["model"].get("routing_mode", "vsa"),
    )
    records, prompts = _common_inputs(cfg)
    eval_cfg = cfg["evaluation"]
    quality_contexts = evaluate_quality_contexts(adapter, records, eval_cfg)
    generations = generate_suite(adapter, prompts, eval_cfg["generation"], seed)
    hellaswag = evaluate_hellaswag(
        adapter,
        max_examples=int(eval_cfg["hellaswag_examples"]),
        seed=seed + 70,
        common_context=int(eval_cfg["common_context_length"]),
        local_path=(_assets(cfg).get("hellaswag") or {}).get("path"),
    )
    result = {
        "schema_version": 2,
        "profile": cfg["_profile"],
        "environment": environment_snapshot(),
        "model": adapter.metadata(),
        "checkpoint": str(checkpoint_path),
        "quality": {**quality_contexts, "hellaswag": hellaswag},
        "generations": generations,
        "ablations": custom_ablations(adapter, eval_cfg, seed),
    }
    out = run_dir / "evaluation" / "custom.json"
    atomic_json(out, result)
    atomic_json(run_dir / "evaluation" / "custom_generations.json", generations)
    return result


def evaluate_official(cfg: dict[str, Any], model_index: int = 0) -> dict[str, Any]:
    seed = int(cfg["project"]["seed"])
    seed_all(seed)
    configure_threads(int(cfg["runtime"]["threads"]))
    assets = _assets(cfg)
    item = validated_official_model_asset(cfg, assets, model_index)
    adapter = OfficialAdapter.load(
        item["id"],
        item["snapshot_path"],
        revision=item.get("resolved_sha") or item.get("requested_revision"),
        device_name=cfg["runtime"]["device"],
        dtype_name=cfg["runtime"]["dtype"],
        trained_context_length=item.get("trained_context_length"),
    )
    records, prompts = _common_inputs(cfg)
    eval_cfg = cfg["evaluation"]
    result = {
        "schema_version": 2,
        "profile": cfg["_profile"],
        "environment": environment_snapshot(),
        "model": adapter.metadata(),
        "quality": {
            **evaluate_quality_contexts(adapter, records, eval_cfg),
            "hellaswag": evaluate_hellaswag(
                adapter,
                max_examples=int(eval_cfg["hellaswag_examples"]),
                seed=seed + 70,
                common_context=int(eval_cfg["common_context_length"]),
                local_path=(_assets(cfg).get("hellaswag") or {}).get("path"),
            ),
        },
        "generations": generate_suite(adapter, prompts, eval_cfg["generation"], seed),
    }
    run_dir = resolve_path(cfg["paths"]["run_dir"]) / cfg["_profile"]
    safe = item["id"].replace("/", "__")
    atomic_json(run_dir / "evaluation" / f"official_{safe}.json", result)
    atomic_json(
        run_dir / "evaluation" / f"official_{safe}_generations.json",
        result["generations"],
    )
    return result
