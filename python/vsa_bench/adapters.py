from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    autocast_context,
    dtype_from_name,
    select_device,
    sha256_file,
    validate_prepared_compatibility,
)
from .data import IndexedStoryStore
from .model import VSAPathMoE, load_checkpoint, parameter_accounting


class LanguageModelAdapter(Protocol):
    name: str
    context_length: int
    vocab_size: int
    device: torch.device
    dtype: torch.dtype

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def bos_id(self) -> int: ...
    def eos_id(self) -> int: ...
    def route(self, text: str) -> int | None: ...
    def prepare_request(self, text: str) -> None: ...
    def score_conditional(
        self, prefix: str, continuation: str, context_limit: int, stride: int
    ) -> dict[str, Any]: ...
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
        greedy: bool,
    ) -> dict[str, Any]: ...
    def metadata(self) -> dict[str, Any]: ...


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
    greedy: bool,
) -> torch.Tensor:
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(float(temperature), 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits = logits.masked_fill(logits < values[:, -1, None], -float("inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        restored = torch.full_like(logits, -float("inf"))
        restored.scatter_(1, sorted_indices, sorted_logits)
        logits = restored
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)


def _assert_additive_boundary(adapter: LanguageModelAdapter, prefix: str, continuation: str) -> tuple[list[int], list[int]]:
    """Return separate encodings only when they exactly equal joint encoding.

    Without this check, a BPE token may cross the raw prefix boundary and make
    conditional bits-per-byte depend on an arbitrary tokenizer segmentation.
    Prepared TinyStories records split immediately before whitespace, so both
    the lossless SentencePiece tokenizer and GPT-2-style tokenizer should pass.
    """
    prefix_ids = adapter.encode(prefix)
    continuation_ids = adapter.encode(continuation)
    combined_ids = adapter.encode(prefix + continuation)
    if [*prefix_ids, *continuation_ids] != combined_ids:
        raise RuntimeError(
            f"Unsafe conditional boundary for {adapter.name}: encode(prefix) + "
            "encode(continuation) != encode(prefix + continuation). Re-run data "
            "preparation with the tokenizer-stable boundary rule."
        )
    return prefix_ids, continuation_ids


def _conditional_windows(
    sequence: list[int], start_target: int, context_limit: int, stride: int
):
    limit = max(2, int(context_limit))
    stride = min(max(1, int(stride)), limit - 1)
    next_position = int(start_target)
    while next_position < len(sequence):
        end = min(next_position + stride, len(sequence))
        begin = max(0, end - (limit + 1))
        window = sequence[begin:end]
        absolute = np.arange(begin + 1, end)
        keep = np.flatnonzero(absolute >= next_position)
        yield window, keep
        next_position = end


@dataclass
class CustomAdapter:
    name: str
    model: VSAPathMoE
    store: IndexedStoryStore
    device: torch.device
    dtype: torch.dtype
    kernel_mode: str = "fused"
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    routing_mode: str = "vsa"
    checkpoint_provenance: dict[str, Any] | None = None

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        prepared_dir: Path,
        *,
        device_name: str,
        dtype_name: str,
        kernel_mode: str = "fused",
        expected_routing_mode: str = "vsa",
    ) -> "CustomAdapter":
        device = select_device(device_name)
        dtype = dtype_from_name(dtype_name, device)
        model, checkpoint_data = load_checkpoint(checkpoint, device="cpu")
        store = IndexedStoryStore(prepared_dir)
        checkpoint_metadata = checkpoint_data.get("metadata", {})
        routing_mode = str(checkpoint_metadata.get("routing_mode", "vsa"))
        validate_prepared_compatibility(
            checkpoint_metadata,
            store.metadata,
            routing_mode=expected_routing_mode,
        )
        if model.config.num_routes != store.num_routes and not (
            routing_mode == "fixed_dense" and model.config.num_routes == 1
        ):
            raise RuntimeError("Checkpoint route count does not match the prepared router")
        model = model.to(device=device, dtype=dtype).eval()
        name = "Dense matched control" if routing_mode == "fixed_dense" else "VSA-PathMoE-10M"
        checkpoint_prepared = checkpoint_metadata.get("prepared_metadata", {})
        checkpoint_environment = checkpoint_metadata.get("environment", {})
        provenance = {
            "checkpoint_format_version": checkpoint_data.get("format_version"),
            "training_profile": checkpoint_metadata.get("profile"),
            "resolved_config_sha256": checkpoint_metadata.get(
                "resolved_config_sha256"
            ),
            "routing_mode": routing_mode,
            "preparation_signature": checkpoint_prepared.get(
                "preparation_signature"
            ),
            "tokenizer_sha256": checkpoint_prepared.get("tokenizer", {}).get(
                "model_sha256"
            ),
            "router_sha256": checkpoint_prepared.get("router", {}).get(
                "path_sha256"
            ),
            "training_git_commit": checkpoint_environment.get("git_commit"),
            "training_git_worktree_dirty": checkpoint_environment.get(
                "git_worktree_dirty"
            ),
            "training_runtime_source_sha256": checkpoint_environment.get(
                "runtime_source_sha256"
            ),
            "training_machine": checkpoint_environment.get("machine"),
            "training_python": checkpoint_environment.get("python"),
            "training_torch_version": checkpoint_environment.get("torch"),
            "training_numpy_version": checkpoint_environment.get("numpy"),
            "training_runtime_dependencies": checkpoint_environment.get(
                "runtime_dependencies"
            ),
            "training_cuda_version": checkpoint_environment.get("cuda_version"),
            "training_cuda_devices": checkpoint_environment.get("cuda_devices"),
            "training_cudnn_version": checkpoint_environment.get("cudnn_version"),
            "training_nvidia_driver_version": checkpoint_environment.get(
                "nvidia_driver_version"
            ),
            "training_containerized": checkpoint_environment.get("containerized"),
            "training_container_image": checkpoint_environment.get("container_image"),
            "training_container_image_digest": checkpoint_environment.get(
                "container_image_digest"
            ),
            "training_container_derived_image_id": checkpoint_environment.get(
                "container_derived_image_id"
            ),
        }
        return cls(
            name,
            model,
            store,
            device,
            dtype,
            kernel_mode,
            str(checkpoint.resolve()),
            sha256_file(checkpoint),
            routing_mode,
            provenance,
        )

    @property
    def context_length(self) -> int:
        return self.model.config.block_size

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.store.tokenizer.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.store.tokenizer.decode(ids)

    def bos_id(self) -> int:
        return int(self.store.tokenizer.bos_id())

    def eos_id(self) -> int:
        return int(self.store.tokenizer.eos_id())

    def route(self, text: str) -> int:
        if self.routing_mode == "fixed_dense":
            return 0
        return self.store.route_prompt(text)

    def route_conditioning_prefix(self, prefix: str) -> int:
        if self.routing_mode == "fixed_dense":
            return 0
        return self.store.route_conditioning_prefix(prefix)

    def prepare_request(self, text: str) -> None:
        self.model.prepare_route(self.route(text))

    @torch.inference_mode()
    def score_conditional(
        self, prefix: str, continuation: str, context_limit: int, stride: int
    ) -> dict[str, Any]:
        prefix_ids, continuation_ids = _assert_additive_boundary(
            self, prefix, continuation
        )
        if not continuation_ids:
            return {
                "nll": 0.0,
                "tokens": 0,
                "bytes": len(continuation.encode("utf-8")),
                "route": self.route_conditioning_prefix(prefix),
            }
        sequence = [self.bos_id(), *prefix_ids, *continuation_ids]
        start_target = 1 + len(prefix_ids)
        route = self.route_conditioning_prefix(prefix)
        self.model.prepare_route(route)
        total_nll = 0.0
        total_tokens = 0
        for window, keep in _conditional_windows(
            sequence, start_target, min(context_limit, self.context_length), stride
        ):
            x = torch.tensor([window[:-1]], dtype=torch.long, device=self.device)
            targets = torch.tensor(window[1:], dtype=torch.long, device=self.device)
            with autocast_context(self.device, self.dtype):
                logits, _, _ = self.model(x, route, kernel_mode=self.kernel_mode)
            if len(keep):
                keep_t = torch.as_tensor(keep, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(
                    logits[0, keep_t], targets[keep_t], reduction="sum"
                )
                total_nll += float(loss)
                total_tokens += int(len(keep))
        return {
            "nll": total_nll,
            "tokens": total_tokens,
            "bytes": len(continuation.encode("utf-8", errors="replace")),
            "route": route,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
        greedy: bool,
    ) -> dict[str, Any]:
        route = self.route(prompt)
        self.model.prepare_route(route)
        prompt_ids = [self.bos_id(), *self.encode(prompt)]
        if len(prompt_ids) + int(max_new_tokens) > self.context_length:
            raise RuntimeError(
                f"Generation request does not fit {self.name}: {len(prompt_ids)} prompt "
                f"tokens + {max_new_tokens} new tokens > context {self.context_length}. "
                "The prompt is never silently truncated; reduce max_new_tokens."
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device.type).manual_seed(seed)
        start = time.perf_counter()
        with autocast_context(self.device, self.dtype):
            logits, _, cache = self.model(
                input_ids,
                route,
                use_cache=True,
                kernel_mode=self.kernel_mode,
            )
        generated: list[int] = []
        for step in range(int(max_new_tokens)):
            next_id = _sample_next(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                greedy=greedy,
            )
            token = int(next_id.item())
            generated.append(token)
            if token == self.eos_id() or step + 1 >= int(max_new_tokens):
                break
            with autocast_context(self.device, self.dtype):
                logits, _, cache = self.model(
                    next_id,
                    route,
                    past_key_values=cache,
                    use_cache=True,
                    kernel_mode=self.kernel_mode,
                )
        return {
            "text": self.decode(generated),
            "route": route,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "elapsed_seconds": time.perf_counter() - start,
        }

    def metadata(self) -> dict[str, Any]:
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        router_arrays = (
            self.store.router.token_content,
            self.store.router.token_order,
            self.store.router.roles,
            self.store.router.proto_a,
            self.store.router.proto_b,
            self.store.router._proto_a16,
            self.store.router._proto_b16,
        )
        router_resident_bytes = sum(int(value.nbytes) for value in router_arrays)
        fused_cache_current_bytes = 0
        fused_cache_maximum_bytes = 0
        fused_cache_capacity_routes = 0
        for block in self.model.blocks:
            mlp = block.mlp
            fused_cache_capacity_routes = max(
                fused_cache_capacity_routes,
                min(int(mlp._cache_limit), int(mlp.num_routes)),
            )
            for w1, w2 in mlp._fused_cache.values():
                fused_cache_current_bytes += (
                    w1.numel() * w1.element_size()
                    + w2.numel() * w2.element_size()
                )
            one_route_elements = (
                (mlp.shared_w1.shape[0] + mlp.expert_w1.shape[1])
                * mlp.shared_w1.shape[1]
                + mlp.shared_w2.shape[0]
                * (mlp.shared_w2.shape[1] + mlp.expert_w2.shape[2])
            )
            fused_cache_maximum_bytes += (
                int(one_route_elements)
                * mlp.shared_w1.element_size()
                * min(int(mlp._cache_limit), int(mlp.num_routes))
            )
        persistent_current = (
            int(parameter_bytes)
            + router_resident_bytes
            + fused_cache_current_bytes
        )
        persistent_maximum = (
            int(parameter_bytes)
            + router_resident_bytes
            + fused_cache_maximum_bytes
        )
        return {
            "name": self.name,
            "kind": (
                "matched_dense_control"
                if self.routing_mode == "fixed_dense"
                else "custom_vsa_pathmoe"
            ),
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "kernel_mode": self.kernel_mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_provenance": self.checkpoint_provenance,
            "checkpoint_bytes": (
                Path(self.checkpoint_path).stat().st_size
                if self.checkpoint_path is not None
                else None
            ),
            "resident_parameter_bytes": int(parameter_bytes),
            "persistent_model_state_current_bytes": persistent_current,
            "persistent_model_state_max_bytes": persistent_maximum,
            "fused_route_cache": {
                "capacity_routes": fused_cache_capacity_routes,
                "current_bytes": fused_cache_current_bytes,
                "maximum_bytes": fused_cache_maximum_bytes,
            },
            "routing_policy": {
                "type": (
                    "fixed route zero"
                    if self.routing_mode == "fixed_dense"
                    else "VSA over fixed conditioning prefix"
                ),
                "conditioning_prefix_characters": self.store.routing_prefix_characters,
                "router_resident_array_bytes": router_resident_bytes,
                "router_artifact_bytes": int(
                    (self.store.root / "router.npz").stat().st_size
                ),
            },
            "parameters": parameter_accounting(self.model),
        }


@dataclass
class OfficialAdapter:
    name: str
    model_id: str
    revision: str | None
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    local_path: str
    trained_context_length: int | None = None

    @classmethod
    def load(
        cls,
        model_id: str,
        local_path: str,
        *,
        revision: str | None,
        device_name: str,
        dtype_name: str,
        trained_context_length: int | None = None,
    ) -> "OfficialAdapter":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = select_device(device_name)
        dtype = dtype_from_name(dtype_name, device)
        tokenizer = AutoTokenizer.from_pretrained(
            local_path, local_files_only=True, use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            torch_dtype=(dtype if device.type != "cpu" or dtype != torch.float32 else None),
        ).to(device)
        model.eval()
        return cls(
            model_id,
            model_id,
            revision,
            model,
            tokenizer,
            device,
            dtype,
            local_path,
            (int(trained_context_length) if trained_context_length else None),
        )

    @property
    def context_length(self) -> int:
        if self.trained_context_length is not None:
            return self.trained_context_length
        cfg = self.model.config
        return int(
            getattr(cfg, "max_position_embeddings", None)
            or getattr(cfg, "n_positions", None)
            or 2048
        )

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def bos_id(self) -> int:
        value = self.tokenizer.bos_token_id
        if value is None:
            value = self.tokenizer.eos_token_id
        if value is None:
            raise RuntimeError("Official tokenizer has no BOS/EOS token")
        return int(value)

    def eos_id(self) -> int:
        value = self.tokenizer.eos_token_id
        return self.bos_id() if value is None else int(value)

    def route(self, text: str) -> None:
        return None

    def prepare_request(self, text: str) -> None:
        return None

    @torch.inference_mode()
    def score_conditional(
        self, prefix: str, continuation: str, context_limit: int, stride: int
    ) -> dict[str, Any]:
        prefix_ids, continuation_ids = _assert_additive_boundary(
            self, prefix, continuation
        )
        if not continuation_ids:
            return {
                "nll": 0.0,
                "tokens": 0,
                "bytes": len(continuation.encode("utf-8")),
            }
        sequence = [self.bos_id(), *prefix_ids, *continuation_ids]
        start_target = 1 + len(prefix_ids)
        total_nll = 0.0
        total_tokens = 0
        for window, keep in _conditional_windows(
            sequence, start_target, min(context_limit, self.context_length), stride
        ):
            x = torch.tensor([window[:-1]], dtype=torch.long, device=self.device)
            targets = torch.tensor(window[1:], dtype=torch.long, device=self.device)
            with autocast_context(self.device, self.dtype):
                output = self.model(input_ids=x, use_cache=False)
            if len(keep):
                keep_t = torch.as_tensor(keep, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(
                    output.logits[0, keep_t], targets[keep_t], reduction="sum"
                )
                total_nll += float(loss)
                total_tokens += int(len(keep))
        return {
            "nll": total_nll,
            "tokens": total_tokens,
            "bytes": len(continuation.encode("utf-8", errors="replace")),
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int,
        greedy: bool,
    ) -> dict[str, Any]:
        prompt_ids = [self.bos_id(), *self.encode(prompt)]
        if len(prompt_ids) + int(max_new_tokens) > self.context_length:
            raise RuntimeError(
                f"Generation request does not fit {self.name}: {len(prompt_ids)} prompt "
                f"tokens + {max_new_tokens} new tokens > context {self.context_length}. "
                "The prompt is never silently truncated; reduce max_new_tokens."
            )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generator = torch.Generator(device=self.device.type).manual_seed(seed)
        start = time.perf_counter()
        with autocast_context(self.device, self.dtype):
            output = self.model(input_ids=input_ids, use_cache=True)
        logits, cache = output.logits, output.past_key_values
        generated: list[int] = []
        for step in range(int(max_new_tokens)):
            next_id = _sample_next(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
                greedy=greedy,
            )
            token = int(next_id.item())
            generated.append(token)
            if token == self.eos_id() or step + 1 >= int(max_new_tokens):
                break
            with autocast_context(self.device, self.dtype):
                output = self.model(
                    input_ids=next_id, past_key_values=cache, use_cache=True
                )
            logits, cache = output.logits, output.past_key_values
        return {
            "text": self.decode(generated),
            "route": None,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated),
            "elapsed_seconds": time.perf_counter() - start,
        }

    def metadata(self) -> dict[str, Any]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        configured_context = int(
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "n_positions", None)
            or 2048
        )
        weight_artifact_bytes = sum(
            path.stat().st_size
            for path in Path(self.local_path).iterdir()
            if path.is_file() and path.suffix in {".bin", ".safetensors"}
        )
        return {
            "name": self.name,
            "kind": "official_huggingface",
            "model_id": self.model_id,
            "revision": self.revision,
            "local_path": self.local_path,
            "weight_artifact_bytes": int(weight_artifact_bytes),
            "architecture": self.model.__class__.__name__,
            "context_length": self.context_length,
            "artifact_config_context_length": configured_context,
            "trained_context_length_override": self.trained_context_length,
            "vocab_size": self.vocab_size,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "parameters": {
                "total_parameters": int(total),
                "fp32_megabytes": float(total * 4 / 1e6),
                "resident_parameter_bytes": int(parameter_bytes),
            },
            "resident_parameter_bytes": int(parameter_bytes),
            "persistent_model_state_current_bytes": int(parameter_bytes),
            "persistent_model_state_max_bytes": int(parameter_bytes),
            "config": self.model.config.to_dict(),
        }
