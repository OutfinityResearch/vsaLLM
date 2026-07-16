from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 2048
    block_size: int = 256
    d_model: int = 192
    n_layer: int = 6
    n_head: int = 6
    shared_hidden: int = 128
    expert_hidden: int = 57
    num_routes: int = 64
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: values[key] for key in allowed if key in values})


class RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, time, channels = x.shape
        head_dim = channels // self.n_head
        q, k, v = self.qkv(x).split(channels, dim=-1)
        q = q.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        past_length = 0
        if past is not None:
            past_length = past[0].shape[2]
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        if past is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif time == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Rare path, but keeps cached multi-token continuation strictly causal.
            query_pos = past_length + torch.arange(time, device=x.device)[:, None]
            key_pos = torch.arange(past_length + time, device=x.device)[None, :]
            mask = key_pos <= query_pos
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(y), ((k, v) if use_cache else None)


class RoutedMLP(nn.Module):
    """Shared FFN plus one route-specific tensor micro-expert.

    In evaluation, the two input matrices and two output matrices are
    concatenated per active route. This computes both branches with two GEMMs
    while retaining independent learned parameters.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, hs, he, routes = (
            cfg.d_model,
            cfg.shared_hidden,
            cfg.expert_hidden,
            cfg.num_routes,
        )
        self.shared_w1 = nn.Parameter(torch.empty(hs, d))
        self.shared_w2 = nn.Parameter(torch.empty(d, hs))
        self.expert_w1 = nn.Parameter(torch.empty(routes, he, d))
        self.expert_w2 = nn.Parameter(torch.empty(routes, d, he))
        self.shared_hidden = hs
        self._fused_cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cache_limit = 4
        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    @property
    def num_routes(self) -> int:
        return int(self.expert_w1.shape[0])

    def clear_fused(self) -> None:
        self._fused_cache.clear()

    def _apply(self, fn):
        # Cached tensors are derived from parameters and otherwise would keep
        # their old device/dtype after model.to(...).
        self.clear_fused()
        return super()._apply(fn)

    def prepare_route(self, route: int) -> None:
        route = int(route)
        if route in self._fused_cache:
            self._fused_cache.move_to_end(route)
            return
        with torch.no_grad():
            w1 = torch.cat((self.shared_w1, self.expert_w1[route]), dim=0).contiguous()
            w2 = torch.cat((self.shared_w2, self.expert_w2[route]), dim=1).contiguous()
        self._fused_cache[route] = (w1, w2)
        while len(self._fused_cache) > self._cache_limit:
            self._fused_cache.popitem(last=False)

    def _fused_weights(self, route: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training and route in self._fused_cache:
            self._fused_cache.move_to_end(route)
            return self._fused_cache[route]
        return (
            torch.cat((self.shared_w1, self.expert_w1[route]), dim=0),
            torch.cat((self.shared_w2, self.expert_w2[route]), dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        route: int,
        *,
        shared_scale: float = 1.0,
        expert_scale: float = 1.0,
        kernel_mode: str = "fused",
    ) -> torch.Tensor:
        route = int(route)
        if not 0 <= route < self.num_routes:
            raise ValueError(f"Route {route} outside [0, {self.num_routes})")
        if shared_scale == 0.0:
            return expert_scale * F.linear(
                F.gelu(F.linear(x, self.expert_w1[route])), self.expert_w2[route]
            )
        if expert_scale == 0.0:
            return shared_scale * F.linear(
                F.gelu(F.linear(x, self.shared_w1)), self.shared_w2
            )
        if kernel_mode == "separate":
            shared = F.linear(F.gelu(F.linear(x, self.shared_w1)), self.shared_w2)
            expert = F.linear(
                F.gelu(F.linear(x, self.expert_w1[route])), self.expert_w2[route]
            )
            return shared_scale * shared + expert_scale * expert
        if kernel_mode != "fused":
            raise ValueError(f"Unknown kernel mode: {kernel_mode}")
        w1, w2 = self._fused_weights(route)
        hidden = F.gelu(F.linear(x, w1))
        if shared_scale != 1.0 or expert_scale != 1.0:
            hidden = torch.cat(
                (
                    hidden[..., : self.shared_hidden] * shared_scale,
                    hidden[..., self.shared_hidden :] * expert_scale,
                ),
                dim=-1,
            )
        return F.linear(hidden, w2)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = RoutedMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        route: int,
        *,
        past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        shared_scale: float = 1.0,
        expert_scale: float = 1.0,
        kernel_mode: str = "fused",
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention, cache = self.attn(self.norm_attn(x), past=past, use_cache=use_cache)
        x = x + attention
        x = x + self.mlp(
            self.norm_mlp(x),
            route,
            shared_scale=shared_scale,
            expert_scale=expert_scale,
            kernel_mode=kernel_mode,
        )
        return x, cache


class VSAPathMoE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def clear_fused(self) -> None:
        for block in self.blocks:
            block.mlp.clear_fused()

    def prepare_route(self, route: int) -> None:
        for block in self.blocks:
            block.mlp.prepare_route(route)

    def forward(
        self,
        input_ids: torch.Tensor,
        route: int,
        targets: torch.Tensor | None = None,
        *,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        shared_scale: float = 1.0,
        expert_scale: float = 1.0,
        kernel_mode: str = "fused",
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        batch, time = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_length + time > self.config.block_size:
            raise ValueError(
                f"Context {past_length + time} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(
            past_length, past_length + time, device=input_ids.device, dtype=torch.long
        )
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, cache = block(
                x,
                route,
                past=past,
                use_cache=use_cache,
                shared_scale=shared_scale,
                expert_scale=expert_scale,
                kernel_mode=kernel_mode,
            )
            if cache is not None:
                new_cache.append(cache)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, (new_cache if use_cache else None)


def parameter_accounting(model: VSAPathMoE) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    all_experts = sum(
        block.mlp.expert_w1.numel() + block.mlp.expert_w2.numel()
        for block in model.blocks
    )
    one_route = sum(
        block.mlp.expert_w1[0].numel() + block.mlp.expert_w2[0].numel()
        for block in model.blocks
    )
    active = total - all_experts + one_route
    backbone_and_shared = total - all_experts
    return {
        "total_parameters": int(total),
        "active_parameters_per_request": int(active),
        "inactive_parameters_per_request": int(total - active),
        "active_fraction": float(active / total),
        "backbone_and_shared_parameters": int(backbone_and_shared),
        "all_route_expert_parameters": int(all_experts),
        "expert_parameters_per_route_all_layers": int(one_route),
        "fp32_total_megabytes": float(total * 4 / 1e6),
        "fp32_active_megabytes": float(active * 4 / 1e6),
        "fp32_expert_per_route_megabytes": float(one_route * 4 / 1e6),
    }


def set_trainable_phase(model: VSAPathMoE, phase: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = phase == "joint"
    if phase == "experts":
        for block in model.blocks:
            block.mlp.expert_w1.requires_grad = True
            block.mlp.expert_w2.requires_grad = True
    elif phase != "joint":
        raise ValueError(f"Unknown training phase: {phase}")


def save_checkpoint(
    path: Path,
    model: VSAPathMoE,
    *,
    optimizer_state: dict[str, Any] | None,
    training_state: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 2,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer_state,
            "training_state": training_state,
            "metadata": metadata,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path, *, device: str | torch.device = "cpu"
) -> tuple[VSAPathMoE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_mapping(checkpoint["model_config"])
    model = VSAPathMoE(cfg)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint
