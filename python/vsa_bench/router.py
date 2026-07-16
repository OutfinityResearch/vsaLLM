from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import entropy_bits


@dataclass
class RouterConfig:
    dimension: int = 512
    prefix_bytes: int = 512
    role_period: int = 64
    product_k: int = 8
    kmeans_iterations: int = 12
    seed: int = 20260727


class VSAProductRouter:
    """Bipolar VSA router with two product-key codebooks.

    The first half bundles byte identity. The second half binds each byte to a
    periodic positional role before bundling. Routing is nearest prototype in
    each half, implemented as int16 dot products (equivalent to Hamming ranking
    for bipolar vectors). A native implementation can replace this with
    XNOR+popcount without changing route semantics.
    """

    def __init__(
        self,
        token_content: np.ndarray,
        token_order: np.ndarray,
        roles: np.ndarray,
        proto_a: np.ndarray,
        proto_b: np.ndarray,
        prefix_bytes: int,
    ) -> None:
        self.token_content = np.asarray(token_content, dtype=np.int8)
        self.token_order = np.asarray(token_order, dtype=np.int8)
        self.roles = np.asarray(roles, dtype=np.int8)
        self.proto_a = np.asarray(proto_a, dtype=np.int8)
        self.proto_b = np.asarray(proto_b, dtype=np.int8)
        self._proto_a16 = self.proto_a.astype(np.int16)
        self._proto_b16 = self.proto_b.astype(np.int16)
        self.prefix_bytes = int(prefix_bytes)
        if self.token_content.shape[0] != 256 or self.token_order.shape[0] != 256:
            raise ValueError("Byte codebooks must have 256 entries")
        if self.proto_a.shape[1] != self.token_content.shape[1]:
            raise ValueError("Prototype A dimension mismatch")
        if self.proto_b.shape[1] != self.token_order.shape[1]:
            raise ValueError("Prototype B dimension mismatch")

    @property
    def dimension(self) -> int:
        return int(self.proto_a.shape[1] + self.proto_b.shape[1])

    @property
    def product_k(self) -> int:
        return int(self.proto_a.shape[0])

    @property
    def num_routes(self) -> int:
        return int(self.proto_a.shape[0] * self.proto_b.shape[0])

    def code(self, text: str) -> np.ndarray:
        raw = np.frombuffer(
            text.encode("utf-8", errors="replace")[: self.prefix_bytes], dtype=np.uint8
        )
        if len(raw) == 0:
            return np.ones(self.dimension, dtype=np.int8)
        content = self.token_content[raw].sum(axis=0, dtype=np.int32)
        role_ids = np.arange(len(raw), dtype=np.int64) % len(self.roles)
        ordered = (self.token_order[raw] * self.roles[role_ids]).sum(axis=0, dtype=np.int32)
        return np.concatenate(
            (np.where(content >= 0, 1, -1), np.where(ordered >= 0, 1, -1))
        ).astype(np.int8)

    def route_from_code(self, code: np.ndarray) -> int:
        code = np.asarray(code, dtype=np.int8)
        half = self.proto_a.shape[1]
        code16 = code.astype(np.int16)
        score_a = self._proto_a16 @ code16[:half]
        score_b = self._proto_b16 @ code16[half:]
        return int(np.argmax(score_a) * self.proto_b.shape[0] + np.argmax(score_b))

    def route(self, text: str) -> int:
        return self.route_from_code(self.code(text))

    def route_with_margin(self, text: str) -> tuple[int, int]:
        code = self.code(text)
        half = self.proto_a.shape[1]
        code16 = code.astype(np.int16)
        a = self._proto_a16 @ code16[:half]
        b = self._proto_b16 @ code16[half:]
        ia = np.argsort(a)[-2:]
        ib = np.argsort(b)[-2:]
        route = int(ia[-1] * self.proto_b.shape[0] + ib[-1])
        margin = int(min(a[ia[-1]] - a[ia[-2]], b[ib[-1]] - b[ib[-2]]))
        return route, margin

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            token_content=self.token_content,
            token_order=self.token_order,
            roles=self.roles,
            proto_a=self.proto_a,
            proto_b=self.proto_b,
            prefix_bytes=np.asarray(self.prefix_bytes, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "VSAProductRouter":
        values = np.load(path)
        return cls(
            values["token_content"],
            values["token_order"],
            values["roles"],
            values["proto_a"],
            values["proto_b"],
            int(values["prefix_bytes"]),
        )


def _binary_kmeans(
    codes: np.ndarray,
    clusters: int,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(codes) < clusters:
        raise ValueError(f"Need at least {clusters} samples, got {len(codes)}")
    prototypes = codes[rng.choice(len(codes), clusters, replace=False)].copy().astype(np.int8)
    history: list[dict[str, Any]] = []
    codes16 = codes.astype(np.int16)
    for iteration in range(iterations):
        assignments = (codes16 @ prototypes.T.astype(np.int16)).argmax(axis=1)
        updated = prototypes.copy()
        counts: list[int] = []
        for cluster in range(clusters):
            members = np.flatnonzero(assignments == cluster)
            counts.append(int(len(members)))
            if len(members):
                updated[cluster] = np.where(codes[members].sum(axis=0) >= 0, 1, -1)
            else:
                updated[cluster] = codes[rng.integers(len(codes))]
        changed = int(np.sum(updated != prototypes))
        prototypes = updated.astype(np.int8)
        best = np.max(codes16 @ prototypes.T.astype(np.int16), axis=1)
        history.append(
            {
                "iteration": iteration,
                "bit_changes": changed,
                "mean_hamming_distance": float((codes.shape[1] - best).mean() / 2),
                "cluster_counts": counts,
            }
        )
        if changed == 0:
            break
    return prototypes, history


def fit_router(prefixes: list[str], cfg: RouterConfig) -> tuple[VSAProductRouter, dict[str, Any]]:
    if cfg.dimension % 2:
        raise ValueError("VSA dimension must be even")
    rng = np.random.default_rng(cfg.seed)
    half = cfg.dimension // 2
    values = np.asarray([-1, 1], dtype=np.int8)
    token_content = rng.choice(values, size=(256, half))
    token_order = rng.choice(values, size=(256, half))
    roles = rng.choice(values, size=(cfg.role_period, half))
    dummy = np.ones((cfg.product_k, half), dtype=np.int8)
    provisional = VSAProductRouter(
        token_content, token_order, roles, dummy, dummy, cfg.prefix_bytes
    )
    codes = np.stack([provisional.code(prefix) for prefix in prefixes])
    proto_a, history_a = _binary_kmeans(
        codes[:, :half], cfg.product_k, cfg.kmeans_iterations, rng
    )
    proto_b, history_b = _binary_kmeans(
        codes[:, half:], cfg.product_k, cfg.kmeans_iterations, rng
    )
    router = VSAProductRouter(
        token_content, token_order, roles, proto_a, proto_b, cfg.prefix_bytes
    )
    routes = np.asarray([router.route_from_code(code) for code in codes], dtype=np.int32)
    counts = np.bincount(routes, minlength=router.num_routes)
    stats = {
        "dimension": cfg.dimension,
        "prefix_bytes": cfg.prefix_bytes,
        "role_period": cfg.role_period,
        "product_k": cfg.product_k,
        "num_routes": router.num_routes,
        "fit_prefix_count": len(prefixes),
        "active_routes": int((counts > 0).sum()),
        "route_counts": counts.tolist(),
        "route_entropy_bits": entropy_bits(counts),
        "route_entropy_max_bits": float(np.log2(router.num_routes)),
        "route_count_cv": float(counts.std() / counts.mean()) if counts.mean() else 0.0,
        "codebook_a_history": history_a,
        "codebook_b_history": history_b,
    }
    return router, stats
