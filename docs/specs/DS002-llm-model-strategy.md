---
id: DS002
title: LLM Model Strategy
status: draft
owner: vsaLLM maintainers
summary: Defines the sparse model, VSA routing policy, official baselines, dense controls, and model-accounting boundaries.
---

# DS002 — LLM Model Strategy

## Introduction

The model strategy combines a small decoder-only Transformer with request-level
VSA routing and route-specific feed-forward micro-experts. The goal is to test
conditional tensor execution on CPU without representing arithmetic activity as
resident-memory usage.

This specification defines the model family and the controls required to
interpret it.

## Core Content

### Sparse model

The default VSA-PathMoE model uses:

- a 2,048-entry lossless SentencePiece vocabulary;
- 256-token Transformer context;
- model dimension 192, six layers, and six attention heads;
- a shared FFN hidden width of 128;
- a route-expert hidden width of 57;
- 64 routes formed as an 8 × 8 product key;
- a 512-dimensional bipolar VSA code.

It contains 10,029,504 total parameters. One request uses 1,755,840 arithmetic
active parameters, or 17.5067% of the total. The expert contribution on one route
across all layers contains 131,328 parameters.

### Routing policy

The router must derive one route from the raw causal conditioning prefix. It
bundles byte identity and bytes bound to periodic positional roles, compares two
code halves with offline binary-k-means prototypes, and combines the prototype
indexes into one product-key route.

The route must remain fixed through the continuation and across all layers.
Each layer has its own weights for the selected route. The target continuation
must not affect route selection. This is request-level conditional computation,
not token-level MoE routing.

The current arrays use `int8` storage and `int16` dot products. Bit packing,
XNOR, and popcount may be investigated later, but must not be described as
implemented or benchmarked behavior.

### Tensor execution

Each block combines always-on attention, the shared FFN branch, and one selected
expert branch. At inference, shared and selected weights are concatenated so the
FFN can execute through two GEMMs. A bounded LRU cache may retain fused matrices
for recently used routes.

The complete expert bank is loaded by the reference adapter. Approximately
7.02 MB of FP32 arithmetic-active values is not a process-memory measurement.
Reports must separate active arithmetic bytes, total parameter bytes, resident
parameter bytes, router resident-array bytes, router artifact bytes, RSS/USS/PSS,
current and maximum persistent model-state bytes, current and maximum fused-cache
bytes, and other cache allocations.

### Baselines and controls

`roneneldan/TinyStories-8M` is the configured primary official baseline.
`roneneldan/TinyStories-33M` is a larger capacity anchor. Official comparisons
are system-level because tokenizer, architecture, and historical training differ.

The matched-active dense control uses `routing_mode=fixed_dense` and exactly
1,755,840 total/active parameters. The matched-total dense control also uses a
fixed route and exactly 10,029,504 total/active parameters. DGX training profiles
exist as `dgx_spark_dense_active` and `dgx_spark_dense_total`.

Matched controls must be trained in separate profiles. They produce custom-only
evaluation and CPU benchmark artifacts; the primary sparse report must ingest
and present them descriptively. The report does not implement a cross-profile
matched-control performance pass/fail gate, so no such gate may be invented in
interpretation. It does aggregate and validate the required control artifacts
fail closed.

### Context strategy

The custom model's native context is 256 tokens. Official TinyStories models use
the protocol-declared trained context of 512 tokens, while their artifact config
may allocate 2,048 positions. Native-context quality is primary and a 256-token
common-context result is diagnostic. Artifact capacity must not be substituted
for trained context.

## Decisions & Questions

### Question #1

Why is routing performed once per request?

Response: The tested hypothesis treats VSA as a low-cost control plane over a
conditioning prefix. Holding the route fixed makes routing cost explicit and
separates the design from token-level learned gates.

### Question #2

Why are both dense controls required?

Response: Matched-active tests a tensor-only model at the same forward parameter
budget. Matched-total tests a tensor-only model that activates the sparse model's
full stored capacity. They answer different causal questions.

### Question #3

Does the active parameter count imply a memory reduction?

Response: No. The full bank is resident in the current implementation. A memory
advantage must be measured through the configured resident-memory protocol.

### Question #4

Is an alternative learned or deterministic router control part of the current
official run?

Response: No. Internal shared-only and permuted-route ablations are implemented,
but a separately trained alternative router or MoE control is not. It may be
added only as a newly preregistered experiment.

## Conclusion

The model strategy tests request-level VSA control over small tensor paths while
keeping storage and compute accounting separate. Official checkpoints establish
deployment comparisons; the two dense profiles provide matched interpretive
controls without an automated matched-control performance pass/fail gate.
