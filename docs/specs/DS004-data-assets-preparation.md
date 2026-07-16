---
id: DS004
title: Data, Assets, and Preparation
status: draft
owner: vsaLLM maintainers
summary: Defines immutable source acquisition, path-independent preparation format v5, causal splitting, and cross-host compatibility.
---

# DS004 — Data, Assets, and Preparation

## Introduction

Data preparation assigns token and route semantics that every checkpoint and
evaluation depends on. The same prepared store must remain verifiable after
transfer from DGX Spark to a separate CPU host, even when cache paths differ.

This specification defines asset identity, preparation format v5, and the
compatibility checks required for cross-host use.

## Core Content

### Source acquisition

The source dataset is `roneneldan/TinyStories`. Full training must use
`TinyStories-train.txt`; `TinyStories-valid.txt` must remain validation/test data.
Official evaluation prompts come from `Evaluation prompts.yaml`.

Mutable requested revisions such as `main` must be resolved to immutable commit
SHAs before download. The asset manifest must retain repository ID, requested
revision, resolved SHA, local path, and protocol-declared trained context for
each official model. The complete configured official-model list must match the
manifest in length, order, ID, requested revision, and trained context.

### Streaming preparation

Preparation must stream the corpus. It may retain bounded reservoir samples for
tokenizer and router fitting, but it must not load every story or token into RAM.
The custom tokenizer must be lossless and use byte fallback. Preparation must
verify raw-text round trips and additive prefix/continuation tokenization.

The causal split targets 96 characters, searches backward for whitespace, then
forward, and uses the exact position only when no nearby whitespace exists. The
separator remains in the continuation. VSA prototypes must be fit from training
prefixes, never target continuations.

### Preparation format v5

`PREPARATION_FORMAT_VERSION` is 5. The preparation signature must be independent
of machine-local cache paths. Its canonical payload contains:

- format version 5;
- project seed;
- the complete data configuration;
- dataset repository ID and resolved SHA;
- the selected training filename;
- the validation filename.

Absolute train, validation, tokenizer, and router paths may be recorded in
metadata for local operation, but must not affect the semantic signature. Raw
train/validation file SHA-256 values, tokenizer model SHA-256, and router artifact
SHA-256 must remain recorded for audit. Format-v5 metadata must also retain the
preparation environment, including Git/source, machine, Python, PyTorch, NumPy,
the recorded runtime-dependency version map, CUDA version and device inventory,
cuDNN, NVIDIA driver, and container identities.

### Prepared artifacts

The store must contain versioned metadata, `PREPARED.json`, the SentencePiece
model and vocabulary, `router.npz`, indexed train/validation/test arrays, and raw
evaluation JSONL splits. Token IDs must fit `uint16`; route IDs must fit `uint8`.

### Compatibility and transfer

Checkpoint loading must compare preparation signature, tokenizer SHA-256, router
SHA-256, and routing mode. Missing values must fail closed. A prepared store may
be copied to another host because format v5 excludes local paths from its
signature, but the destination must regenerate its asset manifest with the same
immutable source revisions and must validate the transferred hashes. The
official handoff does not transfer asset manifests or caches.

The x86-64 host must use the deployment-only asset mode. That manifest contains
validation text, prompts, and immutable official-model snapshots but omits the
full training corpus and HellaSwag. Preparation must reject a deployment-only
manifest, preventing accidental retraining from validation data.

For every `dgx_spark*` profile, preparation itself must invoke the strict DGX
preflight before reading or reusing a prepared store. Consequently, the x86-64
benchmark host must verify and use the immutable transferred store through the
handoff manifest; it must not invoke DGX-profile preparation or rebuild the
store.

`--force` rebuilds shared prepared data and therefore invalidates checkpoints
whose token or route semantics no longer match. It must not be used during an
active official run.

## Decisions & Questions

### Question #1

Why does format v5 exclude cache paths from the signature?

Response: Local Hugging Face cache and downloaded-asset paths may differ between
DGX Spark and the CPU host. They do not change token or route semantics and must
not prevent a verified transfer. The complete data config, including its
prepared-store setting, remains part of the signature.

### Question #2

What identifies dataset semantics if local paths are excluded?

Response: The signature uses repository ID, immutable resolved revision,
selected source filenames, project seed, and complete data config. Metadata also
records raw file hashes for audit.

### Question #3

May a transferred store be accepted from `PREPARED.json` alone?

Response: No. Checkpoint compatibility must also verify tokenizer and router
hashes, and the handoff must verify the complete transferred prepared-store
inventory and hashes before use.

### Question #4

May validation data be substituted for full training on the official run?

Response: No. `use_valid_as_train` is permitted only for smoke or explicitly
separate pilot profiles. It changes the preparation signature and cannot support
the full claim.

## Conclusion

Preparation format v5 makes semantic identity portable without trusting local
paths. Immutable source revisions, causal prefix construction, lossless
tokenization, and fail-closed checkpoint compatibility must remain intact across
the DGX-to-CPU handoff.
