---
id: DS006
title: Evaluation and Blind Review
status: draft
owner: vsaLLM maintainers
summary: Defines CUDA BF16 quality evaluation, official-baseline validation, sparse ablations, generation, and provenance-bound blind review.
---

# DS006 — Evaluation and Blind Review

## Introduction

Quality evaluation must bind each score and generation to the frozen profile,
exact checkpoint, immutable official revision, and deterministic input set.
Official DGX profiles evaluate on CUDA BF16 before artifacts are transferred for
CPU FP32 benchmarking.

This specification defines the evaluation contract and blind-review identity.

## Core Content

### Runtime and artifact identity

The `dgx_spark` profile must run custom and every configured official evaluation
on CUDA BF16. The `dgx_spark_dense_active` and `dgx_spark_dense_total` profiles
must run only their custom evaluation on the same CUDA BF16 contract; duplicating
official evaluation inside control profiles is not part of the workflow. Every
evaluation artifact must record profile, environment, runtime-source hash, model
device/dtype, and model provenance.

Report validation must require evaluation to match the training machine,
Python, PyTorch, NumPy, recorded runtime-dependency version map, CUDA version,
CUDA device inventory, cuDNN version, NVIDIA driver version, container state,
exact NGC image label, base digest, derived image ID, clean Git commit, and
executable-source identity. A CUDA BF16 artifact from a different nominally
compatible environment is not interchangeable.

Custom evaluation must use the exact selected checkpoint. The adapter must
record its SHA-256, format version, training profile, resolved-config SHA,
routing mode, preparation signature, tokenizer/router hashes, training Git
identity, clean-worktree state, and training runtime-source hash.

Official evaluation must load only the configured model list from the validated
asset manifest. The model ID, requested revision, immutable resolved SHA, order,
and trained-context override must match the frozen configuration.

### Conditional quality

The default full evaluation uses 2,000 deterministically selected test stories.
For each story it must retain raw ID, continuation NLL, continuation token count,
UTF-8 byte count, and conditional BPB inputs.

`heldout_native` is primary: custom uses context 256 and official TinyStories
uses the declared trained context 512. `heldout_common_context` caps all models
at 256 and is diagnostic. Scoring must verify additive tokenization of the raw
prefix and continuation for each tokenizer.

HellaSwag is a secondary task but an enabled HellaSwag gate still participates
in the implemented core verdict. `hellaswag_examples=0` means the complete
configured validation split, a positive value means a deterministic subset, and
`-1` means disabled. Reports must preserve this scope.

### Sparse ablations

Sparse evaluation must produce `full`, `shared_only`, `expert_only`,
`permuted_route`, and `fixed_route_zero` results on deterministic batches. The
shared-only and permuted-route differences feed the implemented expert and
routing gates.

Fixed-dense controls have one route. Their evaluation must produce `full` and
mark routing ablations as skipped with a reason. A skipped dense ablation must
not be converted into an internal routing pass or used to invent a
matched-control performance gate.

### Generation

The full generation suite uses 44 configured prompts. It must store one greedy
continuation and three sampled continuations per prompt using the frozen
temperature 0.75, top-k 40, top-p 0.95, and deterministic seeds. Inputs must not
be silently truncated.

Complete `custom.json` and `official_<model>.json` artifacts must embed their
generation suites. Separate generation files are convenience exports and must
not replace the provenance-bearing complete evaluations.

### Blind review

Blind review is defined against `comparison.primary_official_model_id` only. The
builder must validate profile, custom kind, custom checkpoint SHA, official
model/revision, prompt count, samples per prompt, sample index, ID order, and raw
prompt equality.

`blindEvaluationId` must hash profile, primary model, selected sample,
prompt-manifest SHA, and complete custom/official evaluation hashes. The secret
key must retain that ID, prompt-manifest SHA, official revision, custom
checkpoint SHA, evaluation hashes, and per-item custom side.

Scores may be consumed only when profile, `blindEvaluationId`, and
`promptManifestSha256` match the key. Regenerating either evaluation invalidates
the previous review.

## Decisions & Questions

### Question #1

Why is quality evaluation performed on DGX CUDA BF16 while the primary benchmark
is CPU FP32?

Response: Quality evaluation and deployment performance are separate protocol
axes. The frozen DGX profile defines training/evaluation precision, while
`benchmark.primary_device` and `benchmark.primary_dtype` define the CPU
deployment measurement selected by the report.

### Question #2

Which context result determines quality non-inferiority?

Response: The native-context result is primary. The 256-token common-context
result is diagnostic and must not replace the primary result after inspection.

### Question #3

May blind review use the standalone generation JSON files?

Response: No. It must use complete evaluation artifacts so generation content is
cryptographically bound to model, checkpoint, revision, profile, and quality
provenance.

### Question #4

Does a dense-control evaluation need routing ablations?

Response: No. They are not applicable to a one-route fixed-dense model and must
remain explicitly skipped. Dense controls are interpreted through their quality,
CPU, and memory artifacts, not a fabricated routing gate.

## Conclusion

Evaluation must remain bound to immutable model and checkpoint identities.
Native-context CBPB, explicit ablations, complete generation artifacts, and a
provenance-bound blind ID prevent cross-run mixing and post-hoc sample changes.
