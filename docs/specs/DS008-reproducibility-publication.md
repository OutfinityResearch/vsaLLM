---
id: DS008
title: Reproducibility, Provenance, and Publication
status: draft
owner: vsaLLM maintainers
summary: Defines frozen configuration, source and artifact provenance, report schema v4, publication bundles, and reproducibility limits.
---

# DS008 — Reproducibility, Provenance, and Publication

## Introduction

A performance number is useful only when its data, code, configuration,
checkpoint, model revisions, workload, and runtime can be identified. Cross-host
DGX-to-CPU execution increases the risk of mixing artifacts and therefore
requires fail-closed provenance.

This specification defines the identities that must be preserved and the
minimum publication bundle.

## Core Content

### Frozen config and executable source

Each profile must have one `resolved_config.json`. Preparation, training, and
evaluation must use that semantic configuration. Only benchmark runtime device,
dtype, and threads may vary through the explicit benchmark exception.

The canonical resolved-config SHA-256 must be stored in training summary and
checkpoint metadata. `runtime_source_sha256` must cover executable Python, the
Node.js `.mjs` protocol scripts, config files, dependency manifests including
the DGX-specific lock, the official DGX Dockerfile, and `.dockerignore`. Every
evaluation and benchmark artifact must record a runtime-source hash. Custom
evaluation and custom benchmark artifacts selected by one report must carry the
same current hash.

The official training run must start from a known Git commit and clean worktree.
The report must reject dirty or unknown training worktree state, malformed Git
identity, or missing source hashes.

### Source and model provenance

Dataset, HellaSwag, and official model revisions must resolve to immutable SHAs.
The manifest's complete official-model list must match frozen config. Official
evaluation and benchmark artifacts must preserve resolved revision, model ID,
trained context, artifact context, actual parameter count, resident parameter
bytes, and weight-artifact bytes.

Preparation format v5 must identify semantics without local cache paths. The
prepared store and checkpoint must still agree on preparation signature,
tokenizer SHA, router SHA, and routing mode.

### Checkpoint and artifact provenance

Custom evaluation and every custom benchmark must use the same checkpoint SHA
and full checkpoint-provenance object. The report must validate training profile,
checkpoint format version 2, resolved-config SHA, preparation signature,
tokenizer/router hashes, training Git commit, clean worktree, and training source
hash against the training summary.

Benchmark artifacts must use schema v3 and one canonical workload manifest.
Blind scores must match the exact evaluation hashes, checkpoint SHA, official
revision, `blindEvaluationId`, and prompt-manifest SHA.

The DGX host must create handoff schema v1 with
`npm run handoff:create -- --profile dgx_spark`. Creation must require a clean
worktree, format-v5 prepared data, the three frozen profiles, both checkpoints,
training summaries, custom evaluations, and the main profile's official
evaluations. The manifest must record the exact Git commit and stream file size
and SHA-256 entries for `data/prepared/dgx_spark` and all three run directories.
Creation and verification must reject duplicate or unsafe paths, roots/profiles
that differ from frozen config, any missing or unmanifested regular file, size
or digest mismatch, and any missing required artifact. Both commands must print
the manifest SHA-256 for independent recording. After transfer, the x86-64 host
must check out that commit and run
`npm run handoff:verify -- --profile dgx_spark` before any benchmark. Asset
manifests and caches are regenerated locally and are not transfer roots.

### Report schema v4

`comparison_<profile>.json` must include:

- `comparisons[]`, one entry per official model in frozen order;
- `primaryOfficialModelId` and primary aliases without fallback;
- `artifactValidation`, including checkpoint provenance, frozen-config/training
  provenance, runtime-source hash, workload SHA, official revisions, selected
  benchmark `{device,dtype,machineArchitecture}`, and thread list;
- per-baseline paired quality uncertainty, every CPU prompt row and timing
  interval, paired CPU environments, conservative primary minima, memory rows,
  measured-load and persistent-state ratios, the controlling conservative
  memory source, gates, and core/final verdicts;
- a top-level `matchedControls` object containing `configured`,
  `configuredProfiles`, `artifactPolicy`, `performancePolicy`, and one validated
  comparison per configured control;
- for each matched control, its role, exact parameter budget, prepared identity,
  config validation, hashed artifact evidence, provenance, sparse/control model
  metadata, paired quality uncertainty, every CPU row and interval, primary
  descriptive CPU aggregate, memory rows, and descriptive memory ratios.

Matched-control entries must state that they are descriptive and must not carry
new thresholds or verdicts. The accepted control artifact policy is custom
evaluation plus primary CPU custom benchmarks; official-model evaluation and a
separate control report are not part of a control profile.

The generated Markdown, HTML, JSON, and CSV outputs must be derived from the same
validated report object.

Within each comparison, `models.custom` and `models.official` must describe the
selected primary-thread CPU FP32 benchmark instances because they provide the
authoritative resident-parameter accounting. CUDA BF16 quality-evaluation model
metadata must remain separately available under `models.evaluation`. The
top-level `customModel` and `customEvaluationModel` aliases must preserve the
same distinction.

### Publication bundle

An official publication must archive:

- base, profile, and resolved configuration;
- Git commit, source hash, clean-worktree evidence, and environment snapshots,
  including Python, PyTorch, NumPy, recorded runtime-dependency versions, CUDA
  device properties, cuDNN, and NVIDIA driver;
- the NGC base RepoDigest, derived container image ID, DGX Dockerfile,
  `.dockerignore`, `requirements-dgx.txt`, and complete `pip freeze --all`
  output;
- the complete CPU-host `pip freeze --all` output;
- the CPU topology mapping used to choose the fixed eight-core CPU set and the
  recorded `environment.cpu_topology` objects, plus governor/performance-policy
  evidence for the complete benchmark matrix;
- immutable asset manifests and prepared-data metadata/hashes;
- `handoff/dgx_spark_manifest.json` and the verified transfer-root inventory;
- `best.pt`, `last.pt`, their SHA-256 values, training summary, and training log;
- complete sparse custom and official evaluation artifacts;
- every primary CPU FP32 benchmark artifact and raw row;
- custom evaluation and complete custom-only CPU matrices for both matched
  controls;
- report JSON, Markdown, HTML, and CSV;
- blind scores and the key after scoring is frozen;
- DGX and CPU hardware/software manifests;
- matched-active and matched-total run bundles.

The secret blind key may be withheld during review but should be released with
the finalized result.

### Reproducibility limits

Fixed seeds and deterministic bootstrap make one run auditable but do not
estimate training variance. A robust claim should include multiple training
seeds, repeated CPU sessions, and a second CPU host. Negative and unavailable
gates must remain published.

No final full-run result is currently available.

## Decisions & Questions

### Question #1

Which identity binds training to evaluation and benchmark?

Response: The profile, resolved-config hash, checkpoint SHA and provenance,
preparation/tokenizer/router hashes, Git identity, and runtime-source hashes form
the binding. No single filename is sufficient.

### Question #2

Can absolute paths differ across hosts?

Response: Yes for downloaded-asset cache paths. Format-v5 preparation identity
excludes those paths. Source revisions and content hashes must remain equal, and
each host may record its own local asset paths; the frozen data config remains
unchanged.

### Question #3

What happens when an artifact predates a required provenance field?

Response: It must be treated as unverifiable and regenerated. The report must not
infer or manually fill provenance for legacy artifacts.

### Question #4

Does one deterministic seed establish reproducibility?

Response: It establishes repeatable construction for that run, not robustness to
training variation. Multiple independently trained seeds are required for the
stronger claim.

## Conclusion

Reproducibility depends on a chain of immutable identities rather than directory
names. The report and publication bundle must preserve that chain across DGX
training, CUDA evaluation, CPU benchmarking, matched controls, and blind review.
