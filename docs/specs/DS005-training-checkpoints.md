---
id: DS005
title: Training and Checkpoint Lifecycle
status: draft
owner: vsaLLM maintainers
summary: Defines DGX Spark BF16 training, two-phase optimization, resumable and best checkpoints, and checkpoint provenance.
---

# DS005 — Training and Checkpoint Lifecycle

## Introduction

Official custom and dense-control training is assigned to DGX Spark CUDA BF16.
Checkpoint identity must survive later CPU FP32 benchmarking under the same
profile without weakening configuration or preparation checks.

This specification defines training phases, checkpoint semantics, resume, and
publication requirements.

## Core Content

### Training profiles and runtime

The sparse official profile is `dgx_spark`. The matched controls are
`dgx_spark_dense_active` and `dgx_spark_dense_total`. All three configure CUDA,
BF16, four CPU helper threads, batch size 32, gradient accumulation 2,
evaluation batch size 16, and a checkpoint cadence of 2,000,000 target tokens.
The sparse profile declares the two control-profile names for provenance and
orchestration; that declaration does not create a cross-profile performance
success gate.

Preparation and training for every `dgx_spark*` profile must invoke the strict
DGX doctor internally. A failed GB10, container, dependency, provenance, or
runtime check must abort the command even if the operator omitted the explicit
preflight command.

The full default budget is 320 million unmasked continuation target tokens in
joint training and 80 million in expert specialization. Calibration may estimate
capacity and duration but must not alter the frozen official profile after test
results are visible.

### Optimization phases

During the joint phase, backbone, shared FFN, and route experts must train
together. Route sampling uses the configured tempered route distribution.

During expert specialization, backbone and shared parameters must freeze and
active experts must be sampled uniformly. Specialization starts from the best
joint checkpoint, not merely the last joint update. Fixed-dense profiles have
one route; their second phase updates that one expert branch and must be
disclosed when interpreting controls.

### Checkpoint files

Checkpoint format version is 2. `last.pt` must contain model, optimizer,
training state, RNG state, GradScaler state, next evaluation and checkpoint
boundaries, and provenance needed for resume. `best.pt` must contain the best
validation model and training state but intentionally omits optimizer state.
Best selection uses validation conditional loss and may be replaced by a better
specialization checkpoint.

Writes must be atomic. Resume must restore deterministic sampling and scaler
state when available. A resume under changed preparation or routing semantics
must fail. The orchestrator's frozen config must be used to prevent training
drift.

Resume must use `last.pt`, not the weights-only `best.pt`. It must fail when the
resolved-config hash or any checked execution-environment field differs. The
checked fields are runtime-source hash, machine, Python, PyTorch, NumPy, the
recorded runtime-dependency version map, CUDA version and device inventory,
cuDNN version, NVIDIA driver version, container label, base container digest,
and derived image ID. Optimizer state, NumPy RNG state, phase counters, and the
next evaluation/checkpoint boundaries must be present; the implementation must
not silently restart any of them.

### Checkpoint provenance

Checkpoint metadata must include:

- training profile and routing mode;
- canonical resolved-config SHA-256;
- complete prepared metadata, including format-v5 signature;
- tokenizer and router artifact hashes;
- parameter accounting;
- Git commit and clean/dirty worktree state;
- executable runtime-source SHA-256;
- recorded training device and dtype through the training summary, accompanied
  by the environment snapshot.

The official report requires a clean training worktree. Missing or malformed
training profile, config hash, preparation signature, tokenizer/router hash,
source hash, Git identity, or checkpoint format must prevent a passing report.
Before DGX training, the prepared-data and current training environments must
also agree on runtime-source hash, Git identity/state, Python, PyTorch, NumPy,
the recorded runtime-dependency version map, CUDA version and device inventory,
cuDNN version, NVIDIA driver version, container label, base digest, and derived
image ID.

### Cross-host use

The exact checkpoint bytes and SHA-256 must be transferred to the CPU benchmark
host with `runs/<profile>/resolved_config.json`, training summary, log, and
prepared data. Evaluation and benchmark artifacts must reference the same
checkpoint SHA and checkpoint provenance. Training remains CUDA BF16; the later
CPU FP32 benchmark is a recorded runtime axis, not a second training profile.

## Decisions & Questions

### Question #1

Why use one profile across GPU training and CPU benchmarking?

Response: The profile identifies one experiment and checkpoint lineage. The
orchestrator explicitly permits device, dtype, and threads to vary only for
benchmark commands, while preserving all semantic configuration.

### Question #2

Which checkpoint should evaluation use?

Response: Evaluation must use the exact `best.pt` selected by validation unless
the frozen protocol explicitly names another checkpoint. Its SHA-256 must match
all downstream custom artifacts.

### Question #3

May a checkpoint trained from a dirty worktree support the official report?

Response: No. The report validator requires `training_git_worktree_dirty` to be
exactly false and requires the recorded Git and runtime-source identities.

### Question #4

Do the dense-control profile declarations imply a pass/fail requirement?

Response: No. They identify required control runs. Their quality, CPU, and memory
artifacts must be ingested by the primary sparse report, but the current report
keeps their performance descriptive and does not convert it into a
matched-control pass/fail gate.

## Conclusion

The checkpoint lifecycle binds CUDA BF16 training to later CPU FP32 deployment
measurements through one profile, immutable config, prepared-data identity, and
checkpoint hashes. Clean-source provenance and atomic, versioned checkpoints are
mandatory for an official result.
