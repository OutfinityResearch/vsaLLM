---
id: DS009
title: Operations and Troubleshooting
status: draft
owner: vsaLLM maintainers
summary: Defines supported operational sequences, environment checks, failure handling, and recovery rules without weakening the protocol.
---

# DS009 — Operations and Troubleshooting

## Introduction

Operational recovery must preserve the experiment rather than bypass its
validators. This specification covers environment setup, command sequencing,
common failures, and permitted recovery actions.

## Core Content

### Environment setup

The execution host must provide Node.js 20 or newer and a supported Python
release. DGX Spark runs must retain PyTorch/CUDA from the digest-pinned NGC base
and install the exact direct versions in `requirements-dgx.txt`. The official
x86-64 host must use `npm run setup -- --locked-cpu`; this force-installs the
official CPU-only `torch==2.10.0` wheel, installs the same exact direct
requirements, and verifies dependency consistency. The generic setup modes are
not valid for the official cross-host comparison.

DGX Spark runs must use Node.js 22 or newer and should pin OS image or container
digest, driver, CUDA/cuDNN, PyTorch, Python, Node, storage mount, and
power/performance settings. Both hosts must archive `pip freeze --all` and the
recorded Python, PyTorch, NumPy, and runtime-dependency version map. The DGX
record must additionally include CUDA device name, compute capability, device
memory, cuDNN version, and NVIDIA driver version.

Before any official artifact is produced, the operator must run unit tests,
offline smoke, and the appropriate preflight for the intended profile. A DGX
run must use `doctor:dgx --strict`; a CPU host must record the general `doctor`
output. Smoke and calibration outputs must be labeled as infrastructure evidence
or estimates.

Configuration validation must run during profile loading and stop invalid model,
data, training, evaluation, or benchmark geometry before downloads or long
training. The strict DGX doctor must also require the GB10 device's compute
capability to equal 12.1 and reject any dataset, HellaSwag, or official-model
revision that is not an immutable 40–64-character hexadecimal SHA.

### Command sequencing

The normal semantic sequence is download, prepare, train, evaluate, benchmark,
report, and blind review. The cross-host official sequence is specified in
DS010. Operators must use explicit steps rather than `npm run full` when training
and CPU benchmarking occur on different hosts.

The first artifact-producing command freezes config. An operator must not delete
`resolved_config.json` to bypass drift or combine outputs from different runs.

### Storage and source handling

Hugging Face caches may live on host-specific storage, but manifests must resolve
the same immutable source SHAs. The official prepared data and run directories
must be transferred with the DS010 checksum manifest. A copied asset manifest
containing invalid absolute paths must be reacquired or regenerated against the
same revisions.

Code and config used for evaluation and benchmark must retain the same
runtime-source identity. Source changes after official training/evaluation
should start a new documented run unless the report's provenance contract
explicitly accepts the separation.

### Failure handling

- Configuration drift must create a new profile/run; it must not be ignored.
- Preparation-signature mismatch must trigger a deliberate rebuild or correct
  store selection; `--force` rebuilds the shared artifact and requires
  revalidation of every dependent lineage.
- A deployment-only asset manifest must never feed preparation or training; it
  exists only to reacquire CPU benchmark inputs after verified handoff.
- Checkpoint incompatibility must not be repaired by copying state dictionaries
  across profiles.
- A benchmark prompt that does not fit must change only in a newly frozen pilot
  or protocol; silent per-model truncation is forbidden.
- Workload-manifest mismatch requires rerunning all compared benchmarks.
- An incomplete CPU topology, an affinity other than the configured eight
  logical CPUs, duplicate physical-core identities, or selected SMT siblings
  must abort the official benchmark; the guardrail must not be bypassed.
- Missing USS/PSS must use the implemented memory fallback order; values must not
  be fabricated.
- Negative or unstable memory deltas should trigger clean-process repetition and
  disclosure, not substitution with active bytes.
- A stale blind key or score must be regenerated from the exact evaluations.

### Monitoring

Training operators should monitor loss, validation, target tokens/s, route usage,
storage growth, thermals, and checkpoint writes. CPU benchmark operators should
minimize competing workloads and record CPU affinity, frequency governor,
thermals, and power state.

An operational incident that changes code, config, assets, prepared semantics,
checkpoint bytes, or primary workload must be recorded and may require a new
run.

## Decisions & Questions

### Question #1

May `--force` be used to fix a preparation error during an official run?

Response: Only by declaring the existing run invalid and rebuilding all dependent
checkpoints and artifacts. It must not be used as an invisible repair.

### Question #2

May benchmark runtime overrides be used on the CPU host?

Response: Yes, but the primary report selects the frozen primary CPU/FP32 runtime.
Overrides for exploratory runtimes must produce distinct filenames and must not
replace primary artifacts.

### Question #3

What is the correct response to missing or incompatible provenance?

Response: Stop and regenerate the artifact from verified inputs. Manual metadata
patching is not permitted.

### Question #4

May calibration choose a different official threshold?

Response: No. Calibration may size batches, estimate duration, and validate
plumbing. It must not inspect test outcomes or revise success gates.

## Conclusion

Operational convenience must not weaken experimental identity. Recovery is
valid only when config, source, preparation, checkpoint, workload, and runtime
contracts remain explicit and verifiable.
