---
id: DS007
title: CPU Performance and Memory Methodology
status: draft
owner: vsaLLM maintainers
summary: Defines CPU FP32 benchmark runtime axes, deterministic workload, timing uncertainty, resident-memory measurement, and report selection.
---

# DS007 — CPU Performance and Memory Methodology

## Introduction

Training and evaluation runtime must not determine the deployment benchmark
runtime. The DGX profiles freeze CUDA BF16 for training/evaluation while allowing
CPU FP32 device, dtype, and thread count as benchmark-only axes after cross-host
transfer.

This specification defines the CPU workload, measurements, artifact naming, and
report selection rules.

## Core Content

### Runtime-axis contract

`benchmark.primary_device` must be `cpu` and `benchmark.primary_dtype` must be
FP32. `benchmark.primary_machine_architecture` must canonicalize to `x86_64` or
`amd64`. Configuration and report validation reject another primary runtime or
architecture. Benchmark-only commands may ignore frozen `runtime.device`,
`runtime.dtype`, and `runtime.threads` when checking config drift; no other
profile field may drift.

When no explicit benchmark runtime override is given, orchestration must use the
configured primary CPU/FP32 values even though `dgx_spark` itself specifies CUDA
BF16. Thread count is measured at 1, 4, and 8 by default.

Before primary measurement, the operator must inspect
`lscpu -e=CPU,CORE,SOCKET,ONLINE` and preregister one fixed CPU set containing
exactly eight distinct `(socket,core)` pairs with no SMT sibling duplicated.
The same CPU set must constrain the sparse model, every official baseline, and
both dense controls. At each 1/4/8-thread point, compared models must use the
same configured thread count and the same eight-core process affinity. The
matrix changes worker-thread count, not the eight-CPU affinity. The
operator must also select and record one stable governor/performance policy and
keep it unchanged throughout the complete matrix.

The sparse profile and its two official matched-control profiles freeze
`benchmark.required_cpu_affinity_logical_cpus=8` and
`benchmark.require_distinct_physical_cores=true`. Before loading a benchmarked
model, the runtime must snapshot the process affinity and Linux sysfs topology
and fail unless it is complete, contains exactly eight logical CPUs mapped to
eight distinct `(physical_package_id,core_id)` pairs, and selects no SMT
siblings.

Runtime-specific artifacts must be named
`custom_cpu_fp32_threads_<N>.json` and
`official_<safe_model_id>_cpu_fp32_threads_<N>.json`. Other device/dtype
artifacts may coexist, but the primary report must ignore them.

### Deterministic workload

The full workload must contain four distinct deterministic prompts at each 128,
320, and 512 character target. It must run 64 greedy decode steps, five warmups,
and 30 measured warm repetitions per case. Custom routing must also retain cold
rows with the fused route cache cleared.

Every case must record target, rendered length, UTF-8 bytes, sample index, source
index, and prompt SHA-256. The artifact must store a canonical workload manifest
and SHA-256. Custom and every official model, at every thread count, must use the
same complete manifest; both matched controls must use it as well. Inputs must
fail rather than truncate when they do not fit the common 256-token context after
decode reservation.

### End-to-end scope

Custom end-to-end timing must include tokenizer encode, causal prompt routing,
VSA code, route selection, fused-matrix preparation/cache lookup, prefill,
KV-cached decode, and text decode. Official timing must include its tokenizer,
prefill, the same decode-step count, and text decode. Dense controls use the
custom tokenizer and a fixed route without VSA computation.

Warm rows measure steady state. Cold rows are a required diagnostic of route
preparation and cache behavior. Generated tokens/s and UTF-8 bytes/s must be
reported alongside latency because tokenizers and generated text differ.

At every thread count, every compared sparse/official and sparse/control pair
must match CPU processor, platform, logical CPU count, affinity, frequency
governors, Python, PyTorch, NumPy, the recorded runtime-dependency version map,
container state, container label/digest, and derived image ID. A mismatch must
stop reporting rather than create an unpaired comparison.

Each artifact must store `environment.cpu_topology`. Its `logical_cpus` rows
contain `logical_cpu`, `physical_package_id`, `core_id`, and
`thread_siblings_list`; its summary contains `complete`,
`affinity_logical_cpu_count`, `affinity_physical_core_count`, and
`affinity_contains_smt_siblings`. The report must independently revalidate this
contract for every selected sparse, official, and control artifact and must
require exact paired equality for topology, affinity, and observed governors.
The guardrail does not impose a particular governor value; it enforces paired
equality while the operator controls and records the chosen stable policy.
The publication bundle must retain the independent `lscpu` output and
governor-control evidence as operational corroboration.

### Timing uncertainty

For each prompt, the report must independently bootstrap custom and official raw
warm rows and compute a ratio of resampled medians. The primary point is one
thread at the 512-character target. End-to-end, decode, and generated-byte gates
must each use the minimum lower-95 bound across the four prompt samples.

The threshold for each implemented CPU gate is 1.20×. Repetitions from one
process quantify operational variability, not between-machine or between-run
variance.

### Resident memory

Each process must capture `before_load`, `after_load`, and `after_benchmark`
snapshots. Artifacts should retain RSS, USS and PSS when available, high-water
RSS and source, load/benchmark/total deltas, load time, resident parameter bytes,
weight-artifact bytes, persistent model-state bytes, and fused-cache accounting.

The measured-load component must choose the first positive comparable pair in
this order: pre-request USS load delta, then pre-request RSS load delta. This
snapshot occurs before the first request and therefore excludes fused-route-cache
population. A second exact component must compare maximum persistent model state:
resident parameters plus router arrays and maximum fused-route-cache capacity for
the custom model, and the corresponding persistent state for the official model.

The resident-memory gate must use the maximum available ratio across measured
load and exact persistent state. Its threshold is custom/official at most 0.80.
Peak and total RSS remain diagnostics and receive no confidence interval from a
single process.

Approximately 7.02 MB of active arithmetic FP32 parameters must never be labeled
as RSS, USS, or total resident model memory. The full expert bank, runtime,
fused-route cache, and KV cache remain relevant allocations.

### Report validation

The report must validate benchmark schema v3, profile, exact primary device and
dtype, configured x86-64 machine architecture, filename/thread identity,
checkpoint SHA and provenance, runtime-source SHA, benchmark config, workload
manifest, official model/revision, and complete thread coverage. It must fail
when any primary artifact is missing or stale.

For each configured matched control, it must also require the exact sparse
thread set, exact canonical workload identity and SHA, paired CPU environment,
control checkpoint lineage, and declared parameter budget. Sparse/control timing
and memory ratios are descriptive and must not reuse official-baseline gate
thresholds.

## Decisions & Questions

### Question #1

How can one DGX profile produce CPU artifacts without config drift?

Response: Benchmark orchestration treats device, dtype, and thread count as
explicit runtime axes. It keeps the same profile, checkpoint lineage, and
semantic config while selecting CPU FP32 for benchmark artifacts.

### Question #2

Why is one thread the primary point?

Response: It reduces ambiguity from thread scheduling and represents the
configured low-resource CPU claim. Four- and eight-thread measurements remain
required secondary scaling diagnostics.

### Question #3

Does resident parameter memory replace process memory?

Response: No. Resident parameter bytes are one component of exact persistent
state. The gate conservatively takes the larger available ratio between that
state and positive comparable USS/RSS pre-request load deltas. The controlling
source must be reported explicitly.

### Question #4

May a CUDA benchmark artifact influence the primary CPU report?

Response: No. The report selects filenames and validates model metadata from
`benchmark.primary_device`, `benchmark.primary_dtype`, and
`benchmark.primary_machine_architecture`; other runtime artifacts are ignored.

## Conclusion

CPU performance is an explicit runtime axis under the same trained profile. The
workload, uncertainty aggregation, runtime-specific filenames, and
resident-memory selection must be identical across custom, official, and
matched-control models, with active arithmetic bytes kept separate from process
memory.
