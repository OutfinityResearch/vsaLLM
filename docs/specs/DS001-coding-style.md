---
id: DS001
title: Coding Style and Engineering Guardrails
status: draft
owner: vsaLLM maintainers
summary: Defines implementation conventions that preserve determinism, artifact compatibility, and fail-closed reporting.
---

# DS001 — Coding Style and Engineering Guardrails

## Introduction

The repository combines Python machine-learning modules with Node.js ESM
orchestration and reporting. Engineering choices directly affect scientific
validity because silent fallback, mutable configuration, or ambiguous artifact
selection can change the comparison.

This specification defines normative implementation rules for code and artifact
changes.

## Core Content

### Configuration as the source of truth

Behavior that affects the experiment must come from the deep-merged base and
profile configuration. Model IDs, primary baseline, runtime axes, contexts,
workload sizes, thresholds, and seeds must not be hard-coded in reporting or
orchestration.

The resolved configuration must be frozen in `runs/<profile>/resolved_config.json`.
Preparation, training, and evaluation must reject semantic config drift.
Benchmark device, dtype, and thread count may vary only as explicit runtime axes
under the benchmark-runtime exception.

Configuration loading must reject unrepresentable or unsafe geometry before a
long command starts. This includes token/route IDs that do not fit their on-disk
dtypes, invalid model dimensions, non-positive data/training/evaluation sizes,
invalid optimization ranges, impossible generation/benchmark contexts, duplicate
models or controls, and a non-CPU/FP32/x86-64 primary benchmark contract.

### Naming and schemas

Python code and raw JSON artifacts should use `snake_case`. JavaScript report
objects may use `camelCase`, but each translation must be explicit and covered by
tests. Artifact formats must carry schema or format versions. A format change
must increment its version when an older consumer could misinterpret the data.

Runtime-specific benchmark filenames must include model identity, device, dtype,
and thread count. Consumers must select the configured primary device, dtype,
and machine architecture exactly; they must not use a “first file found”
fallback.

### Determinism and explicit failure

Random behavior must derive from the frozen project seed or a documented stable
offset. Hashes must use canonical serialization when object ordering could vary.
Text boundaries, prompt manifests, model lists, and sample indexes must be
validated before comparison.

The implementation must fail closed when required provenance is missing,
malformed, stale, reordered, or inconsistent. It must not silently truncate
inputs, substitute a baseline, accept an unverifiable legacy checkpoint, or turn
`skipped`/`unavailable` into `pass`.

### File writes and source identity

Structured artifacts should be written atomically through a temporary file and
rename. Generated artifacts must include enough identity to associate them with
profile, model, checkpoint, and runtime.

`runtime_source_sha256` covers executable Python, Node scripts, config files,
dependency manifests, and official DGX container build inputs. Training,
evaluation, benchmark, and report changes that affect execution must be made
before the official clean-worktree run. The final report rejects training with
a dirty or unknown Git worktree state.

### Testing expectations

Changes to configuration, preparation, checkpoints, routing, evaluation,
benchmark schemas, report selection, or provenance must include focused tests.
Tests should cover both acceptance and rejection paths. Smoke fixtures may prove
plumbing, but test data and generated fixture scores must not be treated as
scientific results.

## Decisions & Questions

### Question #1

What is authoritative when documentation and code disagree?

Response: The frozen resolved config and versioned artifact validators are
authoritative for an executed run. Documentation must be corrected before the
run and must not override an artifact contract after the fact.

### Question #2

How should compatibility failures be handled?

Response: Compatibility failures must stop the command with a specific error.
Consumers must not guess a revision, profile, runtime, route mapping, tokenizer,
or checkpoint identity.

### Question #3

Which runtime fields may change after training?

Response: Only benchmark device, dtype, and thread count may change through the
explicit benchmark-runtime axis. The benchmark still must use the frozen
`benchmark.primary_device` and `benchmark.primary_dtype` for primary reporting.

### Question #4

May an old artifact be upgraded in place?

Response: Only a deterministic, tested migration with explicit schema identity
may do so. Otherwise the artifact must be regenerated from its source inputs.

### Question #5

When should the oversized report implementation be split into smaller modules?

Response: Not immediately before the frozen official run. A split changes the
runtime-source file set and therefore `runtime_source_sha256`; the current tests
also do not guarantee byte-identical JSON, Markdown, HTML, and CSV rendering or
validation order. Before a subsequent protocol revision, first add golden output
fixtures, then separate provenance, statistics, comparisons, matched controls,
and renderers behind an API-compatible facade.

## Conclusion

Repository style is part of the experimental control system.
Configuration-driven behavior, versioned schemas, atomic writes, deterministic
construction, and fail-closed validation must be preserved across all future
changes.
