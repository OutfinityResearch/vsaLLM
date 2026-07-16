---
id: DS003
title: Experimental Methodology and Claims
status: draft
owner: vsaLLM maintainers
summary: Defines evaluation metrics, preregistered gates, uncertainty rules, and valid interpretation of official and matched-control comparisons.
---

# DS003 — Experimental Methodology and Claims

## Introduction

The experiment must compare models with different tokenizers, architectures, and
runtime implementations without selecting favorable outcomes after inspection.
This specification defines the primary metrics, gates, uncertainty procedures,
and interpretation limits.

## Core Content

### Primary quality comparison

Cross-tokenizer quality must use conditional bits per UTF-8 byte (CBPB), not
per-token perplexity. Both models must receive the same raw prefix and score the
same raw continuation. Tokenization must satisfy additive prefix/continuation
boundaries, and story-level artifacts must preserve NLL and continuation byte
count.

The primary comparison uses each model's protocol-native context: 256 for custom
and 512 for official TinyStories checkpoints. A 256-token matched-context result
must remain diagnostic.

Quality non-inferiority must use a paired bootstrap over identical story IDs. At
each resample, aggregate NLL/bytes must be recomputed for each model. The gate
passes only when the upper 95% bound of custom/official aggregate BPB is at most
1.02.

### CPU performance comparison

The primary benchmark runtime is configured as CPU FP32 on x86-64. The full
workload must use four deterministic prompts at each 128, 320, and 512 character
target, 64 greedy decode steps, five warmups, and 30 measured warm repetitions
per prompt. Custom and official artifacts must share the complete
workload-manifest SHA.

Speed uncertainty must use independent bootstrap resampling of each process's
raw warm rows and a ratio of resampled medians. At the primary one-thread,
512-character point, each speed gate must use the minimum lower-95 bound across
the four prompt samples. End-to-end, decode throughput, and generated UTF-8
bytes/s each require at least 1.20×.

### Resident memory comparison

The measured-load component must use the first positive comparable pair in this
order: pre-request USS load delta, then pre-request RSS load delta. A separate
exact component must compare maximum persistent model state, including resident
parameters, router arrays, and maximum fused-route-cache capacity where
applicable. The gate must use the larger available component ratio and require
custom/official at most 0.80. Peak and total RSS remain diagnostics. One process
loading observation has no statistical confidence interval.

Arithmetic active parameter bytes must not enter the resident-memory gate.

### Secondary and internal criteria

The normalized HellaSwag delta must be at least -0.01 when HellaSwag is enabled.
The custom sparse checkpoint must show at least 0.005 loss improvement for
shared-only minus full and at least 0.005 for permuted-route minus full. Blind
preference against the configured primary baseline must be at least 0.45, with a
tie weighted as 0.5.

HellaSwag is a secondary domain-transfer task, but its configured gate remains
part of the implemented core verdict while the task is enabled. Blind preference
is an additional final-verdict gate. A single evaluator or a 45% operational
threshold must not be presented as statistically significant human preference.

### Baseline and verdict selection

TinyStories-8M is the configured primary baseline. TinyStories-33M is reported
separately. Report schema v4 must contain one comparison per configured official
model and must copy top-level verdicts only from the primary baseline. Missing
primary artifacts must stop reporting; no baseline fallback is permitted.

The primary report must fail closed unless it can load the custom evaluation and
complete CPU matrix from both configured matched-control profiles. It must
compare them descriptively with the sparse artifacts under the shared workload.
They do not have cross-profile performance success gates and must not be folded
into a fabricated pass/fail criterion.

Before comparison, the report must validate the controls' roles, fixed-dense
geometry, exact active/total parameter budgets, frozen config and training
identity, clean source provenance, prepared/tokenizer/router identity, complete
thread set, workload, and paired CPU environment. Control quality must use
paired story IDs, CPU rows must retain their bootstrap intervals, and memory
must retain measured-load and persistent-state ratios. These quantities remain
descriptive.

### Result status

One training seed does not estimate training variance. Fixture, smoke,
calibration, and incomplete DGX artifacts must be labeled non-results. The final
claim remains pending until the full official and control runs are archived.

## Decisions & Questions

### Question #1

Why is CBPB primary instead of perplexity?

Response: Token-level perplexity depends on tokenizer segmentation. CBPB scores
the same raw bytes and supports an auditable cross-tokenizer comparison.

### Question #2

Why does the timing gate use the worst prompt lower bound?

Response: The primary claim is intended to cover the preregistered workload, not
the best route or prompt. The minimum lower-95 bound prevents favorable prompt
selection.

### Question #3

What happens when a required gate value is unavailable?

Response: The gate must remain `pending` or `unavailable`, and the aggregate
verdict must not pass. Missing values must not be replaced by zero or a favorable
fallback.

### Question #4

Do matched dense controls determine the primary top-level verdict?

Response: No. They are required evidence for architectural interpretation, but
the current implementation aggregates and validates their evidence without
defining a matched-control performance pass/fail gate.

## Conclusion

The methodology uses identical raw inputs, conservative uncertainty bounds,
explicit resident-memory measurement, and a fixed primary baseline. All claims
must follow the implemented gates, and unmatched or incomplete evidence must
remain visible rather than being converted into a positive verdict.
