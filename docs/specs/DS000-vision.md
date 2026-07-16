---
id: DS000
title: VSA-PathMoE Research Vision and Evidence Boundary
status: draft
owner: vsaLLM maintainers
summary: Defines the research objective, required evidence, claim scope, and current non-result status.
---

# DS000 — VSA-PathMoE Research Vision and Evidence Boundary

## Introduction

The project evaluates whether request-level Vector Symbolic Architecture (VSA)
routing can coordinate small tensor experts while retaining useful language
model quality and reducing CPU inference cost. The work is an executable,
falsifiable research protocol. It is not a product performance statement and it
does not currently contain a validated final result.

This specification defines what the project is trying to establish and the
minimum evidence required before any positive claim may be published.

## Core Content

### Research objective

The primary hypothesis is conjunctive. Under the frozen TinyStories protocol,
VSA-PathMoE must:

- remain non-inferior to the configured primary official baseline in conditional
  bits per UTF-8 byte;
- exceed the configured CPU latency and throughput thresholds on the complete
  preregistered workload;
- remain below the configured resident-memory ratio threshold;
- meet the configured normalized HellaSwag delta while that task is enabled;
- show that the selected expert and route identity contribute to quality.

Passing one component must not be presented as passing the whole hypothesis.

### Evidence hierarchy

The project distinguishes the following levels of evidence:

1. Unit and smoke tests establish implementation behavior only.
2. Internal ablations establish whether experts and route identity are used by a
   checkpoint.
3. Official-baseline comparisons establish system-level deployment behavior.
4. Matched dense controls help interpret whether sparse activation is useful.
5. Multiple training seeds and hardware replications establish robustness.
6. A separate prior-art review is required for a novelty or priority claim.

Reports must state the highest level actually supported and must publish failed
gates as well as passed gates.

### Scope boundaries

TinyStories is a restricted synthetic domain. A positive result may support a
claim about the tested models, frozen workload, software revision, and recorded
hardware. It must not be generalized to all language models, all domains, or all
CPUs.

Training and quality evaluation are assigned to CUDA BF16 on the DGX Spark
profiles. The primary performance artifacts are CPU FP32 runtime axes collected
under the same profile after checkpoint and artifact transfer. The implemented
primary report requires a frozen commodity x86-64 host. A result collected on
the DGX Spark ARM CPU may be reported separately but cannot replace the primary
x86-64 artifacts.

### Current status

The code, profiles, controls, provenance checks, and report generator pass the
repository's local test and offline-fixture checks. Those checks establish
implementation behavior only. The complete DGX Spark training/evaluation run
and the cross-host CPU benchmarks have not been completed. Documentation and
fixture outputs must therefore use “result pending” language.

## Decisions & Questions

### Question #1

What is the primary scientific claim?

Response: The primary claim is joint quality non-inferiority, CPU efficiency,
resident-memory efficiency, and demonstrated expert/routing contribution under
one frozen TinyStories protocol.

### Question #2

Does comparison with official Hugging Face checkpoints isolate the causal effect
of VSA?

Response: No. It is a system-level comparison because tokenizer, vocabulary,
backbone, historical training, and runtime differ. Internal ablations and matched
dense controls are required for architectural interpretation.

### Question #3

Does the project currently establish a new class of MoE?

Response: No. Existing work already covers sparse MoE, top-1 routing, hash
routing, product-key lookup, and VSA. The project may describe its implemented
combination, but a priority claim requires a separate systematic review.

### Question #4

Which exact x86-64 CPU host will support the primary performance statement?

Response: The exact machine remains unresolved and must be preregistered before
measurement. It must be a representative commodity x86-64 host because the
implemented report rejects another primary architecture. DGX Spark ARM may
remain an explicitly separate exploratory measurement.

## Conclusion

The project must remain evidence-led. Until the complete DGX and CPU artifacts,
matched controls, and provenance checks exist, it is a protocol with a pending
result. Any later claim must remain within the domain, hardware, model, and
uncertainty boundaries defined here.
