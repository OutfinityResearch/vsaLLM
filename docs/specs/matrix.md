# Specification Matrix

Generated from DS frontmatter by `scripts/generate_specs_matrix.mjs`. Edit the DS files and rerun the generator instead of editing this file manually.

| Specification | Title | Status | Owner | Summary |
| --- | --- | --- | --- | --- |
| [DS000](/specsLoader.html?spec=DS000-vision.md) | VSA-PathMoE Research Vision and Evidence Boundary | [[status:draft]] | vsaLLM maintainers | Defines the research objective, required evidence, claim scope, and current non-result status. |
| [DS001](/specsLoader.html?spec=DS001-coding-style.md) | Coding Style and Engineering Guardrails | [[status:draft]] | vsaLLM maintainers | Defines implementation conventions that preserve determinism, artifact compatibility, and fail-closed reporting. |
| [DS002](/specsLoader.html?spec=DS002-llm-model-strategy.md) | LLM Model Strategy | [[status:draft]] | vsaLLM maintainers | Defines the sparse model, VSA routing policy, official baselines, dense controls, and model-accounting boundaries. |
| [DS003](/specsLoader.html?spec=DS003-experimental-methodology.md) | Experimental Methodology and Claims | [[status:draft]] | vsaLLM maintainers | Defines evaluation metrics, preregistered gates, uncertainty rules, and valid interpretation of official and matched-control comparisons. |
| [DS004](/specsLoader.html?spec=DS004-data-assets-preparation.md) | Data, Assets, and Preparation | [[status:draft]] | vsaLLM maintainers | Defines immutable source acquisition, path-independent preparation format v5, causal splitting, and cross-host compatibility. |
| [DS005](/specsLoader.html?spec=DS005-training-checkpoints.md) | Training and Checkpoint Lifecycle | [[status:draft]] | vsaLLM maintainers | Defines DGX Spark BF16 training, two-phase optimization, resumable and best checkpoints, and checkpoint provenance. |
| [DS006](/specsLoader.html?spec=DS006-evaluation-blind-review.md) | Evaluation and Blind Review | [[status:draft]] | vsaLLM maintainers | Defines CUDA BF16 quality evaluation, official-baseline validation, sparse ablations, generation, and provenance-bound blind review. |
| [DS007](/specsLoader.html?spec=DS007-cpu-performance-memory.md) | CPU Performance and Memory Methodology | [[status:draft]] | vsaLLM maintainers | Defines CPU FP32 benchmark runtime axes, deterministic workload, timing uncertainty, resident-memory measurement, and report selection. |
| [DS008](/specsLoader.html?spec=DS008-reproducibility-publication.md) | Reproducibility, Provenance, and Publication | [[status:draft]] | vsaLLM maintainers | Defines frozen configuration, source and artifact provenance, report schema v4, publication bundles, and reproducibility limits. |
| [DS009](/specsLoader.html?spec=DS009-operations-troubleshooting.md) | Operations and Troubleshooting | [[status:draft]] | vsaLLM maintainers | Defines supported operational sequences, environment checks, failure handling, and recovery rules without weakening the protocol. |
| [DS010](/specsLoader.html?spec=DS010-dgx-spark-handoff.md) | DGX Spark Cross-Host Handoff | [[status:draft]] | vsaLLM maintainers | Defines the official DGX Spark CUDA BF16 run, transfer to a CPU host, CPU FP32 runtime-axis benchmarks, controls, and final artifact assembly. |
