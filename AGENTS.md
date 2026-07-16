# Repository Guidance

## Scope

This repository implements and evaluates VSA-PathMoE, a small decoder-only
Mixture-of-Experts research system whose request-level route is selected by a
causal Vector Symbolic Architecture (VSA) router. These instructions govern the
Python runtime, Node.js orchestration and reporting, configuration profiles,
tests, technical documentation, and versioned Design Specifications (DS).

The repository is a downstream consumer of the agent skills under
`.agents/skills/`; those imported skills are tooling, not the subject of the
host project's public documentation. Keep skill-specific examples and guidance
inside the corresponding skill folder. Do not create imported-skill pages or
imported-skill DS files in the host `docs/` tree.

## Mandatory Reading Order

1. Read this file and `README.md` for project scope and result status.
2. Read `docs/specs/DS001-coding-style.md` before changing code. DS001 is the
   authority for coding style, source layout, artifact compatibility, and test
   organization.
3. Read the DS files relevant to the work under `docs/specs/`. The DS
   specifications are the source of truth for documented behavior and
   structure.
4. Read `docs/index.html` and the relevant long-form HTML chapter before making
   documentation-related changes.
5. Read a skill's complete `SKILL.md` and required references before using it.

## Current Skill Catalog

The currently imported project-local skills are:

- `achilles-specs` — adds AchillesAgentLib integration rules to project
  bootstrap and DS001.
- `antropic-skill-build` — establishes the portable Anthropic-style skill
  baseline.
- `article-build` — incrementally rebuilds self-contained research articles.
- `create-akus` — creates Achilles-compatible Agentic Knowledge Units from WAC
  sources.
- `cskill-build` — separates stable C-Skill specifications from generated
  JavaScript.
- `dgskill-build` — defines guarded dynamic-code-generation skill conventions.
- `gamp-specs` — maintains the AGENTS, HTML documentation, and DS structure.
- `manage-ploinky-agents` — governs Ploinky manifests, MCP policy, service
  exposure, authentication, and routing.
- `oskill-build` — defines preparation and execution loops for orchestration
  skills.
- `review-specs` — reviews and updates DS contracts without blurring their
  boundaries.

Update this catalog whenever a skill folder is added, removed, or renamed.
When a new skill family, coding-style rule, or project-bootstrap rule is
introduced, update the `gamp-specs` skill itself as part of the same change.

## Repository Rules

- Write documentation, specifications, code comments, and persistent generated
  prose in English.
- Preserve scientific fail-closed behavior. Never silently substitute a model,
  revision, checkpoint, runtime, workload, route mapping, or missing result.
- Treat `config/base.json` plus the selected profile as the experiment input;
  preserve `runs/<profile>/resolved_config.json` once artifacts exist.
- Keep data preparation, tokenizer, router, checkpoint, workload, and model
  provenance versioned and hash-linked across stages.
- When source behavior, interfaces, architecture, workflows, or constraints
  change, update both the relevant HTML documentation and DS specifications in
  the same change set.
- Keep DS numbering contiguous and gap-free. In every ordinary DS file,
  `Decisions & Questions` uses numbered question subchapters. Put rationale,
  conflict resolution, and unresolved alternatives in the affected DS file;
  do not maintain a separate repository decision log.
- Preserve user changes in a dirty worktree and keep generated scientific
  artifacts out of source edits unless the task explicitly requires them.
- Add focused acceptance and rejection tests for changes to configuration,
  preparation, routing, checkpoints, evaluation, benchmark schemas, reporting,
  or provenance.
- Use repository-relative paths in public documentation; never publish local
  workstation paths.

## Runtime Defaults

- Seed: `20260716`; tokenizer vocabulary: 2,048; custom context: 256 tokens.
- VSA-PathMoE: six layers, width 192, six heads, 64 request-level routes,
  10,029,504 total parameters, and 1,755,840 arithmetic active parameters per
  request.
- Primary external baseline: `roneneldan/TinyStories-8M`; the 33M checkpoint is
  the larger capacity anchor.
- Official DGX Spark training and quality evaluation use profile `dgx_spark`,
  CUDA, and BF16. Matched dense controls use `dgx_spark_dense_active` and
  `dgx_spark_dense_total`.
- The deployment benchmark uses CPU FP32 on a commodity `x86_64` host at 1, 4,
  and 8 threads. DGX Spark host-CPU timing does not establish the ordinary-CPU
  claim.
- Prepared-data format is version 5; checkpoint format is version 2. DGX
  profiles write a recoverable checkpoint every 2,000,000 target tokens.
- `npm run full -- --profile dgx_spark` is intentionally disabled. Execute DGX
  stages and the cross-host CPU benchmark explicitly.
- Scientific status remains `protocol implemented; result pending` until the
  complete frozen run, dense controls, and commodity-CPU artifacts validate.

## Key Paths

- `docs/index.html` — public technical-documentation entry point.
- `docs/specsLoader.html?spec=matrix.md` — rendered DS entry point.
- `docs/specs/` — authoritative Design Specifications.
- `docs/specs/DS001-coding-style.md` — coding and test authority.
- `config/base.json`, `config/*.json` — base experiment contract and profiles.
- `python/vsa_bench/` — preparation, routing, model, training, evaluation, and
  benchmark implementation.
- `scripts/` — Node.js orchestration, report, blind-review, and verification
  tools.
- `scripts/handoff.mjs`, `handoff/` — schema-v1 DGX-to-x86 transfer inventory
  and its generated manifest.
- `tests/` — Node.js and Python regression tests.
- `containers/dgx-spark/Dockerfile` — DGX Spark container recipe; the NGC tag
  must be resolved and pinned by immutable digest before an official run.
- `requirements-dgx.txt` — exact direct DGX dependencies; PyTorch remains
  supplied by the immutable NGC base.
- `.agents/skills/` — imported agent skills, each carrying its own guidance.
