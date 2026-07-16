# VSA-PathMoE — controlled TinyStories research protocol

VSA-PathMoE is an executable experiment for request-level sparse language-model
routing. A causal Vector Symbolic Architecture (VSA) router selects one small
tensor expert per layer while attention and a shared feed-forward path remain
active. The repository compares that design with published TinyStories models
and with two locally trained dense controls.

> **Scientific status: protocol implemented; result pending.** Tests, fixtures,
> parameter accounting, and artifact validation establish that the protocol is
> executable. They do not establish a quality, speed, memory, superiority, or
> novelty result. Only a complete frozen DGX-to-x86 run may support a narrowly
> qualified statement about the measured checkpoints and hosts.

## Documentation authority

The maintained technical documentation starts at
[`docs/index.html`](docs/index.html). The authoritative Design Specifications
(DS) are available through
[`docs/specsLoader.html?spec=matrix.md`](docs/specsLoader.html?spec=matrix.md).
In particular:

- [`DS001`](docs/specs/DS001-coding-style.md) defines coding, source-layout,
  compatibility, and test rules;
- [`DS003`](docs/specs/DS003-experimental-methodology.md) defines the claims,
  metrics, controls, and interpretation limits;
- [`DS010`](docs/specs/DS010-dgx-spark-handoff.md) defines the official staged
  DGX Spark to commodity-x86 execution contract;
- [`docs/dgx-runbook.html`](docs/dgx-runbook.html) is the operational runbook.

The DS set is the source of truth when explanatory text differs. The previous
numbered Markdown chapters have been consolidated into DS000–DS010 and the HTML
site. To view fetched partials and rendered specifications locally, serve the
documentation root and open <http://localhost:8000/>:

```bash
python3 -m http.server --directory docs 8000
```

## Experiment boundary

The default sparse model has six decoder layers, width 192, 64 request-stable
routes, 10,029,504 total parameters, and 1,755,840 arithmetic active parameters
per request. Those counts describe the implemented model and the selected
tensor path; they are not measurements of process memory or inference speed.

The frozen comparison has three distinct roles:

- `roneneldan/TinyStories-8M` is the primary external baseline;
- `roneneldan/TinyStories-33M` is a larger capacity anchor, not a matched
  training control;
- `dgx_spark_dense_active` and `dgx_spark_dense_total` are fixed-dense controls
  matched to the sparse model's active and total parameter budgets.

Quality uses the same raw continuation boundary and conditional bits per UTF-8
byte. HellaSwag, routing ablations, expert ablations, blind review, isolated
CPU timing, and measured plus persistent model-state memory are separate pieces
of evidence. The report must retain failures and uncertainty rather than
selecting a convenient baseline or silently substituting an artifact.

## Official execution boundary

The official workflow deliberately crosses two hosts:

1. An NVIDIA DGX Spark trains and evaluates the sparse model and both dense
   controls with CUDA BF16 inside the pinned NGC-derived container.
2. A handoff manifest freezes the prepared store and all three run directories,
   including their full file inventory, sizes, and SHA-256 values.
3. A preregistered commodity `x86_64` host verifies the handoff and runs the
   CPU FP32 1/4/8-thread matrices under one recorded physical-core affinity.
4. Reporting and blind-review artifacts are generated only after the x86 stage
   and both control matrices exist.

`npm run full -- --profile dgx_spark` is intentionally disabled because a
single-host pipeline cannot represent this boundary.

## Preflight and container identity

Commit the reviewed source and begin the official run from a clean worktree.
Run `npm ci`, the package-integrity check, the full tests, and the offline smoke
on that exact revision. The Dockerfile defaults to the human-readable
`nvcr.io/nvidia/pytorch:25.11-py3` tag, but an official build must resolve the
real registry RepoDigest and pass the digest-qualified image reference as its
`PYTORCH_IMAGE` build argument. The repository intentionally does not publish
an unverified digest.

The derived container must expose the recorded base digest through
`VSA_CONTAINER_IMAGE_DIGEST` and its own image ID through
`VSA_DERIVED_IMAGE_ID`. Follow the exact build, mount, cache, UID/GID, and GPU
commands in the [DGX runbook](docs/dgx-runbook.html). Do not run
`npm run setup` inside NGC: PyTorch and CUDA belong to the immutable base image,
while `requirements-dgx.txt` supplies only exact direct experiment dependencies.

## DGX Spark stage

Inside the correctly identified derived container, run the sparse stages
explicitly:

```bash
npm ci
npm run verify:package
npm test
npm run smoke:offline
npm run doctor:dgx -- --profile dgx_spark --strict
npm run download:full -- --profile dgx_spark
npm run prepare -- --profile dgx_spark
npm run train -- --profile dgx_spark
npm run evaluate -- --profile dgx_spark
```

Then validate, train, and custom-evaluate both matched controls against the same
immutable assets and format-v5 prepared store:

```bash
npm run doctor:dgx -- --profile dgx_spark_dense_active --strict
npm run prepare -- --profile dgx_spark_dense_active
npm run train -- --profile dgx_spark_dense_active
npm run evaluate:custom -- --profile dgx_spark_dense_active

npm run doctor:dgx -- --profile dgx_spark_dense_total --strict
npm run prepare -- --profile dgx_spark_dense_total
npm run train -- --profile dgx_spark_dense_total
npm run evaluate:custom -- --profile dgx_spark_dense_total
```

Do not duplicate official-model evaluation in the control profiles and do not
generate the final report on DGX. Once all required artifacts exist and the
worktree is clean, create the transfer inventory:

```bash
npm run handoff:create -- --profile dgx_spark
```

Record the printed manifest SHA-256 through an independent channel. Transfer
only the roots named by the manifest plus the manifest itself; do not transfer
`.venv`, the container filesystem, asset manifests, or Hugging Face caches.

## Commodity-x86 stage

On the designated x86 host, check out the manifest's exact Git commit, place
the transferred roots at their repository-relative paths, and run:

```bash
npm ci
npm run verify:package
npm run setup -- --locked-cpu
npm run handoff:verify -- --profile dgx_spark
npm test
npm run doctor -- --profile dgx_spark
npm run download:deployment -- --profile dgx_spark
```

The deployment download reacquires validation text, prompts, and official model
snapshots at their immutable revisions. It omits the training corpus and
HellaSwag. Never rerun `prepare` for a `dgx_spark*` profile on x86.

After recording topology, select eight logical CPU IDs that map to eight
distinct physical cores without selected SMT siblings. Use the same affinity
and stable host policy for every matrix:

```bash
CPUSET=<eight-comma-separated-logical-CPU-ids>
taskset -c "$CPUSET" npm run benchmark:matrix -- --profile dgx_spark \
  --device cpu --dtype fp32
taskset -c "$CPUSET" npm run benchmark:custom:matrix -- \
  --profile dgx_spark_dense_active --device cpu --dtype fp32
taskset -c "$CPUSET" npm run benchmark:custom:matrix -- \
  --profile dgx_spark_dense_total --device cpu --dtype fp32
npm run report -- --profile dgx_spark
npm run blind -- --profile dgx_spark
```

All three DGX profiles require exactly eight affinity IDs and distinct physical
cores. Before loading a custom or official model, the benchmark validates the
current affinity against complete Linux sysfs topology and rejects selected SMT
siblings. The 1/4/8 values are worker-thread counts; the eight-CPU affinity does
not change. The report revalidates and exactly pairs sparse, official, and
control affinity/topology/governor records, and exposes topology in JSON,
Markdown, HTML, and CSV. Governors are recorded and paired, but the code does
not impose a particular governor value.

The report is fail-closed: missing controls, mismatched hashes or preparation
identities, incomplete thread matrices, wrong runtime axes, non-x86 primary
artifacts, or provenance drift are errors. Never repair a failed lineage by
editing generated JSON.

## Repository map

- `config/` contains the base experiment contract and named profiles.
- `python/vsa_bench/` implements preparation, routing, models, training,
  evaluation, and benchmarking.
- `scripts/` provides orchestration, handoff, reporting, blind review, package
  verification, and documentation checks.
- `containers/dgx-spark/Dockerfile` defines the NGC-derived execution image.
- `tests/` contains Node.js and Python acceptance and rejection tests.
- `docs/` contains the technical site; `docs/specs/` contains the authoritative
  DS contracts.

External datasets, model weights, container images, and Python packages are not
redistributed by this repository. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
for the acquisition boundary.
