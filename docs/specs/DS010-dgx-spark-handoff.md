---
id: DS010
title: DGX Spark Cross-Host Handoff
status: draft
owner: vsaLLM maintainers
summary: Defines the official DGX Spark CUDA BF16 run, transfer to a CPU host, CPU FP32 runtime-axis benchmarks, controls, and final artifact assembly.
---

# DS010 — DGX Spark Cross-Host Handoff

## Introduction

The official workflow trains and evaluates sparse and matched-dense checkpoints
on DGX Spark CUDA BF16, then transfers the immutable run lineage to a separate
host for CPU FP32 benchmarks. All stages remain under their original profile so
the report can validate one checkpoint and configuration lineage.

This is an execution contract, not evidence that the run has completed. No final
scientific result currently exists.

## Core Content

### Preconditions

Before official execution, all code and configs must be committed and the
worktree must be clean. The operator must pin the DGX software stack and archive
`doctor`, environment, storage, and power-mode information. Tests and offline
smoke must pass on the exact source revision.

The Dockerfile's default `nvcr.io/nvidia/pytorch:25.11-py3` value is a mutable
tag, not an immutable identity. The operator must resolve the trusted image's
registry RepoDigest after pulling it and pass the digest-qualified reference as
the `PYTORCH_IMAGE` build argument. The derived image must retain PyTorch and
CUDA from that NGC base: `requirements-dgx.txt` pins the direct experiment
dependencies and must never install or replace `torch`.

The runtime must set `VSA_CONTAINER_IMAGE_DIGEST` to `sha256:<64 lowercase hex
characters>` or the equivalent `image@sha256:<64 lowercase hex characters>`, and
must set `VSA_DERIVED_IMAGE_ID` to `sha256:<64 lowercase hex characters>`. The
repository does not publish an unverified digest. `doctor:dgx --strict` must fail
until both resolved values are present. The exact base RepoDigest, derived image
ID, Dockerfile, `requirements-dgx.txt`, and complete `pip freeze --all` output
must be archived with the run.

The strict DGX preflight must also confirm an ARM64 host, NVIDIA GB10 GPU with
compute capability 12.1, available CUDA, BF16 support, execution inside a
container, the exact `nvcr.io/nvidia/pytorch:25.11-py3` image label, a valid
derived image ID, exact
installed versions for every direct dependency in `requirements-dgx.txt`,
Node.js 22 or newer, at least 110,000,000,000 bytes of system memory, at least
20,000,000,000 bytes of free project-filesystem storage, a clean Git worktree,
immutable hexadecimal revisions for the dataset, HellaSwag, and every official
model, and the configured CUDA/BF16 profile runtime. Unified memory is assessed
as system memory rather than as discrete `nvidia-smi` VRAM.

The three official training profiles are:

- `dgx_spark` for VSA-PathMoE;
- `dgx_spark_dense_active` for the 1,755,840-parameter fixed-dense control;
- `dgx_spark_dense_total` for the 10,029,504-parameter fixed-dense control.

They must use the same immutable assets and format-v5 prepared store. Their
thresholds and training budgets must not be changed after test inspection.
Both `prepare` and `train` independently run the strict DGX preflight for every
`dgx_spark*` profile; the explicit doctor steps below provide archived evidence
but are not the only enforcement point.

### DGX Spark sparse run

Run explicit stages rather than `npm run full`:

```bash
npm run verify:package
mkdir -p runs/dgx_spark/artifacts
python -m pip check
python -m pip freeze --all > runs/dgx_spark/artifacts/pip_freeze_dgx.txt
npm test
npm run smoke:offline
npm run doctor:dgx -- --profile dgx_spark --strict \
  > runs/dgx_spark/artifacts/doctor_dgx_spark.txt
npm run download:full -- --profile dgx_spark
npm run prepare -- --profile dgx_spark
npm run train -- --profile dgx_spark
npm run evaluate -- --profile dgx_spark
```

The resulting training and evaluation artifacts must report CUDA BF16 and the
same profile/checkpoint lineage. Do not run the final report yet because primary
CPU FP32 benchmark artifacts are intentionally absent.

### DGX Spark dense controls

For each control, freeze its own resolved config, validate shared preparation,
train, and evaluate:

```bash
mkdir -p runs/dgx_spark_dense_active/artifacts
npm run doctor:dgx -- --profile dgx_spark_dense_active --strict \
  > runs/dgx_spark_dense_active/artifacts/doctor_dgx_dense_active.txt
npm run prepare -- --profile dgx_spark_dense_active
npm run train -- --profile dgx_spark_dense_active
npm run evaluate:custom -- --profile dgx_spark_dense_active

mkdir -p runs/dgx_spark_dense_total/artifacts
npm run doctor:dgx -- --profile dgx_spark_dense_total --strict \
  > runs/dgx_spark_dense_total/artifacts/doctor_dgx_dense_total.txt
npm run prepare -- --profile dgx_spark_dense_total
npm run train -- --profile dgx_spark_dense_total
npm run evaluate:custom -- --profile dgx_spark_dense_total
```

The existing asset manifest is shared. If a profile needs to reacquire assets,
it must resolve the same immutable SHAs. Dense custom evaluation must retain
skipped routing ablations. Official evaluation must not be duplicated in the
control profiles. Cross-profile artifacts are aggregated and validated fail
closed, but no matched-control performance pass/fail gate is implemented.

After all three profiles have the required checkpoints, summaries, and custom
evaluations, and the main sparse profile has every official evaluation, create
the transfer manifest from the clean worktree:

```bash
npm run handoff:create -- --profile dgx_spark
```

### Transfer bundle

`handoff/dgx_spark_manifest.json` is schema v1 and must record the exact Git
commit, prepared format, profile list, transfer roots, and each regular file's
byte count and streaming SHA-256. Its roots are `data/prepared/dgx_spark` and the
complete run directories for `dgx_spark`, `dgx_spark_dense_active`, and
`dgx_spark_dense_total`. Symlinks are forbidden.

Creation and verification must validate roots and profiles against frozen
config, require the exact full regular-file set, and reject duplicate or unsafe
paths, missing or unmanifested files, size/digest mismatches, and missing
required artifacts. Both commands print the manifest SHA-256; the operator must
record that value out of band before and after transport.

Transfer those roots plus the manifest. Asset manifests and Hugging Face caches
must not be transferred as handoff roots. On the CPU host, check out the exact
manifest commit and verify every transferred byte before loading a checkpoint:

```bash
npm run handoff:verify -- --profile dgx_spark
```

The operator must then regenerate `data/assets_dgx_spark.json` in deployment-only
mode against the same resolved revisions. This reacquires validation text,
prompts, and official-model snapshots while skipping the training corpus and
HellaSwag. The transferred prepared store remains immutable and is validated by
the handoff hashes; do not run `prepare` for a `dgx_spark*` profile on x86-64.
The command rejects the deployment-only manifest in any case. The publication
archive must also retain the base RepoDigest, derived image ID, Dockerfile,
`.dockerignore`, `requirements-dgx.txt`, `pip freeze --all`, and DGX environment
records.

### CPU FP32 benchmark stage

After the sparse run has been transferred and local assets resolve correctly:

```bash
npm run verify:package
npm run setup -- --locked-cpu
.venv/bin/python -m pip freeze --all \
  > runs/dgx_spark/artifacts/pip_freeze_cpu_x86.txt
npm test
npm run doctor -- --profile dgx_spark > runs/dgx_spark/artifacts/doctor_cpu_x86.txt
npm run download:deployment -- --profile dgx_spark
lscpu -e=CPU,CORE,SOCKET,ONLINE \
  > runs/dgx_spark/artifacts/cpu_topology_x86.txt
CPUSET=<eight-comma-separated-logical-CPU-ids>
taskset -c "$CPUSET" npm run benchmark:matrix -- --profile dgx_spark \
  --device cpu --dtype fp32
```

Benchmark orchestration must default to the frozen primary runtime axes CPU and
FP32, even though the profile's training runtime is CUDA BF16. It must write
runtime-specific 1/4/8-thread artifacts and preserve the exact checkpoint SHA
used by DGX evaluation.

The locked CPU setup force-installs the official CPU-only `torch==2.10.0` wheel,
then installs the exact direct versions from `requirements-dgx.txt` and runs
dependency consistency checks. Benchmark artifacts must record Python, PyTorch,
NumPy, and the runtime-dependency version map; every paired artifact must match
those values exactly. DGX preparation, training, resume, and CUDA evaluation
provenance must additionally preserve the CUDA device inventory (name, compute
capability, and total memory), cuDNN version, and NVIDIA driver version.

The three DGX profiles freeze
`benchmark.required_cpu_affinity_logical_cpus=8` and
`benchmark.require_distinct_physical_cores=true`. Each custom or official
benchmark command snapshots `environment.cpu_topology` from Linux sysfs before
model loading and fails unless the current affinity has exactly eight logical
CPUs, eight distinct `(physical_package_id,core_id)` pairs, complete sibling
metadata, and no selected SMT siblings.

Run the same CPU FP32 benchmark stage for transferred dense checkpoints:

```bash
taskset -c "$CPUSET" npm run benchmark:custom:matrix \
  -- --profile dgx_spark_dense_active --device cpu --dtype fp32
taskset -c "$CPUSET" npm run benchmark:custom:matrix \
  -- --profile dgx_spark_dense_total --device cpu --dtype fp32
```

Replace the `CPUSET` placeholder only after inspecting the saved topology. It
must identify exactly eight distinct `(socket,core)` pairs and must not include
SMT siblings. The identical value must be used for the sparse matrix, both
official models executed within it, and both control matrices. Set and record a
stable governor/performance policy before the first matrix and keep it unchanged
through the last. The report revalidates every recorded topology object and
requires exact paired equality for affinity, topology, and governors across
sparse, official, and control artifacts. The saved `lscpu` topology and
governor-control records remain mandatory independent evidence. No particular
governor value is enforced by code; the operator-selected policy must remain
stable and exactly paired.

The primary CPU host must be a preregistered commodity x86-64 system. The report
rejects a primary artifact whose recorded architecture is not x86-64/amd64. A
DGX Spark host-CPU measurement may be retained as an additional ARM result but
cannot replace the configured primary artifacts. Because architecture is not in
the benchmark filename, any ARM artifact must be copied to a separate
exploratory archive before the x86-64 run and must never occupy or overwrite a
primary CPU FP32 artifact slot.

### Reporting and blind review

Generate the single validated sparse report and its blind-review form:

```bash
npm run report -- --profile dgx_spark
npm run blind -- --profile dgx_spark
```

The sparse report must select only `cpu/fp32` benchmark files, record primary
runtime device/dtype, and fail if evaluation and benchmark source/checkpoint
provenance differ. It must also load both configured control profiles, validate
their custom evaluation and complete CPU matrices, and integrate their quality,
CPU, and memory measurements descriptively. It must fail when a required control
artifact or provenance field is missing. No control performance result
contributes a pass/fail gate.

The report JSON is schema v4. Its top-level `matchedControls` object must retain
the configured roles/profiles, artifact and performance policies, exact budget
and prepared-identity checks, hashed evidence paths, provenance, and complete
quality, CPU, and memory comparisons for both controls. The rendered report
formats must surface that descriptive evidence from the same validated object.

Blind review must be completed without revealing the key. After the score export
is saved, rerun the sparse report. The final publication bundle must include all
source, config, checkpoint, asset, workload, environment, report, and control
artifacts listed in DS008.

### Stop conditions

The official workflow must stop if any checksum, resolved config, preparation
signature, tokenizer/router hash, checkpoint SHA, training profile, Git/source
identity, official revision, evaluation hash, workload manifest, CPU
affinity/topology, paired governor record, or primary runtime does not match.
Operators must not repair the mismatch by editing JSON metadata.

## Decisions & Questions

### Question #1

Why must `npm run full` be avoided for the official cross-host run?

Response: `full` continues directly into benchmark and report on the training
host. The official workflow intentionally stops after CUDA BF16 evaluation,
transfers the lineage, and creates primary CPU FP32 artifacts on the designated
CPU host.

### Question #2

How can local asset paths change without invalidating preparation?

Response: Preparation format v5 excludes machine-local downloaded-asset paths
and instead binds dataset ID, immutable revision, filenames, seed, and the
complete data config. Transferred tokenizer/router hashes and source identities
must still match.

### Question #3

Are matched-control profiles part of the sparse top-level verdict?

Response: No. They are mandatory interpretive evidence and must be published,
but no automated cross-profile matched-control performance pass/fail gate is
currently implemented.

### Question #4

How will the transfer bundle be transported and independently retained?

Response: This remains an operational choice, but checksum verification is
mandatory at both ends.

Options:

- create a content-addressed archive with a signed SHA-256 manifest;
- use a resumable file transfer and independently generated pre/post SHA-256
  manifests.

### Question #5

Which commodity x86-64 host is the primary CPU replication target?

Response: The exact host remains unresolved. It must be fixed before CPU
measurements and recorded in the publication bundle. The implemented report
requires x86-64/amd64; DGX Spark ARM may only be an additional exploratory
measurement.

## Conclusion

The DGX handoff preserves one profile and checkpoint lineage while changing only
the permitted benchmark runtime axes. CUDA BF16 training/evaluation, verified
format-v5 transfer, CPU FP32 benchmarks, integrated descriptive control evidence,
and fail-closed provenance are all required before any final result may be stated.
