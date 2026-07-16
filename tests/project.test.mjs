import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { ROOT, comparableRunConfig } from '../scripts/lib.mjs';

const readText = (file) => fs.readFileSync(path.join(ROOT, file), 'utf8');
const readJson = (file) => JSON.parse(readText(file));

const htmlPages = [
  'docs/index.html',
  'docs/architecture.html',
  'docs/protocol.html',
  'docs/data.html',
  'docs/training.html',
  'docs/evaluation.html',
  'docs/cpu-memory.html',
  'docs/reproducibility.html',
  'docs/operations.html',
  'docs/dgx-runbook.html',
  'docs/sources.html',
  'docs/specsLoader.html',
];

const specificationFiles = [
  'DS000-vision.md',
  'DS001-coding-style.md',
  'DS002-llm-model-strategy.md',
  'DS003-experimental-methodology.md',
  'DS004-data-assets-preparation.md',
  'DS005-training-checkpoints.md',
  'DS006-evaluation-blind-review.md',
  'DS007-cpu-performance-memory.md',
  'DS008-reproducibility-publication.md',
  'DS009-operations-troubleshooting.md',
  'DS010-dgx-spark-handoff.md',
];

test('canonical guidance, HTML documentation, and runtime entry points exist', () => {
  const required = [
    'AGENTS.md',
    'README.md',
    'PACKAGE_INFO.json',
    'THIRD_PARTY_NOTICES.md',
    'fileSizesCheck.sh',
    'config/base.json',
    'config/dgx_spark.json',
    'config/dgx_spark_dense_active.json',
    'config/dgx_spark_dense_total.json',
    'containers/dgx-spark/Dockerfile',
    'requirements-dgx.txt',
    'python/vsa_bench/cli.py',
    'python/vsa_bench/dgx.py',
    'scripts/experiment.mjs',
    'scripts/handoff.mjs',
    'docs/styles.css',
    'docs/partials/header.html',
    'docs/partials/footer.html',
    'docs/partials-loader.js',
    'docs/specs/matrix.md',
    ...htmlPages,
  ];
  for (const file of required) {
    assert.equal(fs.existsSync(path.join(ROOT, file)), true, file);
  }
});

test('AGENTS and the maintained HTML site expose the DS authority', () => {
  const guidance = readText('AGENTS.md');
  const headings = [
    '## Scope',
    '## Mandatory Reading Order',
    '## Current Skill Catalog',
    '## Repository Rules',
    '## Runtime Defaults',
    '## Key Paths',
  ];
  let previous = -1;
  for (const heading of headings) {
    const position = guidance.indexOf(heading);
    assert.ok(position > previous, `${heading} must exist in canonical order`);
    previous = position;
  }
  assert.match(guidance, /DS\s+specifications are the source of truth/);
  assert.match(guidance, /DS001-coding-style\.md/);
  assert.match(guidance, /docs\/index\.html/);
  assert.match(guidance, /DGX-to-x86|DGX.*x86/s);

  for (const file of htmlPages) {
    const html = readText(file);
    assert.match(html, /<html lang="en">/, file);
    assert.match(html, /mermaid@11\/dist\/mermaid\.esm\.min\.mjs/, file);
    assert.match(html, /data-include="partials\/header\.html"/, file);
    assert.match(html, /data-include="partials\/footer\.html"/, file);
    assert.match(html, /src="partials-loader\.js"/, file);
  }

  const header = readText('docs/partials/header.html');
  assert.equal((header.match(/<nav\b/g) ?? []).length, 1);
  assert.match(header, /specsLoader\.html\?spec=matrix\.md/);
  assert.match(readText('docs/index.html'), /Protocol implemented; final DGX and commodity-CPU result pending/);
});

test('the DS catalog is contiguous, structured, and reachable through the matrix', () => {
  const specsDir = path.join(ROOT, 'docs', 'specs');
  const actual = fs.readdirSync(specsDir)
    .filter((file) => /^DS\d{3}-.*\.md$/.test(file))
    .sort();
  assert.deepEqual(actual, specificationFiles);

  const matrix = readText('docs/specs/matrix.md');
  specificationFiles.forEach((file, index) => {
    const id = `DS${String(index).padStart(3, '0')}`;
    const content = readText(path.join('docs', 'specs', file));
    assert.match(content, new RegExp(`^id: ${id}$`, 'm'), file);
    for (const field of ['title', 'status', 'owner', 'summary']) {
      assert.match(content, new RegExp(`^${field}: .+$`, 'm'), `${file}: ${field}`);
    }
    for (const section of [
      '## Introduction',
      '## Core Content',
      '## Decisions & Questions',
      '## Conclusion',
    ]) {
      assert.match(content, new RegExp(`^${section.replace(/[&]/g, '\\&')}$`, 'm'), `${file}: ${section}`);
    }
    assert.match(content, /^### Question #\d+/m, `${file}: numbered question`);
    assert.ok(
      matrix.includes(`/specsLoader.html?spec=${file}`),
      `${file} must be linked through specsLoader.html`,
    );
  });
});

test('production config targets the registered sparse model and CPU comparison', () => {
  const cfg = readJson('config/base.json');
  assert.equal(cfg.data.router_product_k ** 2, 64);
  assert.equal(cfg.model.expert_hidden, 57);
  assert.equal(cfg.model.d_model, 192);
  assert.equal(cfg.evaluation.common_context_length, cfg.model.block_size);
  assert.equal(cfg.evaluation.primary_quality_context, 'native');
  assert.equal(cfg.evaluation.run_common_context_diagnostic, true);
  assert.equal(cfg.comparison.primary_official_model_id, 'roneneldan/TinyStories-8M');
  assert.equal(cfg.benchmark.primary_device, 'cpu');
  assert.equal(cfg.benchmark.primary_dtype, 'fp32');
  assert.equal(cfg.benchmark.primary_machine_architecture, 'x86_64');
  assert.equal(cfg.benchmark.primary_threads, 1);
});

test('DGX profiles pin immutable shared assets and matched dense controls', () => {
  const sparse = readJson('config/dgx_spark.json');
  const active = readJson('config/dgx_spark_dense_active.json');
  const total = readJson('config/dgx_spark_dense_total.json');
  const profiles = [sparse, active, total];

  assert.deepEqual(sparse.comparison.matched_control_profiles, {
    active_parameter_budget: 'dgx_spark_dense_active',
    total_parameter_budget: 'dgx_spark_dense_total',
  });
  assert.deepEqual(
    sparse.sources.official_models.map((item) => item.id),
    ['roneneldan/TinyStories-8M', 'roneneldan/TinyStories-33M'],
  );

  for (const profile of profiles) {
    assert.equal(profile.runtime.device, 'cuda');
    assert.equal(profile.runtime.dtype, 'bf16');
    assert.equal(profile.benchmark.required_cpu_affinity_logical_cpus, 8);
    assert.equal(profile.benchmark.require_distinct_physical_cores, true);
    assert.equal(profile.data.prepared_dir, 'data/prepared/dgx_spark');
    assert.equal(profile.training.checkpoint_every_target_tokens, 2_000_000);
    assert.match(profile.sources.dataset_revision, /^[0-9a-f]{40,64}$/);
    assert.match(profile.sources.hellaswag_revision, /^[0-9a-f]{40,64}$/);
    for (const model of profile.sources.official_models) {
      assert.match(model.revision, /^[0-9a-f]{40,64}$/);
    }
    assert.deepEqual(profile.sources, sparse.sources);
  }

  assert.equal(active.model.routing_mode, 'fixed_dense');
  assert.equal(active.model.shared_hidden, 128);
  assert.equal(total.model.routing_mode, 'fixed_dense');
  assert.equal(total.model.shared_hidden, 3719);
});

test('DGX container and dependency lock preserve the pinned runtime boundary', () => {
  const dockerfile = readText('containers/dgx-spark/Dockerfile');
  const requirements = readText('requirements-dgx.txt');
  const pins = requirements.split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'));

  assert.match(dockerfile, /ARG PYTORCH_IMAGE=nvcr\.io\/nvidia\/pytorch:25\.11-py3/);
  assert.match(dockerfile, /ARG NODE_IMAGE=node:22\.16\.0-bookworm-slim/);
  assert.match(dockerfile, /COPY requirements-dgx\.txt/);
  assert.match(dockerfile, /VSA_CONTAINER_IMAGE=nvcr\.io\/nvidia\/pytorch:25\.11-py3/);
  assert.doesNotMatch(dockerfile, /pip install[^\n]*torch/i);
  assert.ok(pins.length > 0);
  for (const pin of pins) assert.match(pin, /^[A-Za-z0-9_.-]+==[^=\s]+$/, pin);
  assert.equal(pins.some((pin) => /^torch==/i.test(pin)), false);
});

test('DGX-to-x86 orchestration retains strict doctor and handoff safeguards', () => {
  const pkg = readJson('package.json');
  assert.equal(pkg.scripts['doctor:dgx'], 'node scripts/experiment.mjs doctor-dgx');
  assert.equal(pkg.scripts['download:deployment'], 'node scripts/experiment.mjs download-deployment');
  assert.equal(pkg.scripts['handoff:create'], 'node scripts/handoff.mjs create');
  assert.equal(pkg.scripts['handoff:verify'], 'node scripts/handoff.mjs verify');
  assert.equal(pkg.scripts['benchmark:custom:matrix'], 'node scripts/experiment.mjs benchmark-custom-matrix');

  const doctor = readText('python/vsa_bench/dgx.py');
  for (const check of [
    'arm64_host',
    'cuda_available',
    'bf16_supported',
    'gb10_gpu',
    'blackwell_compute_capability',
    'clean_git_worktree',
    'containerized',
    'pinned_container_digest',
    'derived_image_id',
    'locked_python_dependencies',
    'immutable_source_revisions',
    'profile_runtime',
  ]) {
    assert.match(doctor, new RegExp(`"${check}"`), check);
  }

  const experiment = readText('scripts/experiment.mjs');
  assert.match(experiment, /The monolithic full command is disabled/);
  const setup = readText('scripts/setup.mjs');
  assert.match(setup, /--locked-cpu/);
  assert.match(setup, /torch==2\.10\.0/);

  const handoff = readText('scripts/handoff.mjs');
  assert.match(handoff, /const SCHEMA_VERSION = 1/);
  assert.match(handoff, /Symlinks are not allowed in handoff/);
  assert.match(handoff, /Create the official handoff only from a clean Git worktree/);
  assert.match(handoff, /Handoff roots\/profiles differ from the frozen main configuration/);
  assert.match(handoff, /Transferred roots contain missing or unmanifested files/);
  assert.match(handoff, /Manifest SHA-256:/);
});

test('root metadata points to the maintained protocol without claiming a result', () => {
  const readme = readText('README.md');
  assert.match(readme, /Scientific status: protocol implemented; result pending/i);
  assert.match(readme, /docs\/index\.html/);
  assert.match(readme, /docs\/specsLoader\.html\?spec=matrix\.md/);
  assert.match(readme, /npm run handoff:create -- --profile dgx_spark/);
  assert.match(readme, /npm run handoff:verify -- --profile dgx_spark/);
  assert.match(readme, /npm run setup -- --locked-cpu/);
  assert.doesNotMatch(readme, /docs\/(00_START_HERE|13_OFFICIAL_RUNBOOK|14_CLAIMS_AND_CONTROLS)\.md/);

  const info = readJson('PACKAGE_INFO.json');
  assert.equal(info.documentation_language, 'English');
  assert.equal(info.scientific_status, 'protocol implemented; result pending');
  assert.equal(info.entry_points.technical_documentation, 'docs/index.html');
  assert.equal(info.entry_points.dgx_handoff_specification, 'docs/specs/DS010-dgx-spark-handoff.md');
  assert.equal(info.execution_contract.benchmark_runtime, 'CPU FP32 at 1, 4, and 8 threads');
  assert.match(info.execution_contract.cpu_affinity_contract, /eight distinct physical cores/);
  assert.equal(info.distribution.external_datasets_included, false);
  assert.equal('local_verification' in info, false);

  const notices = readText('THIRD_PARTY_NOTICES.md');
  assert.match(notices, /does not redistribute/i);
  assert.match(notices, /nvcr\.io\/nvidia\/pytorch:25\.11-py3/);
  assert.match(notices, /node:22\.16\.0-bookworm-slim/);
  assert.match(notices, /torch==2\.10\.0/);
  assert.match(notices, /Mermaid 11/);

  const gitignore = readText('.gitignore');
  assert.match(gitignore, /^data\/\*$/m);
  assert.match(gitignore, /^!data\/prepared\/$/m);
  assert.match(gitignore, /^!data\/prepared\/\.gitkeep$/m);
});

test('superseded numbered Markdown chapters have been consolidated into DS and HTML', () => {
  const obsolete = fs.readdirSync(path.join(ROOT, 'docs')).filter(
    (file) => /^\d{2}_.+\.md$/.test(file)
      || ['EXPERIMENT_CHECKLIST.md', 'TESTED_LOCALLY.md'].includes(file),
  );
  assert.deepEqual(obsolete, []);
});

test('benchmark config comparison ignores only the transferred runtime axes', () => {
  const previous = {
    runtime: { threads: 8, device: 'cpu', dtype: 'fp32', allocator: 'native' },
    model: { d: 32 },
  };
  const current = {
    runtime: { threads: 1, device: 'cuda', dtype: 'bf16', allocator: 'native' },
    model: { d: 32 },
  };
  assert.deepEqual(
    comparableRunConfig(previous, { ignoreBenchmarkRuntime: true }),
    comparableRunConfig(current, { ignoreBenchmarkRuntime: true }),
  );
  assert.notDeepEqual(comparableRunConfig(previous), comparableRunConfig(current));

  const unrelatedDrift = {
    ...current,
    runtime: { ...current.runtime, allocator: 'changed' },
  };
  assert.notDeepEqual(
    comparableRunConfig(previous, { ignoreBenchmarkRuntime: true }),
    comparableRunConfig(unrelatedDrift, { ignoreBenchmarkRuntime: true }),
  );
  assert.equal(previous.runtime.threads, 8, 'comparison must not mutate the frozen config');
});
