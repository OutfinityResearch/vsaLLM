import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

process.env.VSA_REPORT_LIBRARY_ONLY = '1';
const {
  bootstrapRatio,
  bootstrapSpeedup,
  canonicalJson,
  conservativePrimary,
  generateReport,
  residentMemoryRatio,
} = await import('../scripts/report.mjs');

const PROFILE = 'unit';
const CUSTOM_SHA = 'c'.repeat(64);
const RUNTIME_SHA = 'a'.repeat(64);
const PREPARATION_SHA = '6'.repeat(64);
const BLIND_ID = 'e'.repeat(64);
const TRAIN_COMMIT = 'f'.repeat(40);
const BASE_IMAGE_DIGEST = `sha256:${'1'.repeat(64)}`;
const DERIVED_IMAGE_ID = `sha256:${'2'.repeat(64)}`;
const REVISION_8M = '1'.repeat(40);
const REVISION_33M = '2'.repeat(40);
const ACTIVE_CONTROL_PROFILE = 'unit_dense_active';
const TOTAL_CONTROL_PROFILE = 'unit_dense_total';
const RUNTIME_DEPENDENCIES = {
  sentencepiece: '0.2.1',
  PyYAML: '6.0.3',
  psutil: '7.2.2',
  huggingface_hub: '1.4.1',
  transformers: '5.3.0',
  datasets: '4.8.3',
  safetensors: '0.7.0',
};
const CUDA_DEVICES = [{
  index: 0,
  name: 'NVIDIA GB10',
  compute_capability: [12, 1],
  total_memory_bytes: 128_000_000_000,
}];
const MODELS = [
  ['example/TinyStories-8M', REVISION_8M, 8_000_000, 100_000_000],
  ['example/TinyStories-33M', REVISION_33M, 33_000_000, 132_000_000],
];

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

function sha256(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function sha256File(file) {
  return sha256(fs.readFileSync(file));
}

function perStory(nll) {
  return [
    { id: 0, nll, bytes: 10 },
    { id: 1, nll, bytes: 10 },
  ];
}

function quality(nll = 10, hellaswag = 0.5) {
  const heldout = {
    conditional_bits_per_byte: nll / 10,
    context_limit: 512,
    per_story: perStory(nll),
  };
  return {
    primary_context_mode: 'native',
    heldout,
    heldout_native: heldout,
    heldout_common_context: { ...heldout, context_limit: 256 },
    hellaswag: { accuracy_length_normalized: hellaswag, examples: 100 },
  };
}

function rawRows(endToEndMs, decodeMs) {
  return Array.from({ length: 6 }, () => ({
    end_to_end_ms: endToEndMs,
    decode_ms: decodeMs,
    generated_tokens: 10,
    generated_utf8_bytes: 20,
  }));
}

function measuredCase(item, official = false) {
  const endToEndMs = official ? 20 : 10;
  const decodeMs = official ? 10 : 5;
  return {
    ...item,
    warm_route_cache: {
      end_to_end_ms: { median_ms: endToEndMs },
      derived: {
        decode_tokens_per_second: official ? 1_000 : 2_000,
        end_to_end_generated_bytes_per_second: official ? 1_000 : 2_000,
        median_prompt_tokens: official ? 128 : 120,
      },
    },
    cold_route_cache: null,
    raw_warm_rows: rawRows(endToEndMs, decodeMs),
    raw_cold_rows: null,
  };
}

function memory(loadUss, loadRss, residentBytes) {
  return {
    snapshots: {
      after_load: {
        rss_bytes: 300_000_000,
        uss_bytes: 250_000_000,
        pss_bytes: 275_000_000,
      },
      after_benchmark: {
        peak_rss_hwm_bytes: 350_000_000,
      },
    },
    deltas: {
      load: {
        rss_delta_bytes: loadRss,
        uss_delta_bytes: loadUss,
        pss_delta_bytes: loadUss,
      },
      total: {
        rss_delta_bytes: loadRss + 5_000_000,
        uss_delta_bytes: loadUss + 5_000_000,
      },
    },
    residentBytes,
  };
}

function config() {
  return {
    _profile: PROFILE,
    project: { name: 'unit matched benchmark', seed: 20260716 },
    paths: { assets_manifest: 'data/assets.json', run_dir: 'runs' },
    sources: {
      official_models: MODELS.map(([id]) => ({
        id,
        revision: 'main',
        trained_context_length: 512,
      })),
    },
    comparison: { primary_official_model_id: MODELS[0][0] },
    data: {
      prepared_dir: 'data/prepared/unit',
      vocab_size: 2048,
      router_product_k: 8,
    },
    model: {
      block_size: 256,
      d_model: 192,
      n_layer: 6,
      n_head: 6,
      shared_hidden: 128,
      expert_hidden: 57,
      dropout: 0,
      vocab_size: 2048,
      num_routes: 64,
    },
    training: {
      batch_size: 32,
      gradient_accumulation: 2,
      joint_target_tokens: 320_000_000,
      expert_specialization_target_tokens: 80_000_000,
      learning_rate: 0.0015,
    },
    evaluation: {
      primary_quality_context: 'native',
      hellaswag_examples: 0,
    },
    benchmark: {
      custom_kernel_mode: 'fused',
      common_context_length: 256,
      prompt_character_targets: [512],
      decode_tokens: 10,
      warmup_repeats: 1,
      repeats: 6,
      prompt_samples_per_target: 4,
      primary_threads: 1,
      primary_prompt_character_target: 512,
      primary_device: 'cpu',
      primary_dtype: 'fp32',
      primary_machine_architecture: 'x86_64',
    },
    success_gates: {
      conditional_bpb_relative_to_official_max: 1.02,
      cpu_end_to_end_speedup_min: 1.2,
      cpu_decode_speedup_min: 1.2,
      cpu_generated_byte_throughput_min: 1.2,
      cpu_resident_memory_ratio_max: 0.8,
      hellaswag_normalized_delta_min: -0.01,
      expert_gain_loss_min: 0.005,
      routing_gain_loss_min: 0.005,
      blind_preference_custom_min: 0.45,
    },
  };
}

function model(kind, options = {}) {
  if (['custom_vsa_pathmoe', 'matched_dense_control'].includes(kind)) {
    const totalParameters = options.totalParameters ?? 10_000_000;
    const activeParameters = options.activeParameters ?? 7_000_000;
    const residentBytes = options.residentBytes ?? 40_000_000;
    const persistentCurrent = options.persistentCurrent ?? residentBytes + 1_000_000;
    const persistentMax = options.persistentMax ?? residentBytes + 5_000_000;
    return {
      name: kind === 'matched_dense_control' ? 'Dense matched control' : 'VSA-test',
      kind,
      checkpoint_sha256: options.checkpointSha ?? CUSTOM_SHA,
      checkpoint_provenance: options.checkpointProvenance,
      context_length: 512,
      device: options.device ?? 'cuda',
      dtype: options.dtype ?? 'bfloat16',
      resident_parameter_bytes: residentBytes,
      persistent_model_state_current_bytes: persistentCurrent,
      persistent_model_state_max_bytes: persistentMax,
      fused_route_cache: {
        capacity_routes: 4,
        current_bytes: 0,
        maximum_bytes: persistentMax - persistentCurrent,
      },
      parameters: {
        total_parameters: totalParameters,
        active_parameters_per_request: activeParameters,
        resident_parameter_bytes: residentBytes,
        fp32_active_megabytes: activeParameters * 4 / 1e6,
        fp32_expert_per_route_megabytes: 0.525312,
      },
      routing_policy: {
        router_resident_array_bytes: 1_024,
        router_artifact_bytes: 2_048,
      },
    };
  }
  return {
    kind: 'official_huggingface',
    model_id: options.id,
    revision: options.revision,
    context_length: 512,
    artifact_config_context_length: 2048,
    device: options.device ?? 'cuda',
    dtype: options.dtype ?? 'bfloat16',
    parameters: {
      total_parameters: options.parameters,
      resident_parameter_bytes: options.residentBytes,
    },
    resident_parameter_bytes: options.residentBytes,
    persistent_model_state_current_bytes: options.residentBytes,
    persistent_model_state_max_bytes: options.residentBytes,
  };
}

function environment(
  runtimeSourceSha256 = RUNTIME_SHA,
  machine = 'x86_64',
  { training = false } = {},
) {
  return {
    runtime_source_sha256: runtimeSourceSha256,
    git_commit: TRAIN_COMMIT,
    git_worktree_dirty: false,
    machine,
    processor: machine === 'x86_64' ? 'Test x86 CPU' : 'Test ARM CPU',
    platform: machine === 'x86_64' ? 'Linux-test-x86_64' : 'Linux-test-aarch64',
    logical_cpus: 8,
    cpu_affinity: [0],
    cpu_topology: {
      logical_cpus: [{
        logical_cpu: 0,
        physical_package_id: 0,
        core_id: 0,
        thread_siblings_list: [0, 8],
      }],
      complete: true,
      affinity_logical_cpu_count: 1,
      affinity_physical_core_count: 1,
      affinity_contains_smt_siblings: false,
    },
    cpu_frequency_governors: ['performance'],
    python: '3.12.12 (unit test)',
    torch: '2.10.0',
    numpy: '2.4.1',
    runtime_dependencies: structuredClone(RUNTIME_DEPENDENCIES),
    cuda_version: training ? '13.0' : null,
    cuda_device_count: training ? 1 : 0,
    cuda_devices: training ? structuredClone(CUDA_DEVICES) : [],
    cudnn_version: training ? 9_150_100 : null,
    nvidia_driver_version: training ? '580.95.05' : null,
    containerized: training,
    container_image: training ? 'nvcr.io/nvidia/pytorch:25.11-py3' : null,
    container_image_digest: training ? BASE_IMAGE_DIGEST : null,
    container_derived_image_id: training ? DERIVED_IMAGE_ID : null,
  };
}

function makeFixture({
  dirty = false,
  benchmarkRuntimeSha = RUNTIME_SHA,
  benchmarkMachine = 'x86_64',
  withControls = false,
  cpuGuard = false,
} = {}) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'vsa-report-'));
  const run = path.join(root, 'runs', PROFILE);
  const cfg = config();
  cfg.runtime = { device: 'cuda', dtype: 'bf16', threads: 4 };
  const requiredCpuCount = cpuGuard === true ? 1 : Number(cpuGuard || 0);
  if (requiredCpuCount > 0) {
    cfg.benchmark.required_cpu_affinity_logical_cpus = requiredCpuCount;
    cfg.benchmark.require_distinct_physical_cores = true;
  }
  const cpuBenchmarkEnvironment = (
    runtimeSourceSha256 = RUNTIME_SHA,
    machine = 'x86_64',
  ) => {
    const value = environment(runtimeSourceSha256, machine);
    if (requiredCpuCount <= 0) return value;
    const affinity = Array.from(
      { length: requiredCpuCount },
      (_, index) => index * 2,
    );
    value.cpu_affinity = affinity;
    value.cpu_topology = {
      logical_cpus: affinity.map((logicalCpu, index) => ({
        logical_cpu: logicalCpu,
        physical_package_id: 0,
        core_id: index,
        thread_siblings_list: [logicalCpu, logicalCpu + 1],
      })),
      complete: true,
      affinity_logical_cpu_count: affinity.length,
      affinity_physical_core_count: affinity.length,
      affinity_contains_smt_siblings: false,
    };
    return value;
  };
  if (withControls) {
    cfg.comparison.matched_control_profiles = {
      active_parameter_budget: ACTIVE_CONTROL_PROFILE,
      total_parameter_budget: TOTAL_CONTROL_PROFILE,
    };
  }
  const resolvedConfigSha256 = sha256(canonicalJson(cfg));
  const checkpointProvenance = {
    checkpoint_format_version: 2,
    training_profile: PROFILE,
    resolved_config_sha256: resolvedConfigSha256,
    routing_mode: 'vsa',
    preparation_signature: PREPARATION_SHA,
    tokenizer_sha256: '7'.repeat(64),
    router_sha256: '8'.repeat(64),
    training_git_commit: TRAIN_COMMIT,
    training_git_worktree_dirty: dirty,
    training_runtime_source_sha256: RUNTIME_SHA,
    training_machine: 'aarch64',
    training_python: '3.12.12 (unit test)',
    training_torch_version: '2.10.0',
    training_numpy_version: '2.4.1',
    training_runtime_dependencies: structuredClone(RUNTIME_DEPENDENCIES),
    training_cuda_version: '13.0',
    training_cuda_devices: structuredClone(CUDA_DEVICES),
    training_cudnn_version: 9_150_100,
    training_nvidia_driver_version: '580.95.05',
    training_containerized: true,
    training_container_image: 'nvcr.io/nvidia/pytorch:25.11-py3',
    training_container_image_digest: BASE_IMAGE_DIGEST,
    training_container_derived_image_id: DERIVED_IMAGE_ID,
  };
  const configPath = path.join(run, 'resolved_config.json');
  writeJson(configPath, cfg);
  writeJson(path.join(run, 'artifacts', 'training_summary.json'), {
    profile: PROFILE,
    parameter_accounting: {
      total_parameters: 10_000_000,
      active_parameters_per_request: 7_000_000,
    },
    provenance: {
      resolved_config_sha256: resolvedConfigSha256,
      preparation_signature: PREPARATION_SHA,
      training_runtime_source_sha256: RUNTIME_SHA,
      training_git_commit: TRAIN_COMMIT,
      training_git_worktree_dirty: dirty,
      training_machine: 'aarch64',
      training_python: '3.12.12 (unit test)',
      training_torch_version: '2.10.0',
      training_numpy_version: '2.4.1',
      training_runtime_dependencies: structuredClone(RUNTIME_DEPENDENCIES),
      training_cuda_version: '13.0',
      training_cuda_devices: structuredClone(CUDA_DEVICES),
      training_cudnn_version: 9_150_100,
      training_nvidia_driver_version: '580.95.05',
      training_containerized: true,
      training_container_image: 'nvcr.io/nvidia/pytorch:25.11-py3',
      training_container_image_digest: BASE_IMAGE_DIGEST,
      training_container_derived_image_id: DERIVED_IMAGE_ID,
    },
  });

  const customEvaluationPath = path.join(run, 'evaluation', 'custom.json');
  writeJson(customEvaluationPath, {
    schema_version: 2,
    profile: PROFILE,
    environment: environment(RUNTIME_SHA, 'aarch64', { training: true }),
    model: model('custom_vsa_pathmoe', { checkpointProvenance }),
    quality: quality(),
    ablations: {
      full: { loss: 1 },
      shared_only: { loss: 1.1 },
      permuted_route: { loss: 1.1 },
    },
  });

  const officialPaths = new Map();
  for (const [id, revision, parameters, residentBytes] of MODELS) {
    const file = path.join(run, 'evaluation', `official_${id.replaceAll('/', '__')}.json`);
    writeJson(file, {
      schema_version: 2,
      profile: PROFILE,
      environment: environment(RUNTIME_SHA, 'aarch64', { training: true }),
      model: model('official_huggingface', {
        id, revision, parameters, residentBytes,
      }),
      quality: quality(),
    });
    officialPaths.set(id, file);
  }

  const manifestCases = Array.from({ length: 4 }, (_, promptSampleIndex) => ({
    prompt_target_characters: 512,
    prompt_characters: 512,
    prompt_utf8_bytes: 512,
    prompt_sample_index: promptSampleIndex,
    prompt_source_index: promptSampleIndex,
    prompt_sha256: String(promptSampleIndex + 3).repeat(64),
  }));
  const manifestSha = sha256(canonicalJson(manifestCases));
  const workload = {
    prompt_samples_per_target: 4,
    case_count: 4,
    manifest_sha256: manifestSha,
    cases: manifestCases,
  };
  const customMemory = memory(50_000_000, 60_000_000, 40_000_000);
  const customBenchmark = {
    schema_version: 3,
    profile: PROFILE,
    environment: cpuBenchmarkEnvironment(benchmarkRuntimeSha, benchmarkMachine),
    threads: 1,
    model: model('custom_vsa_pathmoe', {
      device: 'cpu', dtype: 'float32', checkpointProvenance,
    }),
    load_seconds: 0.2,
    memory: customMemory,
    benchmark_config: cfg.benchmark,
    workload,
    cases: manifestCases.map((item) => measuredCase(item)),
  };
  writeJson(
    path.join(run, 'benchmark', 'custom_cpu_fp32_threads_1.json'),
    customBenchmark,
  );
  // A different runtime can coexist and must not enter the CPU/FP32 comparison.
  writeJson(path.join(run, 'benchmark', 'custom_cuda_bf16_threads_1.json'), {
    stale: true,
  });

  for (const [id, revision, parameters, residentBytes] of MODELS) {
    const officialMemory = memory(100_000_000, 120_000_000, residentBytes);
    writeJson(
      path.join(
        run,
        'benchmark',
        `official_${id.replaceAll('/', '__')}_cpu_fp32_threads_1.json`,
      ),
      {
        schema_version: 3,
        profile: PROFILE,
        environment: cpuBenchmarkEnvironment(),
        threads: 1,
        model: model('official_huggingface', {
          id,
          revision,
          parameters,
          residentBytes,
          device: 'cpu',
          dtype: 'float32',
        }),
        load_seconds: 0.4,
        memory: officialMemory,
        benchmark_config: cfg.benchmark,
        workload,
        cases: manifestCases.map((item) => measuredCase(item, true)),
      },
    );
  }

  const controls = {};
  if (withControls) {
    const specifications = [
      {
        role: 'active_parameter_budget',
        profile: ACTIVE_CONTROL_PROFILE,
        checkpointSha: 'd'.repeat(64),
        totalParameters: 7_000_000,
        residentBytes: 28_000_000,
        persistentMax: 30_000_000,
        loadUss: 40_000_000,
        loadRss: 48_000_000,
        nll: 12,
        hellaswag: 0.49,
        sharedHidden: 101,
      },
      {
        role: 'total_parameter_budget',
        profile: TOTAL_CONTROL_PROFILE,
        checkpointSha: '9'.repeat(64),
        totalParameters: 10_000_000,
        residentBytes: 40_000_000,
        persistentMax: 42_000_000,
        loadUss: 70_000_000,
        loadRss: 84_000_000,
        nll: 9,
        hellaswag: 0.52,
        sharedHidden: 371,
      },
    ];
    for (const specification of specifications) {
      const controlRun = path.join(root, 'runs', specification.profile);
      const controlConfig = structuredClone(cfg);
      controlConfig._profile = specification.profile;
      delete controlConfig.comparison.matched_control_profiles;
      controlConfig.model.routing_mode = 'fixed_dense';
      controlConfig.model.num_routes = 1;
      controlConfig.model.shared_hidden = specification.sharedHidden;
      const controlConfigSha = sha256(canonicalJson(controlConfig));
      const controlCheckpointProvenance = {
        ...structuredClone(checkpointProvenance),
        training_profile: specification.profile,
        resolved_config_sha256: controlConfigSha,
        routing_mode: 'fixed_dense',
      };
      const controlParameters = {
        total_parameters: specification.totalParameters,
        active_parameters_per_request: specification.totalParameters,
      };
      const controlConfigPath = path.join(controlRun, 'resolved_config.json');
      const controlSummaryPath = path.join(
        controlRun,
        'artifacts',
        'training_summary.json',
      );
      const controlEvaluationPath = path.join(
        controlRun,
        'evaluation',
        'custom.json',
      );
      const controlBenchmarkPath = path.join(
        controlRun,
        'benchmark',
        'custom_cpu_fp32_threads_1.json',
      );
      writeJson(controlConfigPath, controlConfig);
      writeJson(controlSummaryPath, {
        profile: specification.profile,
        parameter_accounting: controlParameters,
        provenance: {
          resolved_config_sha256: controlConfigSha,
          preparation_signature: PREPARATION_SHA,
          training_runtime_source_sha256: RUNTIME_SHA,
          training_git_commit: TRAIN_COMMIT,
          training_git_worktree_dirty: false,
          training_machine: 'aarch64',
          training_python: '3.12.12 (unit test)',
          training_torch_version: '2.10.0',
          training_numpy_version: '2.4.1',
          training_runtime_dependencies: structuredClone(RUNTIME_DEPENDENCIES),
          training_cuda_version: '13.0',
          training_cuda_devices: structuredClone(CUDA_DEVICES),
          training_cudnn_version: 9_150_100,
          training_nvidia_driver_version: '580.95.05',
          training_containerized: true,
          training_container_image: 'nvcr.io/nvidia/pytorch:25.11-py3',
          training_container_image_digest: BASE_IMAGE_DIGEST,
          training_container_derived_image_id: DERIVED_IMAGE_ID,
        },
      });
      const controlModelOptions = {
        checkpointSha: specification.checkpointSha,
        checkpointProvenance: controlCheckpointProvenance,
        totalParameters: specification.totalParameters,
        activeParameters: specification.totalParameters,
        residentBytes: specification.residentBytes,
        persistentCurrent: specification.residentBytes,
        persistentMax: specification.persistentMax,
      };
      writeJson(controlEvaluationPath, {
        schema_version: 2,
        profile: specification.profile,
        environment: environment(RUNTIME_SHA, 'aarch64', { training: true }),
        model: model('matched_dense_control', controlModelOptions),
        quality: quality(specification.nll, specification.hellaswag),
        ablations: { full: { loss: 1 } },
      });
      writeJson(controlBenchmarkPath, {
        schema_version: 3,
        profile: specification.profile,
        environment: cpuBenchmarkEnvironment(),
        threads: 1,
        model: model('matched_dense_control', {
          ...controlModelOptions,
          device: 'cpu',
          dtype: 'float32',
        }),
        load_seconds: 0.3,
        memory: memory(
          specification.loadUss,
          specification.loadRss,
          specification.residentBytes,
        ),
        benchmark_config: controlConfig.benchmark,
        workload,
        cases: manifestCases.map((item) => measuredCase(item, true)),
      });
      controls[specification.role] = {
        ...specification,
        run: controlRun,
        configPath: controlConfigPath,
        summaryPath: controlSummaryPath,
        evaluationPath: controlEvaluationPath,
        benchmarkPath: controlBenchmarkPath,
      };
    }
  }

  const primaryOfficialPath = officialPaths.get(MODELS[0][0]);
  writeJson(path.join(root, 'results', `blind_key_${PROFILE}.json`), {
    profile: PROFILE,
    officialModelId: MODELS[0][0],
    blindEvaluationId: BLIND_ID,
    promptManifestSha256: 'b'.repeat(64),
    provenance: {
      customEvaluationSha256: sha256File(customEvaluationPath),
      customCheckpointSha256: CUSTOM_SHA,
      officialEvaluationSha256: sha256File(primaryOfficialPath),
      officialRevision: REVISION_8M,
    },
    items: [{ id: 0, customSide: 'A' }],
  });
  writeJson(path.join(root, 'results', `blind_scores_${PROFILE}.json`), {
    profile: PROFILE,
    blindEvaluationId: BLIND_ID,
    promptManifestSha256: 'b'.repeat(64),
    items: [{ id: 0, choice: 'A' }],
  });
  return { root, run, controls };
}

test('quality and speed bootstraps use deterministic bounds', () => {
  const qualityBound = bootstrapRatio(perStory(20), perStory(10), 200);
  assert.equal(qualityBound.lower95, 2);
  assert.equal(qualityBound.median, 2);
  assert.equal(qualityBound.upper95, 2);

  const speedBound = bootstrapSpeedup(
    rawRows(10, 5),
    rawRows(20, 10),
    'unit-speed',
    200,
  );
  assert.equal(speedBound.endToEndSpeedup.lower95, 2);
  assert.equal(speedBound.decodeSpeedup.lower95, 2);
  assert.equal(speedBound.generatedByteThroughputRatio.lower95, 2);
});

test('primary CPU aggregation uses the minimum lower95 over all four prompts', () => {
  const rows = Array.from({ length: 4 }, (_, promptSampleIndex) => ({
    threads: 1,
    promptTargetCharacters: 512,
    promptSampleIndex,
    endToEndSpeedup: 3 - promptSampleIndex * 0.1,
    decodeSpeedup: 4,
    generatedByteThroughputRatio: 5,
    speedBootstrap95: {
      endToEndSpeedup: { lower95: 2 - promptSampleIndex * 0.1 },
      decodeSpeedup: { lower95: 3 },
      generatedByteThroughputRatio: { lower95: 4 },
    },
  }));
  const result = conservativePrimary(rows, 1, 512, 4);
  assert.equal(result.worstCasePoint.endToEndSpeedup, 2.7);
  assert.equal(result.worstCaseLower95.endToEndSpeedup, 1.7);
  assert.throws(() => conservativePrimary(rows.slice(1), 1, 512, 4), /needs 4 samples/);
});

test('resident memory gate takes the maximum measured and persistent-state ratio', () => {
  const artifact = (uss, rss, resident, persistent = resident) => ({
    model: {
      resident_parameter_bytes: resident,
      persistent_model_state_current_bytes: resident,
      persistent_model_state_max_bytes: persistent,
    },
    memory: {
      snapshots: {},
      deltas: { load: { uss_delta_bytes: uss, rss_delta_bytes: rss }, total: {} },
    },
  });
  const uss = residentMemoryRatio(artifact(5, 7, 11), artifact(10, 14, 22));
  assert.equal(uss.measuredLoad.source, 'uss_load_delta_bytes');
  assert.equal(uss.measuredLoad.ratio, 0.5);
  assert.equal(uss.persistentState.ratio, 0.5);
  assert.equal(uss.ratio, 0.5);
  assert.equal(
    residentMemoryRatio(artifact(0, 7, 11), artifact(0, 14, 22)).source,
    'rss_load_delta_bytes',
  );
  assert.equal(
    residentMemoryRatio(artifact(0, 0, 11), artifact(0, 0, 22)).source,
    'persistent_model_state_max_bytes',
  );
  const persistentControls = residentMemoryRatio(
    artifact(4, 4, 11, 15),
    artifact(10, 10, 22, 20),
  );
  assert.equal(persistentControls.measuredLoad.ratio, 0.4);
  assert.equal(persistentControls.persistentState.ratio, 0.75);
  assert.equal(persistentControls.ratio, 0.75);
  assert.equal(persistentControls.source, 'persistent_model_state_max_bytes');
});

test('multi-baseline report validates provenance and emits every CPU row', (t) => {
  const fixture = makeFixture();
  t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));
  const report = generateReport({ root: fixture.root, profile: PROFILE });
  assert.equal(report.comparisons.length, 2);
  assert.equal(report.primaryOfficialModelId, MODELS[0][0]);
  assert.equal(report.customModel.device, 'cpu');
  assert.equal(report.primary.models.evaluation.custom.device, 'cuda');
  assert.equal(report.primary.cpu.rows.length, 4);
  assert.equal(report.primary.cpu.primary.worstCaseLower95.endToEndSpeedup, 2);
  assert.equal(report.primary.memory.primaryResidentGateMetric.source, 'uss_load_delta_bytes');
  assert.equal(report.primary.memory.primaryResidentGateMetric.ratio, 0.5);
  assert.equal(report.primary.memory.primaryResidentGateMetric.persistentState.ratio, 0.45);
  assert.equal(report.coreVerdict, 'pass');
  assert.equal(report.finalVerdict, 'pass');
  assert.equal(report.blind.status, 'complete');
  assert.deepEqual(report.artifactValidation.benchmarkRuntime, {
    device: 'cpu', dtype: 'fp32', machineArchitecture: 'x86_64',
  });
  assert.equal(report.artifactValidation.training.trainingMachine, 'aarch64');
  assert.equal(
    report.artifactValidation.training.trainingContainerImageDigest,
    BASE_IMAGE_DIGEST,
  );

  const markdown = fs.readFileSync(
    path.join(fixture.root, 'results', `REPORT_${PROFILE}.md`),
    'utf8',
  );
  assert.match(markdown, /7 MB active != RSS/);
  assert.match(markdown, /do not causally isolate VSA/);
  assert.match(markdown, /excludes custom fused-route-cache population/);
  assert.match(markdown, /Test x86 CPU/);
  assert.match(markdown, /example\/TinyStories-33M/);
  const csv = fs.readFileSync(
    path.join(fixture.root, 'results', `cpu_metrics_${PROFILE}.csv`),
    'utf8',
  ).trim().split('\n');
  assert.equal(csv.length, 1 + 2 * 4);
});

test('report rejects dirty training and mismatched custom runtime source', (t) => {
  const dirty = makeFixture({ dirty: true });
  const staleRuntime = makeFixture({ benchmarkRuntimeSha: 'd'.repeat(64) });
  const armBenchmark = makeFixture({ benchmarkMachine: 'aarch64' });
  t.after(() => {
    fs.rmSync(dirty.root, { recursive: true, force: true });
    fs.rmSync(staleRuntime.root, { recursive: true, force: true });
    fs.rmSync(armBenchmark.root, { recursive: true, force: true });
  });
  assert.throws(
    () => generateReport({ root: dirty.root, profile: PROFILE }),
    /dirty Git worktree/,
  );
  assert.throws(
    () => generateReport({ root: staleRuntime.root, profile: PROFILE }),
    /runtime source differs from custom evaluation/,
  );
  assert.throws(
    () => generateReport({ root: armBenchmark.root, profile: PROFILE }),
    /machine architecture aarch64 != configured x86_64/,
  );
});

test('report rejects frozen-config and checkpoint-provenance drift', (t) => {
  const configDrift = makeFixture();
  const checkpointDrift = makeFixture();
  t.after(() => {
    fs.rmSync(configDrift.root, { recursive: true, force: true });
    fs.rmSync(checkpointDrift.root, { recursive: true, force: true });
  });

  const summaryPath = path.join(
    configDrift.run, 'artifacts', 'training_summary.json',
  );
  const summary = JSON.parse(fs.readFileSync(summaryPath));
  summary.provenance.resolved_config_sha256 = '0'.repeat(64);
  writeJson(summaryPath, summary);
  assert.throws(
    () => generateReport({ root: configDrift.root, profile: PROFILE }),
    /differs from frozen resolved_config\.json/,
  );

  const benchmarkPath = path.join(
    checkpointDrift.run, 'benchmark', 'custom_cpu_fp32_threads_1.json',
  );
  const benchmark = JSON.parse(fs.readFileSync(benchmarkPath));
  benchmark.model.checkpoint_provenance.router_sha256 = '9'.repeat(64);
  writeJson(benchmarkPath, benchmark);
  assert.throws(
    () => generateReport({ root: checkpointDrift.root, profile: PROFILE }),
    /checkpoint provenance differs from custom evaluation/,
  );
});

test('report rejects dirty execution artifacts and unpaired CPU environments', (t) => {
  const dirtyEvaluation = makeFixture();
  const environmentMismatch = makeFixture();
  t.after(() => {
    fs.rmSync(dirtyEvaluation.root, { recursive: true, force: true });
    fs.rmSync(environmentMismatch.root, { recursive: true, force: true });
  });

  const customEvaluationPath = path.join(
    dirtyEvaluation.run, 'evaluation', 'custom.json',
  );
  const customEvaluation = JSON.parse(fs.readFileSync(customEvaluationPath));
  customEvaluation.environment.git_worktree_dirty = true;
  writeJson(customEvaluationPath, customEvaluation);
  assert.throws(
    () => generateReport({ root: dirtyEvaluation.root, profile: PROFILE }),
    /custom evaluation was produced from a dirty/,
  );

  const officialBenchmarkPath = path.join(
    environmentMismatch.run,
    'benchmark',
    'official_example__TinyStories-8M_cpu_fp32_threads_1.json',
  );
  const officialBenchmark = JSON.parse(fs.readFileSync(officialBenchmarkPath));
  officialBenchmark.environment.processor = 'Different CPU';
  writeJson(officialBenchmarkPath, officialBenchmark);
  assert.throws(
    () => generateReport({ root: environmentMismatch.root, profile: PROFILE }),
    /custom\/official environment differs in processor/,
  );
});

test('report never mixes stale blind scores with a newer key', (t) => {
  const fixture = makeFixture();
  t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));
  const scorePath = path.join(fixture.root, 'results', `blind_scores_${PROFILE}.json`);
  const scores = JSON.parse(fs.readFileSync(scorePath));
  scores.blindEvaluationId = 'old-blind-evaluation';
  writeJson(scorePath, scores);
  const report = generateReport({ root: fixture.root, profile: PROFILE });
  assert.equal(report.blind.status, 'unavailable');
  assert.equal(report.blind.preference, null);
  assert.equal(report.finalVerdict, 'pending');
});

test('main report aggregates custom-only matched controls descriptively', (t) => {
  const fixture = makeFixture({ withControls: true });
  t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));
  const report = generateReport({ root: fixture.root, profile: PROFILE });
  assert.equal(report.schemaVersion, 4);
  assert.equal(report.matchedControls.configured, true);
  assert.equal(report.matchedControls.comparisons.length, 2);
  const active = report.matchedControls.comparisons.find(
    (item) => item.role === 'active_parameter_budget',
  );
  const total = report.matchedControls.comparisons.find(
    (item) => item.role === 'total_parameter_budget',
  );
  assert.equal(active.artifactContract, 'validated');
  assert.equal(active.parameterBudget.expectedControlTotalParameters, 7_000_000);
  assert.equal(active.parameterBudget.controlTotalParameters, 7_000_000);
  assert.equal(total.parameterBudget.expectedControlTotalParameters, 10_000_000);
  assert.equal(total.parameterBudget.controlTotalParameters, 10_000_000);
  assert.equal(active.quality.pairedBootstrapSparseToControlRatio95.upper95, 10 / 12);
  assert.equal(active.cpu.rows.length, 4);
  assert.equal(active.cpu.primary.worstCaseLower95.endToEndSpeedup, 2);
  assert.equal(active.memory.primaryRatios.measuredLoad.sparseToControlRatio, 1.25);
  assert.equal(active.memory.primaryRatios.persistentState.sparseToControlRatio, 1.5);
  assert.equal(active.memory.primaryRatios.descriptiveMaximumRatio, 1.5);
  assert.equal(Object.hasOwn(active, 'gates'), false);
  assert.equal(
    active.evidence.commands.evaluation,
    `npm run evaluate:custom -- --profile ${ACTIVE_CONTROL_PROFILE}`,
  );
  assert.equal(
    active.evidence.commands.cpuBenchmarkMatrix,
    `npm run benchmark:custom:matrix -- --profile ${ACTIVE_CONTROL_PROFILE}`,
  );
  assert.equal(
    fs.existsSync(path.join(fixture.root, 'results', `REPORT_${ACTIVE_CONTROL_PROFILE}.md`)),
    false,
  );

  const markdown = fs.readFileSync(
    path.join(fixture.root, 'results', `REPORT_${PROFILE}.md`),
    'utf8',
  );
  assert.match(markdown, /Matched dense controls/);
  assert.match(markdown, /active_parameter_budget: unit_dense_active/);
  assert.match(markdown, /descriptive matched-control comparison/i);
  assert.match(markdown, /sentencepiece=0\.2\.1/);
  const html = fs.readFileSync(
    path.join(fixture.root, 'results', `REPORT_${PROFILE}.html`),
    'utf8',
  );
  assert.match(html, /total_parameter_budget: unit_dense_total/);
  assert.match(html, /No official artifacts or separate control reports are consumed/);
  const csv = fs.readFileSync(
    path.join(fixture.root, 'results', `cpu_metrics_${PROFILE}.csv`),
    'utf8',
  ).trim().split('\n');
  assert.equal(csv.length, 1 + 4 * 4);
  assert.match(csv[0], /comparison_kind/);
  assert.equal(csv.filter((line) => line.startsWith('matched_dense_control,')).length, 8);
});

test('matched controls fail closed on profile, config, preparation, source, budget, and workload tampering', (t) => {
  const cases = [
    {
      name: 'profile',
      expected: /profile wrong_profile != unit_dense_active/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const artifact = JSON.parse(fs.readFileSync(control.evaluationPath));
        artifact.profile = 'wrong_profile';
        writeJson(control.evaluationPath, artifact);
      },
    },
    {
      name: 'config',
      expected: /training configuration differs from the sparse main run/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const value = JSON.parse(fs.readFileSync(control.configPath));
        value.training.joint_target_tokens += 1;
        writeJson(control.configPath, value);
      },
    },
    {
      name: 'preparation',
      expected: /training identity differs from the sparse main run in preparationSignature/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const different = '3'.repeat(64);
        const summary = JSON.parse(fs.readFileSync(control.summaryPath));
        summary.provenance.preparation_signature = different;
        writeJson(control.summaryPath, summary);
        for (const file of [control.evaluationPath, control.benchmarkPath]) {
          const artifact = JSON.parse(fs.readFileSync(file));
          artifact.model.checkpoint_provenance.preparation_signature = different;
          writeJson(file, artifact);
        }
      },
    },
    {
      name: 'source',
      expected: /training identity differs from the sparse main run in trainingRuntimeSourceSha256/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const different = '4'.repeat(64);
        const summary = JSON.parse(fs.readFileSync(control.summaryPath));
        summary.provenance.training_runtime_source_sha256 = different;
        writeJson(control.summaryPath, summary);
        for (const file of [control.evaluationPath, control.benchmarkPath]) {
          const artifact = JSON.parse(fs.readFileSync(file));
          artifact.environment.runtime_source_sha256 = different;
          artifact.model.checkpoint_provenance.training_runtime_source_sha256 = different;
          writeJson(file, artifact);
        }
      },
    },
    {
      name: 'budget',
      expected: /total parameter budget 7000001 != sparse active_parameter_budget budget 7000000/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const summary = JSON.parse(fs.readFileSync(control.summaryPath));
        summary.parameter_accounting.total_parameters += 1;
        summary.parameter_accounting.active_parameters_per_request += 1;
        writeJson(control.summaryPath, summary);
        for (const file of [control.evaluationPath, control.benchmarkPath]) {
          const artifact = JSON.parse(fs.readFileSync(file));
          artifact.model.parameters.total_parameters += 1;
          artifact.model.parameters.active_parameters_per_request += 1;
          writeJson(file, artifact);
        }
      },
    },
    {
      name: 'workload',
      expected: /workload differs from the sparse main run/,
      mutate(fixture) {
        const control = fixture.controls.active_parameter_budget;
        const artifact = JSON.parse(fs.readFileSync(control.benchmarkPath));
        const promptSha = 'b'.repeat(64);
        artifact.workload.cases[0].prompt_sha256 = promptSha;
        artifact.workload.manifest_sha256 = sha256(canonicalJson(artifact.workload.cases));
        artifact.cases[0].prompt_sha256 = promptSha;
        writeJson(control.benchmarkPath, artifact);
      },
    },
  ];
  for (const item of cases) {
    const fixture = makeFixture({ withControls: true });
    t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));
    item.mutate(fixture);
    assert.throws(
      () => generateReport({ root: fixture.root, profile: PROFILE }),
      item.expected,
      item.name,
    );
  }
});

test('structured runtime provenance is compared by value and rejects real drift', (t) => {
  const accepted = makeFixture();
  const dependencyDrift = makeFixture();
  const checkpointDeviceDrift = makeFixture();
  t.after(() => {
    fs.rmSync(accepted.root, { recursive: true, force: true });
    fs.rmSync(dependencyDrift.root, { recursive: true, force: true });
    fs.rmSync(checkpointDeviceDrift.root, { recursive: true, force: true });
  });
  assert.doesNotThrow(() => generateReport({ root: accepted.root, profile: PROFILE }));

  const evaluationPath = path.join(
    dependencyDrift.run,
    'evaluation',
    'custom.json',
  );
  const evaluation = JSON.parse(fs.readFileSync(evaluationPath));
  evaluation.environment.runtime_dependencies.transformers = 'different';
  writeJson(evaluationPath, evaluation);
  assert.throws(
    () => generateReport({ root: dependencyDrift.root, profile: PROFILE }),
    /training-environment field runtime_dependencies differs/,
  );

  const benchmarkPath = path.join(
    checkpointDeviceDrift.run,
    'benchmark',
    'custom_cpu_fp32_threads_1.json',
  );
  const benchmark = JSON.parse(fs.readFileSync(benchmarkPath));
  benchmark.model.checkpoint_provenance.training_cuda_devices[0]
    .compute_capability = [9, 0];
  writeJson(benchmarkPath, benchmark);
  assert.throws(
    () => generateReport({ root: checkpointDeviceDrift.root, profile: PROFILE }),
    /checkpoint training_cuda_devices differs from training summary/,
  );
});

test('report revalidates configured CPU affinity topology and pairs it exactly', (t) => {
  const accepted = makeFixture({ withControls: true, cpuGuard: 2 });
  const wrongCount = makeFixture({ cpuGuard: 2 });
  const incomplete = makeFixture({ cpuGuard: 2 });
  const smt = makeFixture({ cpuGuard: 2 });
  const controlMismatch = makeFixture({ withControls: true, cpuGuard: 2 });
  t.after(() => {
    for (const fixture of [accepted, wrongCount, incomplete, smt, controlMismatch]) {
      fs.rmSync(fixture.root, { recursive: true, force: true });
    }
  });

  const report = generateReport({ root: accepted.root, profile: PROFILE });
  assert.equal(
    report.primary.memory.rows[0].environment.cpu_topology
      .affinity_physical_core_count,
    2,
  );
  const markdown = fs.readFileSync(
    path.join(accepted.root, 'results', `REPORT_${PROFILE}.md`),
    'utf8',
  );
  assert.match(markdown, /CPU topology/);

  const customBenchmarkPath = (fixture) => path.join(
    fixture.run,
    'benchmark',
    'custom_cpu_fp32_threads_1.json',
  );
  const wrong = JSON.parse(fs.readFileSync(customBenchmarkPath(wrongCount)));
  wrong.environment.cpu_affinity.pop();
  writeJson(customBenchmarkPath(wrongCount), wrong);
  assert.throws(
    () => generateReport({ root: wrongCount.root, profile: PROFILE }),
    /requires exactly 2 affinity logical CPUs; found 1/,
  );

  const partial = JSON.parse(fs.readFileSync(customBenchmarkPath(incomplete)));
  partial.environment.cpu_topology.complete = false;
  writeJson(customBenchmarkPath(incomplete), partial);
  assert.throws(
    () => generateReport({ root: incomplete.root, profile: PROFILE }),
    /CPU topology is incomplete/,
  );

  const siblings = JSON.parse(fs.readFileSync(customBenchmarkPath(smt)));
  const topology = siblings.environment.cpu_topology;
  topology.logical_cpus[0].thread_siblings_list = [0, 2];
  topology.logical_cpus[1].physical_package_id = 0;
  topology.logical_cpus[1].core_id = 0;
  topology.logical_cpus[1].thread_siblings_list = [0, 2];
  topology.affinity_physical_core_count = 1;
  topology.affinity_contains_smt_siblings = true;
  writeJson(customBenchmarkPath(smt), siblings);
  assert.throws(
    () => generateReport({ root: smt.root, profile: PROFILE }),
    /requires distinct physical cores; found 1 cores for 2 logical CPUs/,
  );

  const control = controlMismatch.controls.active_parameter_budget;
  const mismatched = JSON.parse(fs.readFileSync(control.benchmarkPath));
  mismatched.environment.cpu_topology.logical_cpus[0].thread_siblings_list = [0, 99];
  writeJson(control.benchmarkPath, mismatched);
  assert.throws(
    () => generateReport({ root: controlMismatch.root, profile: PROFILE }),
    /sparse\/control environment differs in cpu_topology/,
  );
});
