import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { ROOT } from './lib.mjs';

const BOOTSTRAP_REPEATS = 1000;
const EXPECTED_DGX_CONTAINER_IMAGE = 'nvcr.io/nvidia/pytorch:25.11-py3';
const MATCHED_CONTROL_ROLES = [
  'active_parameter_budget',
  'total_parameter_budget',
];
const REQUIRED_RUNTIME_DEPENDENCIES = [
  'sentencepiece',
  'PyYAML',
  'psutil',
  'huggingface_hub',
  'transformers',
  'datasets',
  'safetensors',
];
const CORE_GATES = [
  'conditionalBpbNonInferiority',
  'cpuEndToEnd',
  'cpuDecode',
  'cpuGeneratedBytes',
  'cpuResidentMemory',
  'hellaswag',
  'expertContribution',
  'routingContribution',
];

function readJson(file, label = file) {
  if (!fs.existsSync(file)) throw new Error(`Missing ${label}: ${file}`);
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (error) {
    throw new Error(`Invalid JSON in ${label} (${file}): ${error.message}`);
  }
}

function maybeRead(file) {
  return fs.existsSync(file) ? readJson(file) : null;
}

function fail(message) {
  throw new Error(`Artifact validation failed: ${message}`);
}

function requireString(value, label) {
  if (typeof value !== 'string' || value.length === 0) fail(`${label} is missing`);
  return value;
}

function requireSha256(value, label) {
  const sha = requireString(value, label);
  if (!/^[0-9a-f]{64}$/i.test(sha)) fail(`${label} is not a SHA-256 digest`);
  return sha.toLowerCase();
}

function requireFinite(value, label) {
  if (!Number.isFinite(value)) fail(`${label} is not finite`);
  return Number(value);
}

function requirePositiveInteger(value, label) {
  const number = requireFinite(value, label);
  if (!Number.isInteger(number) || number <= 0) {
    fail(`${label} must be a positive integer`);
  }
  return number;
}

function validateRuntimeDependencies(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    fail(`${label} is missing`);
  }
  for (const dependency of REQUIRED_RUNTIME_DEPENDENCIES) {
    requireString(value[dependency], `${label}.${dependency}`);
  }
  return canonicalize(value);
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]),
    );
  }
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

function sha256Canonical(value) {
  return crypto.createHash('sha256').update(canonicalJson(value)).digest('hex');
}

function sha256File(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function validateFrozenConfigProfile(config, profile, label = 'frozen resolved config') {
  if (config?._profile !== profile) {
    fail(`${label} _profile ${config?._profile} != ${profile}`);
  }
}

function parameterBudgetIdentity(parameters, label) {
  if (!parameters || typeof parameters !== 'object') {
    fail(`${label} parameter accounting is missing`);
  }
  const totalParameters = requirePositiveInteger(
    parameters.total_parameters,
    `${label} total_parameters`,
  );
  const activeParametersPerRequest = requirePositiveInteger(
    parameters.active_parameters_per_request,
    `${label} active_parameters_per_request`,
  );
  if (activeParametersPerRequest > totalParameters) {
    fail(`${label} active_parameters_per_request exceeds total_parameters`);
  }
  return { totalParameters, activeParametersPerRequest };
}

function validateParameterBudgetIdentity(left, right, label) {
  const leftBudget = parameterBudgetIdentity(left, `${label} first artifact`);
  const rightBudget = parameterBudgetIdentity(right, `${label} second artifact`);
  if (canonicalJson(leftBudget) !== canonicalJson(rightBudget)) {
    fail(`${label} active/total parameter accounting differs`);
  }
  return leftBudget;
}

function safeModelId(modelId) {
  return modelId.replaceAll('/', '__');
}

function runtimeSlug(value, label) {
  const slug = requireString(value, label);
  if (!/^[a-z0-9_-]+$/.test(slug)) {
    fail(`${label} must be a lower-case [a-z0-9_-] slug`);
  }
  return slug;
}

function canonicalDtype(value) {
  return ({ fp32: 'float32', bf16: 'bfloat16', fp16: 'float16' })[value] ?? value;
}

function canonicalMachineArchitecture(value) {
  const normalized = String(value ?? '').toLowerCase();
  if (['x86_64', 'amd64'].includes(normalized)) return 'x86_64';
  if (['aarch64', 'arm64'].includes(normalized)) return 'aarch64';
  return normalized || null;
}

function validateRuntimeIdentity(artifact, device, dtype, label) {
  if (artifact.model?.device !== device) {
    fail(`${label} model.device ${artifact.model?.device} != configured ${device}`);
  }
  if (canonicalDtype(artifact.model?.dtype) !== canonicalDtype(dtype)) {
    fail(`${label} model.dtype ${artifact.model?.dtype} != configured ${dtype}`);
  }
}

function validateBenchmarkMachine(artifact, expectedArchitecture, label) {
  const actual = canonicalMachineArchitecture(artifact.environment?.machine);
  if (!actual) fail(`${label} environment.machine is missing`);
  if (actual !== expectedArchitecture) {
    fail(
      `${label} machine architecture ${artifact.environment.machine} != `
      + `configured ${expectedArchitecture}`,
    );
  }
}

function validateExecutionProvenance(artifact, training, label) {
  if (artifact.environment?.git_worktree_dirty !== false) {
    fail(`${label} was produced from a dirty or unknown Git worktree`);
  }
  if (artifact.environment?.git_commit !== training.trainingGitCommit) {
    fail(`${label} Git commit differs from the training commit`);
  }
}

function validateTrainingExecutionEnvironment(artifact, training, label) {
  const checks = {
    machine: training.trainingMachine,
    python: training.trainingPython,
    torch: training.trainingTorchVersion,
    numpy: training.trainingNumpyVersion,
    runtime_dependencies: training.trainingRuntimeDependencies,
    cuda_version: training.trainingCudaVersion,
    cuda_devices: training.trainingCudaDevices,
    cudnn_version: training.trainingCudnnVersion,
    nvidia_driver_version: training.trainingNvidiaDriverVersion,
    containerized: training.trainingContainerized,
    container_image: training.trainingContainerImage,
    container_image_digest: training.trainingContainerImageDigest,
    container_derived_image_id: training.trainingContainerDerivedImageId,
  };
  for (const [field, expected] of Object.entries(checks)) {
    const actual = field === 'machine'
      ? canonicalMachineArchitecture(artifact.environment?.[field])
      : artifact.environment?.[field];
    if (canonicalJson(actual) !== canonicalJson(expected)) {
      fail(`${label} training-environment field ${field} differs from training summary`);
    }
  }
}

const CPU_ENVIRONMENT_FIELDS = [
  'processor',
  'platform',
  'logical_cpus',
  'cpu_affinity',
  'cpu_topology',
  'cpu_frequency_governors',
  'python',
  'torch',
  'numpy',
  'runtime_dependencies',
  'containerized',
  'container_image',
  'container_image_digest',
  'container_derived_image_id',
];

function validateBenchmarkEnvironmentPair(
  custom,
  official,
  label,
  firstName = 'custom',
  secondName = 'official',
) {
  validateRuntimeDependencies(
    custom.environment?.runtime_dependencies,
    `${label} ${firstName} runtime_dependencies`,
  );
  validateRuntimeDependencies(
    official.environment?.runtime_dependencies,
    `${label} ${secondName} runtime_dependencies`,
  );
  for (const field of CPU_ENVIRONMENT_FIELDS) {
    if (!Object.hasOwn(custom.environment ?? {}, field)
        || !Object.hasOwn(official.environment ?? {}, field)) {
      fail(`${label} environment field ${field} is missing`);
    }
    if (canonicalJson(custom.environment?.[field])
        !== canonicalJson(official.environment?.[field])) {
      fail(`${label} ${firstName}/${secondName} environment differs in ${field}`);
    }
  }
}

function benchmarkEnvironmentSummary(environment) {
  return Object.fromEntries(
    CPU_ENVIRONMENT_FIELDS.map((field) => [field, environment?.[field] ?? null]),
  );
}

function validateBenchmarkCpuAffinity(artifact, config, label) {
  const benchmark = config.benchmark ?? {};
  const requiredRaw = benchmark.required_cpu_affinity_logical_cpus ?? 0;
  if (!Number.isInteger(Number(requiredRaw)) || Number(requiredRaw) < 0) {
    fail(`${label} required_cpu_affinity_logical_cpus is invalid`);
  }
  const required = Number(requiredRaw);
  const requireDistinct = benchmark.require_distinct_physical_cores ?? false;
  if (typeof requireDistinct !== 'boolean') {
    fail(`${label} require_distinct_physical_cores is not boolean`);
  }
  if (required === 0 && !requireDistinct) return;

  const affinity = artifact.environment?.cpu_affinity;
  const topology = artifact.environment?.cpu_topology;
  if (!Array.isArray(affinity)
      || affinity.some((cpu) => !Number.isInteger(cpu) || cpu < 0)
      || new Set(affinity).size !== affinity.length) {
    fail(`${label} CPU affinity is missing or malformed`);
  }
  if (required > 0 && affinity.length !== required) {
    fail(`${label} requires exactly ${required} affinity logical CPUs; found ${affinity.length}`);
  }
  if (!topology || typeof topology !== 'object' || topology.complete !== true) {
    fail(`${label} CPU topology is incomplete`);
  }
  const rows = topology.logical_cpus;
  if (!Array.isArray(rows) || rows.length !== affinity.length) {
    fail(`${label} CPU topology rows do not match affinity`);
  }
  const rowIds = rows.map((row) => row?.logical_cpu);
  if (rowIds.some((cpu) => !Number.isInteger(cpu) || cpu < 0)
      || new Set(rowIds).size !== rowIds.length
      || canonicalJson([...rowIds].sort((a, b) => a - b))
        !== canonicalJson([...affinity].sort((a, b) => a - b))) {
    fail(`${label} CPU topology logical CPUs do not match affinity`);
  }
  const selected = new Set(affinity);
  const pairs = new Set();
  let derivedSmt = false;
  for (const row of rows) {
    if (!Number.isInteger(row.physical_package_id) || !Number.isInteger(row.core_id)) {
      fail(`${label} CPU topology has an incomplete core identity`);
    }
    const siblings = row.thread_siblings_list;
    if (!Array.isArray(siblings)
        || siblings.some((cpu) => !Number.isInteger(cpu) || cpu < 0)
        || new Set(siblings).size !== siblings.length
        || !siblings.includes(row.logical_cpu)) {
      fail(`${label} CPU topology has malformed SMT siblings`);
    }
    const pair = `${row.physical_package_id}:${row.core_id}`;
    if (pairs.has(pair)) derivedSmt = true;
    pairs.add(pair);
    if (siblings.filter((cpu) => selected.has(cpu)).length > 1) derivedSmt = true;
  }
  if (topology.affinity_logical_cpu_count !== affinity.length) {
    fail(`${label} CPU topology logical count is inconsistent`);
  }
  if (topology.affinity_physical_core_count !== pairs.size) {
    fail(`${label} CPU topology physical-core count is inconsistent`);
  }
  if (topology.affinity_contains_smt_siblings !== derivedSmt) {
    fail(`${label} CPU topology SMT summary is inconsistent`);
  }
  if (requireDistinct && pairs.size !== affinity.length) {
    fail(
      `${label} requires distinct physical cores; found ${pairs.size} cores `
      + `for ${affinity.length} logical CPUs`,
    );
  }
  if (requireDistinct && derivedSmt) {
    fail(`${label} CPU affinity contains SMT siblings`);
  }
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.map(Number).sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

function quantile(sorted, probability) {
  if (!sorted.length) return null;
  return sorted[Math.floor((sorted.length - 1) * probability)];
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function deterministicSeed(label) {
  return crypto.createHash('sha256').update(label).digest().readUInt32LE(0);
}

function pairedStories(customRows, officialRows) {
  if (!Array.isArray(customRows) || !Array.isArray(officialRows)) {
    fail('per-story quality rows are missing');
  }
  const officialById = new Map();
  for (const row of officialRows) {
    const id = String(row.id);
    if (officialById.has(id)) fail(`duplicate official story id ${id}`);
    officialById.set(id, row);
  }
  const customIds = new Set();
  const pairs = customRows.map((customRow) => {
    const id = String(customRow.id);
    if (customIds.has(id)) fail(`duplicate custom story id ${id}`);
    customIds.add(id);
    const officialRow = officialById.get(id);
    if (!officialRow) fail(`official quality artifact is missing story id ${id}`);
    const customBytes = requireFinite(customRow.bytes, `custom story ${id} bytes`);
    const officialBytes = requireFinite(officialRow.bytes, `official story ${id} bytes`);
    if (customBytes !== officialBytes) fail(`UTF-8 byte count differs for story ${id}`);
    return [customRow, officialRow];
  });
  if (pairs.length !== officialRows.length) {
    fail('custom and official quality artifacts contain different story sets');
  }
  return pairs;
}

export function bootstrapRatio(customRows, officialRows, repeats = BOOTSTRAP_REPEATS) {
  const pairs = pairedStories(customRows, officialRows);
  if (pairs.length < 2) return null;
  const random = mulberry32(0x6d2b79f5);
  const values = [];
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    let customNll = 0;
    let customBytes = 0;
    let officialNll = 0;
    let officialBytes = 0;
    for (let index = 0; index < pairs.length; index += 1) {
      const [customRow, officialRow] = pairs[Math.floor(random() * pairs.length)];
      customNll += requireFinite(customRow.nll, 'custom story NLL');
      customBytes += requireFinite(customRow.bytes, 'custom story bytes');
      officialNll += requireFinite(officialRow.nll, 'official story NLL');
      officialBytes += requireFinite(officialRow.bytes, 'official story bytes');
    }
    if (customBytes <= 0 || officialBytes <= 0 || officialNll <= 0) {
      fail('quality bootstrap encountered non-positive bytes or official NLL');
    }
    values.push((customNll / customBytes) / (officialNll / officialBytes));
  }
  values.sort((a, b) => a - b);
  return {
    pairedStories: pairs.length,
    repeats,
    method: 'paired story bootstrap of aggregate NLL/UTF-8-byte ratio',
    lower95: quantile(values, 0.025),
    median: quantile(values, 0.5),
    upper95: quantile(values, 0.975),
  };
}

function resampleRows(rows, random) {
  return Array.from({ length: rows.length }, () => rows[Math.floor(random() * rows.length)]);
}

function rawMetrics(rows, label) {
  if (!Array.isArray(rows) || rows.length < 2) fail(`${label} raw_warm_rows are missing`);
  const endToEndMs = median(rows.map((row) => requireFinite(row.end_to_end_ms, `${label} end_to_end_ms`)));
  const decodeMs = median(rows.map((row) => requireFinite(row.decode_ms, `${label} decode_ms`)));
  const generatedTokens = median(rows.map((row) => requireFinite(row.generated_tokens, `${label} generated_tokens`)));
  const generatedBytes = median(rows.map((row) => requireFinite(row.generated_utf8_bytes, `${label} generated_utf8_bytes`)));
  return {
    endToEndMs,
    decodeTokensPerSecond: decodeMs > 0 ? generatedTokens / (decodeMs / 1000) : null,
    generatedBytesPerSecond: endToEndMs > 0 ? generatedBytes / (endToEndMs / 1000) : null,
  };
}

export function bootstrapSpeedup(
  customRows,
  officialRows,
  seedLabel,
  repeats = BOOTSTRAP_REPEATS,
) {
  if (!Array.isArray(customRows) || !Array.isArray(officialRows)
      || customRows.length < 2 || officialRows.length < 2) {
    return null;
  }
  const random = mulberry32(deterministicSeed(`speed:${seedLabel}`));
  const samples = {
    endToEndSpeedup: [],
    decodeSpeedup: [],
    generatedByteThroughputRatio: [],
  };
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const custom = rawMetrics(resampleRows(customRows, random), 'custom bootstrap');
    const official = rawMetrics(resampleRows(officialRows, random), 'official bootstrap');
    if (custom.endToEndMs > 0) {
      samples.endToEndSpeedup.push(official.endToEndMs / custom.endToEndMs);
    }
    if (official.decodeTokensPerSecond > 0) {
      samples.decodeSpeedup.push(
        custom.decodeTokensPerSecond / official.decodeTokensPerSecond,
      );
    }
    if (official.generatedBytesPerSecond > 0) {
      samples.generatedByteThroughputRatio.push(
        custom.generatedBytesPerSecond / official.generatedBytesPerSecond,
      );
    }
  }
  const summarize = (values) => {
    if (values.length !== repeats) return null;
    values.sort((a, b) => a - b);
    return {
      lower95: quantile(values, 0.025),
      median: quantile(values, 0.5),
      upper95: quantile(values, 0.975),
    };
  };
  return {
    repeats,
    method: 'independent bootstrap of per-process raw rows; ratio of resampled medians',
    endToEndSpeedup: summarize(samples.endToEndSpeedup),
    decodeSpeedup: summarize(samples.decodeSpeedup),
    generatedByteThroughputRatio: summarize(samples.generatedByteThroughputRatio),
  };
}

function validateProfile(artifact, profile, label) {
  if (artifact.profile !== profile) fail(`${label} profile ${artifact.profile} != ${profile}`);
}

function validateEvaluationSchema(artifact, label) {
  if (artifact.schema_version !== 2) fail(`${label} must use evaluation schema_version 2`);
}

function validateTrainingProvenance(runDir, profile, config) {
  const file = path.join(runDir, 'artifacts', 'training_summary.json');
  const summary = readJson(file, 'training summary');
  validateProfile(summary, profile, 'training summary');
  const parameterAccounting = parameterBudgetIdentity(
    summary.parameter_accounting,
    `${profile} training summary`,
  );
  const provenance = summary.provenance;
  if (!provenance || typeof provenance !== 'object') {
    fail('training summary has no provenance object; retrain with the current harness');
  }
  if (provenance.training_git_worktree_dirty !== false) {
    fail(
      provenance.training_git_worktree_dirty === true
        ? 'training was performed from a dirty Git worktree'
        : 'training Git worktree cleanliness is unknown',
    );
  }
  const resolvedConfigSha256 = sha256Canonical(config);
  if (provenance.resolved_config_sha256 !== resolvedConfigSha256) {
    fail('training resolved_config_sha256 differs from frozen resolved_config.json');
  }
  const trainingRuntimeSourceSha256 = requireSha256(
    provenance.training_runtime_source_sha256,
    'training runtime source SHA',
  );
  const preparationSignature = requireSha256(
    provenance.preparation_signature,
    'training preparation signature',
  );
  const trainingGitCommit = requireString(
    provenance.training_git_commit,
    'training Git commit',
  );
  if (!/^[0-9a-f]{40,64}$/i.test(trainingGitCommit)) {
    fail('training Git commit is malformed');
  }
  const trainingMachine = canonicalMachineArchitecture(
    requireString(provenance.training_machine, 'training machine'),
  );
  const trainingPython = requireString(
    provenance.training_python,
    'training Python version',
  );
  const trainingTorchVersion = requireString(
    provenance.training_torch_version,
    'training PyTorch version',
  );
  const trainingNumpyVersion = requireString(
    provenance.training_numpy_version,
    'training NumPy version',
  );
  const trainingRuntimeDependencies = validateRuntimeDependencies(
    provenance.training_runtime_dependencies,
    'training runtime dependencies',
  );
  const trainingCudaVersion = provenance.training_cuda_version ?? null;
  const trainingCudaDevices = provenance.training_cuda_devices ?? null;
  const trainingCudnnVersion = provenance.training_cudnn_version ?? null;
  const trainingNvidiaDriverVersion = provenance.training_nvidia_driver_version ?? null;
  const trainingContainerized = provenance.training_containerized;
  if (typeof trainingContainerized !== 'boolean') {
    fail('training containerized flag is missing');
  }
  const trainingContainerImage = provenance.training_container_image ?? null;
  const trainingContainerImageDigest = provenance.training_container_image_digest ?? null;
  const trainingContainerDerivedImageId =
    provenance.training_container_derived_image_id ?? null;
  const dgxTraining = config.runtime?.device === 'cuda'
    && canonicalDtype(config.runtime?.dtype) === 'bfloat16';
  if (dgxTraining) {
    if (trainingMachine !== 'aarch64') {
      fail('CUDA/BF16 training must have an ARM64 training machine');
    }
    if (!trainingCudaVersion) fail('CUDA/BF16 training has no CUDA version');
    if (!Array.isArray(trainingCudaDevices) || !trainingCudaDevices.length) {
      fail('CUDA/BF16 training has no CUDA device provenance');
    }
    if (canonicalJson(trainingCudaDevices[0]?.compute_capability) !== '[12,1]') {
      fail('CUDA/BF16 DGX Spark training device compute capability is not 12.1');
    }
    if (!Number.isFinite(Number(trainingCudnnVersion))) {
      fail('CUDA/BF16 training has no cuDNN version');
    }
    requireString(
      trainingNvidiaDriverVersion,
      'CUDA/BF16 training NVIDIA driver version',
    );
    if (trainingContainerized !== true) fail('CUDA/BF16 training was not containerized');
    if (trainingContainerImage !== EXPECTED_DGX_CONTAINER_IMAGE) {
      fail(`CUDA/BF16 training image is not ${EXPECTED_DGX_CONTAINER_IMAGE}`);
    }
    for (const [label, value] of [
      ['base image digest', trainingContainerImageDigest],
      ['derived image ID', trainingContainerDerivedImageId],
    ]) {
      if (!/^sha256:[0-9a-f]{64}$/i.test(String(value ?? ''))) {
        fail(`CUDA/BF16 training ${label} is missing or malformed`);
      }
    }
  }
  return {
    trainingSummaryPath: path.relative(runDir, file),
    trainingSummarySha256: sha256File(file),
    resolvedConfigSha256,
    trainingRuntimeSourceSha256,
    trainingGitCommit,
    trainingGitWorktreeDirty: false,
    preparationSignature,
    trainingMachine,
    trainingPython,
    trainingTorchVersion,
    trainingNumpyVersion,
    trainingRuntimeDependencies,
    trainingCudaVersion,
    trainingCudaDevices,
    trainingCudnnVersion,
    trainingNvidiaDriverVersion,
    trainingContainerized,
    trainingContainerImage,
    trainingContainerImageDigest,
    trainingContainerDerivedImageId,
    parameterAccounting,
  };
}

function runtimeSourceSha(artifact, label) {
  const value = requireString(
    artifact.environment?.runtime_source_sha256,
    `${label} environment.runtime_source_sha256`,
  );
  if (!/^[0-9a-f]{64}$/i.test(value)) fail(`${label} runtime source SHA is malformed`);
  return value.toLowerCase();
}

function validateCheckpointProvenance(
  model,
  training,
  profile,
  label,
  expected = null,
) {
  const provenance = model?.checkpoint_provenance;
  if (!provenance || typeof provenance !== 'object') {
    fail(`${label} has no model.checkpoint_provenance`);
  }
  if (provenance.training_profile !== profile) {
    fail(`${label} checkpoint training_profile differs from the run profile`);
  }
  if (provenance.resolved_config_sha256 !== training.resolvedConfigSha256) {
    fail(`${label} checkpoint resolved config SHA differs from training summary`);
  }
  if (!training.preparationSignature
      || provenance.preparation_signature !== training.preparationSignature) {
    fail(`${label} checkpoint preparation signature differs from training summary`);
  }
  if (provenance.training_git_worktree_dirty !== false) {
    fail(`${label} checkpoint was trained from a dirty or unknown Git worktree`);
  }
  if (!training.trainingGitCommit
      || provenance.training_git_commit !== training.trainingGitCommit) {
    fail(`${label} checkpoint training Git commit differs from training summary`);
  }
  if (provenance.training_runtime_source_sha256 !== training.trainingRuntimeSourceSha256) {
    fail(`${label} checkpoint training source hash differs from training summary`);
  }
  const trainingEnvironmentChecks = [
    ['training_machine', training.trainingMachine, canonicalMachineArchitecture],
    ['training_python', training.trainingPython, (value) => value],
    ['training_torch_version', training.trainingTorchVersion, (value) => value],
    ['training_numpy_version', training.trainingNumpyVersion, (value) => value],
    [
      'training_runtime_dependencies',
      training.trainingRuntimeDependencies,
      canonicalize,
    ],
    ['training_cuda_version', training.trainingCudaVersion, (value) => value ?? null],
    ['training_cuda_devices', training.trainingCudaDevices, canonicalize],
    ['training_cudnn_version', training.trainingCudnnVersion, (value) => value ?? null],
    [
      'training_nvidia_driver_version',
      training.trainingNvidiaDriverVersion,
      (value) => value ?? null,
    ],
    ['training_containerized', training.trainingContainerized, (value) => value],
    ['training_container_image', training.trainingContainerImage, (value) => value ?? null],
    [
      'training_container_image_digest',
      training.trainingContainerImageDigest,
      (value) => value ?? null,
    ],
    [
      'training_container_derived_image_id',
      training.trainingContainerDerivedImageId,
      (value) => value ?? null,
    ],
  ];
  for (const [field, expectedValue, normalize] of trainingEnvironmentChecks) {
    if (canonicalJson(normalize(provenance[field])) !== canonicalJson(expectedValue)) {
      fail(`${label} checkpoint ${field} differs from training summary`);
    }
  }
  if (Number(provenance.checkpoint_format_version) !== 2) {
    fail(`${label} checkpoint format is not the current version 2`);
  }
  for (const field of [
    'resolved_config_sha256',
    'tokenizer_sha256',
    'router_sha256',
    'training_runtime_source_sha256',
  ]) {
    if (!/^[0-9a-f]{64}$/i.test(String(provenance[field] ?? ''))) {
      fail(`${label} checkpoint provenance ${field} is missing or malformed`);
    }
  }
  if (!/^[0-9a-f]{40,64}$/i.test(String(provenance.training_git_commit ?? ''))) {
    fail(`${label} checkpoint training Git commit is missing or malformed`);
  }
  if (expected && canonicalJson(provenance) !== canonicalJson(expected)) {
    fail(`${label} checkpoint provenance differs from custom evaluation`);
  }
  return provenance;
}

function validateRevision(configuredRevision, actualRevision, label) {
  requireString(actualRevision, `${label} revision`);
  const immutable = /^[0-9a-f]{40,64}$/i;
  if (!immutable.test(actualRevision)) {
    fail(`${label} revision ${actualRevision} is not an immutable commit SHA`);
  }
  if (immutable.test(String(configuredRevision))
      && actualRevision.toLowerCase() !== String(configuredRevision).toLowerCase()) {
    fail(`${label} revision ${actualRevision} != configured ${configuredRevision}`);
  }
}

function validateBenchmarkConfig(artifact, config, label) {
  if (artifact.schema_version !== 3) fail(`${label} must use benchmark schema_version 3`);
  if (canonicalJson(artifact.benchmark_config) !== canonicalJson(config.benchmark)) {
    fail(`${label} benchmark_config differs from resolved_config.json`);
  }
}

function validateWorkload(artifact, config, label) {
  const workload = artifact.workload;
  if (!workload || !Array.isArray(workload.cases)) fail(`${label} workload is missing`);
  const manifest = requireSha256(workload.manifest_sha256, `${label} workload manifest`);
  const recomputed = sha256Canonical(workload.cases);
  if (manifest !== recomputed) fail(`${label} workload manifest SHA does not match its cases`);
  if (Number(workload.case_count) !== workload.cases.length) {
    fail(`${label} workload case_count is inconsistent`);
  }
  const expectedSamples = Number(config.benchmark.prompt_samples_per_target);
  const expectedTargets = config.benchmark.prompt_character_targets.map(Number);
  if (Number(workload.prompt_samples_per_target) !== expectedSamples) {
    fail(`${label} prompt_samples_per_target differs from resolved config`);
  }
  if (workload.cases.length !== expectedTargets.length * expectedSamples) {
    fail(`${label} workload case count differs from resolved config`);
  }
  const manifestBySha = new Map();
  for (const item of workload.cases) {
    const sha = requireSha256(item.prompt_sha256, `${label} prompt SHA`);
    if (manifestBySha.has(sha)) fail(`${label} contains duplicate prompt SHA ${sha}`);
    manifestBySha.set(sha, item);
  }
  if (!Array.isArray(artifact.cases) || artifact.cases.length !== workload.cases.length) {
    fail(`${label} measured cases do not match workload case count`);
  }
  for (const item of artifact.cases) {
    const manifestItem = manifestBySha.get(item.prompt_sha256);
    if (!manifestItem) fail(`${label} measured an unknown prompt ${item.prompt_sha256}`);
    for (const field of [
      'prompt_target_characters',
      'prompt_characters',
      'prompt_utf8_bytes',
      'prompt_sample_index',
      'prompt_source_index',
      'prompt_sha256',
    ]) {
      if (item[field] !== manifestItem[field]) {
        fail(`${label} case ${item.prompt_sha256} differs in ${field}`);
      }
    }
  }
  return manifest;
}

function modelTotalParameters(model) {
  return Number(model?.parameters?.total_parameters);
}

function residentParameterBytes(model) {
  const value = model?.resident_parameter_bytes ?? model?.parameters?.resident_parameter_bytes;
  return Number.isFinite(value) ? Number(value) : null;
}

function persistentModelStateMaxBytes(model) {
  const value = model?.persistent_model_state_max_bytes;
  return Number.isFinite(value) ? Number(value) : null;
}

function validatePersistentStateMetadata(model, label) {
  const resident = residentParameterBytes(model);
  const current = Number(model?.persistent_model_state_current_bytes);
  const maximum = persistentModelStateMaxBytes(model);
  if (!(Number.isFinite(resident) && resident > 0)) {
    fail(`${label} resident_parameter_bytes is missing or non-positive`);
  }
  if (!(Number.isFinite(current) && current >= resident)) {
    fail(`${label} persistent_model_state_current_bytes is missing or too small`);
  }
  if (!(Number.isFinite(maximum) && maximum >= current)) {
    fail(`${label} persistent_model_state_max_bytes is missing or too small`);
  }
}

function memoryValue(artifact, stage, field) {
  const value = artifact?.memory?.snapshots?.[stage]?.[field];
  return Number.isFinite(value) ? Number(value) : null;
}

function memoryDelta(artifact, stage, field) {
  const value = artifact?.memory?.deltas?.[stage]?.[field];
  return Number.isFinite(value) ? Number(value) : null;
}

function memorySummary(artifact) {
  return {
    threads: Number(artifact.threads),
    loadSeconds: Number(artifact.load_seconds),
    residentParameterBytes: residentParameterBytes(artifact.model),
    persistentModelStateCurrentBytes:
      Number(artifact.model?.persistent_model_state_current_bytes),
    persistentModelStateMaxBytes: persistentModelStateMaxBytes(artifact.model),
    fusedRouteCacheCurrentBytes:
      Number.isFinite(artifact.model?.fused_route_cache?.current_bytes)
        ? Number(artifact.model.fused_route_cache.current_bytes)
        : null,
    fusedRouteCacheMaximumBytes:
      Number.isFinite(artifact.model?.fused_route_cache?.maximum_bytes)
        ? Number(artifact.model.fused_route_cache.maximum_bytes)
        : null,
    fusedRouteCacheCapacityRoutes:
      Number.isFinite(artifact.model?.fused_route_cache?.capacity_routes)
        ? Number(artifact.model.fused_route_cache.capacity_routes)
        : null,
    routerResidentArrayBytes:
      Number.isFinite(artifact.model?.routing_policy?.router_resident_array_bytes)
        ? Number(artifact.model.routing_policy.router_resident_array_bytes)
        : null,
    rssAfterLoadBytes: memoryValue(artifact, 'after_load', 'rss_bytes'),
    ussAfterLoadBytes: memoryValue(artifact, 'after_load', 'uss_bytes'),
    pssAfterLoadBytes: memoryValue(artifact, 'after_load', 'pss_bytes'),
    rssLoadDeltaBytes: memoryDelta(artifact, 'load', 'rss_delta_bytes'),
    ussLoadDeltaBytes: memoryDelta(artifact, 'load', 'uss_delta_bytes'),
    pssLoadDeltaBytes: memoryDelta(artifact, 'load', 'pss_delta_bytes'),
    rssTotalDeltaBytes: memoryDelta(artifact, 'total', 'rss_delta_bytes'),
    ussTotalDeltaBytes: memoryDelta(artifact, 'total', 'uss_delta_bytes'),
    peakRssAfterBenchmarkBytes: memoryValue(
      artifact,
      'after_benchmark',
      'peak_rss_hwm_bytes',
    ),
  };
}

export function residentMemoryRatio(customArtifact, officialArtifact) {
  const custom = memorySummary(customArtifact);
  const official = memorySummary(officialArtifact);
  const measuredCandidates = [
    ['uss_load_delta_bytes', custom.ussLoadDeltaBytes, official.ussLoadDeltaBytes],
    ['rss_load_delta_bytes', custom.rssLoadDeltaBytes, official.rssLoadDeltaBytes],
  ];
  let measuredLoad = null;
  for (const [source, customBytes, officialBytes] of measuredCandidates) {
    if (Number.isFinite(customBytes) && customBytes > 0
        && Number.isFinite(officialBytes) && officialBytes > 0) {
      measuredLoad = { source, customBytes, officialBytes, ratio: customBytes / officialBytes };
      break;
    }
  }
  const persistentState = (
    Number.isFinite(custom.persistentModelStateMaxBytes)
      && custom.persistentModelStateMaxBytes > 0
      && Number.isFinite(official.persistentModelStateMaxBytes)
      && official.persistentModelStateMaxBytes > 0
  ) ? {
      source: 'persistent_model_state_max_bytes',
      customBytes: custom.persistentModelStateMaxBytes,
      officialBytes: official.persistentModelStateMaxBytes,
      ratio: custom.persistentModelStateMaxBytes / official.persistentModelStateMaxBytes,
    } : null;
  const components = [measuredLoad, persistentState].filter(Boolean);
  const controlling = components.reduce(
    (current, item) => (!current || item.ratio > current.ratio ? item : current),
    null,
  );
  return {
    source: controlling?.source ?? null,
    customBytes: controlling?.customBytes ?? null,
    officialBytes: controlling?.officialBytes ?? null,
    ratio: controlling?.ratio ?? null,
    aggregation: 'maximum of available measured-load and persistent-state ratios',
    measuredLoad,
    persistentState,
    measurementTiming:
      'load delta is captured before the first request and excludes fused-route-cache population',
  };
}

function validateCustomBenchmark(
  artifact,
  fileThread,
  profile,
  config,
  checkpointSha,
  expectedRuntimeSourceSha,
  primaryDevice,
  primaryDtype,
  primaryMachineArchitecture,
  trainingProvenance,
  expectedCheckpointProvenance,
) {
  validateProfile(artifact, profile, `custom benchmark threads=${fileThread}`);
  validateExecutionProvenance(
    artifact,
    trainingProvenance,
    `custom benchmark threads=${fileThread}`,
  );
  validateBenchmarkConfig(artifact, config, `custom benchmark threads=${fileThread}`);
  if (Number(artifact.threads) !== fileThread) fail(`custom benchmark filename/thread mismatch`);
  if (!['custom_vsa_pathmoe', 'matched_dense_control'].includes(artifact.model?.kind)) {
    fail(`custom benchmark threads=${fileThread} model.kind is invalid`);
  }
  validatePersistentStateMetadata(
    artifact.model,
    `custom benchmark threads=${fileThread}`,
  );
  validateRuntimeIdentity(
    artifact,
    primaryDevice,
    primaryDtype,
    `custom benchmark threads=${fileThread}`,
  );
  validateBenchmarkMachine(
    artifact,
    primaryMachineArchitecture,
    `custom benchmark threads=${fileThread}`,
  );
  validateBenchmarkCpuAffinity(
    artifact,
    config,
    `custom benchmark threads=${fileThread}`,
  );
  if (artifact.model?.checkpoint_sha256 !== checkpointSha) {
    fail(`custom benchmark threads=${fileThread} uses a stale/different checkpoint`);
  }
  validateCheckpointProvenance(
    artifact.model,
    trainingProvenance,
    profile,
    `custom benchmark threads=${fileThread}`,
    expectedCheckpointProvenance,
  );
  const runtimeSha = runtimeSourceSha(artifact, `custom benchmark threads=${fileThread}`);
  if (runtimeSha !== expectedRuntimeSourceSha) {
    fail(`custom benchmark threads=${fileThread} runtime source differs from custom evaluation`);
  }
  return validateWorkload(artifact, config, `custom benchmark threads=${fileThread}`);
}

function validateOfficialArtifact(artifact, source, expectedRevision, profile, label) {
  validateProfile(artifact, profile, label);
  if (artifact.model?.kind !== 'official_huggingface') {
    fail(`${label} model.kind is not official_huggingface`);
  }
  if (artifact.model?.model_id !== source.id) {
    fail(`${label} model_id ${artifact.model?.model_id} != ${source.id}`);
  }
  validateRevision(source.revision, artifact.model?.revision, label);
  if (expectedRevision && artifact.model.revision !== expectedRevision) {
    fail(`${label} revision ${artifact.model.revision} != evaluation ${expectedRevision}`);
  }
}

function validateOfficialBenchmark(
  artifact,
  fileThread,
  source,
  evaluation,
  profile,
  config,
  expectedManifest,
  primaryDevice,
  primaryDtype,
  primaryMachineArchitecture,
  expectedRuntimeSourceSha,
  trainingProvenance,
) {
  const label = `${source.id} benchmark threads=${fileThread}`;
  validateExecutionProvenance(artifact, trainingProvenance, label);
  validateOfficialArtifact(artifact, source, evaluation.model.revision, profile, label);
  validateBenchmarkConfig(artifact, config, label);
  if (Number(artifact.threads) !== fileThread) fail(`${label} filename/thread mismatch`);
  validateRuntimeIdentity(artifact, primaryDevice, primaryDtype, label);
  validateBenchmarkMachine(artifact, primaryMachineArchitecture, label);
  validateBenchmarkCpuAffinity(artifact, config, label);
  validatePersistentStateMetadata(artifact.model, label);
  if (runtimeSourceSha(artifact, label) !== expectedRuntimeSourceSha) {
    fail(`${label} runtime source differs from custom evaluation`);
  }
  const manifest = validateWorkload(artifact, config, label);
  if (manifest !== expectedManifest) {
    fail(`${label} workload.manifest_sha256 differs from custom benchmark`);
  }
}

function aggregateCase(caseItem, label) {
  const warm = caseItem?.warm_route_cache;
  if (!warm) fail(`${label} warm_route_cache is missing`);
  return {
    endToEndMs: requireFinite(warm.end_to_end_ms?.median_ms, `${label} E2E median`),
    decodeTokensPerSecond: requireFinite(
      warm.derived?.decode_tokens_per_second,
      `${label} decode tokens/s`,
    ),
    generatedBytesPerSecond: requireFinite(
      warm.derived?.end_to_end_generated_bytes_per_second,
      `${label} generated bytes/s`,
    ),
    promptTokens: requireFinite(warm.derived?.median_prompt_tokens, `${label} prompt tokens`),
  };
}

function pairBenchmarkCases(customArtifact, officialArtifact, officialModelId) {
  const officialBySha = new Map();
  for (const item of officialArtifact.cases) {
    if (officialBySha.has(item.prompt_sha256)) {
      fail(`${officialModelId} benchmark has duplicate prompt ${item.prompt_sha256}`);
    }
    officialBySha.set(item.prompt_sha256, item);
  }
  return customArtifact.cases.map((customCase) => {
    const officialCase = officialBySha.get(customCase.prompt_sha256);
    if (!officialCase) fail(`${officialModelId} is missing prompt ${customCase.prompt_sha256}`);
    for (const field of ['prompt_target_characters', 'prompt_sample_index']) {
      if (customCase[field] !== officialCase[field]) {
        fail(`${officialModelId} prompt ${customCase.prompt_sha256} differs in ${field}`);
      }
    }
    const custom = aggregateCase(customCase, `custom ${customCase.prompt_sha256}`);
    const official = aggregateCase(
      officialCase,
      `${officialModelId} ${customCase.prompt_sha256}`,
    );
    const confidence = bootstrapSpeedup(
      customCase.raw_warm_rows,
      officialCase.raw_warm_rows,
      `${officialModelId}:${customArtifact.threads}:${customCase.prompt_sha256}`,
    );
    return {
      officialModelId,
      officialRevision: officialArtifact.model.revision,
      threads: Number(customArtifact.threads),
      promptTargetCharacters: Number(customCase.prompt_target_characters),
      promptCharacters: Number(customCase.prompt_characters),
      promptSampleIndex: Number(customCase.prompt_sample_index),
      promptSourceIndex: Number(customCase.prompt_source_index),
      promptSha256: customCase.prompt_sha256,
      customEndToEndMs: custom.endToEndMs,
      officialEndToEndMs: official.endToEndMs,
      endToEndSpeedup: custom.endToEndMs > 0
        ? official.endToEndMs / custom.endToEndMs
        : null,
      customDecodeTokensPerSecond: custom.decodeTokensPerSecond,
      officialDecodeTokensPerSecond: official.decodeTokensPerSecond,
      decodeSpeedup: official.decodeTokensPerSecond > 0
        ? custom.decodeTokensPerSecond / official.decodeTokensPerSecond
        : null,
      customGeneratedBytesPerSecond: custom.generatedBytesPerSecond,
      officialGeneratedBytesPerSecond: official.generatedBytesPerSecond,
      generatedByteThroughputRatio: official.generatedBytesPerSecond > 0
        ? custom.generatedBytesPerSecond / official.generatedBytesPerSecond
        : null,
      customPromptTokens: custom.promptTokens,
      officialPromptTokens: official.promptTokens,
      customColdEndToEndMs:
        customCase.cold_route_cache?.end_to_end_ms?.median_ms ?? null,
      speedBootstrap95: confidence,
    };
  });
}

function minimumIfComplete(rows, selector) {
  const values = rows.map(selector);
  return values.every(Number.isFinite) ? Math.min(...values) : null;
}

export function conservativePrimary(
  rows,
  primaryThreads,
  primaryPromptTarget,
  expectedSamples,
) {
  const selected = rows
    .filter((row) => row.threads === Number(primaryThreads)
      && row.promptTargetCharacters === Number(primaryPromptTarget))
    .sort((a, b) => a.promptSampleIndex - b.promptSampleIndex);
  if (selected.length !== Number(expectedSamples)) {
    fail(
      `primary CPU point needs ${expectedSamples} samples at threads=${primaryThreads}, `
      + `target=${primaryPromptTarget}; found ${selected.length}`,
    );
  }
  const sampleIndexes = selected.map((row) => row.promptSampleIndex);
  if (new Set(sampleIndexes).size !== selected.length
      || sampleIndexes.some((value, index) => value !== index)) {
    fail('primary CPU prompt sample indexes are incomplete or duplicated');
  }
  return {
    threads: Number(primaryThreads),
    promptTargetCharacters: Number(primaryPromptTarget),
    sampleCount: selected.length,
    aggregation: 'minimum across prompt samples (worst case)',
    confidenceAggregation: 'minimum lower-95 bootstrap bound across prompt samples',
    samples: selected,
    worstCasePoint: {
      endToEndSpeedup: minimumIfComplete(selected, (row) => row.endToEndSpeedup),
      decodeSpeedup: minimumIfComplete(selected, (row) => row.decodeSpeedup),
      generatedByteThroughputRatio: minimumIfComplete(
        selected,
        (row) => row.generatedByteThroughputRatio,
      ),
    },
    worstCaseLower95: {
      endToEndSpeedup: minimumIfComplete(
        selected,
        (row) => row.speedBootstrap95?.endToEndSpeedup?.lower95,
      ),
      decodeSpeedup: minimumIfComplete(
        selected,
        (row) => row.speedBootstrap95?.decodeSpeedup?.lower95,
      ),
      generatedByteThroughputRatio: minimumIfComplete(
        selected,
        (row) => row.speedBootstrap95?.generatedByteThroughputRatio?.lower95,
      ),
    },
  };
}

function gate(value, threshold, operator, source) {
  if (!Number.isFinite(value)) {
    return { status: 'pending', value: null, threshold, operator, source };
  }
  const passed = operator === '<=' ? value <= threshold : value >= threshold;
  return { status: passed ? 'pass' : 'fail', value, threshold, operator, source };
}

function unavailableGate(threshold, operator, source, reason) {
  return { status: 'unavailable', value: null, threshold, operator, source, reason };
}

function verdictFor(gates, names) {
  const statuses = names.map((name) => gates[name].status);
  if (statuses.every((status) => status === 'pass')) return 'pass';
  if (statuses.some((status) => status === 'fail')) return 'fail';
  return 'pending';
}

function blindEvaluation(
  resultDir,
  profile,
  primaryModelId,
  custom,
  primaryOfficial,
  customEvaluationSha256,
  officialEvaluationSha256,
) {
  const scoresPath = path.join(resultDir, `blind_scores_${profile}.json`);
  const keyPath = path.join(resultDir, `blind_key_${profile}.json`);
  const scores = maybeRead(scoresPath);
  const key = maybeRead(keyPath);
  if (!key) return { status: 'pending', preference: null, reason: 'blind key not generated' };
  if (!key || !Array.isArray(key.items)) {
    return { status: 'unavailable', preference: null, reason: 'blind key does not use object.items schema' };
  }
  if (key.profile !== profile || key.officialModelId !== primaryModelId) {
    return { status: 'unavailable', preference: null, reason: 'blind key profile/model is stale' };
  }
  const keyCheckpoint = key.provenance?.customCheckpointSha256;
  const keyRevision = key.provenance?.officialRevision;
  if (!keyCheckpoint || !keyRevision
      || !key.provenance?.customEvaluationSha256
      || !key.provenance?.officialEvaluationSha256) {
    return { status: 'unavailable', preference: null, reason: 'blind key lacks full latest provenance' };
  }
  if (keyCheckpoint !== custom.model.checkpoint_sha256) {
    return { status: 'unavailable', preference: null, reason: 'blind key checkpoint SHA is stale' };
  }
  if (keyRevision !== primaryOfficial.model.revision) {
    return { status: 'unavailable', preference: null, reason: 'blind key official revision is stale' };
  }
  if (key.provenance.customEvaluationSha256 !== customEvaluationSha256
      || key.provenance.officialEvaluationSha256 !== officialEvaluationSha256) {
    return { status: 'unavailable', preference: null, reason: 'blind key evaluation artifact hashes are stale' };
  }
  if (!key.blindEvaluationId || !key.promptManifestSha256) {
    return { status: 'unavailable', preference: null, reason: 'blind key lacks evaluation/manifest identity' };
  }
  if (!/^[0-9a-f]{64}$/i.test(key.blindEvaluationId)
      || !/^[0-9a-f]{64}$/i.test(key.promptManifestSha256)) {
    return { status: 'unavailable', preference: null, reason: 'blind key identity digests are malformed' };
  }
  if (!scores) return {
    status: 'pending',
    preference: null,
    officialModelId: primaryModelId,
    blindEvaluationId: key.blindEvaluationId ?? null,
    promptManifestSha256: key.promptManifestSha256 ?? null,
    reason: 'blind scores not completed',
  };
  if (scores.profile !== profile
      || !key.blindEvaluationId
      || scores.blindEvaluationId !== key.blindEvaluationId
      || !key.promptManifestSha256
      || scores.promptManifestSha256 !== key.promptManifestSha256
      || !Array.isArray(scores.items)) {
    return {
      status: 'unavailable',
      preference: null,
      reason: 'blind scores do not match profile, blindEvaluationId, and prompt manifest',
    };
  }
  const keyById = new Map(key.items.map((item) => [Number(item.id), item.customSide]));
  let wins = 0;
  let ties = 0;
  let count = 0;
  for (const item of scores.items ?? []) {
    const side = keyById.get(Number(item.id));
    if (!side || !['A', 'B', 'tie'].includes(item.choice)) continue;
    count += 1;
    if (item.choice === 'tie') ties += 1;
    else if (item.choice === side) wins += 1;
  }
  if (!count) return { status: 'pending', preference: null, reason: 'no blind choices completed' };
  return {
    status: 'complete',
    preference: (wins + 0.5 * ties) / count,
    examples: count,
    officialModelId: primaryModelId,
    blindEvaluationId: key.blindEvaluationId,
    promptManifestSha256: key.promptManifestSha256,
  };
}

function qualityComparison(custom, official, config) {
  const expectedPrimaryMode = String(config.evaluation?.primary_quality_context);
  if (!['native', 'common'].includes(expectedPrimaryMode)) {
    fail('evaluation.primary_quality_context is missing or invalid');
  }
  for (const [label, artifact] of [['custom', custom], ['official', official]]) {
    if (artifact.quality?.primary_context_mode !== expectedPrimaryMode) {
      fail(`${label} primary quality context differs from resolved config`);
    }
    for (const field of ['heldout', 'heldout_native', 'heldout_common_context']) {
      if (!artifact.quality?.[field]
          || !Array.isArray(artifact.quality[field].per_story)) {
        fail(`${label} quality.${field} or its per_story rows are missing`);
      }
    }
  }
  const customHeldout = custom.quality.heldout;
  const officialHeldout = official.quality.heldout;
  const customCommon = custom.quality.heldout_common_context;
  const officialCommon = official.quality.heldout_common_context;
  if (Number(customCommon.context_limit) !== Number(officialCommon.context_limit)) {
    fail('custom and official common-context quality limits differ');
  }
  const customBpb = requireFinite(
    customHeldout.conditional_bits_per_byte,
    'custom conditional bits/byte',
  );
  const officialBpb = requireFinite(
    officialHeldout.conditional_bits_per_byte,
    'official conditional bits/byte',
  );
  const customCommonBpb = requireFinite(
    customCommon.conditional_bits_per_byte,
    'custom common-context bits/byte',
  );
  const officialCommonBpb = requireFinite(
    officialCommon.conditional_bits_per_byte,
    'official common-context bits/byte',
  );
  const nativeBootstrap = bootstrapRatio(
    customHeldout.per_story,
    officialHeldout.per_story,
  );
  const commonBootstrap = bootstrapRatio(
    customCommon.per_story,
    officialCommon.per_story,
  );
  const customHella = custom.quality.hellaswag?.accuracy_length_normalized;
  const officialHella = official.quality.hellaswag?.accuracy_length_normalized;
  const configuredHellaExamples = Number(config.evaluation.hellaswag_examples);
  return {
    primaryContextMode: custom.quality.primary_context_mode ?? 'native',
    customConditionalBitsPerByte: customBpb,
    officialConditionalBitsPerByte: officialBpb,
    conditionalBitsPerByteRatio: officialBpb > 0 ? customBpb / officialBpb : null,
    pairedBootstrapRatio95: nativeBootstrap,
    customNativeContextLimit:
      custom.quality.heldout_native?.context_limit ?? customHeldout.context_limit,
    officialNativeContextLimit:
      official.quality.heldout_native?.context_limit ?? officialHeldout.context_limit,
    customCommonContextBitsPerByte: customCommonBpb,
    officialCommonContextBitsPerByte: officialCommonBpb,
    commonContextBitsPerByteRatio:
      officialCommonBpb > 0 ? customCommonBpb / officialCommonBpb : null,
    commonContextPairedBootstrapRatio95: commonBootstrap,
    commonContextLimit: customCommon.context_limit,
    customHellaSwagNormalized: Number.isFinite(customHella) ? Number(customHella) : null,
    officialHellaSwagNormalized:
      Number.isFinite(officialHella) ? Number(officialHella) : null,
    customHellaSwagExamples:
      Number.isFinite(custom.quality.hellaswag?.examples)
        ? Number(custom.quality.hellaswag.examples)
        : null,
    officialHellaSwagExamples:
      Number.isFinite(official.quality.hellaswag?.examples)
        ? Number(official.quality.hellaswag.examples)
        : null,
    hellaswagConfiguredExamples: configuredHellaExamples,
    hellaswagEvaluationScope: configuredHellaExamples === 0
      ? 'full validation split'
      : (configuredHellaExamples > 0
        ? `deterministic subset capped at ${configuredHellaExamples} examples`
        : 'skipped by configuration'),
    hellaswagDelta:
      Number.isFinite(customHella) && Number.isFinite(officialHella)
        ? Number(customHella) - Number(officialHella)
        : null,
  };
}

function specialization(custom) {
  const fullLoss = custom.ablations?.full?.loss;
  const sharedLoss = custom.ablations?.shared_only?.loss;
  const wrongLoss = custom.ablations?.permuted_route?.loss;
  return {
    fullLoss: Number.isFinite(fullLoss) ? Number(fullLoss) : null,
    sharedOnlyLoss: Number.isFinite(sharedLoss) ? Number(sharedLoss) : null,
    permutedRouteLoss: Number.isFinite(wrongLoss) ? Number(wrongLoss) : null,
    expertGain:
      Number.isFinite(sharedLoss) && Number.isFinite(fullLoss)
        ? Number(sharedLoss) - Number(fullLoss)
        : null,
    routingGain:
      Number.isFinite(wrongLoss) && Number.isFinite(fullLoss)
        ? Number(wrongLoss) - Number(fullLoss)
        : null,
  };
}

function buildBaselineComparison({
  source,
  custom,
  official,
  customBenchmarks,
  officialBenchmarks,
  config,
  specializationResult,
  blind,
  isPrimary,
}) {
  const rows = [];
  const memoryRows = [];
  for (const [threads, customArtifact] of customBenchmarks) {
    const officialArtifact = officialBenchmarks.get(threads);
    rows.push(...pairBenchmarkCases(customArtifact, officialArtifact, source.id));
    memoryRows.push({
      threads,
      environment: benchmarkEnvironmentSummary(customArtifact.environment),
      custom: memorySummary(customArtifact),
      official: memorySummary(officialArtifact),
    });
  }
  rows.sort((a, b) => a.threads - b.threads
    || a.promptTargetCharacters - b.promptTargetCharacters
    || a.promptSampleIndex - b.promptSampleIndex);
  memoryRows.sort((a, b) => a.threads - b.threads);
  const primary = conservativePrimary(
    rows,
    config.benchmark.primary_threads,
    config.benchmark.primary_prompt_character_target,
    config.benchmark.prompt_samples_per_target,
  );
  const customPrimaryBenchmark = customBenchmarks.get(Number(config.benchmark.primary_threads));
  const officialPrimaryBenchmark = officialBenchmarks.get(Number(config.benchmark.primary_threads));
  const residentMemory = residentMemoryRatio(
    customPrimaryBenchmark,
    officialPrimaryBenchmark,
  );
  const quality = qualityComparison(custom, official, config);
  const thresholds = config.success_gates;
  const blindGate = isPrimary
    ? (Number.isFinite(blind.preference)
      ? gate(
        blind.preference,
        thresholds.blind_preference_custom_min,
        '>=',
        `blind A/B vs ${source.id}`,
      )
      : unavailableGate(
        thresholds.blind_preference_custom_min,
        '>=',
        `blind A/B vs ${source.id}`,
        blind.reason,
      ))
    : unavailableGate(
      thresholds.blind_preference_custom_min,
      '>=',
      `blind A/B vs ${source.id}`,
      'blind evaluation is defined only for the configured primary baseline',
    );
  const gates = {
    conditionalBpbNonInferiority: gate(
      quality.pairedBootstrapRatio95?.upper95,
      thresholds.conditional_bpb_relative_to_official_max,
      '<=',
      'upper95 paired bootstrap ratio',
    ),
    cpuEndToEnd: gate(
      primary.worstCaseLower95.endToEndSpeedup,
      thresholds.cpu_end_to_end_speedup_min,
      '>=',
      'minimum lower95 across primary prompt samples',
    ),
    cpuDecode: gate(
      primary.worstCaseLower95.decodeSpeedup,
      thresholds.cpu_decode_speedup_min,
      '>=',
      'minimum lower95 across primary prompt samples',
    ),
    cpuGeneratedBytes: gate(
      primary.worstCaseLower95.generatedByteThroughputRatio,
      thresholds.cpu_generated_byte_throughput_min,
      '>=',
      'minimum lower95 across primary prompt samples',
    ),
    cpuResidentMemory: gate(
      residentMemory.ratio,
      thresholds.cpu_resident_memory_ratio_max,
      '<=',
      residentMemory.source
        ? `maximum conservative ratio; controlling=${residentMemory.source}`
        : 'no valid resident-memory measurement',
    ),
    hellaswag: gate(
      quality.hellaswagDelta,
      thresholds.hellaswag_normalized_delta_min,
      '>=',
      'length-normalized accuracy delta',
    ),
    expertContribution: gate(
      specializationResult.expertGain,
      thresholds.expert_gain_loss_min,
      '>=',
      'shared-only loss minus full loss',
    ),
    routingContribution: gate(
      specializationResult.routingGain,
      thresholds.routing_gain_loss_min,
      '>=',
      'permuted-route loss minus full loss',
    ),
    blindPreference: blindGate,
  };
  return {
    officialModelId: source.id,
    configuredRevision: source.revision,
    resolvedRevision: official.model.revision,
    isPrimary,
    coreVerdict: verdictFor(gates, CORE_GATES),
    finalVerdict: verdictFor(gates, [...CORE_GATES, 'blindPreference']),
    models: {
      // The primary benchmark metadata is authoritative for CPU/FP32 resident bytes.
      custom: customPrimaryBenchmark.model,
      official: officialPrimaryBenchmark.model,
      evaluation: { custom: custom.model, official: official.model },
    },
    quality,
    specialization: specializationResult,
    cpu: { rows, primary },
    memory: {
      rows: memoryRows,
      primaryResidentGateMetric: residentMemory,
      note: 'The gate is the maximum of the pre-request USS/RSS load ratio and exact persistent-state maximum ratio. Load delta excludes fused-cache population; peak/total RSS remains diagnostic.',
    },
    blind: isPrimary ? blind : {
      status: 'not_applicable',
      preference: null,
      reason: 'blind evaluation is tied to the primary official baseline',
    },
    gates,
  };
}

function configuredMatchedControlProfiles(config, mainProfile) {
  const configured = config.comparison?.matched_control_profiles;
  if (configured === undefined || configured === null) return [];
  if (typeof configured !== 'object' || Array.isArray(configured)) {
    fail('comparison.matched_control_profiles must be an object');
  }
  const keys = Object.keys(configured).sort();
  const expectedKeys = [...MATCHED_CONTROL_ROLES].sort();
  if (canonicalJson(keys) !== canonicalJson(expectedKeys)) {
    fail(
      'comparison.matched_control_profiles must configure exactly '
      + MATCHED_CONTROL_ROLES.join(' and '),
    );
  }
  const profiles = MATCHED_CONTROL_ROLES.map((role) => {
    const profile = requireString(
      configured[role],
      `comparison.matched_control_profiles.${role}`,
    );
    if (!/^[A-Za-z0-9_.-]+$/.test(profile)) {
      fail(`matched-control profile ${profile} is not a safe profile name`);
    }
    if (profile === mainProfile) fail(`matched-control profile ${profile} is the main profile`);
    return { role, profile };
  });
  if (new Set(profiles.map((item) => item.profile)).size !== profiles.length) {
    fail('matched-control profile names must be unique');
  }
  return profiles;
}

function validateMatchedControlConfig(mainConfig, controlConfig, role, profile) {
  validateFrozenConfigProfile(controlConfig, profile, `${profile} frozen resolved config`);
  if (controlConfig.model?.routing_mode !== 'fixed_dense') {
    fail(`${profile} model.routing_mode must be fixed_dense`);
  }
  if (Number(controlConfig.model?.num_routes) !== 1) {
    fail(`${profile} model.num_routes must be exactly 1`);
  }
  if (Object.keys(controlConfig.comparison?.matched_control_profiles ?? {}).length) {
    fail(`${profile} must not recursively configure matched controls`);
  }
  for (const section of [
    'project',
    'paths',
    'sources',
    'runtime',
    'data',
    'training',
    'evaluation',
    'benchmark',
  ]) {
    if (canonicalJson(mainConfig[section]) !== canonicalJson(controlConfig[section])) {
      fail(`${profile} ${section} configuration differs from the sparse main run`);
    }
  }
  if (mainConfig.comparison?.primary_official_model_id
      !== controlConfig.comparison?.primary_official_model_id) {
    fail(`${profile} primary official model differs from the sparse main run`);
  }
  const comparableModel = (value) => {
    const copy = structuredClone(value ?? {});
    for (const field of [
      'routing_mode',
      'num_routes',
      'shared_hidden',
      'expert_hidden',
    ]) delete copy[field];
    return copy;
  };
  if (canonicalJson(comparableModel(mainConfig.model))
      !== canonicalJson(comparableModel(controlConfig.model))) {
    fail(`${profile} backbone model configuration differs from the sparse main run`);
  }
  return {
    role,
    profile,
    routingMode: controlConfig.model.routing_mode,
    allowedModelDifferences: [
      'routing_mode',
      'num_routes',
      'shared_hidden',
      'expert_hidden',
    ],
  };
}

function validateMatchedTrainingIdentity(mainTraining, controlTraining, profile) {
  const exactFields = [
    'trainingRuntimeSourceSha256',
    'trainingGitCommit',
    'trainingGitWorktreeDirty',
    'preparationSignature',
    'trainingMachine',
    'trainingPython',
    'trainingTorchVersion',
    'trainingNumpyVersion',
    'trainingRuntimeDependencies',
    'trainingCudaVersion',
    'trainingCudaDevices',
    'trainingCudnnVersion',
    'trainingNvidiaDriverVersion',
    'trainingContainerized',
    'trainingContainerImage',
    'trainingContainerImageDigest',
    'trainingContainerDerivedImageId',
  ];
  for (const field of exactFields) {
    if (canonicalJson(mainTraining[field]) !== canonicalJson(controlTraining[field])) {
      fail(`${profile} training identity differs from the sparse main run in ${field}`);
    }
  }
}

function validateMatchedPreparedIdentity(mainCheckpoint, controlCheckpoint, profile) {
  if (mainCheckpoint.routing_mode !== 'vsa') {
    fail('sparse main checkpoint routing_mode must be vsa when matched controls are configured');
  }
  if (controlCheckpoint.routing_mode !== 'fixed_dense') {
    fail(`${profile} checkpoint routing_mode must be fixed_dense`);
  }
  for (const field of [
    'preparation_signature',
    'tokenizer_sha256',
    'router_sha256',
  ]) {
    if (controlCheckpoint[field] !== mainCheckpoint[field]) {
      fail(`${profile} checkpoint ${field} differs from the sparse main run`);
    }
  }
}

function matchedQualityComparison(sparse, control, config, profile) {
  const quality = qualityComparison(sparse, control, config);
  if (Number.isFinite(quality.customHellaSwagExamples)
      && Number.isFinite(quality.officialHellaSwagExamples)
      && quality.customHellaSwagExamples !== quality.officialHellaSwagExamples) {
    fail(`${profile} HellaSwag example count differs from the sparse main run`);
  }
  return {
    primaryContextMode: quality.primaryContextMode,
    ratioSemantics: 'sparse/control; lower CBPB is better; no matched-control threshold',
    sparseConditionalBitsPerByte: quality.customConditionalBitsPerByte,
    controlConditionalBitsPerByte: quality.officialConditionalBitsPerByte,
    sparseToControlConditionalBitsPerByteRatio:
      quality.conditionalBitsPerByteRatio,
    pairedBootstrapSparseToControlRatio95: quality.pairedBootstrapRatio95,
    sparseNativeContextLimit: quality.customNativeContextLimit,
    controlNativeContextLimit: quality.officialNativeContextLimit,
    sparseCommonContextBitsPerByte: quality.customCommonContextBitsPerByte,
    controlCommonContextBitsPerByte: quality.officialCommonContextBitsPerByte,
    sparseToControlCommonContextBitsPerByteRatio:
      quality.commonContextBitsPerByteRatio,
    commonContextPairedBootstrapSparseToControlRatio95:
      quality.commonContextPairedBootstrapRatio95,
    commonContextLimit: quality.commonContextLimit,
    sparseHellaSwagNormalized: quality.customHellaSwagNormalized,
    controlHellaSwagNormalized: quality.officialHellaSwagNormalized,
    sparseHellaSwagExamples: quality.customHellaSwagExamples,
    controlHellaSwagExamples: quality.officialHellaSwagExamples,
    hellaswagConfiguredExamples: quality.hellaswagConfiguredExamples,
    hellaswagEvaluationScope: quality.hellaswagEvaluationScope,
    sparseMinusControlHellaSwagDelta: quality.hellaswagDelta,
  };
}

function matchedCpuRows(sparseArtifact, controlArtifact, profile) {
  return pairBenchmarkCases(sparseArtifact, controlArtifact, profile).map((row) => ({
    threads: row.threads,
    promptTargetCharacters: row.promptTargetCharacters,
    promptCharacters: row.promptCharacters,
    promptSampleIndex: row.promptSampleIndex,
    promptSourceIndex: row.promptSourceIndex,
    promptSha256: row.promptSha256,
    sparseEndToEndMs: row.customEndToEndMs,
    controlEndToEndMs: row.officialEndToEndMs,
    sparseEndToEndSpeedupRelativeToControl: row.endToEndSpeedup,
    sparseDecodeTokensPerSecond: row.customDecodeTokensPerSecond,
    controlDecodeTokensPerSecond: row.officialDecodeTokensPerSecond,
    sparseDecodeThroughputRatio: row.decodeSpeedup,
    sparseGeneratedBytesPerSecond: row.customGeneratedBytesPerSecond,
    controlGeneratedBytesPerSecond: row.officialGeneratedBytesPerSecond,
    sparseGeneratedByteThroughputRatio: row.generatedByteThroughputRatio,
    sparsePromptTokens: row.customPromptTokens,
    controlPromptTokens: row.officialPromptTokens,
    sparseColdEndToEndMs: row.customColdEndToEndMs,
    speedBootstrap95: row.speedBootstrap95,
    // These aliases feed the existing threshold-free conservative aggregation helper.
    endToEndSpeedup: row.endToEndSpeedup,
    decodeSpeedup: row.decodeSpeedup,
    generatedByteThroughputRatio: row.generatedByteThroughputRatio,
  }));
}

function relabelMatchedMemoryRatio(ratio) {
  const component = (value) => (value ? {
    source: value.source,
    sparseBytes: value.customBytes,
    controlBytes: value.officialBytes,
    sparseToControlRatio: value.ratio,
  } : null);
  return {
    aggregation: ratio.aggregation,
    ratioSemantics: 'sparse/control; lower memory ratio is smaller; no matched-control threshold',
    measuredLoad: component(ratio.measuredLoad),
    persistentState: component(ratio.persistentState),
    descriptiveMaximumRatio: ratio.ratio,
    descriptiveMaximumSource: ratio.source,
    measurementTiming: ratio.measurementTiming,
  };
}

function matchedParameterBudget(
  role,
  sparseBudget,
  controlBudget,
  profile,
) {
  if (controlBudget.activeParametersPerRequest !== controlBudget.totalParameters) {
    fail(`${profile} fixed-dense active and total parameter counts differ`);
  }
  const expected = role === 'active_parameter_budget'
    ? sparseBudget.activeParametersPerRequest
    : sparseBudget.totalParameters;
  if (controlBudget.totalParameters !== expected) {
    fail(
      `${profile} total parameter budget ${controlBudget.totalParameters} != `
      + `sparse ${role} budget ${expected}`,
    );
  }
  return {
    role,
    expectedFromSparseField: role === 'active_parameter_budget'
      ? 'active_parameters_per_request'
      : 'total_parameters',
    sparseActiveParametersPerRequest: sparseBudget.activeParametersPerRequest,
    sparseTotalParameters: sparseBudget.totalParameters,
    expectedControlTotalParameters: expected,
    controlTotalParameters: controlBudget.totalParameters,
    controlActiveParametersPerRequest: controlBudget.activeParametersPerRequest,
    exactMatchValidated: true,
  };
}

function buildMatchedControlComparisons({
  root,
  mainProfile,
  mainConfig,
  mainTraining,
  sparseEvaluation,
  sparseEvaluationRuntimeSourceSha256,
  sparseCheckpointProvenance,
  sparseBenchmarks,
  workloadManifest,
  primaryDevice,
  primaryDtype,
  primaryMachineArchitecture,
}) {
  const configured = configuredMatchedControlProfiles(mainConfig, mainProfile);
  if (!configured.length) {
    return {
      configured: false,
      configuredProfiles: {},
      comparisons: [],
      note: 'No matched dense controls are configured for this profile.',
    };
  }
  if (sparseEvaluation.model?.kind !== 'custom_vsa_pathmoe') {
    fail('matched-control aggregation requires a custom_vsa_pathmoe sparse main model');
  }
  const sparseBudget = parameterBudgetIdentity(
    sparseEvaluation.model?.parameters,
    'sparse main evaluation',
  );
  if (canonicalJson(sparseBudget) !== canonicalJson(mainTraining.parameterAccounting)) {
    fail('sparse main evaluation active/total parameter accounting differs from training summary');
  }
  for (const [threads, sparseBenchmark] of sparseBenchmarks) {
    const benchmarkBudget = parameterBudgetIdentity(
      sparseBenchmark.model?.parameters,
      `sparse main benchmark threads=${threads}`,
    );
    if (canonicalJson(benchmarkBudget) !== canonicalJson(sparseBudget)) {
      fail(`sparse main benchmark threads=${threads} active/total parameter accounting is stale`);
    }
  }

  const expectedThreads = [...sparseBenchmarks.keys()].sort((a, b) => a - b);
  const comparisons = configured.map(({ role, profile }) => {
    const runDir = path.join(root, 'runs', profile);
    const configPath = path.join(runDir, 'resolved_config.json');
    const controlConfig = readJson(configPath, `${profile} frozen resolved config`);
    const configValidation = validateMatchedControlConfig(
      mainConfig,
      controlConfig,
      role,
      profile,
    );
    const controlTraining = validateTrainingProvenance(runDir, profile, controlConfig);
    validateMatchedTrainingIdentity(mainTraining, controlTraining, profile);

    const evaluationPath = path.join(runDir, 'evaluation', 'custom.json');
    const controlEvaluation = readJson(evaluationPath, `${profile} custom evaluation`);
    validateProfile(controlEvaluation, profile, `${profile} custom evaluation`);
    validateEvaluationSchema(controlEvaluation, `${profile} custom evaluation`);
    validateExecutionProvenance(
      controlEvaluation,
      controlTraining,
      `${profile} custom evaluation`,
    );
    validateTrainingExecutionEnvironment(
      controlEvaluation,
      controlTraining,
      `${profile} custom evaluation`,
    );
    validateRuntimeIdentity(
      controlEvaluation,
      controlConfig.runtime.device,
      controlConfig.runtime.dtype,
      `${profile} custom evaluation`,
    );
    if (controlEvaluation.model?.kind !== 'matched_dense_control') {
      fail(`${profile} custom evaluation model.kind must be matched_dense_control`);
    }
    const controlRuntimeSourceSha256 = runtimeSourceSha(
      controlEvaluation,
      `${profile} custom evaluation`,
    );
    if (controlRuntimeSourceSha256 !== sparseEvaluationRuntimeSourceSha256) {
      fail(`${profile} evaluation runtime source differs from the sparse main run`);
    }
    const checkpointSha256 = requireSha256(
      controlEvaluation.model?.checkpoint_sha256,
      `${profile} custom evaluation checkpoint SHA`,
    );
    const checkpointProvenance = validateCheckpointProvenance(
      controlEvaluation.model,
      controlTraining,
      profile,
      `${profile} custom evaluation`,
    );
    validateMatchedPreparedIdentity(
      sparseCheckpointProvenance,
      checkpointProvenance,
      profile,
    );
    const controlBudget = parameterBudgetIdentity(
      controlEvaluation.model?.parameters,
      `${profile} custom evaluation`,
    );
    if (canonicalJson(controlBudget) !== canonicalJson(controlTraining.parameterAccounting)) {
      fail(`${profile} evaluation active/total parameter accounting differs from training summary`);
    }
    const parameterBudget = matchedParameterBudget(
      role,
      sparseBudget,
      controlBudget,
      profile,
    );

    const benchmarkDir = path.join(runDir, 'benchmark');
    if (!fs.existsSync(benchmarkDir)) {
      fail(`${profile} benchmark directory is missing: ${benchmarkDir}`);
    }
    const pattern = new RegExp(
      `^custom_${primaryDevice}_${primaryDtype}_threads_(\\d+)\\.json$`,
    );
    const files = fs.readdirSync(benchmarkDir)
      .filter((name) => pattern.test(name))
      .sort();
    const actualThreads = files.map((name) => Number(name.match(pattern)[1]))
      .sort((a, b) => a - b);
    if (new Set(actualThreads).size !== actualThreads.length) {
      fail(`${profile} has duplicate primary CPU benchmark thread counts`);
    }
    if (canonicalJson(actualThreads) !== canonicalJson(expectedThreads)) {
      fail(
        `${profile} primary CPU benchmark thread set ${actualThreads.join(',')} != `
        + `sparse main thread set ${expectedThreads.join(',')}`,
      );
    }
    const controlBenchmarks = new Map();
    const benchmarkEvidence = {};
    for (const threads of expectedThreads) {
      const name = `custom_${primaryDevice}_${primaryDtype}_threads_${threads}.json`;
      const file = path.join(benchmarkDir, name);
      const artifact = readJson(file, `${profile} ${name}`);
      const manifest = validateCustomBenchmark(
        artifact,
        threads,
        profile,
        controlConfig,
        checkpointSha256,
        controlRuntimeSourceSha256,
        primaryDevice,
        primaryDtype,
        primaryMachineArchitecture,
        controlTraining,
        checkpointProvenance,
      );
      if (artifact.model?.kind !== 'matched_dense_control') {
        fail(`${profile} benchmark threads=${threads} model.kind must be matched_dense_control`);
      }
      if (manifest !== workloadManifest) {
        fail(`${profile} benchmark threads=${threads} workload differs from the sparse main run`);
      }
      const sparseBenchmark = sparseBenchmarks.get(threads);
      if (canonicalJson(artifact.workload) !== canonicalJson(sparseBenchmark.workload)) {
        fail(`${profile} benchmark threads=${threads} workload is not exactly paired`);
      }
      validateBenchmarkEnvironmentPair(
        sparseBenchmark,
        artifact,
        `${profile} benchmark threads=${threads}`,
        'sparse',
        'control',
      );
      const benchmarkBudget = parameterBudgetIdentity(
        artifact.model?.parameters,
        `${profile} benchmark threads=${threads}`,
      );
      if (canonicalJson(benchmarkBudget) !== canonicalJson(controlBudget)) {
        fail(`${profile} benchmark threads=${threads} active/total parameter accounting is stale`);
      }
      benchmarkEvidence[threads] = {
        path: path.relative(root, file),
        sha256: sha256File(file),
      };
      controlBenchmarks.set(threads, artifact);
    }

    const rows = [];
    const memoryRows = [];
    for (const [threads, sparseBenchmark] of sparseBenchmarks) {
      const controlBenchmark = controlBenchmarks.get(threads);
      rows.push(...matchedCpuRows(sparseBenchmark, controlBenchmark, profile));
      memoryRows.push({
        threads,
        environment: benchmarkEnvironmentSummary(sparseBenchmark.environment),
        sparse: memorySummary(sparseBenchmark),
        control: memorySummary(controlBenchmark),
        ratios: relabelMatchedMemoryRatio(
          residentMemoryRatio(sparseBenchmark, controlBenchmark),
        ),
      });
    }
    rows.sort((a, b) => a.threads - b.threads
      || a.promptTargetCharacters - b.promptTargetCharacters
      || a.promptSampleIndex - b.promptSampleIndex);
    memoryRows.sort((a, b) => a.threads - b.threads);
    const primaryCpu = conservativePrimary(
      rows,
      mainConfig.benchmark.primary_threads,
      mainConfig.benchmark.primary_prompt_character_target,
      mainConfig.benchmark.prompt_samples_per_target,
    );
    const primaryMemory = memoryRows.find(
      (item) => item.threads === Number(mainConfig.benchmark.primary_threads),
    );
    if (!primaryMemory) fail(`${profile} primary memory comparison is missing`);
    const quality = matchedQualityComparison(
      sparseEvaluation,
      controlEvaluation,
      mainConfig,
      profile,
    );
    return {
      role,
      profile,
      comparisonType: 'matched_dense_control',
      interpretation: 'descriptive only; no new success threshold or verdict',
      artifactContract: 'validated',
      configValidation,
      parameterBudget,
      preparedIdentity: {
        preparationSignature: checkpointProvenance.preparation_signature,
        tokenizerSha256: checkpointProvenance.tokenizer_sha256,
        routerSha256: checkpointProvenance.router_sha256,
      },
      evidence: {
        resolvedConfig: {
          path: path.relative(root, configPath),
          sha256: controlTraining.resolvedConfigSha256,
        },
        trainingSummary: {
          path: path.relative(root, path.join(runDir, 'artifacts', 'training_summary.json')),
          sha256: controlTraining.trainingSummarySha256,
        },
        customEvaluation: {
          path: path.relative(root, evaluationPath),
          sha256: sha256File(evaluationPath),
        },
        customBenchmarks: benchmarkEvidence,
        commands: {
          evaluation: `npm run evaluate:custom -- --profile ${profile}`,
          cpuBenchmarkMatrix: `npm run benchmark:custom:matrix -- --profile ${profile}`,
        },
      },
      provenance: {
        checkpointSha256,
        checkpoint: checkpointProvenance,
        training: controlTraining,
        runtimeSourceSha256: controlRuntimeSourceSha256,
        workloadManifestSha256: workloadManifest,
        benchmarkThreads: expectedThreads,
      },
      models: {
        sparse: sparseBenchmarks.get(Number(mainConfig.benchmark.primary_threads)).model,
        control: controlBenchmarks.get(Number(mainConfig.benchmark.primary_threads)).model,
        evaluation: {
          sparse: sparseEvaluation.model,
          control: controlEvaluation.model,
        },
      },
      quality,
      cpu: {
        ratioSemantics: 'control time / sparse time, or sparse/control throughput; above 1 means sparse is faster; descriptive only',
        rows,
        primary: primaryCpu,
      },
      memory: {
        ratioSemantics: 'sparse/control; below 1 means sparse uses less memory; descriptive only',
        rows: memoryRows,
        primaryRatios: primaryMemory.ratios,
      },
    };
  });
  return {
    configured: true,
    configuredProfiles: Object.fromEntries(
      configured.map((item) => [item.role, item.profile]),
    ),
    artifactPolicy: 'fail closed; custom evaluation and primary CPU custom benchmarks only',
    performancePolicy: 'descriptive only; existing official-baseline gates are unchanged',
    comparisons,
  };
}

function fmt(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : 'n/a';
}

function pct(value) {
  return Number.isFinite(value) ? `${(100 * value).toFixed(2)}%` : 'n/a';
}

function bytes(value) {
  return Number.isFinite(value) ? `${(Number(value) / 1e6).toFixed(2)} MB` : 'n/a';
}

function markdownCell(value) {
  if (value === null || value === undefined) return 'n/a';
  const rendered = typeof value === 'string' ? value : JSON.stringify(value);
  return rendered.replaceAll('|', '\\|').replaceAll('\n', ' ');
}

function icon(status) {
  if (status === 'pass') return '✅';
  if (status === 'fail') return '❌';
  return '⏳';
}

function gateMarkdown(gates) {
  return Object.entries(gates).map(([name, item]) => (
    `| ${name} | ${icon(item.status)} ${item.status} | ${fmt(item.value, 4)} | `
    + `${item.operator} ${fmt(item.threshold, 4)} | ${item.source}${item.reason ? ` — ${item.reason}` : ''} |`
  )).join('\n');
}

function dependencySummary(value) {
  if (!value || typeof value !== 'object') return 'n/a';
  return Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, version]) => `${name}=${version ?? 'missing'}`)
    .join('; ');
}

function modelAccountingRows(report) {
  const custom = report.customModel;
  const rows = [{ id: custom.name, model: custom, revision: custom.checkpoint_sha256 }];
  for (const comparison of report.comparisons) {
    rows.push({
      id: comparison.officialModelId,
      model: comparison.models.official,
      revision: comparison.resolvedRevision,
    });
  }
  for (const comparison of report.matchedControls?.comparisons ?? []) {
    rows.push({
      id: `${comparison.profile} (${comparison.role})`,
      model: comparison.models.control,
      revision: comparison.provenance.checkpointSha256,
    });
  }
  return rows.map(({ id, model, revision }) => {
    const parameters = model.parameters ?? {};
    return `| ${id} | ${revision ?? 'n/a'} | ${fmt(modelTotalParameters(model), 0)} | `
      + `${bytes(residentParameterBytes(model))} | ${bytes(persistentModelStateMaxBytes(model))} | `
      + `${bytes(model.fused_route_cache?.maximum_bytes)} | ${bytes(parameters.fp32_active_megabytes * 1e6)} | `
      + `${bytes(parameters.fp32_expert_per_route_megabytes * 1e6)} | `
      + `${bytes(model.routing_policy?.router_resident_array_bytes)} / ${bytes(model.routing_policy?.router_artifact_bytes)} | `
      + `${model.context_length ?? 'n/a'} / ${model.artifact_config_context_length ?? 'n/a'} |`;
  }).join('\n');
}

function matchedControlsMarkdown(matchedControls) {
  if (!matchedControls?.configured) {
    return `## Matched dense controls\n\n${matchedControls?.note ?? 'Not configured.'}\n`;
  }
  const sections = matchedControls.comparisons.map((comparison) => {
    const quality = comparison.quality;
    const budget = comparison.parameterBudget;
    const cpuRows = comparison.cpu.rows.map((row) => (
      `| ${row.threads} | ${row.promptTargetCharacters} | ${row.promptSampleIndex} | `
      + `${fmt(row.sparseEndToEndMs)} | ${fmt(row.controlEndToEndMs)} | `
      + `${fmt(row.sparseEndToEndSpeedupRelativeToControl)} / ${fmt(row.speedBootstrap95?.endToEndSpeedup?.lower95)} | `
      + `${fmt(row.sparseDecodeThroughputRatio)} / ${fmt(row.speedBootstrap95?.decodeSpeedup?.lower95)} | `
      + `${fmt(row.sparseGeneratedByteThroughputRatio)} / ${fmt(row.speedBootstrap95?.generatedByteThroughputRatio?.lower95)} | `
      + `${fmt(row.sparsePromptTokens, 0)} / ${fmt(row.controlPromptTokens, 0)} | ${row.promptSha256.slice(0, 12)} |`
    )).join('\n');
    const memoryRows = comparison.memory.rows.map((row) => (
      `| ${row.threads} | ${bytes(row.sparse.persistentModelStateMaxBytes)} | `
      + `${bytes(row.control.persistentModelStateMaxBytes)} | `
      + `${fmt(row.ratios.persistentState?.sparseToControlRatio, 4)}× | `
      + `${bytes(row.ratios.measuredLoad?.sparseBytes)} | `
      + `${bytes(row.ratios.measuredLoad?.controlBytes)} | `
      + `${fmt(row.ratios.measuredLoad?.sparseToControlRatio, 4)}× `
      + `(${row.ratios.measuredLoad?.source ?? 'unavailable'}) | `
      + `${fmt(row.ratios.descriptiveMaximumRatio, 4)}× `
      + `(${row.ratios.descriptiveMaximumSource ?? 'unavailable'}) |`
    )).join('\n');
    const evidenceRows = [
      ['Frozen config', comparison.evidence.resolvedConfig],
      ['Training summary', comparison.evidence.trainingSummary],
      ['Custom evaluation', comparison.evidence.customEvaluation],
      ...Object.entries(comparison.evidence.customBenchmarks).map(([threads, item]) => (
        [`Custom CPU benchmark, ${threads} threads`, item]
      )),
    ].map(([label, item]) => (
      `| ${label} | ${item.path} | ${item.sha256} |`
    )).join('\n');
    const environment = comparison.memory.rows[0]?.environment ?? {};
    return `### ${comparison.role}: ${comparison.profile}

This is a **descriptive matched-control comparison**. It adds no success threshold and does not change the official-baseline verdict.

Exact parameter contract: control total **${fmt(budget.controlTotalParameters, 0)}** = sparse **${budget.expectedFromSparseField} ${fmt(budget.expectedControlTotalParameters, 0)}**. The fixed-dense control has ${fmt(budget.controlActiveParametersPerRequest, 0)} active parameters per request.

Prepared identity: \`${comparison.preparedIdentity.preparationSignature}\`; tokenizer \`${comparison.preparedIdentity.tokenizerSha256}\`; router \`${comparison.preparedIdentity.routerSha256}\`.

Commands:

- \`${comparison.evidence.commands.evaluation}\`
- \`${comparison.evidence.commands.cpuBenchmarkMatrix}\`

| Evidence | Path | SHA-256 |
|---|---|---|
${evidenceRows}

#### Quality — sparse/control

| Metric | Sparse | Control | Sparse/control ratio or delta |
|---|---:|---:|---:|
| Primary-context CBPB | ${fmt(quality.sparseConditionalBitsPerByte, 4)} | ${fmt(quality.controlConditionalBitsPerByte, 4)} | ${fmt(quality.sparseToControlConditionalBitsPerByteRatio, 4)}× |
| Paired story bootstrap CBPB | — | — | lower ${fmt(quality.pairedBootstrapSparseToControlRatio95?.lower95, 4)}×; median ${fmt(quality.pairedBootstrapSparseToControlRatio95?.median, 4)}×; upper ${fmt(quality.pairedBootstrapSparseToControlRatio95?.upper95, 4)}× |
| Common-context CBPB | ${fmt(quality.sparseCommonContextBitsPerByte, 4)} | ${fmt(quality.controlCommonContextBitsPerByte, 4)} | ${fmt(quality.sparseToControlCommonContextBitsPerByteRatio, 4)}× |
| Normalized HellaSwag | ${pct(quality.sparseHellaSwagNormalized)} | ${pct(quality.controlHellaSwagNormalized)} | ${Number.isFinite(quality.sparseMinusControlHellaSwagDelta) ? `${(100 * quality.sparseMinusControlHellaSwagDelta).toFixed(2)} pp` : 'n/a'} |

CBPB is paired by exact story ID and UTF-8 byte count. HellaSwag scope: **${quality.hellaswagEvaluationScope}** (sparse n=${quality.sparseHellaSwagExamples ?? 'n/a'}, control n=${quality.controlHellaSwagExamples ?? 'n/a'}).

#### CPU — sparse relative to control

| Threads | Target characters | Sample | Sparse ms | Control ms | Sparse E2E speedup point/lower95 | Decode ratio point/lower95 | Bytes/s ratio point/lower95 | Prompt tokens S/C | Prompt SHA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
${cpuRows}

Primary descriptive aggregate at ${comparison.cpu.primary.threads} thread(s), ${comparison.cpu.primary.promptTargetCharacters} target characters: minimum point E2E ${fmt(comparison.cpu.primary.worstCasePoint.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCasePoint.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCasePoint.generatedByteThroughputRatio)}×; minimum lower95 E2E ${fmt(comparison.cpu.primary.worstCaseLower95.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCaseLower95.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCaseLower95.generatedByteThroughputRatio)}×.

Paired runtime: ${markdownCell(environment.processor)}, ${markdownCell(environment.platform)}, Python ${markdownCell(environment.python)}, PyTorch ${markdownCell(environment.torch)}, NumPy ${markdownCell(environment.numpy)}. Dependencies: ${markdownCell(dependencySummary(environment.runtime_dependencies))}. Affinity: ${markdownCell(environment.cpu_affinity)}; topology: ${markdownCell(environment.cpu_topology)}; governors: ${markdownCell(environment.cpu_frequency_governors)}.

#### Memory — sparse/control

| Threads | Sparse persistent max | Control persistent max | Persistent ratio | Sparse measured load | Control measured load | Measured ratio/source | Descriptive max/source |
|---:|---:|---:|---:|---:|---:|---:|---:|
${memoryRows}

Measured load is the pre-request USS delta, with RSS fallback, and excludes fused-route-cache population. Persistent state is exact model metadata. The maximum is descriptive and is not a gate.
`;
  }).join('\n');
  return `## Matched dense controls

Configured controls are custom-only and validated fail-closed against the sparse run. No official-model artifacts or separate control reports are consumed. Performance results below are descriptive; the preregistered official-baseline gates remain unchanged.

${sections}`;
}

function markdownReport(report) {
  const primary = report.primary;
  const sections = report.comparisons.map((comparison) => {
    const quality = comparison.quality;
    const cpuRows = comparison.cpu.rows.map((row) => (
      `| ${row.threads} | ${row.promptTargetCharacters} | ${row.promptSampleIndex} | `
      + `${fmt(row.customEndToEndMs)} | ${fmt(row.officialEndToEndMs)} | `
      + `${fmt(row.endToEndSpeedup)} / ${fmt(row.speedBootstrap95?.endToEndSpeedup?.lower95)} | `
      + `${fmt(row.decodeSpeedup)} / ${fmt(row.speedBootstrap95?.decodeSpeedup?.lower95)} | `
      + `${fmt(row.generatedByteThroughputRatio)} / ${fmt(row.speedBootstrap95?.generatedByteThroughputRatio?.lower95)} | `
      + `${fmt(row.customPromptTokens, 0)} / ${fmt(row.officialPromptTokens, 0)} | ${row.promptSha256.slice(0, 12)} |`
    )).join('\n');
    const memoryRows = comparison.memory.rows.flatMap((row) => [
      ['custom', row.custom],
      [comparison.officialModelId, row.official],
    ].map(([name, item]) => (
      `| ${row.threads} | ${name} | ${fmt(item.loadSeconds, 3)} | `
      + `${bytes(item.residentParameterBytes)} | ${bytes(item.persistentModelStateMaxBytes)} | `
      + `${bytes(item.fusedRouteCacheMaximumBytes)} | ${bytes(item.rssAfterLoadBytes)} | `
      + `${bytes(item.ussAfterLoadBytes)} | ${bytes(item.rssLoadDeltaBytes)} | `
      + `${bytes(item.ussLoadDeltaBytes)} | ${bytes(item.rssTotalDeltaBytes)} | `
      + `${bytes(item.ussTotalDeltaBytes)} | ${bytes(item.peakRssAfterBenchmarkBytes)} |`
    ))).join('\n');
    const environmentRows = comparison.memory.rows.map((row) => {
      const item = row.environment;
      const container = item.containerized
        ? `${item.container_image ?? 'unknown'} @ ${item.container_image_digest ?? 'unknown'}; derived=${item.container_derived_image_id ?? 'unknown'}`
        : 'host process';
      return `| ${row.threads} | ${markdownCell(item.processor)} | ${markdownCell(item.platform)} | `
        + `${markdownCell(item.logical_cpus)} | ${markdownCell(item.cpu_affinity)} | `
        + `${markdownCell(item.cpu_topology)} | `
        + `${markdownCell(item.cpu_frequency_governors)} | ${markdownCell(item.python)} | `
        + `${markdownCell(item.torch)} | ${markdownCell(item.numpy)} | `
        + `${markdownCell(dependencySummary(item.runtime_dependencies))} | `
        + `${markdownCell(container)} |`;
    }).join('\n');
    return `## Comparison against ${comparison.officialModelId}${comparison.isPrimary ? ' — PRIMARY' : ''}

**Core verdict:** ${icon(comparison.coreVerdict)} **${comparison.coreVerdict.toUpperCase()}**  
**Final verdict:** ${icon(comparison.finalVerdict)} **${comparison.finalVerdict.toUpperCase()}**

### Quality

| Metric | Custom | Official | Ratio / delta |
|---|---:|---:|---:|
| Conditional bits/byte, primary context | ${fmt(quality.customConditionalBitsPerByte, 4)} | ${fmt(quality.officialConditionalBitsPerByte, 4)} | ${fmt(quality.conditionalBitsPerByteRatio, 4)}× |
| Upper 95% paired bootstrap CBPB | — | — | ${fmt(quality.pairedBootstrapRatio95?.upper95, 4)}× |
| Conditional bits/byte, common context | ${fmt(quality.customCommonContextBitsPerByte, 4)} | ${fmt(quality.officialCommonContextBitsPerByte, 4)} | ${fmt(quality.commonContextBitsPerByteRatio, 4)}× |
| Normalized HellaSwag | ${pct(quality.customHellaSwagNormalized)} | ${pct(quality.officialHellaSwagNormalized)} | ${Number.isFinite(quality.hellaswagDelta) ? `${(100 * quality.hellaswagDelta).toFixed(2)} pp` : 'n/a'} |

The quality bootstrap pairs the same stories and recomputes the CBPB ratio from aggregate NLL and UTF-8 bytes. The gate uses the upper 95% bound, not the point estimate.

HellaSwag scope: **${quality.hellaswagEvaluationScope}** (custom n=${quality.customHellaSwagExamples ?? 'n/a'}, official n=${quality.officialHellaSwagExamples ?? 'n/a'}). A configured value of 0 means the full validation split, not zero examples.

### CPU — all cases

“Point / lower95” shows the ratio of medians and the lower 95% bootstrap bound. Raw rows from the custom and official processes are resampled independently.

| Threads | Target characters | Sample | Custom ms | Official ms | E2E speedup point/lower95 | Decode point/lower95 | Bytes/s point/lower95 | Prompt tokens C/O | Prompt SHA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
${cpuRows}

The primary point is exactly **${comparison.cpu.primary.threads} threads, ${comparison.cpu.primary.promptTargetCharacters} target characters, ${comparison.cpu.primary.sampleCount} prompts**. Worst-case point: E2E ${fmt(comparison.cpu.primary.worstCasePoint.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCasePoint.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCasePoint.generatedByteThroughputRatio)}×. Minimum lower95: E2E ${fmt(comparison.cpu.primary.worstCaseLower95.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCaseLower95.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCaseLower95.generatedByteThroughputRatio)}×.

### Paired CPU environment

Custom and official processes must match every field below at each thread count.

| Threads | Processor | Platform | Logical CPUs | Affinity | CPU topology | Governors | Python | PyTorch | NumPy | Runtime dependencies | Container |
|---:|---|---|---:|---|---|---|---|---|---|---|---|
${environmentRows}

### Memory and load

| Threads | Model | Load s | Resident parameters | Persistent state max | Fused cache max | RSS after-load | USS after-load | RSS load delta | USS load delta | RSS total delta | USS total delta | Peak RSS |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
${memoryRows}

Measured pre-request load ratio (${comparison.memory.primaryResidentGateMetric.measuredLoad?.source ?? 'unavailable'}): ${bytes(comparison.memory.primaryResidentGateMetric.measuredLoad?.customBytes)} / ${bytes(comparison.memory.primaryResidentGateMetric.measuredLoad?.officialBytes)} = **${fmt(comparison.memory.primaryResidentGateMetric.measuredLoad?.ratio, 4)}×**. This snapshot is taken before the first request, so it excludes custom fused-route-cache population.

Exact persistent-state maximum ratio: ${bytes(comparison.memory.primaryResidentGateMetric.persistentState?.customBytes)} / ${bytes(comparison.memory.primaryResidentGateMetric.persistentState?.officialBytes)} = **${fmt(comparison.memory.primaryResidentGateMetric.persistentState?.ratio, 4)}×**. The memory gate uses the maximum available conservative ratio, **${fmt(comparison.memory.primaryResidentGateMetric.ratio, 4)}×**, controlled by **${comparison.memory.primaryResidentGateMetric.source ?? 'no valid measurement'}**. Total and peak RSS remain separate diagnostics because they include allocator and KV-cache effects.

### Gates

| Criterion | Status | Value | Threshold | Source |
|---|---|---:|---:|---|
${gateMarkdown(comparison.gates)}
`;
  }).join('\n');
  const spec = report.specialization;
  return `# Multi-baseline comparison report: ${report.profile}

**Configured primary baseline:** ${report.primaryOfficialModelId}  
**Primary benchmark runtime:** ${report.artifactValidation.benchmarkRuntime.device}/${report.artifactValidation.benchmarkRuntime.dtype} on ${report.artifactValidation.benchmarkRuntime.machineArchitecture}  
**Primary core verdict:** ${icon(primary.coreVerdict)} **${primary.coreVerdict.toUpperCase()}**  
**Primary final verdict:** ${icon(primary.finalVerdict)} **${primary.finalVerdict.toUpperCase()}**

Configuration is read exclusively from runs/${report.profile}/resolved_config.json. Artifacts are accepted only when model ID, revision SHA, checkpoint SHA, provenance, and workload manifest agree.

Training provenance: **${report.artifactValidation.training.trainingMachine}**, Python **${report.artifactValidation.training.trainingPython}**, PyTorch **${report.artifactValidation.training.trainingTorchVersion}**, NumPy **${report.artifactValidation.training.trainingNumpyVersion}**, CUDA **${report.artifactValidation.training.trainingCudaVersion ?? 'n/a'}**, cuDNN **${report.artifactValidation.training.trainingCudnnVersion ?? 'n/a'}**, NVIDIA driver **${report.artifactValidation.training.trainingNvidiaDriverVersion ?? 'n/a'}**, CUDA devices **${markdownCell(report.artifactValidation.training.trainingCudaDevices)}**, container **${report.artifactValidation.training.trainingContainerImage ?? 'none'} @ ${report.artifactValidation.training.trainingContainerImageDigest ?? 'n/a'}**, derived image **${report.artifactValidation.training.trainingContainerDerivedImageId ?? 'n/a'}**. Runtime dependencies: **${markdownCell(dependencySummary(report.artifactValidation.training.trainingRuntimeDependencies))}**. Training, evaluation, and CPU benchmark artifacts are required to share the exact clean Git commit and runtime-source hash; evaluation must also match the full training environment.

> **Memory warning: 7 MB active != RSS.** Roughly 7 MB of arithmetic-active parameters per request does not imply 7 MB process memory. The full expert bank is resident, and RSS/USS include the runtime and caches.
>
> **Causality warning:** Hugging Face baselines are system-level comparisons and do not causally isolate VSA from tokenizer, vocabulary, backbone, or runtime differences. ${report.matchedControls.configured ? 'The configured matched dense controls are reported descriptively below; an alternative MoE/gate control is still required for a complete architectural claim.' : 'Matched dense controls and an alternative MoE/gate control are required for an architectural claim.'}

## Model accounting

| Model | Checkpoint/revision SHA | Total parameters | Resident parameter bytes | Persistent state max | Fused cache max | Active arithmetic FP32 | Expert/route FP32 | Router arrays/artifact | Effective context / artifact config |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
${modelAccountingRows(report)}

Resident parameter bytes in this table come from the selected CPU/FP32 primary-thread benchmark artifacts. CUDA/BF16 evaluation metadata remains available under each comparison's models.evaluation object.

The effective official context uses the published training override (512 when configured), separately from the raw artifact-config value, which may be 2048.

## Expert-use evidence

| Variant | Loss |
|---|---:|
| Full model | ${fmt(spec.fullLoss, 4)} |
| Shared-only | ${fmt(spec.sharedOnlyLoss, 4)} |
| Permuted route | ${fmt(spec.permutedRouteLoss, 4)} |

Expert gain: **${fmt(spec.expertGain, 4)}**. Routing gain: **${fmt(spec.routingGain, 4)}**.

${sections}
${matchedControlsMarkdown(report.matchedControls)}

## Primary blind evaluation

Status: **${report.blind.status}**. Custom preference: **${pct(report.blind.preference)}**. ${report.blind.reason ?? ''}
`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function matchedControlsHtml(matchedControls) {
  if (!matchedControls?.configured) {
    return `<section><h2>Matched dense controls</h2><p>${escapeHtml(matchedControls?.note ?? 'Not configured.')}</p></section>`;
  }
  const sections = matchedControls.comparisons.map((comparison) => {
    const quality = comparison.quality;
    const budget = comparison.parameterBudget;
    const cpuRows = comparison.cpu.rows.map((row) => (
      `<tr><td>${row.threads}</td><td>${row.promptTargetCharacters}</td>`
      + `<td>${row.promptSampleIndex}</td><td>${fmt(row.sparseEndToEndMs)}</td>`
      + `<td>${fmt(row.controlEndToEndMs)}</td>`
      + `<td>${fmt(row.sparseEndToEndSpeedupRelativeToControl)} / ${fmt(row.speedBootstrap95?.endToEndSpeedup?.lower95)}</td>`
      + `<td>${fmt(row.sparseDecodeThroughputRatio)} / ${fmt(row.speedBootstrap95?.decodeSpeedup?.lower95)}</td>`
      + `<td>${fmt(row.sparseGeneratedByteThroughputRatio)} / ${fmt(row.speedBootstrap95?.generatedByteThroughputRatio?.lower95)}</td></tr>`
    )).join('');
    const memoryRows = comparison.memory.rows.map((row) => (
      `<tr><td>${row.threads}</td><td>${bytes(row.sparse.persistentModelStateMaxBytes)}</td>`
      + `<td>${bytes(row.control.persistentModelStateMaxBytes)}</td>`
      + `<td>${fmt(row.ratios.persistentState?.sparseToControlRatio, 4)}×</td>`
      + `<td>${bytes(row.ratios.measuredLoad?.sparseBytes)}</td>`
      + `<td>${bytes(row.ratios.measuredLoad?.controlBytes)}</td>`
      + `<td>${fmt(row.ratios.measuredLoad?.sparseToControlRatio, 4)}× (${escapeHtml(row.ratios.measuredLoad?.source ?? 'unavailable')})</td>`
      + `<td>${fmt(row.ratios.descriptiveMaximumRatio, 4)}× (${escapeHtml(row.ratios.descriptiveMaximumSource ?? 'unavailable')})</td></tr>`
    )).join('');
    const evidenceRows = [
      ['Frozen config', comparison.evidence.resolvedConfig],
      ['Training summary', comparison.evidence.trainingSummary],
      ['Custom evaluation', comparison.evidence.customEvaluation],
      ...Object.entries(comparison.evidence.customBenchmarks).map(([threads, item]) => (
        [`Custom CPU benchmark, ${threads} threads`, item]
      )),
    ].map(([label, item]) => (
      `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(item.path)}</td>`
      + `<td><code>${escapeHtml(item.sha256)}</code></td></tr>`
    )).join('');
    const environment = comparison.memory.rows[0]?.environment ?? {};
    return `<section><h3>${escapeHtml(comparison.role)}: ${escapeHtml(comparison.profile)}</h3><p><b>Descriptive only:</b> no new success threshold and no change to the official-baseline verdict.</p><p>Exact parameter contract: control total <b>${fmt(budget.controlTotalParameters, 0)}</b> = sparse ${escapeHtml(budget.expectedFromSparseField)} <b>${fmt(budget.expectedControlTotalParameters, 0)}</b>. Fixed-dense control active parameters: ${fmt(budget.controlActiveParametersPerRequest, 0)}.</p><p>Commands: <code>${escapeHtml(comparison.evidence.commands.evaluation)}</code>; <code>${escapeHtml(comparison.evidence.commands.cpuBenchmarkMatrix)}</code>.</p><table><tr><th>Evidence</th><th>Path</th><th>SHA-256</th></tr>${evidenceRows}</table><h4>Quality — sparse/control</h4><table><tr><th>Metric</th><th>Sparse</th><th>Control</th><th>Ratio/delta</th></tr><tr><td>Primary-context CBPB</td><td>${fmt(quality.sparseConditionalBitsPerByte, 4)}</td><td>${fmt(quality.controlConditionalBitsPerByte, 4)}</td><td>${fmt(quality.sparseToControlConditionalBitsPerByteRatio, 4)}×; bootstrap lower/median/upper ${fmt(quality.pairedBootstrapSparseToControlRatio95?.lower95, 4)} / ${fmt(quality.pairedBootstrapSparseToControlRatio95?.median, 4)} / ${fmt(quality.pairedBootstrapSparseToControlRatio95?.upper95, 4)}</td></tr><tr><td>Common-context CBPB</td><td>${fmt(quality.sparseCommonContextBitsPerByte, 4)}</td><td>${fmt(quality.controlCommonContextBitsPerByte, 4)}</td><td>${fmt(quality.sparseToControlCommonContextBitsPerByteRatio, 4)}×</td></tr><tr><td>Normalized HellaSwag</td><td>${pct(quality.sparseHellaSwagNormalized)}</td><td>${pct(quality.controlHellaSwagNormalized)}</td><td>${Number.isFinite(quality.sparseMinusControlHellaSwagDelta) ? `${(100 * quality.sparseMinusControlHellaSwagDelta).toFixed(2)} pp` : 'n/a'}</td></tr></table><p>CBPB is paired by exact story ID and UTF-8 byte count. HellaSwag scope: ${escapeHtml(quality.hellaswagEvaluationScope)}.</p><h4>CPU — sparse relative to control</h4><table><tr><th>Threads</th><th>Target</th><th>Sample</th><th>Sparse ms</th><th>Control ms</th><th>E2E point/lower95</th><th>Decode point/lower95</th><th>Bytes/s point/lower95</th></tr>${cpuRows}</table><p>Primary minimum point: E2E ${fmt(comparison.cpu.primary.worstCasePoint.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCasePoint.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCasePoint.generatedByteThroughputRatio)}×. Minimum lower95: E2E ${fmt(comparison.cpu.primary.worstCaseLower95.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCaseLower95.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCaseLower95.generatedByteThroughputRatio)}×.</p><p>Paired runtime: ${escapeHtml(environment.processor ?? 'n/a')}; Python ${escapeHtml(environment.python ?? 'n/a')}; PyTorch ${escapeHtml(environment.torch ?? 'n/a')}; NumPy ${escapeHtml(environment.numpy ?? 'n/a')}; dependencies ${escapeHtml(dependencySummary(environment.runtime_dependencies))}; affinity ${escapeHtml(JSON.stringify(environment.cpu_affinity))}; topology ${escapeHtml(JSON.stringify(environment.cpu_topology))}; governors ${escapeHtml(JSON.stringify(environment.cpu_frequency_governors))}.</p><h4>Memory — sparse/control</h4><table><tr><th>Threads</th><th>Sparse persistent</th><th>Control persistent</th><th>Persistent ratio</th><th>Sparse measured</th><th>Control measured</th><th>Measured ratio/source</th><th>Descriptive max/source</th></tr>${memoryRows}</table><p>Measured load is pre-request USS with RSS fallback and excludes fused-cache population. The maximum is descriptive, not a gate.</p></section>`;
  }).join('');
  return `<section><h2>Matched dense controls</h2><p>Configured controls are custom-only and validated fail-closed. No official artifacts or separate control reports are consumed. Performance is descriptive and the preregistered gates are unchanged.</p>${sections}</section>`;
}

function htmlReport(report) {
  const comparisonSections = report.comparisons.map((comparison) => {
    const cpu = comparison.cpu.rows.map((row) => `<tr><td>${row.threads}</td><td>${row.promptTargetCharacters}</td><td>${row.promptSampleIndex}</td><td>${fmt(row.customEndToEndMs)}</td><td>${fmt(row.officialEndToEndMs)}</td><td>${fmt(row.endToEndSpeedup)} / ${fmt(row.speedBootstrap95?.endToEndSpeedup?.lower95)}</td><td>${fmt(row.decodeSpeedup)} / ${fmt(row.speedBootstrap95?.decodeSpeedup?.lower95)}</td><td>${fmt(row.generatedByteThroughputRatio)} / ${fmt(row.speedBootstrap95?.generatedByteThroughputRatio?.lower95)}</td></tr>`).join('');
    const memory = comparison.memory.rows.flatMap((row) => [
      ['custom', row.custom],
      [comparison.officialModelId, row.official],
    ].map(([name, item]) => `<tr><td>${row.threads}</td><td>${escapeHtml(name)}</td><td>${fmt(item.loadSeconds)}</td><td>${bytes(item.residentParameterBytes)}</td><td>${bytes(item.persistentModelStateMaxBytes)}</td><td>${bytes(item.fusedRouteCacheMaximumBytes)}</td><td>${bytes(item.rssAfterLoadBytes)}</td><td>${bytes(item.ussAfterLoadBytes)}</td><td>${bytes(item.rssLoadDeltaBytes)}</td><td>${bytes(item.ussLoadDeltaBytes)}</td><td>${bytes(item.rssTotalDeltaBytes)}</td><td>${bytes(item.peakRssAfterBenchmarkBytes)}</td></tr>`)).join('');
    const environments = comparison.memory.rows.map((row) => {
      const item = row.environment;
      const container = item.containerized
        ? `${item.container_image ?? 'unknown'} @ ${item.container_image_digest ?? 'unknown'}; derived=${item.container_derived_image_id ?? 'unknown'}`
        : 'host process';
      return `<tr><td>${row.threads}</td><td>${escapeHtml(item.processor ?? 'n/a')}</td><td>${escapeHtml(item.platform ?? 'n/a')}</td><td>${item.logical_cpus ?? 'n/a'}</td><td>${escapeHtml(JSON.stringify(item.cpu_affinity))}</td><td>${escapeHtml(JSON.stringify(item.cpu_topology))}</td><td>${escapeHtml(JSON.stringify(item.cpu_frequency_governors))}</td><td>${escapeHtml(item.python ?? 'n/a')}</td><td>${escapeHtml(item.torch ?? 'n/a')}</td><td>${escapeHtml(item.numpy ?? 'n/a')}</td><td>${escapeHtml(dependencySummary(item.runtime_dependencies))}</td><td>${escapeHtml(container)}</td></tr>`;
    }).join('');
    const gates = Object.entries(comparison.gates).map(([name, item]) => `<tr><td>${name}</td><td class="${item.status}">${icon(item.status)} ${item.status}</td><td>${fmt(item.value, 4)}</td><td>${item.operator} ${fmt(item.threshold, 4)}</td><td>${escapeHtml(item.source ?? '')}</td></tr>`).join('');
    return `<section><h2>${escapeHtml(comparison.officialModelId)}${comparison.isPrimary ? ' — PRIMARY' : ''}</h2><p>Core: <b class="${comparison.coreVerdict}">${icon(comparison.coreVerdict)} ${comparison.coreVerdict.toUpperCase()}</b> · Final: <b class="${comparison.finalVerdict}">${icon(comparison.finalVerdict)} ${comparison.finalVerdict.toUpperCase()}</b></p><h3>Quality</h3><table><tr><th>Metric</th><th>Custom</th><th>Official</th><th>Ratio/delta</th></tr><tr><td>Primary-context CBPB</td><td>${fmt(comparison.quality.customConditionalBitsPerByte, 4)}</td><td>${fmt(comparison.quality.officialConditionalBitsPerByte, 4)}</td><td>${fmt(comparison.quality.conditionalBitsPerByteRatio, 4)}×; upper95 ${fmt(comparison.quality.pairedBootstrapRatio95?.upper95, 4)}×</td></tr><tr><td>Common-context CBPB</td><td>${fmt(comparison.quality.customCommonContextBitsPerByte, 4)}</td><td>${fmt(comparison.quality.officialCommonContextBitsPerByte, 4)}</td><td>${fmt(comparison.quality.commonContextBitsPerByteRatio, 4)}×</td></tr><tr><td>Normalized HellaSwag</td><td>${pct(comparison.quality.customHellaSwagNormalized)}</td><td>${pct(comparison.quality.officialHellaSwagNormalized)}</td><td>${Number.isFinite(comparison.quality.hellaswagDelta) ? `${(100 * comparison.quality.hellaswagDelta).toFixed(2)} pp` : 'n/a'}</td></tr></table><p>HellaSwag scope: ${escapeHtml(comparison.quality.hellaswagEvaluationScope)}. A configured value of 0 means the full validation split.</p><h3>CPU — point / lower95</h3><table><tr><th>Threads</th><th>Target</th><th>Sample</th><th>Custom ms</th><th>Official ms</th><th>E2E</th><th>Decode</th><th>Bytes/s</th></tr>${cpu}</table><p>Primary point: ${comparison.cpu.primary.threads} threads / ${comparison.cpu.primary.promptTargetCharacters} target characters. Minimum lower95: E2E ${fmt(comparison.cpu.primary.worstCaseLower95.endToEndSpeedup)}×, decode ${fmt(comparison.cpu.primary.worstCaseLower95.decodeSpeedup)}×, bytes/s ${fmt(comparison.cpu.primary.worstCaseLower95.generatedByteThroughputRatio)}×.</p><h3>Paired CPU environment</h3><table><tr><th>Threads</th><th>Processor</th><th>Platform</th><th>Logical CPUs</th><th>Affinity</th><th>CPU topology</th><th>Governors</th><th>Python</th><th>PyTorch</th><th>NumPy</th><th>Runtime dependencies</th><th>Container</th></tr>${environments}</table><h3>Memory</h3><table><tr><th>Threads</th><th>Model</th><th>Load s</th><th>Resident parameters</th><th>Persistent state max</th><th>Fused cache max</th><th>RSS after-load</th><th>USS after-load</th><th>RSS load delta</th><th>USS load delta</th><th>RSS total delta</th><th>Peak RSS</th></tr>${memory}</table><p>Measured pre-request ratio (${escapeHtml(comparison.memory.primaryResidentGateMetric.measuredLoad?.source ?? 'unavailable')}): ${fmt(comparison.memory.primaryResidentGateMetric.measuredLoad?.ratio, 4)}×. It excludes fused-cache population. Exact persistent-state maximum ratio: ${fmt(comparison.memory.primaryResidentGateMetric.persistentState?.ratio, 4)}×. The gate uses their conservative maximum, ${fmt(comparison.memory.primaryResidentGateMetric.ratio, 4)}×, controlled by ${escapeHtml(comparison.memory.primaryResidentGateMetric.source ?? 'n/a')}. Peak/total RSS is diagnostic.</p><h3>Gates</h3><table><tr><th>Criterion</th><th>Status</th><th>Value</th><th>Threshold</th><th>Source</th></tr>${gates}</table></section>`;
  }).join('');
  const training = report.artifactValidation.training;
  const trainingRow = `<tr><td colspan="8">Training provenance: ${escapeHtml(training.trainingMachine)}, Python ${escapeHtml(training.trainingPython)}, PyTorch ${escapeHtml(training.trainingTorchVersion)}, NumPy ${escapeHtml(training.trainingNumpyVersion)}, dependencies ${escapeHtml(dependencySummary(training.trainingRuntimeDependencies))}, CUDA ${escapeHtml(training.trainingCudaVersion ?? 'n/a')}, CUDA devices ${escapeHtml(JSON.stringify(training.trainingCudaDevices))}, cuDNN ${escapeHtml(training.trainingCudnnVersion ?? 'n/a')}, NVIDIA driver ${escapeHtml(training.trainingNvidiaDriverVersion ?? 'n/a')}, ${escapeHtml(training.trainingContainerImage ?? 'no container')} @ ${escapeHtml(training.trainingContainerImageDigest ?? 'n/a')}, derived image ${escapeHtml(training.trainingContainerDerivedImageId ?? 'n/a')}</td></tr>`;
  const accountingModels = [
    { label: report.customModel.name, model: report.customModel },
    ...report.comparisons.map((item) => ({
      label: item.officialModelId,
      model: item.models.official,
    })),
    ...(report.matchedControls?.comparisons ?? []).map((item) => ({
      label: `${item.profile} (${item.role})`,
      model: item.models.control,
    })),
  ];
  const modelRows = trainingRow + accountingModels.map(({ label, model }) => `<tr><td>${escapeHtml(label)}</td><td>${fmt(modelTotalParameters(model), 0)}</td><td>${bytes(residentParameterBytes(model))}</td><td>${bytes(persistentModelStateMaxBytes(model))}</td><td>${bytes(model.fused_route_cache?.maximum_bytes)}</td><td>${bytes((model.parameters?.fp32_active_megabytes ?? NaN) * 1e6)}</td><td>${bytes((model.parameters?.fp32_expert_per_route_megabytes ?? NaN) * 1e6)}</td><td>${model.context_length ?? 'n/a'} / ${model.artifact_config_context_length ?? 'n/a'}</td></tr>`).join('');
  return `<!doctype html><html lang="en"><meta charset="utf-8"><title>VSA multi-baseline ${escapeHtml(report.profile)}</title><style>body{font-family:system-ui;max-width:1450px;margin:35px auto;padding:0 20px;line-height:1.45}table{border-collapse:collapse;width:100%;margin:10px 0 26px;font-size:13px}th,td{border:1px solid #ccc;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}.pass{color:#087f23;font-weight:700}.fail{color:#b00020;font-weight:700}.pending,.unavailable{color:#8a5a00;font-weight:700}.warning{padding:14px;border-left:5px solid #b26a00;background:#fff7e6;margin:14px 0}section{border-top:3px solid #333;margin-top:35px}</style><body><h1>VSA-PathMoE — multi-baseline report</h1><p>Profile: <b>${escapeHtml(report.profile)}</b> · Primary: <b>${escapeHtml(report.primaryOfficialModelId)}</b> · Benchmark runtime: <b>${escapeHtml(report.artifactValidation.benchmarkRuntime.device)}/${escapeHtml(report.artifactValidation.benchmarkRuntime.dtype)} on ${escapeHtml(report.artifactValidation.benchmarkRuntime.machineArchitecture)}</b></p><p>Core: <b class="${report.coreVerdict}">${icon(report.coreVerdict)} ${report.coreVerdict.toUpperCase()}</b> · Final: <b class="${report.finalVerdict}">${icon(report.finalVerdict)} ${report.finalVerdict.toUpperCase()}</b></p><div class="warning"><b>7 MB active != RSS.</b> The full expert bank and runtime remain resident; parameter bytes, persistent-state maximum, pre-request USS/RSS, and peak memory are reported separately.</div><div class="warning"><b>HF baselines do not causally isolate VSA.</b> Tokenizer, vocabulary, backbone, and runtime differ; matched dense controls are reported descriptively below, and an alternative router/MoE control remains necessary.</div><h2>Accounting</h2><table><tr><th>Model</th><th>Parameters</th><th>Resident parameter bytes</th><th>Persistent state max</th><th>Fused cache max</th><th>Active arithmetic</th><th>Expert/route</th><th>Effective/artifact context</th></tr>${modelRows}</table><p>Resident and persistent-state bytes use the selected CPU/FP32 primary-thread benchmark artifacts.</p>${comparisonSections}${matchedControlsHtml(report.matchedControls)}<h2>Primary blind evaluation</h2><p>${escapeHtml(report.blind.status)} · ${pct(report.blind.preference)} · ${escapeHtml(report.blind.reason ?? '')}</p></body></html>`;
}

function csvCell(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '';
  const string = String(value);
  return /[",\n]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
}

function csvReport(report) {
  const header = [
    'comparison_kind', 'comparison_role', 'comparison_id',
    'official_model_id', 'official_revision', 'is_primary', 'threads',
    'matched_control_resolved_config_path', 'matched_control_training_summary_path',
    'matched_control_evaluation_path', 'matched_control_benchmark_path',
    'matched_control_evaluation_command', 'matched_control_benchmark_command',
    'benchmark_device', 'benchmark_dtype', 'benchmark_machine_architecture',
    'training_machine', 'training_python_version', 'training_torch_version',
    'training_numpy_version', 'training_runtime_dependencies', 'training_cuda_version',
    'training_cuda_devices', 'training_cudnn_version', 'training_nvidia_driver_version',
    'training_container_image', 'training_container_image_digest',
    'training_container_derived_image_id',
    'cpu_processor', 'cpu_platform', 'cpu_logical_cpus', 'cpu_affinity',
    'cpu_topology', 'cpu_frequency_governors', 'cpu_python_version', 'cpu_torch_version',
    'cpu_numpy_version', 'cpu_runtime_dependencies',
    'cpu_containerized', 'cpu_container_image', 'cpu_container_image_digest',
    'cpu_container_derived_image_id',
    'prompt_target_characters', 'prompt_sample_index', 'prompt_sha256',
    'custom_end_to_end_ms', 'official_end_to_end_ms', 'end_to_end_speedup',
    'end_to_end_speedup_lower95', 'decode_speedup', 'decode_speedup_lower95',
    'generated_byte_throughput_ratio', 'generated_byte_throughput_lower95',
    'custom_prompt_tokens', 'official_prompt_tokens', 'custom_total_parameters',
    'official_total_parameters', 'custom_resident_parameter_bytes',
    'official_resident_parameter_bytes', 'custom_persistent_model_state_max_bytes',
    'official_persistent_model_state_max_bytes', 'custom_fused_route_cache_max_bytes',
    'official_fused_route_cache_max_bytes', 'custom_load_seconds', 'official_load_seconds',
    'custom_rss_after_load_bytes', 'official_rss_after_load_bytes',
    'custom_uss_after_load_bytes', 'official_uss_after_load_bytes',
    'custom_rss_load_delta_bytes', 'official_rss_load_delta_bytes',
    'custom_uss_load_delta_bytes', 'official_uss_load_delta_bytes',
    'custom_rss_total_delta_bytes', 'official_rss_total_delta_bytes',
    'custom_peak_rss_bytes', 'official_peak_rss_bytes',
    'measured_load_memory_ratio', 'measured_load_memory_source',
    'persistent_state_memory_ratio', 'conservative_memory_gate_ratio',
    'conservative_memory_controlling_source', 'descriptive_max_memory_ratio',
    'descriptive_max_memory_source',
    'sparse_cbpb', 'control_cbpb', 'sparse_to_control_cbpb_ratio',
    'cbpb_bootstrap_lower95', 'cbpb_bootstrap_median', 'cbpb_bootstrap_upper95',
    'sparse_hellaswag_normalized', 'control_hellaswag_normalized',
    'sparse_minus_control_hellaswag_delta',
    'parameter_budget_basis', 'expected_control_total_parameters',
    'actual_control_total_parameters',
  ];
  const records = [];
  const runtime = report.artifactValidation.benchmarkRuntime;
  const addTraining = (record, training) => Object.assign(record, {
    training_machine: training.trainingMachine,
    training_python_version: training.trainingPython,
    training_torch_version: training.trainingTorchVersion,
    training_numpy_version: training.trainingNumpyVersion,
    training_runtime_dependencies: JSON.stringify(training.trainingRuntimeDependencies),
    training_cuda_version: training.trainingCudaVersion,
    training_cuda_devices: JSON.stringify(training.trainingCudaDevices),
    training_cudnn_version: training.trainingCudnnVersion,
    training_nvidia_driver_version: training.trainingNvidiaDriverVersion,
    training_container_image: training.trainingContainerImage,
    training_container_image_digest: training.trainingContainerImageDigest,
    training_container_derived_image_id: training.trainingContainerDerivedImageId,
  });
  const addEnvironment = (record, environment) => Object.assign(record, {
    cpu_processor: environment.processor,
    cpu_platform: environment.platform,
    cpu_logical_cpus: environment.logical_cpus,
    cpu_affinity: JSON.stringify(environment.cpu_affinity),
    cpu_topology: JSON.stringify(environment.cpu_topology),
    cpu_frequency_governors: JSON.stringify(environment.cpu_frequency_governors),
    cpu_python_version: environment.python,
    cpu_torch_version: environment.torch,
    cpu_numpy_version: environment.numpy,
    cpu_runtime_dependencies: JSON.stringify(environment.runtime_dependencies),
    cpu_containerized: environment.containerized,
    cpu_container_image: environment.container_image,
    cpu_container_image_digest: environment.container_image_digest,
    cpu_container_derived_image_id: environment.container_derived_image_id,
  });
  for (const comparison of report.comparisons) {
    const memoryByThread = new Map(comparison.memory.rows.map((row) => [row.threads, row]));
    for (const row of comparison.cpu.rows) {
      const memory = memoryByThread.get(row.threads);
      const record = {
        comparison_kind: 'official_huggingface',
        comparison_id: comparison.officialModelId,
        official_model_id: comparison.officialModelId,
        official_revision: comparison.resolvedRevision,
        is_primary: comparison.isPrimary,
        threads: row.threads,
        benchmark_device: runtime.device,
        benchmark_dtype: runtime.dtype,
        benchmark_machine_architecture: runtime.machineArchitecture,
        prompt_target_characters: row.promptTargetCharacters,
        prompt_sample_index: row.promptSampleIndex,
        prompt_sha256: row.promptSha256,
        custom_end_to_end_ms: row.customEndToEndMs,
        official_end_to_end_ms: row.officialEndToEndMs,
        end_to_end_speedup: row.endToEndSpeedup,
        end_to_end_speedup_lower95: row.speedBootstrap95?.endToEndSpeedup?.lower95,
        decode_speedup: row.decodeSpeedup,
        decode_speedup_lower95: row.speedBootstrap95?.decodeSpeedup?.lower95,
        generated_byte_throughput_ratio: row.generatedByteThroughputRatio,
        generated_byte_throughput_lower95:
          row.speedBootstrap95?.generatedByteThroughputRatio?.lower95,
        custom_prompt_tokens: row.customPromptTokens,
        official_prompt_tokens: row.officialPromptTokens,
        custom_total_parameters: modelTotalParameters(comparison.models.custom),
        official_total_parameters: modelTotalParameters(comparison.models.official),
        custom_resident_parameter_bytes: residentParameterBytes(comparison.models.custom),
        official_resident_parameter_bytes: residentParameterBytes(comparison.models.official),
        custom_persistent_model_state_max_bytes: memory.custom.persistentModelStateMaxBytes,
        official_persistent_model_state_max_bytes: memory.official.persistentModelStateMaxBytes,
        custom_fused_route_cache_max_bytes: memory.custom.fusedRouteCacheMaximumBytes,
        official_fused_route_cache_max_bytes: memory.official.fusedRouteCacheMaximumBytes,
        custom_load_seconds: memory.custom.loadSeconds,
        official_load_seconds: memory.official.loadSeconds,
        custom_rss_after_load_bytes: memory.custom.rssAfterLoadBytes,
        official_rss_after_load_bytes: memory.official.rssAfterLoadBytes,
        custom_uss_after_load_bytes: memory.custom.ussAfterLoadBytes,
        official_uss_after_load_bytes: memory.official.ussAfterLoadBytes,
        custom_rss_load_delta_bytes: memory.custom.rssLoadDeltaBytes,
        official_rss_load_delta_bytes: memory.official.rssLoadDeltaBytes,
        custom_uss_load_delta_bytes: memory.custom.ussLoadDeltaBytes,
        official_uss_load_delta_bytes: memory.official.ussLoadDeltaBytes,
        custom_rss_total_delta_bytes: memory.custom.rssTotalDeltaBytes,
        official_rss_total_delta_bytes: memory.official.rssTotalDeltaBytes,
        custom_peak_rss_bytes: memory.custom.peakRssAfterBenchmarkBytes,
        official_peak_rss_bytes: memory.official.peakRssAfterBenchmarkBytes,
        measured_load_memory_ratio:
          comparison.memory.primaryResidentGateMetric.measuredLoad?.ratio,
        measured_load_memory_source:
          comparison.memory.primaryResidentGateMetric.measuredLoad?.source,
        persistent_state_memory_ratio:
          comparison.memory.primaryResidentGateMetric.persistentState?.ratio,
        conservative_memory_gate_ratio: comparison.memory.primaryResidentGateMetric.ratio,
        conservative_memory_controlling_source:
          comparison.memory.primaryResidentGateMetric.source,
      };
      addTraining(record, report.artifactValidation.training);
      addEnvironment(record, memory.environment);
      records.push(record);
    }
  }
  for (const comparison of report.matchedControls?.comparisons ?? []) {
    const memoryByThread = new Map(comparison.memory.rows.map((row) => [row.threads, row]));
    for (const row of comparison.cpu.rows) {
      const memory = memoryByThread.get(row.threads);
      const quality = comparison.quality;
      const record = {
        comparison_kind: 'matched_dense_control',
        comparison_role: comparison.role,
        comparison_id: comparison.profile,
        is_primary: false,
        threads: row.threads,
        matched_control_resolved_config_path: comparison.evidence.resolvedConfig.path,
        matched_control_training_summary_path: comparison.evidence.trainingSummary.path,
        matched_control_evaluation_path: comparison.evidence.customEvaluation.path,
        matched_control_benchmark_path:
          comparison.evidence.customBenchmarks[row.threads]?.path,
        matched_control_evaluation_command: comparison.evidence.commands.evaluation,
        matched_control_benchmark_command: comparison.evidence.commands.cpuBenchmarkMatrix,
        benchmark_device: runtime.device,
        benchmark_dtype: runtime.dtype,
        benchmark_machine_architecture: runtime.machineArchitecture,
        prompt_target_characters: row.promptTargetCharacters,
        prompt_sample_index: row.promptSampleIndex,
        prompt_sha256: row.promptSha256,
        custom_end_to_end_ms: row.sparseEndToEndMs,
        official_end_to_end_ms: row.controlEndToEndMs,
        end_to_end_speedup: row.sparseEndToEndSpeedupRelativeToControl,
        end_to_end_speedup_lower95: row.speedBootstrap95?.endToEndSpeedup?.lower95,
        decode_speedup: row.sparseDecodeThroughputRatio,
        decode_speedup_lower95: row.speedBootstrap95?.decodeSpeedup?.lower95,
        generated_byte_throughput_ratio: row.sparseGeneratedByteThroughputRatio,
        generated_byte_throughput_lower95:
          row.speedBootstrap95?.generatedByteThroughputRatio?.lower95,
        custom_prompt_tokens: row.sparsePromptTokens,
        official_prompt_tokens: row.controlPromptTokens,
        custom_total_parameters: modelTotalParameters(comparison.models.sparse),
        official_total_parameters: modelTotalParameters(comparison.models.control),
        custom_resident_parameter_bytes: residentParameterBytes(comparison.models.sparse),
        official_resident_parameter_bytes: residentParameterBytes(comparison.models.control),
        custom_persistent_model_state_max_bytes: memory.sparse.persistentModelStateMaxBytes,
        official_persistent_model_state_max_bytes: memory.control.persistentModelStateMaxBytes,
        custom_fused_route_cache_max_bytes: memory.sparse.fusedRouteCacheMaximumBytes,
        official_fused_route_cache_max_bytes: memory.control.fusedRouteCacheMaximumBytes,
        custom_load_seconds: memory.sparse.loadSeconds,
        official_load_seconds: memory.control.loadSeconds,
        custom_rss_after_load_bytes: memory.sparse.rssAfterLoadBytes,
        official_rss_after_load_bytes: memory.control.rssAfterLoadBytes,
        custom_uss_after_load_bytes: memory.sparse.ussAfterLoadBytes,
        official_uss_after_load_bytes: memory.control.ussAfterLoadBytes,
        custom_rss_load_delta_bytes: memory.sparse.rssLoadDeltaBytes,
        official_rss_load_delta_bytes: memory.control.rssLoadDeltaBytes,
        custom_uss_load_delta_bytes: memory.sparse.ussLoadDeltaBytes,
        official_uss_load_delta_bytes: memory.control.ussLoadDeltaBytes,
        custom_rss_total_delta_bytes: memory.sparse.rssTotalDeltaBytes,
        official_rss_total_delta_bytes: memory.control.rssTotalDeltaBytes,
        custom_peak_rss_bytes: memory.sparse.peakRssAfterBenchmarkBytes,
        official_peak_rss_bytes: memory.control.peakRssAfterBenchmarkBytes,
        measured_load_memory_ratio: memory.ratios.measuredLoad?.sparseToControlRatio,
        measured_load_memory_source: memory.ratios.measuredLoad?.source,
        persistent_state_memory_ratio:
          memory.ratios.persistentState?.sparseToControlRatio,
        descriptive_max_memory_ratio: memory.ratios.descriptiveMaximumRatio,
        descriptive_max_memory_source: memory.ratios.descriptiveMaximumSource,
        sparse_cbpb: quality.sparseConditionalBitsPerByte,
        control_cbpb: quality.controlConditionalBitsPerByte,
        sparse_to_control_cbpb_ratio:
          quality.sparseToControlConditionalBitsPerByteRatio,
        cbpb_bootstrap_lower95:
          quality.pairedBootstrapSparseToControlRatio95?.lower95,
        cbpb_bootstrap_median:
          quality.pairedBootstrapSparseToControlRatio95?.median,
        cbpb_bootstrap_upper95:
          quality.pairedBootstrapSparseToControlRatio95?.upper95,
        sparse_hellaswag_normalized: quality.sparseHellaSwagNormalized,
        control_hellaswag_normalized: quality.controlHellaSwagNormalized,
        sparse_minus_control_hellaswag_delta:
          quality.sparseMinusControlHellaSwagDelta,
        parameter_budget_basis: comparison.parameterBudget.expectedFromSparseField,
        expected_control_total_parameters:
          comparison.parameterBudget.expectedControlTotalParameters,
        actual_control_total_parameters: comparison.parameterBudget.controlTotalParameters,
      };
      addTraining(record, comparison.provenance.training);
      addEnvironment(record, memory.environment);
      records.push(record);
    }
  }
  return `${header.join(',')}\n${records.map((record) => (
    header.map((field) => csvCell(record[field])).join(',')
  )).join('\n')}\n`;
}

function resolveProfile() {
  const profileArg = process.argv.findIndex((item) => item === '--profile');
  return profileArg >= 0
    ? process.argv[profileArg + 1]
    : (process.env.VSA_PROFILE || 'full_cpu');
}

export function generateReport({ root = ROOT, profile = resolveProfile() } = {}) {
  const runDir = path.join(root, 'runs', profile);
  const evalDir = path.join(runDir, 'evaluation');
  const benchDir = path.join(runDir, 'benchmark');
  const resultDir = path.join(root, 'results');
  const configPath = path.join(runDir, 'resolved_config.json');
  const config = readJson(configPath, 'frozen resolved config');
  validateFrozenConfigProfile(config, profile);
  const trainingProvenance = validateTrainingProvenance(runDir, profile, config);
  const sources = config.sources?.official_models;
  if (!Array.isArray(sources) || !sources.length) fail('no official models are configured');
  const sourceIds = sources.map((item) => item.id);
  if (new Set(sourceIds).size !== sourceIds.length) fail('configured official model IDs are duplicated');
  const primaryModelId = requireString(
    config.comparison?.primary_official_model_id,
    'comparison.primary_official_model_id',
  );
  if (!sourceIds.includes(primaryModelId)) {
    fail(`primary official model ${primaryModelId} is not configured`);
  }
  const primaryDevice = runtimeSlug(
    config.benchmark?.primary_device,
    'benchmark.primary_device',
  );
  const primaryDtype = runtimeSlug(
    config.benchmark?.primary_dtype,
    'benchmark.primary_dtype',
  );
  const primaryMachineSlug = runtimeSlug(
    config.benchmark?.primary_machine_architecture,
    'benchmark.primary_machine_architecture',
  );
  const primaryMachineArchitecture = canonicalMachineArchitecture(primaryMachineSlug);
  if (primaryMachineArchitecture !== 'x86_64') {
    fail('benchmark.primary_machine_architecture must be x86_64 or amd64');
  }

  const customEvaluationPath = path.join(evalDir, 'custom.json');
  const custom = readJson(customEvaluationPath, 'custom evaluation');
  validateProfile(custom, profile, 'custom evaluation');
  validateEvaluationSchema(custom, 'custom evaluation');
  validateExecutionProvenance(custom, trainingProvenance, 'custom evaluation');
  validateTrainingExecutionEnvironment(
    custom,
    trainingProvenance,
    'custom evaluation',
  );
  validateRuntimeIdentity(
    custom,
    config.runtime?.device,
    config.runtime?.dtype,
    'custom evaluation',
  );
  if (!['custom_vsa_pathmoe', 'matched_dense_control'].includes(custom.model?.kind)) {
    fail('custom evaluation model.kind is invalid');
  }
  const customEvaluationRuntimeSourceSha256 = runtimeSourceSha(custom, 'custom evaluation');
  const customEvaluationSha256 = sha256File(customEvaluationPath);
  const checkpointSha = requireSha256(
    custom.model?.checkpoint_sha256,
    'custom evaluation checkpoint SHA',
  );
  const checkpointProvenance = validateCheckpointProvenance(
    custom.model,
    trainingProvenance,
    profile,
    'custom evaluation',
  );
  const officialEvaluations = new Map();
  const officialEvaluationSha256 = new Map();
  for (const source of sources) {
    const file = path.join(evalDir, `official_${safeModelId(source.id)}.json`);
    const evaluation = readJson(file, `${source.id} evaluation`);
    validateEvaluationSchema(evaluation, `${source.id} evaluation`);
    validateOfficialArtifact(evaluation, source, null, profile, `${source.id} evaluation`);
    validateExecutionProvenance(
      evaluation,
      trainingProvenance,
      `${source.id} evaluation`,
    );
    validateTrainingExecutionEnvironment(
      evaluation,
      trainingProvenance,
      `${source.id} evaluation`,
    );
    if (runtimeSourceSha(evaluation, `${source.id} evaluation`)
        !== customEvaluationRuntimeSourceSha256) {
      fail(`${source.id} evaluation runtime source differs from custom evaluation`);
    }
    officialEvaluations.set(source.id, evaluation);
    officialEvaluationSha256.set(source.id, sha256File(file));
  }

  if (!fs.existsSync(benchDir)) fail(`benchmark directory is missing: ${benchDir}`);
  const customPattern = new RegExp(
    `^custom_${primaryDevice}_${primaryDtype}_threads_(\\d+)\\.json$`,
  );
  const customFiles = fs.readdirSync(benchDir)
    .filter((name) => customPattern.test(name))
    .sort();
  if (!customFiles.length) {
    fail(`no custom ${primaryDevice}/${primaryDtype} benchmark artifacts found`);
  }
  const customBenchmarks = new Map();
  let workloadManifest = null;
  for (const name of customFiles) {
    const fileThread = Number(name.match(customPattern)[1]);
    if (customBenchmarks.has(fileThread)) fail(`duplicate custom benchmark thread ${fileThread}`);
    const artifact = readJson(path.join(benchDir, name), name);
    const manifest = validateCustomBenchmark(
      artifact,
      fileThread,
      profile,
      config,
      checkpointSha,
      customEvaluationRuntimeSourceSha256,
      primaryDevice,
      primaryDtype,
      primaryMachineArchitecture,
      trainingProvenance,
      checkpointProvenance,
    );
    if (workloadManifest && manifest !== workloadManifest) {
      fail(`custom benchmark workload differs across thread counts`);
    }
    workloadManifest = manifest;
    customBenchmarks.set(fileThread, artifact);
  }
  if (!customBenchmarks.has(Number(config.benchmark.primary_threads))) {
    fail(`custom benchmark for primary_threads=${config.benchmark.primary_threads} is missing`);
  }

  const officialBenchmarksByModel = new Map();
  for (const source of sources) {
    const perThread = new Map();
    const evaluation = officialEvaluations.get(source.id);
    for (const [threads] of customBenchmarks) {
      const name = `official_${safeModelId(source.id)}_${primaryDevice}_${primaryDtype}_threads_${threads}.json`;
      const artifact = readJson(path.join(benchDir, name), name);
      validateOfficialBenchmark(
        artifact,
        threads,
        source,
        evaluation,
        profile,
        config,
        workloadManifest,
        primaryDevice,
        primaryDtype,
        primaryMachineArchitecture,
        customEvaluationRuntimeSourceSha256,
        trainingProvenance,
      );
      validateBenchmarkEnvironmentPair(
        customBenchmarks.get(threads),
        artifact,
        `${source.id} benchmark threads=${threads}`,
      );
      perThread.set(threads, artifact);
    }
    officialBenchmarksByModel.set(source.id, perThread);
  }

  fs.mkdirSync(resultDir, { recursive: true });
  const primaryOfficial = officialEvaluations.get(primaryModelId);
  const blind = blindEvaluation(
    resultDir,
    profile,
    primaryModelId,
    custom,
    primaryOfficial,
    customEvaluationSha256,
    officialEvaluationSha256.get(primaryModelId),
  );
  const specializationResult = specialization(custom);
  const comparisons = sources.map((source) => buildBaselineComparison({
    source,
    custom,
    official: officialEvaluations.get(source.id),
    customBenchmarks,
    officialBenchmarks: officialBenchmarksByModel.get(source.id),
    config,
    specializationResult,
    blind,
    isPrimary: source.id === primaryModelId,
  }));
  const primary = comparisons.find((item) => item.officialModelId === primaryModelId);
  if (!primary) fail(`primary baseline ${primaryModelId} comparison is missing`);
  const matchedControls = buildMatchedControlComparisons({
    root,
    mainProfile: profile,
    mainConfig: config,
    mainTraining: trainingProvenance,
    sparseEvaluation: custom,
    sparseEvaluationRuntimeSourceSha256: customEvaluationRuntimeSourceSha256,
    sparseCheckpointProvenance: checkpointProvenance,
    sparseBenchmarks: customBenchmarks,
    workloadManifest,
    primaryDevice,
    primaryDtype,
    primaryMachineArchitecture,
  });
  const report = {
    schemaVersion: 4,
    profile,
    resolvedConfigPath: path.relative(root, configPath),
    primaryOfficialModelId: primaryModelId,
    coreVerdict: primary.coreVerdict,
    finalVerdict: primary.finalVerdict,
    artifactValidation: {
      customCheckpointSha256: checkpointSha,
      customCheckpointProvenance: checkpointProvenance,
      customEvaluationSha256,
      customRuntimeSourceSha256: customEvaluationRuntimeSourceSha256,
      training: trainingProvenance,
      workloadManifestSha256: workloadManifest,
      officialResolvedRevisions: Object.fromEntries(
        comparisons.map((item) => [item.officialModelId, item.resolvedRevision]),
      ),
      benchmarkThreads: [...customBenchmarks.keys()].sort((a, b) => a - b),
      benchmarkRuntime: {
        device: primaryDevice,
        dtype: primaryDtype,
        machineArchitecture: primaryMachineArchitecture,
      },
      mainEvidence: {
        resolvedConfig: {
          path: path.relative(root, configPath),
          sha256: trainingProvenance.resolvedConfigSha256,
        },
        trainingSummary: {
          path: path.relative(root, path.join(runDir, 'artifacts', 'training_summary.json')),
          sha256: trainingProvenance.trainingSummarySha256,
        },
        customEvaluation: {
          path: path.relative(root, customEvaluationPath),
          sha256: customEvaluationSha256,
        },
        customBenchmarks: Object.fromEntries(
          [...customBenchmarks.keys()].sort((a, b) => a - b).map((threads) => {
            const file = path.join(
              benchDir,
              `custom_${primaryDevice}_${primaryDtype}_threads_${threads}.json`,
            );
            return [threads, { path: path.relative(root, file), sha256: sha256File(file) }];
          }),
        ),
      },
    },
    customModel: primary.models.custom,
    customEvaluationModel: custom.model,
    specialization: specializationResult,
    blind,
    comparisons,
    matchedControls,
    primary,
    // Primary aliases preserve simple downstream consumption without selecting a fallback.
    models: primary.models,
    quality: primary.quality,
    cpu: primary.cpu,
    memory: primary.memory,
    gates: primary.gates,
    blindPreference: blind.preference,
    limitations: [
      'Active arithmetic parameter bytes are not process RSS or USS.',
      'Official Hugging Face comparisons are system-level and do not causally isolate VSA.',
      matchedControls.configured
        ? 'Matched dense controls are descriptive and introduce no new success thresholds.'
        : 'Matched dense controls are not configured for this profile.',
      'An alternative-routing control is still required for a complete architectural claim.',
    ],
  };

  fs.writeFileSync(
    path.join(resultDir, `comparison_${profile}.json`),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  fs.writeFileSync(path.join(resultDir, `REPORT_${profile}.md`), markdownReport(report));
  fs.writeFileSync(path.join(resultDir, `REPORT_${profile}.html`), htmlReport(report));
  fs.writeFileSync(path.join(resultDir, `cpu_metrics_${profile}.csv`), csvReport(report));
  console.log(
    `Created multi-baseline report for ${profile}: ${comparisons.length} official models, `
    + `${matchedControls.comparisons.length} matched controls, primary=${primaryModelId}`,
  );
  return report;
}

if (process.env.VSA_REPORT_LIBRARY_ONLY !== '1') {
  generateReport();
}
