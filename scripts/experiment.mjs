import process from 'node:process';
import fs from 'node:fs';
import path from 'node:path';
import {
  ROOT,
  comparableRunConfig,
  parseCommonArgs,
  pythonBaseArgs,
  runPython,
} from './lib.mjs';

const [command = 'help', ...argv] = process.argv.slice(2);
const common = parseCommonArgs(argv);
const base = pythonBaseArgs(common);
const invoke = (subcommand, extra = []) => runPython([...base, subcommand, ...extra]);
let frozenConfig = null;

function freezeConfig({ benchmarkRuntime = false } = {}) {
  if (frozenConfig) return frozenConfig;
  // Benchmark-only commands may move to a different host. Device, dtype and
  // thread count are axes recorded in each artifact; preparation, training and
  // evaluation still freeze their actual runtime overrides.
  const freezeBase = pythonBaseArgs(
    benchmarkRuntime
      ? { ...common, threads: null, device: null, dtype: null }
      : common,
  );
  const resolved = JSON.parse(runPython([...freezeBase, 'show-config'], { capture: true }));
  const runDir = path.join(ROOT, 'runs', common.profile);
  const target = path.join(runDir, 'resolved_config.json');
  if (fs.existsSync(target)) {
    const previous = JSON.parse(fs.readFileSync(target, 'utf8'));
    const previousComparable = comparableRunConfig(previous, {
      ignoreBenchmarkRuntime: benchmarkRuntime,
    });
    const resolvedComparable = comparableRunConfig(resolved, {
      ignoreBenchmarkRuntime: benchmarkRuntime,
    });
    if (JSON.stringify(previousComparable) !== JSON.stringify(resolvedComparable)) {
      throw new Error(
        `Configuration drift for ${common.profile}: ${target} differs from the current ` +
        'resolved config. Use a new profile/run directory instead of mixing artifacts.'
      );
    }
    frozenConfig = previous;
  } else {
    fs.mkdirSync(runDir, { recursive: true });
    const temporary = `${target}.tmp`;
    fs.writeFileSync(temporary, `${JSON.stringify(resolved, null, 2)}\n`);
    fs.renameSync(temporary, target);
    frozenConfig = resolved;
  }
  return frozenConfig;
}

function officialModelIndexes() {
  const config = freezeConfig();
  return config.sources.official_models.map((_, index) => index);
}

function evaluateAllOfficial() {
  for (const index of officialModelIndexes()) {
    invoke('evaluate-official', ['--model-index', String(index)]);
  }
}

function benchmarkAllOfficial(threadBase) {
  for (const index of officialModelIndexes()) {
    runPython([...threadBase, 'benchmark-official', '--model-index', String(index)]);
  }
}

function benchmarkMatrix(extra = []) {
  const config = freezeConfig({ benchmarkRuntime: true });
  const threadValues = common.threads != null ? [common.threads] : [1, 4, 8];
  const device = common.device ?? config.benchmark.primary_device;
  const dtype = common.dtype ?? config.benchmark.primary_dtype;
  for (const threads of threadValues) {
    const threadBase = pythonBaseArgs({ ...common, threads, device, dtype });
    runPython([...threadBase, 'benchmark-custom', ...extra]);
    benchmarkAllOfficial(threadBase);
  }
}

function benchmarkCustomMatrix(extra = []) {
  const config = freezeConfig({ benchmarkRuntime: true });
  const threadValues = common.threads != null ? [common.threads] : [1, 4, 8];
  const device = common.device ?? config.benchmark.primary_device;
  const dtype = common.dtype ?? config.benchmark.primary_dtype;
  for (const threads of threadValues) {
    const threadBase = pythonBaseArgs({ ...common, threads, device, dtype });
    runPython([...threadBase, 'benchmark-custom', ...extra]);
  }
}

function help() {
  console.log(`
VSA-PathMoE TinyStories experiment

Commands:
  doctor                 inspect environment and parameter accounting
  doctor-dgx             fail-closed DGX Spark/NGC preflight
  download               download valid/prompts/official; full train for non-smoke profiles
  download-full          force TinyStories-train.txt download
  download-deployment    download evaluation/runtime assets without the train corpus
  prepare                tokenizer + causal VSA router + indexed stories
  train                  joint phase + expert-only specialization
  evaluate               custom and every configured official model, separate processes
  evaluate-custom        custom checkpoint only (used by matched controls)
  benchmark              custom and every configured official model for one thread count
  benchmark-matrix       benchmark 1, 4 and 8 threads (or only --threads N)
  benchmark-custom-matrix  custom-only 1, 4 and 8 thread matrix for matched controls
  report                 Markdown/HTML/CSV report and preregistered gates
  blind                  blinded A/B generation form
  smoke                  small online pipeline; checks plumbing only
  full                   download-full -> prepare -> train -> evaluate -> benchmark -> report

Common options:
  --profile smoke|calibration_cpu|cpu_quick|full_cpu|full_gpu|dgx_spark|dgx_spark_calibration|dense_active_cpu|dense_total_cpu|dgx_spark_dense_active|dgx_spark_dense_total|official_matrix
  --threads N
  --device cpu|cuda|mps|auto
  --dtype fp32|bf16|fp16|auto
`);
}

switch (command) {
  case 'doctor': invoke('doctor'); break;
  case 'doctor-dgx': invoke('doctor-dgx', common.rest); break;
  case 'download': freezeConfig(); invoke('download', ['--mode', 'auto']); break;
  case 'download-full': freezeConfig(); invoke('download', ['--mode', 'full']); break;
  case 'download-deployment': freezeConfig(); invoke('download', ['--mode', 'deployment']); break;
  case 'prepare': freezeConfig(); invoke('prepare', common.rest); break;
  case 'train': freezeConfig(); invoke('train', common.rest); break;
  case 'evaluate':
    freezeConfig();
    invoke('evaluate-custom', common.rest);
    evaluateAllOfficial();
    break;
  case 'evaluate-custom':
    freezeConfig();
    invoke('evaluate-custom', common.rest);
    break;
  case 'benchmark':
    {
      const config = freezeConfig({ benchmarkRuntime: true });
      const benchmarkCommon = {
        ...common,
        device: common.device ?? config.benchmark.primary_device,
        dtype: common.dtype ?? config.benchmark.primary_dtype,
      };
      const benchmarkBase = pythonBaseArgs(benchmarkCommon);
      runPython([...benchmarkBase, 'benchmark-custom', ...common.rest]);
      benchmarkAllOfficial(benchmarkBase);
    }
    break;
  case 'benchmark-matrix':
    benchmarkMatrix(common.rest);
    break;
  case 'benchmark-custom-matrix':
    benchmarkCustomMatrix(common.rest);
    break;
  case 'report':
    freezeConfig();
    process.env.VSA_PROFILE = common.profile;
    await import('./report.mjs');
    break;
  case 'blind':
    freezeConfig();
    process.env.VSA_PROFILE = common.profile;
    await import('./blind-eval.mjs');
    break;
  case 'smoke':
    freezeConfig();
    invoke('download', ['--mode', 'minimal']);
    invoke('prepare', ['--force']);
    invoke('train');
    invoke('evaluate-custom');
    evaluateAllOfficial();
    invoke('benchmark-custom');
    benchmarkAllOfficial(base);
    process.env.VSA_PROFILE = common.profile;
    await import('./report.mjs');
    await import('./blind-eval.mjs');
    break;
  case 'full':
    if (common.profile === 'dgx_spark') {
      throw new Error(
        'The dgx_spark protocol is staged: run download/prepare/train/evaluate on Spark, ' +
        'then benchmark/report on the designated x86 host. The monolithic full command is disabled.',
      );
    }
    freezeConfig();
    invoke('download', ['--mode', 'full']);
    invoke('prepare');
    invoke('train', common.rest);
    invoke('evaluate-custom');
    evaluateAllOfficial();
    benchmarkMatrix();
    process.env.VSA_PROFILE = common.profile;
    await import('./report.mjs');
    await import('./blind-eval.mjs');
    break;
  case 'help':
  case '--help':
  case '-h': help(); break;
  default:
    help();
    throw new Error(`Unknown command: ${command}`);
}
