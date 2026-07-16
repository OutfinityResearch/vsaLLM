import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function pythonExecutable() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const candidates = process.platform === 'win32'
    ? [path.join(ROOT, '.venv', 'Scripts', 'python.exe'), 'python']
    : [path.join(ROOT, '.venv', 'bin', 'python'), 'python3', 'python'];
  for (const candidate of candidates) {
    if (candidate.includes(path.sep) && !fs.existsSync(candidate)) continue;
    const test = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (test.status === 0) return candidate;
  }
  throw new Error('Python not found. Set PYTHON or run npm run setup.');
}

export function parseCommonArgs(argv) {
  const out = { profile: process.env.VSA_PROFILE || 'smoke', threads: null, device: null, dtype: null, rest: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (['--profile', '--threads', '--device', '--dtype'].includes(item)) {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${item}`);
      const value = argv[++i];
      if (item === '--profile') out.profile = value;
      else if (item === '--threads') out.threads = Number(value);
      else if (item === '--device') out.device = value;
      else if (item === '--dtype') out.dtype = value;
    } else {
      out.rest.push(item);
    }
  }
  return out;
}

export function pythonBaseArgs(common) {
  const args = ['-m', 'vsa_bench.cli', '--profile', common.profile];
  if (common.device) args.push('--device', common.device);
  if (common.dtype) args.push('--dtype', common.dtype);
  if (common.threads != null) args.push('--threads', String(common.threads));
  return args;
}

export function comparableRunConfig(config, { ignoreBenchmarkRuntime = false } = {}) {
  const copy = structuredClone(config);
  if (ignoreBenchmarkRuntime && copy.runtime && typeof copy.runtime === 'object') {
    delete copy.runtime.threads;
    delete copy.runtime.device;
    delete copy.runtime.dtype;
  }
  return copy;
}

export function runPython(args, { capture = false, env = {} } = {}) {
  const python = pythonExecutable();
  const result = spawnSync(python, args, {
    cwd: ROOT,
    env: {
      ...process.env,
      PYTHONPATH: [path.join(ROOT, 'python'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      ...env,
    },
    encoding: capture ? 'utf8' : undefined,
    stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Python failed (${result.status}): ${python} ${args.join(' ')}`);
  return capture ? result.stdout : null;
}
