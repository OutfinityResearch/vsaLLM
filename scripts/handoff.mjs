import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ROOT } from './lib.mjs';

const SCHEMA_VERSION = 1;

function argument(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  if (index + 1 >= process.argv.length) throw new Error(`Missing value for ${name}`);
  return process.argv[index + 1];
}

function relative(value) {
  const absolute = path.resolve(ROOT, value);
  const result = path.relative(ROOT, absolute);
  if (!result || result.startsWith('..') || path.isAbsolute(result)) {
    throw new Error(`Handoff path must remain inside the repository: ${value}`);
  }
  return result.split(path.sep).join('/');
}

function readJson(file, label) {
  if (!fs.existsSync(file)) throw new Error(`Missing ${label}: ${file}`);
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function git(args) {
  const result = spawnSync('git', args, { cwd: ROOT, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`git ${args.join(' ')} failed`);
  return result.stdout.trim();
}

function regularFiles(root) {
  const output = [];
  const visit = (current) => {
    const stat = fs.lstatSync(current);
    if (stat.isSymbolicLink()) throw new Error(`Symlinks are not allowed in handoff: ${current}`);
    if (stat.isFile()) {
      output.push(current);
      return;
    }
    if (!stat.isDirectory()) throw new Error(`Unsupported handoff entry: ${current}`);
    for (const entry of fs.readdirSync(current).sort()) visit(path.join(current, entry));
  };
  visit(root);
  return output;
}

function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(file);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function expectedRoots(profile) {
  if (!/^[a-z0-9_]+$/.test(String(profile))) {
    throw new Error(`Unsafe profile name: ${profile}`);
  }
  const mainRun = path.join(ROOT, 'runs', profile);
  const config = readJson(
    path.join(mainRun, 'resolved_config.json'),
    `${profile} frozen configuration`,
  );
  const controls = Object.values(config.comparison?.matched_control_profiles ?? {});
  if (new Set(controls).size !== controls.length) {
    throw new Error('Matched-control profile names must be unique');
  }
  for (const control of controls) {
    if (!/^[a-z0-9_]+$/.test(String(control))) {
      throw new Error(`Unsafe matched-control profile name: ${control}`);
    }
  }
  const profiles = [profile, ...controls];
  const prepared = relative(config.data.prepared_dir);
  return { config, profiles, roots: [prepared, ...profiles.map((item) => `runs/${item}`)] };
}

function validateRequiredArtifacts(config, profiles, preparedRoot) {
  const prepared = path.join(ROOT, preparedRoot);
  const metadata = readJson(path.join(prepared, 'metadata.json'), 'prepared metadata');
  if (metadata.format_version !== 5) {
    throw new Error(`Handoff requires prepared-data format 5, found ${metadata.format_version}`);
  }
  for (const profile of profiles) {
    const run = path.join(ROOT, 'runs', profile);
    for (const required of [
      'resolved_config.json',
      'artifacts/training_summary.json',
      'checkpoints/best.pt',
      'checkpoints/last.pt',
      'evaluation/custom.json',
    ]) {
      const requiredPath = path.join(run, required);
      if (!fs.existsSync(requiredPath) || !fs.statSync(requiredPath).isFile()) {
        throw new Error(`Missing handoff artifact: runs/${profile}/${required}`);
      }
    }
  }
  for (const item of config.sources.official_models) {
    const safe = item.id.replaceAll('/', '__');
    const file = path.join(ROOT, 'runs', profiles[0], 'evaluation', `official_${safe}.json`);
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      throw new Error(`Missing official evaluation: ${file}`);
    }
  }
}

async function create(profile, manifestPath) {
  if (git(['status', '--porcelain', '--untracked-files=normal'])) {
    throw new Error('Create the official handoff only from a clean Git worktree');
  }
  const { config, profiles, roots } = expectedRoots(profile);
  validateRequiredArtifacts(config, profiles, roots[0]);
  const files = roots
    .flatMap((root) => regularFiles(path.join(ROOT, root)))
    .sort((left, right) => relative(left).localeCompare(relative(right)));
  const entries = [];
  for (const file of files) {
    entries.push({
      path: relative(file),
      bytes: fs.statSync(file).size,
      sha256: await sha256File(file),
    });
  }
  const manifest = {
    schema_version: SCHEMA_VERSION,
    profile,
    git_commit: git(['rev-parse', 'HEAD']),
    prepared_format_version: 5,
    roots,
    profiles,
    files: entries,
  };
  const target = path.resolve(ROOT, manifestPath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const temporary = `${target}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(manifest, null, 2)}\n`);
  fs.renameSync(temporary, target);
  const manifestDigest = await sha256File(target);
  console.log(`Wrote ${entries.length} hashes to ${path.relative(ROOT, target)}`);
  console.log(`Manifest SHA-256: ${manifestDigest}`);
}

async function verify(manifestPath) {
  const target = path.resolve(ROOT, manifestPath);
  const manifest = readJson(target, 'handoff manifest');
  if (manifest.schema_version !== SCHEMA_VERSION) throw new Error('Unsupported handoff schema');
  if (manifest.git_commit !== git(['rev-parse', 'HEAD'])) {
    throw new Error(`Checkout ${manifest.git_commit} before verifying this handoff`);
  }
  const expected = expectedRoots(manifest.profile);
  if (JSON.stringify(manifest.roots) !== JSON.stringify(expected.roots)
      || JSON.stringify(manifest.profiles) !== JSON.stringify(expected.profiles)) {
    throw new Error('Handoff roots/profiles differ from the frozen main configuration');
  }
  if (manifest.prepared_format_version !== 5) {
    throw new Error('Handoff prepared format is not version 5');
  }
  if (!Array.isArray(manifest.files) || !manifest.files.length) {
    throw new Error('Handoff manifest has no files');
  }
  const actualPaths = expected.roots
    .flatMap((root) => regularFiles(path.join(ROOT, root)))
    .map(relative)
    .sort();
  const manifestPaths = manifest.files.map((entry) => relative(entry.path)).sort();
  if (new Set(manifestPaths).size !== manifestPaths.length) {
    throw new Error('Handoff manifest contains duplicate paths');
  }
  if (JSON.stringify(manifestPaths) !== JSON.stringify(actualPaths)) {
    throw new Error('Transferred roots contain missing or unmanifested files');
  }
  for (const entry of manifest.files) {
    const entryPath = relative(entry.path);
    if (!expected.roots.some(
      (root) => entryPath === root || entryPath.startsWith(`${root}/`),
    )) {
      throw new Error(`Manifest path is outside declared roots: ${entry.path}`);
    }
    if (!Number.isSafeInteger(entry.bytes) || entry.bytes < 0) {
      throw new Error(`Invalid byte count: ${entry.path}`);
    }
    if (!/^[0-9a-f]{64}$/.test(entry.sha256)) {
      throw new Error(`Invalid SHA-256: ${entry.path}`);
    }
    const file = path.join(ROOT, entryPath);
    if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
      throw new Error(`Missing transferred file: ${entry.path}`);
    }
    const bytes = fs.statSync(file).size;
    if (bytes !== entry.bytes) throw new Error(`Size mismatch: ${entry.path}`);
    const digest = await sha256File(file);
    if (digest !== entry.sha256) throw new Error(`SHA-256 mismatch: ${entry.path}`);
  }
  validateRequiredArtifacts(expected.config, expected.profiles, expected.roots[0]);
  console.log(`Verified ${manifest.files.length} files for ${manifest.profile}`);
  console.log(`Manifest SHA-256: ${await sha256File(target)}`);
}

export { relative as handoffRelative, sha256File };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [command = 'help'] = process.argv.slice(2);
  const profile = argument('--profile', 'dgx_spark');
  const manifestPath = argument('--manifest', `handoff/${profile}_manifest.json`);
  if (command === 'create') await create(profile, manifestPath);
  else if (command === 'verify') await verify(manifestPath);
  else {
    console.log('Usage: node scripts/handoff.mjs create|verify [--profile dgx_spark] [--manifest path]');
    if (command !== 'help') throw new Error(`Unknown handoff command: ${command}`);
  }
}
