import fs from 'node:fs';
import path from 'node:path';
import { ROOT, runPython } from './lib.mjs';

for (const target of [
  path.join(ROOT, 'data', 'prepared', 'offline_fixture'),
  path.join(ROOT, 'runs', 'offline_fixture'),
]) fs.rmSync(target, { recursive: true, force: true });
const base = ['-m', 'vsa_bench.cli', '--profile', 'offline_fixture'];
runPython([...base, 'prepare', '--force']);
runPython([...base, 'train']);
runPython([...base, 'evaluate-custom']);
runPython([...base, 'benchmark-custom']);
console.log('Offline smoke pipeline passed. Official-model stages intentionally require network assets.');
