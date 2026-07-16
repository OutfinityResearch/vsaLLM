import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { spawnSync } from 'node:child_process';
import { ROOT } from './lib.mjs';

function run(command, args) {
  console.log(`> ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { cwd: ROOT, stdio: 'inherit', env: process.env });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`Command failed with ${result.status}`);
}

const systemPython = process.env.PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
const venv = path.join(ROOT, '.venv');
const vpy = process.platform === 'win32'
  ? path.join(venv, 'Scripts', 'python.exe')
  : path.join(venv, 'bin', 'python');
if (!fs.existsSync(vpy)) run(systemPython, ['-m', 'venv', venv]);
run(vpy, ['-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel']);
const lockedCpu = process.argv.includes('--locked-cpu');
if (lockedCpu) {
  run(vpy, [
    '-m', 'pip', 'install', '--force-reinstall', 'torch==2.10.0',
    '--index-url', 'https://download.pytorch.org/whl/cpu',
  ]);
  run(vpy, [
    '-c',
    "import torch; assert torch.__version__.split('+')[0] == '2.10.0'; assert torch.version.cuda is None",
  ]);
} else if (process.argv.includes('--cpu-torch')) {
  run(vpy, ['-m', 'pip', 'install', 'torch', '--index-url', 'https://download.pytorch.org/whl/cpu']);
} else {
  const probe = spawnSync(vpy, ['-c', 'import torch'], { stdio: 'ignore' });
  if (probe.status !== 0) {
    console.log('PyTorch is absent. Installing the default PyPI build. For a specific CUDA build, install it manually before rerunning setup.');
    run(vpy, ['-m', 'pip', 'install', 'torch']);
  }
}
if (lockedCpu) {
  run(vpy, ['-m', 'pip', 'install', '-r', 'requirements-dgx.txt']);
  run(vpy, ['-m', 'pip', 'install', '--no-deps', '-e', '.']);
  run(vpy, ['-m', 'pip', 'check']);
} else {
  run(vpy, ['-m', 'pip', 'install', '-e', '.', 'pytest']);
}
console.log('\nSetup complete. Next: npm run doctor -- --profile smoke');
