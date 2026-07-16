import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './lib.mjs';

const manifestPath = path.join(ROOT, 'MANIFEST_SHA256.txt');
if (!fs.existsSync(manifestPath)) {
  throw new Error('MANIFEST_SHA256.txt is missing. Use the packaged archive, not an unfinished source directory.');
}

const lines = fs.readFileSync(manifestPath, 'utf8')
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter(Boolean);
let checked = 0;
for (const line of lines) {
  const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
  if (!match) throw new Error(`Malformed manifest line: ${line}`);
  const [, expected, relative] = match;
  const target = path.resolve(ROOT, relative);
  if (!target.startsWith(`${ROOT}${path.sep}`)) throw new Error(`Unsafe manifest path: ${relative}`);
  if (!fs.existsSync(target) || !fs.statSync(target).isFile()) throw new Error(`Missing file: ${relative}`);
  const hash = crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex');
  if (hash !== expected) throw new Error(`Checksum mismatch: ${relative}`);
  checked += 1;
}
console.log(`Package integrity OK: ${checked} files verified.`);
