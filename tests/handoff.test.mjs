import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { ROOT } from '../scripts/lib.mjs';
import { handoffRelative, sha256File } from '../scripts/handoff.mjs';

test('handoff paths cannot escape the repository', () => {
  assert.equal(handoffRelative('runs/dgx_spark'), 'runs/dgx_spark');
  assert.throws(() => handoffRelative('../outside'), /inside the repository/);
  assert.throws(() => handoffRelative(ROOT), /inside the repository/);
});

test('handoff hashes files without loading a transfer bundle', async () => {
  const file = path.join(ROOT, 'package.json');
  const expected = crypto.createHash('sha256').update(await readFile(file)).digest('hex');
  assert.equal(await sha256File(file), expected);
});
