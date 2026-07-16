import assert from 'node:assert/strict';
import test from 'node:test';

import { validateBlindEvaluations } from '../scripts/blind-lib.mjs';

function evaluation(kind, prompts = ['First prompt', 'Second prompt']) {
  const official = kind === 'official_huggingface';
  return {
    profile: 'unit',
    model: official
      ? { kind, model_id: 'example/Tiny-8M', revision: 'immutable-sha' }
      : { kind, checkpoint_sha256: 'checkpoint-sha' },
    generations: {
      prompt_count: prompts.length,
      samples_per_prompt: 2,
      items: prompts.map((prompt, id) => ({
        id,
        prompt,
        greedy: { text: `greedy-${kind}-${id}` },
        samples: [
          { text: `sample-0-${kind}-${id}` },
          { text: `sample-1-${kind}-${id}` },
        ],
      })),
    },
  };
}

function inputs() {
  return {
    profile: 'unit',
    primaryModelId: 'example/Tiny-8M',
    expectedOfficialRevision: 'immutable-sha',
    sampleIndex: 1,
    expectedPromptCount: 2,
    expectedSamplesPerPrompt: 2,
    customEvaluation: evaluation('custom_vsa_pathmoe'),
    officialEvaluation: evaluation('official_huggingface'),
  };
}

test('blind evaluation requires aligned full-provenance artifacts', () => {
  const result = validateBlindEvaluations(inputs());
  assert.equal(result.pairs.length, 2);
  assert.equal(result.pairs[0].customText, 'sample-1-custom_vsa_pathmoe-0');
  assert.equal(result.pairs[0].officialText, 'sample-1-official_huggingface-0');
  assert.match(result.promptManifestSha256, /^[0-9a-f]{64}$/);
});

test('blind evaluation rejects prompt and id misalignment', () => {
  const promptMismatch = inputs();
  promptMismatch.officialEvaluation.generations.items[0].prompt = 'stale prompt';
  assert.throws(
    () => validateBlindEvaluations(promptMismatch),
    /prompts differ/,
  );

  const idMismatch = inputs();
  idMismatch.officialEvaluation.generations.items[1].id = 7;
  assert.throws(() => validateBlindEvaluations(idMismatch), /ids are not aligned/);
});

test('blind evaluation rejects stale provenance and missing selected samples', () => {
  const wrongModel = inputs();
  wrongModel.officialEvaluation.model.model_id = 'example/Other';
  assert.throws(() => validateBlindEvaluations(wrongModel), /expected example\/Tiny-8M/);

  const wrongProfile = inputs();
  wrongProfile.customEvaluation.profile = 'old-run';
  assert.throws(() => validateBlindEvaluations(wrongProfile), /artifact profiles/);

  const staleRevision = inputs();
  staleRevision.officialEvaluation.model.revision = 'old-sha';
  assert.throws(() => validateBlindEvaluations(staleRevision), /current asset revision/);

  const missingSample = inputs();
  missingSample.customEvaluation.generations.items[0].samples.pop();
  assert.throws(() => validateBlindEvaluations(missingSample), /has 1 samples/);
});

test('blind evaluation rejects counts that differ from frozen config', () => {
  const staleCount = inputs();
  staleCount.expectedPromptCount = 3;
  assert.throws(() => validateBlindEvaluations(staleCount), /expected 3/);

  const badIndex = inputs();
  badIndex.sampleIndex = 2;
  assert.throws(() => validateBlindEvaluations(badIndex), /outside samples_per_prompt/);
});
