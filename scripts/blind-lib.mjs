import crypto from 'node:crypto';

function fail(message) {
  throw new Error(`Invalid blind-evaluation inputs: ${message}`);
}

function generationSuite(evaluation, label) {
  if (!evaluation || typeof evaluation !== 'object' || Array.isArray(evaluation)) {
    fail(`${label} evaluation is not an object`);
  }
  const suite = evaluation.generations;
  if (!suite || typeof suite !== 'object' || !Array.isArray(suite.items)) {
    fail(`${label} evaluation has no embedded generation suite`);
  }
  return suite;
}

function selectedText(row, sampleIndex, expectedSamples, label) {
  if (!row || typeof row !== 'object') fail(`${label} generation row is invalid`);
  if (typeof row.prompt !== 'string') fail(`${label} prompt is not a string`);
  if (!Number.isInteger(Number(row.id))) fail(`${label} id ${row.id} is not an integer`);
  if (!row.greedy || typeof row.greedy.text !== 'string') {
    fail(`${label} row ${row.id} has no greedy text`);
  }
  if (!Array.isArray(row.samples) || row.samples.length !== expectedSamples) {
    fail(
      `${label} row ${row.id} has ${row.samples?.length ?? 0} samples; ` +
      `expected ${expectedSamples}`,
    );
  }
  const selected = row.samples[sampleIndex];
  if (!selected || typeof selected.text !== 'string') {
    fail(`${label} row ${row.id} has no sample at index ${sampleIndex}`);
  }
  return selected.text;
}

export function validateBlindEvaluations({
  profile,
  primaryModelId,
  expectedOfficialRevision,
  sampleIndex,
  expectedPromptCount,
  expectedSamplesPerPrompt,
  customEvaluation,
  officialEvaluation,
}) {
  if (!Number.isInteger(sampleIndex) || sampleIndex < 0) {
    fail(`blind_sample_index must be a non-negative integer, got ${sampleIndex}`);
  }
  if (sampleIndex >= expectedSamplesPerPrompt) {
    fail(
      `blind_sample_index ${sampleIndex} is outside samples_per_prompt=` +
      `${expectedSamplesPerPrompt}`,
    );
  }
  if (customEvaluation.profile !== profile || officialEvaluation.profile !== profile) {
    fail(
      `artifact profiles ${JSON.stringify(customEvaluation.profile)}/` +
      `${JSON.stringify(officialEvaluation.profile)} do not both match ` +
      `${JSON.stringify(profile)}`,
    );
  }
  const customKind = customEvaluation.model?.kind;
  if (!['custom_vsa_pathmoe', 'matched_dense_control'].includes(customKind)) {
    fail(`custom artifact model kind is ${customKind ?? 'missing'}`);
  }
  if (!customEvaluation.model?.checkpoint_sha256) {
    fail('custom artifact has no checkpoint SHA-256 provenance');
  }
  if (officialEvaluation.model?.kind !== 'official_huggingface') {
    fail('official artifact model kind is missing or invalid');
  }
  if (officialEvaluation.model?.model_id !== primaryModelId) {
    fail(
      `official artifact is ${officialEvaluation.model?.model_id ?? 'unknown'}, ` +
      `expected ${primaryModelId}`,
    );
  }
  if (!officialEvaluation.model?.revision) {
    fail('official artifact has no immutable revision provenance');
  }
  if (
    expectedOfficialRevision &&
    officialEvaluation.model.revision !== expectedOfficialRevision
  ) {
    fail(
      `official artifact revision ${officialEvaluation.model.revision} does not match ` +
      `current asset revision ${expectedOfficialRevision}`,
    );
  }

  const custom = generationSuite(customEvaluation, 'custom');
  const official = generationSuite(officialEvaluation, 'official');
  for (const [label, suite] of [['custom', custom], ['official', official]]) {
    if (Number(suite.prompt_count) !== suite.items.length) {
      fail(`${label} declared prompt_count does not match its item count`);
    }
    if (suite.items.length !== expectedPromptCount) {
      fail(
        `${label} has ${suite.items.length} prompts; expected ${expectedPromptCount} ` +
        'from the frozen configuration',
      );
    }
    if (Number(suite.samples_per_prompt) !== expectedSamplesPerPrompt) {
      fail(
        `${label} samples_per_prompt=${suite.samples_per_prompt}; expected ` +
        `${expectedSamplesPerPrompt}`,
      );
    }
  }

  const seen = new Set();
  const pairs = custom.items.map((row, index) => {
    const other = official.items[index];
    const id = Number(row.id);
    if (seen.has(id)) fail(`duplicate custom id ${id}`);
    seen.add(id);
    if (Number(other?.id) !== id) {
      fail(`row ${index} ids are not aligned: ${row.id}/${other?.id}`);
    }
    if (other.prompt !== row.prompt) {
      fail(`row ${id} prompts differ between custom and official artifacts`);
    }
    return {
      id,
      prompt: row.prompt,
      customText: selectedText(
        row, sampleIndex, expectedSamplesPerPrompt, 'custom',
      ),
      officialText: selectedText(
        other, sampleIndex, expectedSamplesPerPrompt, 'official',
      ),
    };
  });
  const promptManifest = pairs.map(({ id, prompt }) => ({ id, prompt }));
  const promptManifestSha256 = crypto
    .createHash('sha256')
    .update(JSON.stringify(promptManifest))
    .digest('hex');
  return { pairs, promptManifestSha256 };
}
