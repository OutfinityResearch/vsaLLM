import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { ROOT } from './lib.mjs';
import { validateBlindEvaluations } from './blind-lib.mjs';

const profileArg = process.argv.findIndex((item) => item === '--profile');
const profile = profileArg >= 0 ? process.argv[profileArg + 1] : (process.env.VSA_PROFILE || 'full_cpu');
const evalDir = path.join(ROOT, 'runs', profile, 'evaluation');
const resolvedPath = path.join(ROOT, 'runs', profile, 'resolved_config.json');
if (!fs.existsSync(resolvedPath)) throw new Error(`Missing frozen config ${resolvedPath}`);
const config = JSON.parse(fs.readFileSync(resolvedPath, 'utf8'));
const primaryModelId = config.comparison.primary_official_model_id;
const sampleIndex = Number(config.evaluation.blind_sample_index ?? 0);
const configuredModels = config.sources.official_models;
const primaryModelIndex = configuredModels.findIndex((item) => item.id === primaryModelId);
if (primaryModelIndex < 0) {
  throw new Error(`Primary baseline ${primaryModelId} is absent from the frozen config`);
}
const assetsPath = path.resolve(ROOT, config.paths.assets_manifest);
if (!fs.existsSync(assetsPath)) throw new Error(`Missing asset manifest ${assetsPath}`);
const assets = JSON.parse(fs.readFileSync(assetsPath, 'utf8'));
if (!Array.isArray(assets.official_models) || assets.official_models.length !== configuredModels.length) {
  throw new Error('Official model asset manifest does not match the frozen model list');
}
for (let index = 0; index < configuredModels.length; index += 1) {
  if (assets.official_models[index]?.id !== configuredModels[index].id) {
    throw new Error(`Official model asset manifest is stale or reordered at row ${index}`);
  }
}
const expectedOfficialRevision = assets.official_models[primaryModelIndex].resolved_sha;
if (!expectedOfficialRevision) {
  throw new Error(`Primary baseline ${primaryModelId} has no resolved asset revision`);
}
const customPath = path.join(evalDir, 'custom.json');
if (!fs.existsSync(customPath)) throw new Error(`Missing ${customPath}`);
const safePrimary = primaryModelId.replaceAll('/', '__');
const officialName = `official_${safePrimary}.json`;
const officialPath = path.join(evalDir, officialName);
if (!fs.existsSync(officialPath)) {
  throw new Error(`Evaluation artifact for primary baseline ${primaryModelId} is missing`);
}
const customEvaluation = JSON.parse(fs.readFileSync(customPath, 'utf8'));
const officialEvaluation = JSON.parse(fs.readFileSync(officialPath, 'utf8'));
const { pairs, promptManifestSha256 } = validateBlindEvaluations({
  profile,
  primaryModelId,
  expectedOfficialRevision,
  sampleIndex,
  expectedPromptCount: Number(config.evaluation.prompt_count),
  expectedSamplesPerPrompt: Number(config.evaluation.generation.samples_per_prompt),
  customEvaluation,
  officialEvaluation,
});
const sha256File = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const customEvaluationSha256 = sha256File(customPath);
const officialEvaluationSha256 = sha256File(officialPath);
const blindEvaluationId = crypto
  .createHash('sha256')
  .update(JSON.stringify({
    profile,
    primaryModelId,
    sampleIndex,
    promptManifestSha256,
    customEvaluationSha256,
    officialEvaluationSha256,
  }))
  .digest('hex');
const packet = [];
const key = [];
for (const row of pairs) {
  const hash = crypto
    .createHash('sha256')
    .update(`${blindEvaluationId}:${row.id}`)
    .digest();
  const customSide = hash[0] % 2 === 0 ? 'A' : 'B';
  packet.push({
    id: row.id,
    prompt: row.prompt,
    A: customSide === 'A' ? row.customText : row.officialText,
    B: customSide === 'B' ? row.customText : row.officialText,
  });
  key.push({ id: row.id, customSide });
}
const resultDir = path.join(ROOT, 'results');
fs.mkdirSync(resultDir, { recursive: true });
fs.writeFileSync(path.join(resultDir, `blind_key_${profile}.json`), JSON.stringify({
  profile,
  officialModelId: primaryModelId,
  generationChoice: `sample_${sampleIndex}`,
  blindEvaluationId,
  promptManifestSha256,
  provenance: {
    customEvaluationSha256,
    customCheckpointSha256: customEvaluation.model.checkpoint_sha256,
    officialEvaluationSha256,
    officialRevision: officialEvaluation.model.revision,
  },
  items: key,
}, null, 2));

const escaped = JSON.stringify(packet).replace(/</g, '\\u003c');
const html = `<!doctype html><html lang="ro"><meta charset="utf-8"><title>Evaluare oarbă ${profile}</title><style>body{font-family:system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#fafafa}.item{background:white;border:1px solid #ddd;border-radius:12px;padding:18px;margin:20px 0}.prompt{white-space:pre-wrap;background:#f2f2f2;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.candidate{white-space:pre-wrap;border:1px solid #ccc;padding:12px;min-height:140px}.choice{display:flex;gap:18px;margin-top:12px}button{padding:12px 18px;font-size:16px}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><body><h1>Evaluare oarbă a continuărilor</h1><p>Judecă numai coerența, gramatica, continuitatea și lipsa repetițiilor. Identitatea modelelor nu este inclusă în acest fișier.</p><div id="items"></div><button id="export">Exportă scorurile JSON</button><script>const packet=${escaped};const root=document.getElementById('items');for(const row of packet){const el=document.createElement('section');el.className='item';el.innerHTML='<h2>Exemplul '+row.id+'</h2><div class="prompt"></div><div class="grid"><div><h3>A</h3><div class="candidate a"></div></div><div><h3>B</h3><div class="candidate b"></div></div></div><div class="choice"><label><input type="radio" name="c'+row.id+'" value="A"> A mai bun</label><label><input type="radio" name="c'+row.id+'" value="B"> B mai bun</label><label><input type="radio" name="c'+row.id+'" value="tie"> Egalitate</label></div><textarea rows="2" style="width:100%;margin-top:10px" placeholder="Notă opțională"></textarea>';el.querySelector('.prompt').textContent=row.prompt;el.querySelector('.a').textContent=row.A;el.querySelector('.b').textContent=row.B;root.appendChild(el)}document.getElementById('export').onclick=()=>{const items=packet.map(row=>{const section=[...document.querySelectorAll('.item')].find(x=>x.querySelector('h2').textContent.endsWith(String(row.id)));return{id:row.id,choice:section.querySelector('input:checked')?.value??null,note:section.querySelector('textarea').value}});const blob=new Blob([JSON.stringify({profile:'${profile}',blindEvaluationId:'${blindEvaluationId}',promptManifestSha256:'${promptManifestSha256}',items},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='blind_scores_${profile}.json';a.click();URL.revokeObjectURL(a.href)};</script></body></html>`;
const out = path.join(resultDir, `blind_eval_${profile}.html`);
fs.writeFileSync(out, html);
console.log(`Created ${out}`);
console.log(`Keep results/blind_key_${profile}.json hidden from evaluators.`);
