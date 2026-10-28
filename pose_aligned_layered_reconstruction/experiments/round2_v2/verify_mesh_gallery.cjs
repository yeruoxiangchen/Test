// Offline interaction and file-pairing check; uses Node built-ins, no browser packages.
// This is not a browser screenshot test. It checks every supported selector state.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../../outputs/round2_v2_mesh_visual_audit_20261018');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.options = []; }
  appendChild(child) { this.children.push(child); if (this.tag === 'select') this.options.push(child); return child; }
  append(...children) { children.forEach(x => this.appendChild(x)); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  addEventListener() {}
}
const elements = new Map();
const ids = [...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]);
for (const id of ids) elements.set(id, new Element(['object','domain','seed','view','frame'].includes(id) ? 'select' : 'div'));
for (const [id, value] of Object.entries({ domain: 'legacy_composite', seed: '42', view: '0', frame: 'native' })) elements.get(id).value = value;
elements.get('frame').options = [{}, {}];
const context = vm.createContext({ document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag)}, console });
vm.runInContext(script, context);
const data = JSON.parse(fs.readFileSync(path.join(root, 'gallery_data.json'), 'utf8'));
const flatten = element => [element, ...element.children.filter(x => typeof x === 'object').flatMap(flatten)];
let states = 0, resources = 0;
for (const object of data.objects) for (const domain of ['legacy_clean', 'legacy_composite']) for (const seed of ['42','43']) for (const view of ['0','1','2']) for (const frame of (object.selected ? ['native','aligned'] : ['native'])) {
  for (const [id, value] of Object.entries({object:String(object.id),domain,seed,view,frame})) elements.get(id).value = value;
  vm.runInContext('update()', context);
  const row = data.rows.find(r => r.object_index === object.id && r.domain === domain);
  const cards = elements.get('cards').children;
  const overlays = elements.get('overlays').children;
  assert.equal(cards.length, 4); assert.equal(overlays.length, 3);
  assert.equal(elements.get('input').src, row.input);
  const padded = String(row.index).padStart(3, '0');
  const imageNames = cards.map(card => card.children.find(c => c.tag === 'img').src);
  assert.equal(imageNames[0], `visuals/renders/gt_${String(object.id).padStart(3,'0')}_v${view}.png`);
  ['parent','v1','raw'].forEach((mode,i) => {
    const basename = `${frame === 'aligned' ? 'aligned_' : ''}${padded}_${seed}_${mode}_v${view}.png`;
    assert.equal(imageNames[i+1], `visuals/renders/${basename}`);
    assert.equal(overlays[i].children.find(c => c.tag === 'img').src, `visuals/overlays/${basename}`);
    assert.equal(data.records[`${row.index}_${seed}_${mode}`].record, `alignment/records/${padded}_${seed}_${mode}.json`);
  });
  for (const node of [...flatten(elements.get('cards')), ...flatten(elements.get('overlays')), elements.get('input')]) {
    for (const key of ['src', 'href']) if (node[key]) { assert.ok(fs.existsSync(path.resolve(root, node[key])), `missing ${node[key]}`); resources++; }
  }
  states++;
}
// Unsupported aligned selections must explicitly revert to native.
const nonSelected = data.objects.find(o => !o.selected);
elements.get('object').value = String(nonSelected.id); elements.get('frame').value = 'aligned';
vm.runInContext('update()', context); assert.equal(elements.get('frame').value, 'native');
elements.get('object').value = '0'; elements.get('prev').onclick(); assert.equal(elements.get('object').value, '31');
elements.get('next').onclick(); assert.equal(elements.get('object').value, '0');
const report = { passed: true, interaction_states: states, resource_references_checked: resources,
  all_object_domain_seed_view_pairings: true, selected_aligned_views_checked: true,
  unsupported_aligned_reverts_to_native: true, previous_next_wrap_checked: true,
  test_scope: 'Node DOM stub and filesystem checks; no browser layout or screenshot validation' };
fs.writeFileSync(path.join(root, 'gallery_checks.json'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
