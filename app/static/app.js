(() => {
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const h = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) e.setAttribute(k, v);
  e.append(...kids);
  return e;
};

// All placement values are in full-resolution scene pixels, the same units the server renders in.
const S = {
  sid: null, scene: { w: 0, h: 0 }, objPhoto: { w: 0, h: 0 }, aspect: 1,
  P: null, D: null, devices: [], refs: [], pxPerMm: null,
  tool: null, picks: [], ext: [],
};
window.flatlay = S;   // handy in the browser console when tuning parameters

// ---------------------------------------------------------------------------------------------
// Start screen
// ---------------------------------------------------------------------------------------------
const files = { scene: null, object: null };

function wireDrop(kind, labelSel, inputSel, thumbSel) {
  const label = $(labelSel), input = $(inputSel), thumb = $(thumbSel);
  const take = (f) => {
    if (!f) return;
    files[kind] = f;
    thumb.src = URL.createObjectURL(f);
    thumb.hidden = false;
    label.classList.add('has');
    $('#go').disabled = !(files.scene && files.object);
  };
  input.addEventListener('change', () => take(input.files[0]));
  label.addEventListener('dragover', (e) => { e.preventDefault(); label.classList.add('over'); });
  label.addEventListener('dragleave', () => label.classList.remove('over'));
  label.addEventListener('drop', (e) => {
    e.preventDefault();
    label.classList.remove('over');
    take(e.dataTransfer.files[0]);
  });
}
wireDrop('scene', '#dropScene', '#fileScene', '#thumbScene');
wireDrop('object', '#dropObject', '#fileObject', '#thumbObject');

async function jsonOrThrow(r) {
  if (r.ok) return r.json();
  let msg = `Request failed (${r.status})`;
  try { msg = (await r.json()).error || msg; } catch (_) { /* body was not JSON */ }
  throw new Error(msg);
}
const post = (path, body) => fetch(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
}).then(jsonOrThrow);

$('#go').addEventListener('click', async () => {
  const msg = $('#startMsg');
  msg.className = '';
  msg.textContent = 'Cutting out the object. This takes a few seconds on large photos.';
  $('#go').disabled = true;
  try {
    const fd = new FormData();
    fd.append('scene', files.scene);
    fd.append('object', files.object);
    const [devs, sess] = await Promise.all([
      fetch('/api/devices').then(jsonOrThrow),
      fetch('/api/session', { method: 'POST', body: fd }).then(jsonOrThrow),
    ]);
    S.devices = devs.devices; S.refs = devs.references;
    S.analysis = sess.analysis; S.cutout = sess.cutout; S.sid = sess.id; S.scene = sess.scene; S.objPhoto = sess.object; S.aspect = sess.aspect;
    S.D = sess.params; S.P = JSON.parse(JSON.stringify(sess.params));
    S.pxPerMm = null; S.ext = [];
    openEditor();
  } catch (e) {
    msg.className = 'err';
    msg.textContent = e.message;
    $('#go').disabled = false;
  }
});

// ---------------------------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------------------------
const frame = $('#frame'), view = $('#view'), hit = $('#hit'), box = $('#box');
const syncers = [];
let uid = 0;

function openEditor() {
  $('#start').hidden = true;
  $('#editor').hidden = false;
  buildControls();
  fitFrame();
  schedule();
}

function status(text, err = false) {
  const s = $('#status');
  s.textContent = text;
  s.className = err ? 'err' : '';
}

function slider(parent, { label, min, max, step, get, set, def, fmt }) {
  const id = 's' + (++uid);
  const out = h('output', { for: id });
  const inp = h('input', { type: 'range', id, min, max, step });
  const show = () => { out.textContent = fmt(get()); };
  const sync = () => { inp.value = get(); show(); };
  inp.addEventListener('input', () => { set(parseFloat(inp.value)); show(); schedule(true); });
  inp.addEventListener('change', () => schedule());
  if (def) inp.addEventListener('dblclick', () => { set(def()); sync(); schedule(); });
  inp.title = def ? 'Double-click to reset' : '';
  parent.append(h('div', { class: 'slider' }, h('label', { for: id }, label), out, inp));
  syncers.push(sync);
  sync();
}

const f2 = (v) => v.toFixed(2);
const px = (v) => Math.round(v) + ' px';

function buildControls() {
  syncers.length = 0;
  for (const id of ['#gPlace', '#gLight', '#gPersp', '#gTone']) $(id).replaceChildren();
  const { w: W, h: H } = S.scene;
  const P = () => S.P, D = () => S.D;

  const place = $('#gPlace');
  slider(place, { label: 'Size (long side)', min: Math.round(0.04 * H), max: Math.round(1.8 * H), step: 1, fmt: px,
    get: () => P().height_px, set: (v) => { P().height_px = v; } });
  slider(place, { label: 'Rotation', min: -180, max: 180, step: 0.1, fmt: (v) => v.toFixed(1) + '°',
    get: () => P().rotation_deg, set: (v) => { P().rotation_deg = v; }, def: () => 0 });
  slider(place, { label: 'Position, across', min: Math.round(-0.3 * W), max: Math.round(1.3 * W), step: 1, fmt: px,
    get: () => P().cx, set: (v) => { P().cx = v; } });
  slider(place, { label: 'Position, down', min: Math.round(-0.3 * H), max: Math.round(1.3 * H), step: 1, fmt: px,
    get: () => P().cy, set: (v) => { P().cy = v; } });

  const light = $('#gLight');
  slider(light, { label: 'Shadow length', min: 0, max: 2, step: 0.01, fmt: (v) => f2(v) + ' × thickness',
    get: () => P().shadow_len, set: (v) => { P().shadow_len = v; }, def: () => D().shadow_len });
  slider(light, { label: 'Shadow softness', min: 0.1, max: 3, step: 0.01, fmt: f2,
    get: () => P().shadow_soft, set: (v) => { P().shadow_soft = v; }, def: () => D().shadow_soft });
  slider(light, { label: 'Shadow strength', min: 0, max: 1.5, step: 0.01, fmt: f2,
    get: () => P().shadow_strength, set: (v) => { P().shadow_strength = v; }, def: () => D().shadow_strength });
  slider(light, { label: 'Contact shadow', min: 0, max: 2, step: 0.01, fmt: f2,
    get: () => P().contact, set: (v) => { P().contact = v; }, def: () => D().contact });

  const persp = $('#gPersp');
  slider(persp, { label: 'Camera height', min: 0.3, max: 4, step: 0.01, fmt: (v) => f2(v) + ' × scene width',
    get: () => P().cam_height, set: (v) => { P().cam_height = v; }, def: () => D().cam_height });
  slider(persp, { label: 'Side brightness', min: 0.3, max: 2.2, step: 0.01, fmt: f2,
    get: () => P().side_brightness, set: (v) => { P().side_brightness = v; }, def: () => 1 });

  const tone = $('#gTone');
  const blackK = () => (D().black_rgb[0] > 0.5 ? P().black_rgb[0] / D().black_rgb[0] : 1);
  slider(tone, { label: 'Black level', min: 0, max: 3, step: 0.01, fmt: (v) => '× ' + f2(v),
    get: blackK, set: (v) => { P().black_rgb = D().black_rgb.map((c) => c * v); }, def: () => 1 });
  slider(tone, { label: 'Brightness', min: 0.5, max: 1.4, step: 0.005, fmt: f2,
    get: () => P().gain, set: (v) => { P().gain = v; }, def: () => D().gain });
  slider(tone, { label: 'Rim light', min: 0, max: 2.5, step: 0.01, fmt: f2,
    get: () => P().rim, set: (v) => { P().rim = v; }, def: () => D().rim });
  slider(tone, { label: 'Sheen', min: 0, max: 3, step: 0.01, fmt: f2,
    get: () => P().sheen, set: (v) => { P().sheen = v; }, def: () => D().sheen });
  slider(tone, { label: 'Softness', min: 0, max: 3, step: 0.05, fmt: (v) => v.toFixed(2) + ' px',
    get: () => P().softness, set: (v) => { P().softness = v; }, def: () => D().softness });
  slider(tone, { label: 'Grain', min: 0, max: 6, step: 0.1, fmt: (v) => v.toFixed(1),
    get: () => P().grain, set: (v) => { P().grain = v; }, def: () => D().grain });

  slider($('#gGlass'), { label: 'Glass opacity', min: 0.05, max: 0.7, step: 0.01, fmt: f2,
    get: () => P().glass_opacity, set: (v) => { P().glass_opacity = v; }, def: () => D().glass_opacity });

  buildDevices();
  syncFields();
  updateLightInfo();
  updateCutNote();
}

function syncAll() { syncers.forEach((f) => f()); syncFields(); positionOverlay(); }

function syncFields() {
  $('#lengthMm').value = S.P.length_mm;
  $('#thickMm').value = S.P.thickness_mm;
  $('#sideFaces').checked = S.P.side_faces;
  $('#material').value = S.P.side_material;
  $('#shape').value = S.P.shape;
  $('#finish').value = S.P.finish;
  $('#standoffMm').value = S.P.standoff_mm;
  $('#flipH').checked = S.P.flip_h;
  $('#glassBox').hidden = S.P.finish !== 'glass';
  $('#shadowSwatch').style.background = `rgb(${S.P.shadow_rgb.map((t) => Math.round(clamp(t, 0, 1) * 255)).join(',')})`;
  $('#applyReal').hidden = !S.pxPerMm;
  $('#scaleNote').textContent = S.pxPerMm
    ? `Scale: ${S.pxPerMm.toFixed(2)} px per mm in this photo.`
    : 'Drag the object to move it, scroll over it to resize.';
  updateDial();
}

function buildDevices() {
  const sel = $('#device');
  sel.replaceChildren(...S.devices.map((d) => h('option', { value: d.id }, d.name)));
  sel.value = S.deviceId || 'custom';
}

$('#device').addEventListener('change', (e) => {
  const d = S.devices.find((x) => x.id === e.target.value);
  if (!d) return;
  S.deviceId = d.id;
  S.P.length_mm = d.length_mm; S.P.thickness_mm = d.thickness_mm;
  S.P.side_material = d.side_material; S.P.clear_edge_frac = d.clear_edge_frac || 0;
  S.P.shape = d.shape || 'slab'; S.P.standoff_mm = d.standoff_mm || 0; S.P.finish = d.finish || 'glossy';
  if (S.pxPerMm) S.P.height_px = d.length_mm * S.pxPerMm;
  syncAll(); schedule();
});
const bindSelect = (id, key) => $(id).addEventListener('change', (e) => { S.P[key] = e.target.value; syncAll(); schedule(); });
bindSelect('#shape', 'shape');
bindSelect('#finish', 'finish');
$('#standoffMm').addEventListener('input', (e) => { S.P.standoff_mm = Math.max(0, parseFloat(e.target.value) || 0); schedule(true); });
$('#standoffMm').addEventListener('change', () => schedule());
$('#flipH').addEventListener('change', (e) => { S.P.flip_h = e.target.checked; schedule(); });
// Typing a size means the object is no longer the chosen preset, and, like picking a preset, it takes that real
// size on screen when the photo's scale is known. syncFields is skipped so a half-typed "0." is not rewritten.
function customSizeEdited() {
  S.deviceId = 'custom';
  $('#device').value = 'custom';
  if (S.pxPerMm) S.P.height_px = S.P.length_mm * S.pxPerMm;
  syncers.forEach((f) => f());
  positionOverlay();
  schedule(true);
}
$('#lengthMm').addEventListener('input', (e) => { S.P.length_mm = Math.max(0.1, parseFloat(e.target.value) || 0.1); customSizeEdited(); });
$('#thickMm').addEventListener('input', (e) => { S.P.thickness_mm = Math.max(0.1, parseFloat(e.target.value) || 0.1); customSizeEdited(); });
$('#lengthMm').addEventListener('change', () => schedule());
$('#thickMm').addEventListener('change', () => schedule());
$('#sideFaces').addEventListener('change', (e) => { S.P.side_faces = e.target.checked; schedule(); });
$('#material').addEventListener('change', (e) => { S.P.side_material = e.target.value; schedule(); });
$('#applyReal').addEventListener('click', () => { S.P.height_px = S.P.length_mm * S.pxPerMm; syncAll(); schedule(); });
$('#resetAll').addEventListener('click', () => { S.P = JSON.parse(JSON.stringify(S.D)); S.deviceId = 'custom'; $('#device').value = 'custom'; syncAll(); schedule(); });

// ---------------------------------------------------------------------------------------------
// Frame, overlay and dragging
// ---------------------------------------------------------------------------------------------
function fitFrame() {
  const st = $('#stage');
  const aw = st.clientWidth - 40, ah = st.clientHeight - 40;
  const k = Math.min(aw / S.scene.w, ah / S.scene.h);
  frame.style.width = Math.max(10, S.scene.w * k) + 'px';
  frame.style.height = Math.max(10, S.scene.h * k) + 'px';
  positionOverlay();
}
new ResizeObserver(() => { if (S.sid) fitFrame(); }).observe($('#stage'));

const kScale = () => frame.clientWidth / S.scene.w;

function objBox() {
  const w = S.P.height_px * S.aspect, hh = S.P.height_px;
  const r = (S.P.rotation_deg * Math.PI) / 180;
  const bw = Math.abs(w * Math.cos(r)) + Math.abs(hh * Math.sin(r));
  const bh = Math.abs(w * Math.sin(r)) + Math.abs(hh * Math.cos(r));
  return { x: S.P.cx - bw / 2, y: S.P.cy - bh / 2, w: bw, h: bh };
}

function positionOverlay() {
  if (!S.sid) return;
  const k = kScale(), b = objBox();
  box.hidden = false;
  Object.assign(box.style, { left: b.x * k + 'px', top: b.y * k + 'px', width: b.w * k + 'px', height: b.h * k + 'px' });
  for (const m of $('#markers').children) {
    m.style.left = m.dataset.x * k + 'px';
    m.style.top = m.dataset.y * k + 'px';
  }
}

function toScene(e) {
  const r = frame.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * S.scene.w, ((e.clientY - r.top) / r.height) * S.scene.h];
}
const inside = (x, y) => { const b = objBox(); return x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h; };

let drag = null;
hit.addEventListener('pointerdown', (e) => {
  const [x, y] = toScene(e);
  if (S.tool === 'behind') return beginStroke(e, x, y);
  if (S.tool) return pickPoint(x, y);
  if (!inside(x, y)) return;
  drag = { x, y, cx: S.P.cx, cy: S.P.cy };
  hit.setPointerCapture(e.pointerId);
  hit.classList.add('dragging');
});
hit.addEventListener('pointermove', (e) => {
  const [x, y] = toScene(e);
  if (stroke) return extendStroke(x, y);
  if (drag) {
    S.P.cx = drag.cx + (x - drag.x);
    S.P.cy = drag.cy + (y - drag.y);
    syncAll(); schedule(true);
  } else {
    hit.classList.toggle('grab', !S.tool && inside(x, y));
  }
});
const endDrag = () => { if (!drag) return; drag = null; hit.classList.remove('dragging'); schedule(); };
hit.addEventListener('pointerup', () => { endDrag(); endStroke(); });
hit.addEventListener('pointercancel', () => { endDrag(); endStroke(); });
hit.addEventListener('wheel', (e) => {
  const [x, y] = toScene(e);
  if (S.tool || !inside(x, y)) return;
  e.preventDefault();
  S.P.height_px = clamp(S.P.height_px * (e.deltaY < 0 ? 1.03 : 1 / 1.03), 20, 4 * S.scene.h);
  syncAll(); schedule(true);
  clearTimeout(hit._wheelT);
  hit._wheelT = setTimeout(() => schedule(), 180);
}, { passive: false });

window.addEventListener('keydown', (e) => {
  if (!S.sid || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName) || $('dialog[open]')) return;
  const step = (e.shiftKey ? 10 : 1) / kScale();
  const d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
  if (!d) return;
  e.preventDefault();
  S.P.cx += d[0]; S.P.cy += d[1];
  syncAll(); schedule();
});

// ---------------------------------------------------------------------------------------------
// Light dial
// ---------------------------------------------------------------------------------------------
function updateDial() {
  const az = (S.P.light_az * Math.PI) / 180;
  const sx = -Math.cos(az), sy = -Math.sin(az);           // the light sits opposite the shadow
  $('#dialSun').setAttribute('cx', 46 * sx);
  $('#dialSun').setAttribute('cy', 46 * sy);
  const len = clamp(12 + 34 * S.P.shadow_len, 8, 44);
  const ray = $('#dialRay');
  ray.setAttribute('x2', len * Math.cos(az));
  ray.setAttribute('y2', len * Math.sin(az));
  const v = sy < -0.38 ? 'top' : sy > 0.38 ? 'bottom' : '';
  const hz = sx > 0.38 ? 'right' : sx < -0.38 ? 'left' : '';
  const words = v && hz ? `${v === 'top' ? 'upper' : 'lower'} ${hz}` : (v || hz);
  $('#dialLabel').textContent = `Light from the ${words}`;
  $('#dial').setAttribute('aria-valuenow', Math.round(S.P.light_az));
}
function dialFromEvent(e) {
  const r = $('#dial').getBoundingClientRect();
  const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
  let az = (Math.atan2(dy, dx) * 180) / Math.PI + 180;
  if (az > 180) az -= 360;
  S.P.light_az = az;
  updateDial(); schedule(true);
}
let dialDrag = false;
$('#dial').addEventListener('pointerdown', (e) => { dialDrag = true; $('#dial').setPointerCapture(e.pointerId); dialFromEvent(e); });
$('#dial').addEventListener('pointermove', (e) => { if (dialDrag) dialFromEvent(e); });
$('#dial').addEventListener('pointerup', () => { dialDrag = false; schedule(); });
$('#dial').addEventListener('keydown', (e) => {
  const d = { ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5 }[e.key];
  if (!d) return;
  e.preventDefault();
  S.P.light_az = ((S.P.light_az + d + 540) % 360) - 180;
  updateDial(); schedule();
});

// ---------------------------------------------------------------------------------------------
// Rendering: at most one request in flight; the newest parameters always win
// ---------------------------------------------------------------------------------------------
let pending = null, rendering = false, lastUrl = null;
function schedule(fast = false) {
  pending = { fast };
  if (!rendering) pump();
}
async function pump() {
  rendering = true;
  while (pending) {
    const { fast } = pending;
    pending = null;
    try {
      const r = await fetch(`/api/session/${S.sid}/render`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ params: S.P, max_w: fast ? 800 : 1400 }),
      });
      if (!r.ok) throw new Error((await r.json()).error || 'Render failed');
      const url = URL.createObjectURL(await r.blob());
      await new Promise((res) => { view.onload = res; view.onerror = res; view.src = url; });
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = url;
      status(`${S.scene.w} × ${S.scene.h} px scene`);
    } catch (e) {
      status(e.message, true);
    }
  }
  rendering = false;
}

// ---------------------------------------------------------------------------------------------
// Point-picking tools (scale reference, shadow sample)
// ---------------------------------------------------------------------------------------------
function startTool(name, text) {
  S.tool = name; S.picks = [];
  $('#markers').replaceChildren();
  hit.classList.add('pick');
  const b = $('#banner'); b.textContent = text; b.hidden = false;
}
function endTool() {
  S.tool = null;
  hit.classList.remove('pick');
  $('#banner').hidden = true;
  $('#markers').replaceChildren();
  $('#behindOpts').hidden = true;
  $('#behindLayer').hidden = true;
}
function pickPoint(x, y) {
  S.picks.push([x, y]);
  const m = h('i'); m.dataset.x = x; m.dataset.y = y;
  $('#markers').append(m);
  positionOverlay();
  if (S.tool === 'scale') {
    if (S.picks.length === 1) $('#banner').textContent = 'Now click the other end.';
    else openRefDialog();
  } else if (S.tool === 'shadow') {
    if (S.picks.length === 1) $('#banner').textContent = 'Now click inside the shadow, on the same surface.';
    else finishShadow();
  }
}
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.tool) endTool(); });

$('#toolScale').addEventListener('click', () => startTool('scale', 'Click one end of something you know the size of.'));
$('#toolShadow').addEventListener('click', () => startTool('shadow', 'Click a lit patch of the surface, near a shadow.'));

async function finishShadow() {
  const [lit, shadow] = S.picks;
  endTool();
  try {
    const r = await post(`/api/session/${S.sid}/calibrate/shadow`, { lit, shadow });
    S.P.shadow_rgb = r.shadow_rgb;
    syncAll(); schedule();
  } catch (e) { status(e.message, true); }
}

const refDlg = $('#refDlg');
function openRefDialog() {
  const sel = $('#refSel');
  sel.replaceChildren(...S.refs.map((r) => h('option', { value: r.mm }, `${r.name}, ${r.mm} mm`)), h('option', { value: '' }, 'Other'));
  $('#refMm').value = S.refs.length ? S.refs[0].mm : '';
  refDlg.showModal();
}
$('#refSel').addEventListener('change', (e) => { $('#refMm').value = e.target.value; });
refDlg.addEventListener('close', () => {
  const picks = S.picks.slice();
  endTool();
  if (refDlg.returnValue !== 'ok') return;
  const mm = parseFloat($('#refMm').value);
  if (!(mm > 0) || picks.length < 2) return;
  S.pxPerMm = Math.hypot(picks[1][0] - picks[0][0], picks[1][1] - picks[0][1]) / mm;
  S.P.height_px = S.P.length_mm * S.pxPerMm;
  syncAll(); schedule();
});

// ---------------------------------------------------------------------------------------------
// Cutout fixer
// ---------------------------------------------------------------------------------------------
const cutDlg = $('#cutDlg'), cutImg = $('#cutImg'), cutBox = $('#cutBox'), cutStage = $('#cutStage');
let cutRect = null, cutStart = null;
$('#fixCut').addEventListener('click', () => {
  cutRect = null; cutPhotoRect = null; cutBox.hidden = true; $('#cutOk').disabled = true;
  cutImg.src = `/api/session/${S.sid}/object.jpg?t=${Date.now()}`;
  cutDlg.showModal();
});
const cutPoint = (e) => {
  const r = cutImg.getBoundingClientRect();
  return [clamp(e.clientX - r.left, 0, r.width), clamp(e.clientY - r.top, 0, r.height)];
};
cutStage.addEventListener('pointerdown', (e) => { cutStart = cutPoint(e); cutStage.setPointerCapture(e.pointerId); });
cutStage.addEventListener('pointermove', (e) => {
  if (!cutStart) return;
  const p = cutPoint(e);
  const x0 = Math.min(cutStart[0], p[0]), y0 = Math.min(cutStart[1], p[1]);
  const x1 = Math.max(cutStart[0], p[0]), y1 = Math.max(cutStart[1], p[1]);
  cutRect = [x0, y0, x1, y1];
  cutBox.hidden = false;
  Object.assign(cutBox.style, { left: x0 + 'px', top: y0 + 'px', width: x1 - x0 + 'px', height: y1 - y0 + 'px' });
});
// The rectangle is converted to photo pixels while the dialog is still laid out; once it closes the image
// has no size to measure against.
let cutPhotoRect = null;
cutStage.addEventListener('pointerup', () => {
  cutStart = null;
  const ok = cutRect && cutRect[2] - cutRect[0] > 8 && cutRect[3] - cutRect[1] > 8;
  $('#cutOk').disabled = !ok;
  const f = S.objPhoto.w / cutImg.clientWidth;
  cutPhotoRect = ok ? cutRect.map((v) => Math.round(v * f)) : null;
});
cutDlg.addEventListener('close', async () => {
  if (cutDlg.returnValue !== 'ok' || !cutPhotoRect) return;
  const rect = cutPhotoRect;
  status('Cutting out again…');
  try {
    const r = await post(`/api/session/${S.sid}/cutout`, { rect });
    S.aspect = r.aspect; S.cutout = r.cutout; S.analysis = r.analysis;
    updateCutNote(); updateLightInfo();
    syncAll(); schedule();
  } catch (e) { status(e.message, true); }
});

// ---------------------------------------------------------------------------------------------
// Canvas extension
// ---------------------------------------------------------------------------------------------
// The optical axis is a property of the original photograph, so it must stay on the same pixel when the scene
// grows; otherwise side faces would shift toward the middle of the enlarged canvas.
function applyExtension(size, shift, dir, wasDefault = false) {
  const oldW = S.scene.w, oldH = S.scene.h;
  const P = S.P;
  if (dir > 0) {
    wasDefault = P.principal_x < 0;
    if (wasDefault) { P.principal_x = oldW / 2 + shift[0]; P.principal_y = oldH / 2 + shift[1]; }
    else { P.principal_x += shift[0]; P.principal_y += shift[1]; }
    S.ext.push({ shift, wasDefault });
  } else {
    if (wasDefault) { P.principal_x = -1; P.principal_y = -1; }
    else { P.principal_x -= shift[0]; P.principal_y -= shift[1]; }
  }
  const ratio = oldW / size[0];                 // keeps the camera height constant in pixels
  P.cam_height *= ratio; S.D.cam_height *= ratio;
  P.cx += dir * shift[0]; P.cy += dir * shift[1];
  S.scene = { w: size[0], h: size[1] };
  buildControls(); fitFrame(); schedule();
  if (!$('#behindLayer').hidden) refreshBehind();
}

$('#extGo').addEventListener('click', async () => {
  const side = $('#extSide').value, pxv = parseInt($('#extPx').value, 10);
  const prog = $('#extProg'), bar = prog.firstElementChild;
  $('#extGo').disabled = true; $('#extUndo').disabled = true;
  prog.hidden = false; bar.style.width = '0';
  status(`Adding ${pxv} px of surface on the ${side}…`);
  try {
    const { job } = await post(`/api/session/${S.sid}/extend`, { side, px: pxv });
    for (;;) {
      await new Promise((r) => setTimeout(r, 800));
      const j = await fetch(`/api/job/${job}`).then(jsonOrThrow);
      bar.style.width = Math.round(j.progress * 100) + '%';
      if (j.state === 'error') throw new Error(j.error);
      if (j.state === 'done') {
        applyExtension(j.size, j.shift, +1);
        break;
      }
    }
    buildControls(); fitFrame(); schedule();
  } catch (e) { status(e.message, true); }
  prog.hidden = true;
  $('#extGo').disabled = false;
  $('#extUndo').disabled = S.ext.length === 0;
});
$('#extUndo').addEventListener('click', async () => {
  try {
    const r = await post(`/api/session/${S.sid}/undo_extend`, {});
    const rec = S.ext.pop();
    applyExtension(r.size, rec.shift, -1, rec.wasDefault);
  } catch (e) { status(e.message, true); }
  $('#extUndo').disabled = S.ext.length === 0;
});

// ---------------------------------------------------------------------------------------------
// Cutout notes
// ---------------------------------------------------------------------------------------------
function updateCutNote() {
  const c = S.cutout;
  if (!c) return;
  $('#cutNote').textContent = c.note || (c.method === 'model' ? 'Cut out with the trained model.' : 'Cut out by colour.');
}

// ---------------------------------------------------------------------------------------------
// Measured light and the notice about the object's own lighting
// ---------------------------------------------------------------------------------------------
function lightFrom(az) {
  // az is where a shadow falls; the light is on the opposite side.
  const a = (az * Math.PI) / 180, sx = -Math.cos(a), sy = -Math.sin(a);
  const v = sy < -0.38 ? 'upper' : sy > 0.38 ? 'lower' : '';
  const hz = sx > 0.38 ? 'right' : sx < -0.38 ? 'left' : '';
  return v && hz ? `${v} ${hz}` : (v ? (v === 'upper' ? 'top' : 'bottom') : hz);
}
function updateLightInfo() {
  const a = S.analysis;
  if (!a) return;
  const sl = a.scene_light, info = $('#lightInfo'), btn = $('#useMeasured');
  const ok = sl.confidence >= 0.35;
  btn.hidden = !ok;
  info.textContent = ok
    ? `Measured from the shadows already in the photo: the light comes from the ${lightFrom(sl.az)} (${Math.round(sl.confidence * 100)}% sure).`
    : 'There are not enough shadows in this photo to measure the light. Set it with the dial, or sample a shadow.';
  const mm = a.mismatch, note = $('#lightNote');
  note.hidden = !mm;
  if (mm) {
    $('#lightNoteText').textContent = `The object's own shading suggests light from the ${lightFrom(mm.object_az)}, `
      + `but the scene's shadows suggest the ${lightFrom(mm.scene_az)}. `
      + (mm.fix ? 'Mirroring the object would fit better.'
                : 'If it looks off, try a photo of the object lit from the same side, or soften its rim light and sheen.');
    $('#flipFix').hidden = !mm.fix;
  }
}
$('#useMeasured').addEventListener('click', () => {
  const sl = S.analysis.scene_light;
  S.P.light_az = sl.az;
  if (sl.shadow_rgb) S.P.shadow_rgb = sl.shadow_rgb;
  syncAll(); schedule();
});
$('#flipFix').addEventListener('click', () => {
  S.P.flip_h = true;
  S.analysis.mismatch = null;
  syncAll(); updateLightInfo(); schedule();
});

// ---------------------------------------------------------------------------------------------
// Painting what stays in front of the object
// ---------------------------------------------------------------------------------------------
const strokeCv = $('#strokeLayer');
let stroke = null;
const brushScene = () => parseFloat($('#brush').value) / kScale();     // brush size is set in screen pixels

function refreshBehind() {
  const img = $('#behindLayer');
  img.src = `/api/session/${S.sid}/behind.png?max_w=1000&t=${Date.now()}`;
}
$('#brush').addEventListener('input', () => { $('#brushOut').textContent = $('#brush').value + ' px'; });
$('#brushOut').textContent = $('#brush').value + ' px';
$('#toolBehind').addEventListener('click', () => {
  startTool('behind', 'Paint over what should stay in front of the object. Press Esc when finished.');
  $('#behindOpts').hidden = false;
  $('#behindLayer').hidden = false;
  refreshBehind();
});
$('#behindDone').addEventListener('click', endTool);
$('#behindClear').addEventListener('click', async () => {
  try { await post(`/api/session/${S.sid}/behind/clear`, {}); refreshBehind(); schedule(); }
  catch (e) { status(e.message, true); }
});

function beginStroke(e, x, y) {
  stroke = { pts: [[x, y]], r: brushScene(), erase: $('#behindErase').checked };
  strokeCv.width = frame.clientWidth; strokeCv.height = frame.clientHeight;
  strokeCv.hidden = false;
  hit.setPointerCapture(e.pointerId);
  drawStroke();
}
function extendStroke(x, y) {
  const last = stroke.pts[stroke.pts.length - 1];
  if (Math.hypot(x - last[0], y - last[1]) * kScale() < 3) return;
  stroke.pts.push([x, y]);
  drawStroke();
}
function drawStroke() {
  const k = kScale(), c = strokeCv.getContext('2d');
  c.clearRect(0, 0, strokeCv.width, strokeCv.height);
  c.lineCap = c.lineJoin = 'round';
  c.lineWidth = 2 * stroke.r * k;
  c.strokeStyle = stroke.erase ? 'rgba(255,255,255,.35)' : 'rgba(230,190,60,.5)';
  c.beginPath();
  stroke.pts.forEach(([px_, py_], i) => (i ? c.lineTo(px_ * k, py_ * k) : c.moveTo(px_ * k, py_ * k)));
  if (stroke.pts.length === 1) c.lineTo(stroke.pts[0][0] * k + 0.01, stroke.pts[0][1] * k);
  c.stroke();
}
async function endStroke() {
  if (!stroke) return;
  const st = stroke; stroke = null;
  status(st.erase ? 'Erasing…' : 'Finding the edge…');
  try {
    await post(`/api/session/${S.sid}/behind`, { points: st.pts, radius: st.r, erase: st.erase, snap: $('#behindSnap').checked });
    refreshBehind();
    schedule();
    status('');
  } catch (e) { status(e.message, true); }
  strokeCv.hidden = true;
}

// ---------------------------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------------------------
async function exportAs(format) {
  status('Rendering at full size…');
  try {
    const r = await fetch(`/api/session/${S.sid}/export`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ params: S.P, format }),
    });
    if (!r.ok) throw new Error((await r.json()).error || 'Export failed');
    const url = URL.createObjectURL(await r.blob());
    const a = h('a', { href: url, download: `composite.${format === 'png' ? 'png' : 'jpg'}` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    status('Saved.');
  } catch (e) { status(e.message, true); }
}
$('#exportJpg').addEventListener('click', () => exportAs('jpg'));
$('#exportPng').addEventListener('click', () => exportAs('png'));
})();
