"use strict";
// Keyhole Viewer front-end: synchronized X-ray frames + acoustic emission + keyhole event marking.

const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then((r) => { if (!r.ok) throw new Error(r.status + " " + p); return r; });

const S = {
  samples: [], key: null, info: null,
  fps: 40000, first: 0, last: 0,          // frame range (virtual 40 kHz frames if no video)
  frame: 0, cursorT: 0,                   // cursor time in ms (acoustic time base)
  view: [-5, 60], full: [-5, 60],         // plot window [ms]
  plotData: null, plotReq: 0,
  ann: null, playing: false,
  frameCache: new Map(), refImg: null, refN: null,
  lo: null, hi: null,
  trace: null, roi: null,
  laser: null,                            // [t_on, t_off] ms detected from A2
  det: null,                              // auto-detection result {acoustic, video}
};

// ------------------------------------------------------------------------- persisted settings
const LS = {
  get(k, d) { try { const v = localStorage.getItem("kv_" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("kv_" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } },
};
const umPerPx = () => +$("#umpx").value || 0;
const fmtDepth = (px) => (umPerPx() ? `${px} px (${(px * umPerPx()).toFixed(0)} µm)` : `${px} px`);

const isRaw = () => $("#rawSwitch").checked;

// ---- slider: power profile behind the track + interpolated power above the thumb
const THUMB = 8;  // approx. half thumb width: the track runs from THUMB to width-THUMB
function drawSliderPower() {
  const cv = $("#sliderPower");
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const pi = S.info && S.info.process;
  if (!pi || !pi.profile || S.last <= S.first) { updateSliderTip(); return; }
  const pmax = Math.max(...pi.profile.P_W) || 1;
  ctx.beginPath(); ctx.moveTo(THUMB, H);
  for (let x = THUMB; x <= W - THUMB; x++) {
    const n = S.first + ((x - THUMB) / (W - 2 * THUMB)) * (S.last - S.first);
    const P = powerAt(frameTime(n)).P || 0;
    ctx.lineTo(x, H - (P / pmax) * (H - 2));
  }
  ctx.lineTo(W - THUMB, H); ctx.closePath();
  ctx.fillStyle = "rgba(255,183,77,.22)"; ctx.fill();
  ctx.strokeStyle = "#ffb74d"; ctx.lineWidth = 1.2; ctx.stroke();
  updateSliderTip();
}

function updateSliderTip() {
  const tip = $("#sliderTip"), sl = $("#slider");
  const pi = S.info && S.info.process;
  if (!pi || !pi.profile) { tip.textContent = ""; tip.style.display = "none"; return; }
  const pa = powerAt(S.cursorT);
  tip.style.display = "block";
  tip.classList.toggle("off", !pa.P);
  tip.textContent = pa.P == null ? "P –" : pa.P ? `${pa.P.toFixed(1)} W` + (pa.x == null ? "" : ` · ${pa.x.toFixed(2)} mm`) : "laser off";
  const W = sl.clientWidth, f = S.last > S.first ? (S.frame - S.first) / (S.last - S.first) : 0;
  const x = THUMB + f * (W - 2 * THUMB), half = tip.offsetWidth / 2;
  tip.style.left = Math.max(half, Math.min(W - half, x)) + "px";
}

// position along the line [mm] and commanded power [W] at time t (ms); mirrors data.power_at()
function powerAt(t) {
  const pi = S.info && S.info.process;
  if (!pi || !pi.profile || !pi.velocity_mm_s || !pi.laser_ms) return { x: null, P: null };
  const [t0, t1] = pi.laser_ms;
  if (t < t0 || t > t1 + 1e-6) return { x: null, P: 0 };
  const x = ((t - t0) * pi.velocity_mm_s) / 1e3, xs = pi.profile.x_mm, ps = pi.profile.P_W;
  let P = ps[ps.length - 1];
  if (x <= xs[0]) P = ps[0];
  else for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { P = ps[i - 1] + ((ps[i] - ps[i - 1]) * (x - xs[i - 1])) / ((xs[i] - xs[i - 1]) || 1); break; }
  return { x, P };
}
const fmtP = (t) => { const r = powerAt(t); return r.P == null ? "–" : r.P.toFixed(1); };

// Linear brightness/contrast for untreated frames.
//  "full":   0.3–99.7 % of all pixels.
//  "sample": the frame is bimodal (sample ~80 % of pixels, bright beam/air band above it); an Otsu split
//            isolates the sample population, which is stretched to its 0.5–99.5 % range with 30 % headroom
//            so brighter vapour (keyhole) is not clipped. Falls back to "full" if the split is not clear.
function rawStretch(v, how) {
  const step = Math.max(1, Math.floor(v.length / 80000)), s = [];
  for (let i = 0; i < v.length; i += step) s.push(v[i]);
  s.sort((a, b) => a - b);
  const q = (arr, p) => arr[Math.min(arr.length - 1, Math.max(0, Math.floor((p / 100) * arr.length)))];
  const full = [q(s, 0.3), q(s, 99.7)];
  if (how !== "sample") return full;
  const a = q(s, 0.1), b = q(s, 99.9), NB = 256, hist = new Float64Array(NB);
  if (!(b > a)) return full;
  for (const x of s) hist[Math.min(NB - 1, Math.max(0, Math.floor(((x - a) / (b - a)) * NB)))]++;
  let tot = 0, sumAll = 0; for (let i = 0; i < NB; i++) { tot += hist[i]; sumAll += i * hist[i]; }
  let w0 = 0, sum0 = 0, best = -1, thrBin = 0;
  for (let i = 0; i < NB; i++) {
    w0 += hist[i]; sum0 += i * hist[i];
    const w1 = tot - w0; if (!w0 || !w1) continue;
    const m0 = sum0 / w0, m1 = (sumAll - sum0) / w1, bv = w0 * w1 * (m0 - m1) ** 2;
    if (bv > best) { best = bv; thrBin = i; }
  }
  const thr = a + ((thrBin + 1) / NB) * (b - a);
  const k = s.findIndex((x) => x >= thr);
  const darkFrac = k < 0 ? 1 : k / s.length;
  if (darkFrac < 0.5 || darkFrac > 0.97) return full;
  const dark = s.slice(0, k);
  const lo = q(dark, 0.5), hi0 = q(dark, 99.5);
  return [lo, hi0 + 0.3 * (hi0 - lo)];
}

function applyRawSwitch() {
  const raw = isRaw();
  $("#videoSection").classList.toggle("rawmode", raw);
  for (const id of ["mode", "refFrame", "lo", "hi", "auto", "autoEach"]) $("#" + id).disabled = raw;
  $("#rawRange").classList.toggle("hidden", !raw);
  LS.set("rawSwitch", raw);
  showFrame(S.frame);
}

// raw image coords <-> displayed (flipped) coords; the mapping is its own inverse
function flipXY(x, y) {
  const cv = $("#video");
  return [$("#flipH").checked ? cv.width - 1 - x : x, $("#flipV").checked ? cv.height - 1 - y : y];
}

// ------------------------------------------------------------------------------------ sidebar
async function loadSamples(refresh) {
  const r = await (await api("/api/samples" + (refresh ? "?refresh=1" : ""))).json();
  S.samples = r.samples;
  renderList();
}

function powerStr(p) {
  const pw = p && (p._power != null && p._power !== "" ? p._power : p["Power in W"]);
  return pw != null && pw !== "" ? String(pw).replace(/;/g, "→") + " W" : "";
}

function renderList() {
  const q = $("#search").value.trim().toUpperCase();
  const both = $("#onlyBoth").checked;
  const el = $("#sampleList");
  el.innerHTML = "";
  let lastGrp = null;
  for (const s of S.samples) {
    if (q && !s.key.toUpperCase().includes(q)) continue;
    if (both && !(s.hasVideo && s.hasAcoustic)) continue;
    const grp = `${s.material} · Wall ${s.wall}`;
    if (grp !== lastGrp) {
      const g = document.createElement("div"); g.className = "grp"; g.textContent = grp; el.appendChild(g);
      lastGrp = grp;
    }
    const d = document.createElement("div");
    d.className = "item" + (s.key === S.key ? " sel" : "");
    d.innerHTML = `<span class="nm">${s.key.replace(s.material + "_", "")}</span>
      <span class="pw">${powerStr(s.params)}</span>
      <span class="badge ${s.hasVideo ? "on" : ""}" title="video">V</span>
      <span class="badge ${s.hasAcoustic ? "on" : ""}" title="acoustic">A</span>
      ${s.events ? `<span class="badge ev" title="marked events">${s.events}</span>` : ""}`;
    d.onclick = () => selectSample(s.key);
    el.appendChild(d);
  }
}

// ------------------------------------------------------------------------------ select sample
async function selectSample(key) {
  stop();
  showView("sample");
  S.key = key;
  history.replaceState(null, "", "#" + key);
  renderList();
  const info = await (await api("/api/sample/" + encodeURIComponent(key))).json();
  if (S.key !== key) return;
  S.info = info;
  S.ann = info.annotations;
  S.frameCache.clear(); S.refImg = null; S.refN = null; S.trace = null; S.roi = null; S.laser = null; S.det = null;
  renderDetection(); $("#detStatus").textContent = "";
  $("#roiBox").style.display = "none"; $("#traceStatus").textContent = "";
  const v = info.video, a = info.acoustic;
  S.fps = v ? v.fps : 40000;
  if (v) { S.first = v.first; S.last = v.last; }
  else if (a && !a.error) { S.first = Math.ceil(-5e-3 * S.fps); S.last = Math.floor(80e-3 * S.fps); }
  else { S.first = 0; S.last = 0; }

  $("#title").textContent = `${info.material} — ${info.id}`;
  const p = info.params || {};
  const parts = [];
  const pi = info.process;
  if (pi && pi.profile) parts.push(`power <b>${pi.profile.text}</b>`);
  if (pi && pi.velocity_mm_s) parts.push(`v <b>${+pi.velocity_mm_s.toFixed(1)} mm/s</b>${pi.v_source === "sheet" ? "" : ` <span class="warn">(from ${pi.v_source})</span>`}`);
  if (pi && pi.laser_ms) parts.push(`line ${pi.line_mm} mm`);
  if (p["Beam in um"]) parts.push(`beam <b>${p["Beam in um"]} µm</b>`);
  if (!pi) parts.push(`<span class="warn">no entry in LinkWallNumberParameters.xlsx</span>`);
  for (const n of (pi && pi.notes) || []) parts.push(`<span class="warn">⚠ ${n}</span>`);
  if (v) {
    const missing = v.declaredLast - v.last;
    parts.push(`video <b>${v.folder}</b> · ${v.fps / 1000} kfps · frames ${v.first}…${v.last}` +
      (missing > 0 ? ` <span style="color:var(--laser)">(${missing} frames not yet copied)</span>` : "") +
      ` · rec ${v.date} ${v.time}`);
  }
  if (a) parts.push(a.error ? `acoustic error: ${a.error}` :
    `acoustic <b>${a.file}</b> · ${a.fs / 1e6} MHz · start ${a.startTime.replace("T", " ").slice(0, 19)}`);
  $("#meta").innerHTML = parts.join(" · ");

  $("#noVideo").classList.toggle("hidden", !!v);
  $("#noAcoustic").classList.toggle("hidden", !!(a && !a.error));
  $("#plotWrap").classList.toggle("hidden", !(a && !a.error));
  $("#syncOffset").value = S.ann.sync_offset_ms || 0;
  $("#notes").value = S.ann.notes || "";
  $("#refFrame").value = v ? Math.max(v.first, -100) : -100;

  const sl = $("#slider");
  sl.min = S.first; sl.max = S.last;
  if (!v) { const c = $("#video").getContext("2d"); c.fillStyle = "#000"; c.fillRect(0, 0, 1024, 512); }

  // default plot window: laser-on window (detected below) else video span
  const vt0 = frameTime(S.first), vt1 = frameTime(S.last);
  S.full = a && !a.error ? [a.tmin_ms, a.tmax_ms] : [vt0, vt1];
  S.view = [Math.max(S.full[0], Math.min(vt0, -5)), Math.min(S.full[1], Math.max(vt1, 60))];
  if (a && !a.error) await detectLaser();
  if (S.laser) S.view = [S.laser[0] - 5, S.laser[1] + 15];
  renderEvents();
  drawSliderPower();
  await seekFrame(Math.max(S.first, Math.min(S.last, 0)));
  if (v) await autoContrast();
  requestPlot();
  // pick up a detection that already ran in this server session
  const j = await (await api(`/api/detect/${key}`)).json();
  if (S.key === key && j.state === "done") { S.det = j; renderDetection(); drawPlot(); showFrame(S.frame); }
}

async function detectLaser() {
  // Coarse A2 over the whole record to find laser gate (A2 > 2.5 V)
  const a = S.info.acoustic;
  const r = await (await api(`/api/acoustic/${S.key}?t0=${a.tmin_ms}&t1=${a.tmax_ms}&n=4000&ch=A2`)).json();
  const d = r.A2; if (!d) return;
  const hi = d.raw ? d.y : d.max;
  let on = null, off = null;
  for (let i = 0; i < hi.length; i++) if (hi[i] > 2.5) { if (on === null) on = d.t[i]; off = d.t[i]; }
  if (on === null) return;
  // refine edges at full resolution
  const ref = async (t) => {
    const q = await (await api(`/api/acoustic/${S.key}?t0=${t - 1.5}&t1=${t + 1.5}&n=4000&ch=A2`)).json();
    return q.A2;
  };
  const e1 = await ref(on), e2 = await ref(off);
  const yy1 = e1.raw ? e1.y : e1.max, yy2 = e2.raw ? e2.y : e2.max;
  let t_on = on, t_off = off;
  for (let i = 0; i < yy1.length; i++) if (yy1[i] > 2.5) { t_on = e1.t[i]; break; }
  for (let i = yy2.length - 1; i >= 0; i--) if (yy2[i] > 2.5) { t_off = e2.t[i]; break; }
  S.laser = [t_on, t_off];
}

// ------------------------------------------------------------------------------- time helpers
const syncOff = () => (S.ann ? +S.ann.sync_offset_ms || 0 : 0);
const frameTime = (n) => (n / S.fps) * 1000 + syncOff();
const frameAt = (t) => Math.max(S.first, Math.min(S.last, Math.round(((t - syncOff()) * S.fps) / 1000)));

async function seekFrame(n, keepT) {
  n = Math.max(S.first, Math.min(S.last, Math.round(n)));
  S.frame = n;
  if (!keepT) S.cursorT = frameTime(n);
  $("#slider").value = n;
  $("#frameNo").value = n;
  updateSliderTip();
  if (S.info && S.info.video) await showFrame(n);
  if ($("#follow").checked) {
    const w = S.view[1] - S.view[0];
    if (S.cursorT < S.view[0] + 0.1 * w || S.cursorT > S.view[1] - 0.1 * w) {
      S.view = [S.cursorT - w / 2, S.cursorT + w / 2];
      requestPlot();
    }
  }
  drawPlot();
  updateReadout();
}

function seekTime(t) {
  S.cursorT = t;
  return seekFrame(frameAt(t), true);
}

// ------------------------------------------------------------------------------------- video
async function getFrame(n) {
  if (S.frameCache.has(n)) return S.frameCache.get(n);
  const key = S.key;
  const p = api(`/api/frame/${key}/${n}`).then(async (r) => ({
    w: +r.headers.get("X-Width"), h: +r.headers.get("X-Height"), d: new Uint16Array(await r.arrayBuffer()),
  }));
  S.frameCache.set(n, p);
  if (S.frameCache.size > 120) S.frameCache.delete(S.frameCache.keys().next().value);
  try { return await p; } catch (e) { S.frameCache.delete(n); throw e; }
}

async function displayValues(n) {
  const f = await getFrame(n);
  const mode = isRaw() ? "raw" : $("#mode").value;
  const out = new Float32Array(f.d.length);
  if (mode === "raw") {
    for (let i = 0; i < out.length; i++) out[i] = f.d[i];
  } else if (mode === "ratio") {
    const rn = Math.max(S.first, Math.min(S.last, +$("#refFrame").value || S.first));
    if (S.refN !== rn) { S.refImg = (await getFrame(rn)).d; S.refN = rn; }
    const r = S.refImg;
    for (let i = 0; i < out.length; i++) out[i] = f.d[i] / (r[i] || 1);
  } else {
    const p = (await getFrame(Math.max(S.first, n - 1))).d;
    for (let i = 0; i < out.length; i++) out[i] = f.d[i] - p[i];
  }
  return { w: f.w, h: f.h, v: out };
}

function percentiles(v, ps) {
  const step = Math.max(1, Math.floor(v.length / 60000));
  const s = [];
  for (let i = 0; i < v.length; i += step) s.push(v[i]);
  s.sort((a, b) => a - b);
  return ps.map((p) => s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))]);
}

let drawToken = 0;
async function showFrame(n) {
  const tok = ++drawToken;
  let img;
  try { img = await displayValues(n); } catch (e) { return; }
  if (tok !== drawToken) return;
  let lo, hi;
  if (isRaw()) {
    // untreated pixels, brightness/contrast stretched to this frame's 0.3–99.7 % range
    [lo, hi] = rawStretch(img.v, $("#rawRange").value);
    if (hi <= lo) hi = lo + 1;
    $("#lo").value = Math.round(lo); $("#hi").value = Math.round(hi);
  } else {
    if ($("#autoEach").checked || S.lo === null) setContrast(...percentiles(img.v, [0.5, 99.5]));
    lo = S.lo; hi = S.hi;
  }
  const cv = $("#video");
  if (cv.width !== img.w) { cv.width = img.w; cv.height = img.h; }
  const ctx = cv.getContext("2d");
  const id = ctx.createImageData(img.w, img.h);
  const k = 255 / ((hi - lo) || 1), inv = $("#invert").checked;
  const px = id.data, v = img.v, W = img.w, H = img.h;
  const fV = $("#flipV").checked, fH = $("#flipH").checked;
  for (let yd = 0, j = 0; yd < H; yd++) {
    const row = (fV ? H - 1 - yd : yd) * W;
    for (let xd = 0; xd < W; xd++, j += 4) {
      let g = (v[row + (fH ? W - 1 - xd : xd)] - lo) * k; g = g < 0 ? 0 : g > 255 ? 255 : g;
      if (inv) g = 255 - g;
      px[j] = px[j + 1] = px[j + 2] = g; px[j + 3] = 255;
    }
  }
  ctx.putImageData(id, 0, 0);
  drawKeyholeOverlay(ctx, n);
  const pa = powerAt(frameTime(n));
  // info block at the bottom-right corner (deep bulk material, away from the keyhole)
  const lines = [`frame ${n}`, `t = ${frameTime(n).toFixed(3)} ms`];
  if (pa.P != null) lines.push(`P = ${pa.P.toFixed(1)} W`);
  if (pa.x != null) lines.push(`x = ${pa.x.toFixed(2)} mm`);
  ctx.font = "bold 17px Consolas, ui-monospace, monospace";
  const lh = 21, bw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16, bh = lines.length * lh + 10;
  const bx = cv.width - bw - 6, by = cv.height - bh - 6;
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(bx, by, bw, bh);
  ctx.fillStyle = "#fff"; ctx.textAlign = "right"; ctx.textBaseline = "alphabetic";
  lines.forEach((l, i) => ctx.fillText(l, cv.width - 14, by + 5 + (i + 1) * lh - 5));
  ctx.textAlign = "left";
  // prefetch next frames while playing
  if (S.playing) for (let k2 = 1; k2 <= 3; k2++) { const m = n + k2 * +$("#speed").value; if (m <= S.last) getFrame(m).catch(() => {}); }
}

function detFrame(n) {
  const an = S.det && S.det.video;
  if (!an) return null;
  const i = Math.round((n - an.frames[0]) / (an.step || 1));
  if (i < 0 || i >= an.frames.length || an.frames[i] !== n) return null;
  return { i, pf: an.per_frame[i], depth: an.depth[i], angle: an.angle[i], rear: an.rear_angle[i] };
}

function drawKeyholeOverlay(ctx, n) {
  const an = S.det && S.det.video;
  if (!an || !$("#overlay").checked) return;
  const cv = $("#video");
  const line = (pts, color, w, dash) => {
    ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash || []);
    ctx.beginPath(); pts.forEach(([x, y], i) => { const [a, b] = flipXY(x, y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    ctx.stroke(); ctx.setLineDash([]);
  };
  // top surface: per-column profile (every 8 px) when available, else the global row
  const sp = an.surface_profile;
  if (sp && sp.length) line(sp.map((y, i) => [Math.min(i * 8, cv.width - 1), y]), "rgba(255,215,64,.45)", 1, [6, 6]);
  else line([[0, an.surface_row], [cv.width - 1, an.surface_row]], "rgba(255,215,64,.45)", 1, [6, 6]);
  const d = detFrame(n);
  if (!d || !d.pf) return;
  const pf = d.pf;
  const surf = pf.surf_y != null ? pf.surf_y : an.surface_row;
  // depth bar: surface -> tip
  line([[pf.tip_x, surf], [pf.tip_x, pf.tip_y]], "#ffb74d", 2);
  const [tx_, ty_] = flipXY(pf.tip_x, pf.tip_y);
  ctx.fillStyle = "#ffb74d"; ctx.beginPath(); ctx.arc(tx_, ty_, 3.5, 0, 2 * Math.PI); ctx.fill();
  // wall fits: x = a + b z, y = surf + dir_mat z
  const wall = (fit, color, w) => {
    if (!fit || !pf.zfit) return;
    const [z0, z1] = [0, pf.depth];
    line([[fit[0] + fit[1] * z0, surf + pf.dir_mat * z0], [fit[0] + fit[1] * z1, surf + pf.dir_mat * z1]], color, w);
  };
  const rearKey = an.front === "fitR" ? "fitL" : "fitR";
  wall(pf[rearKey], "rgba(200,200,200,.6)", 1.5);
  wall(pf[an.front], "#4dd0e1", 2.5);
  // label next to keyhole
  const txt = `depth ${fmtDepth(d.depth)}   front ${d.angle == null ? "–" : d.angle.toFixed(1) + "°"}`;
  ctx.font = "bold 14px system-ui";
  const tw = ctx.measureText(txt).width;
  const [lx, ly] = flipXY(pf.tip_x, surf);
  const keyholeBelow = ty_ > ly;   // put the label on the opposite (gas) side of the surface
  const bx = lx + 24 + tw < cv.width ? lx + 24 : Math.max(4, lx - 24 - tw);
  const by = Math.max(30, Math.min(cv.height - 24, ly + (keyholeBelow ? -40 : 20)));
  ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(bx - 4, by, tw + 8, 20);
  ctx.fillStyle = "#fff"; ctx.fillText(txt, bx, by + 15);
}

function setContrast(lo, hi) {
  const dec = $("#mode").value === "ratio" ? 4 : 1;
  S.lo = +(+lo).toFixed(dec); S.hi = +(+hi).toFixed(dec);
  if (S.hi <= S.lo) S.hi = S.lo + (dec > 1 ? 0.01 : 1);
  $("#lo").value = S.lo; $("#hi").value = S.hi;
}

async function autoContrast() {
  if (isRaw()) return showFrame(S.frame);
  const img = await displayValues(S.frame);
  setContrast(...percentiles(img.v, [0.5, 99.5]));
  showFrame(S.frame);
}

// ----------------------------------------------------------------------------------- playback
function play() {
  if (S.playing) return stop();
  S.playing = true; $("#bPlay").textContent = "⏸";
  const tick = async () => {
    if (!S.playing) return;
    const n = S.frame + +$("#speed").value;
    if (n > S.last) return stop();
    await seekFrame(n);
    requestAnimationFrame(tick);
  };
  tick();
}
function stop() { S.playing = false; $("#bPlay").textContent = "▶"; }

// --------------------------------------------------------------------------------------- plot
let plotTimer = null;
function requestPlot() {
  clearTimeout(plotTimer);
  plotTimer = setTimeout(fetchPlot, 40);
  drawPlot();
}

async function fetchPlot() {
  const a = S.info && S.info.acoustic;
  if (!a || a.error) return;
  const id = ++S.plotReq;
  const w = $("#plot").clientWidth || 1000;
  const [t0, t1] = S.view;
  const r = await (await api(`/api/acoustic/${S.key}?t0=${t0}&t1=${t1}&n=${Math.round(w)}&ch=A1,A2,RMS`)).json();
  if (id !== S.plotReq) return;
  S.plotData = { t0, t1, d: r };
  drawPlot();
}

const PAD = { l: 58, r: 40, t: 16, b: 26 };
function plotGeom() {
  const cv = $("#plot");
  const W = cv.clientWidth, H = cv.clientHeight;
  const showR = $("#showRms").checked || !!rightSeries();
  const inner = H - PAD.t - PAD.b;
  const h1 = showR ? Math.round(inner * 0.62) : inner;
  const gap = showR ? 18 : 0;
  return { W, H, x0: PAD.l, x1: W - PAD.r,
    p1: { y0: PAD.t, y1: PAD.t + h1 },
    p2: showR ? { y0: PAD.t + h1 + gap, y1: H - PAD.b } : null };
}
const tx = (g, t) => g.x0 + ((t - S.view[0]) / (S.view[1] - S.view[0])) * (g.x1 - g.x0);
const xt = (g, x) => S.view[0] + ((x - g.x0) / (g.x1 - g.x0)) * (S.view[1] - S.view[0]);

function niceTicks(a, b, n) {
  const span = b - a; if (!(span > 0)) return [a];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function yrange(d, sym) {
  if (!d) return [-1, 1];
  let lo = Infinity, hi = -Infinity;
  const A = d.raw ? d.y : d.min, B = d.raw ? d.y : d.max;
  for (let i = 0; i < A.length; i++) { if (A[i] < lo) lo = A[i]; if (B[i] > hi) hi = B[i]; }
  if (!isFinite(lo)) return [-1, 1];
  if (sym) { const m = Math.max(Math.abs(lo), Math.abs(hi)) * 1.08 || 1; return [-m, m]; }
  const pad = (hi - lo) * 0.06 || 1; return [Math.min(0, lo - pad), hi + pad];
}

function drawSeries(ctx, g, panel, d, yr, color) {
  const ty = (y) => panel.y1 - ((y - yr[0]) / (yr[1] - yr[0])) * (panel.y1 - panel.y0);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1;
  ctx.beginPath();
  if (d.raw) {
    for (let i = 0; i < d.t.length; i++) { const x = tx(g, d.t[i]), y = ty(d.y[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  } else {
    for (let i = 0; i < d.t.length; i++) {
      const x = Math.round(tx(g, d.t[i])) + 0.5;
      ctx.moveTo(x, ty(d.max[i])); ctx.lineTo(x, ty(d.min[i]) + 0.5);
    }
  }
  ctx.stroke();
  return ty;
}

function drawAxisY(ctx, g, panel, yr, label, color) {
  ctx.fillStyle = "#8892a0"; ctx.font = "11px system-ui"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  const ty = (y) => panel.y1 - ((y - yr[0]) / (yr[1] - yr[0])) * (panel.y1 - panel.y0);
  for (const v of niceTicks(yr[0], yr[1], 4)) {
    const y = ty(v);
    ctx.strokeStyle = "#232a33"; ctx.beginPath(); ctx.moveTo(g.x0, y + 0.5); ctx.lineTo(g.x1, y + 0.5); ctx.stroke();
    ctx.fillText(+v.toPrecision(3), g.x0 - 6, y);
  }
  ctx.save(); ctx.translate(12, (panel.y0 + panel.y1) / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillStyle = color; ctx.fillText(label, 0, 0); ctx.restore();
}

function drawPlot() {
  const cv = $("#plot");
  if (!S.info || !S.info.acoustic || S.info.acoustic.error) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const g = plotGeom();
  const panels = [g.p1, g.p2].filter(Boolean);

  // video range + laser shading
  for (const p of panels) {
    if (S.info.video) {
      const a = tx(g, frameTime(S.first)), b = tx(g, frameTime(S.last));
      ctx.fillStyle = "rgba(78,161,255,.05)"; ctx.fillRect(Math.max(g.x0, a), p.y0, Math.min(g.x1, b) - Math.max(g.x0, a), p.y1 - p.y0);
    }
  }
  const pd = S.plotData;
  ctx.save(); ctx.beginPath(); ctx.rect(g.x0, 0, g.x1 - g.x0, H); ctx.clip();
  if (pd && pd.d.A2) {
    const d = pd.d.A2, hi = d.raw ? d.y : d.max;
    ctx.fillStyle = "rgba(255,183,77,.10)";
    let start = null;
    for (let i = 0; i <= d.t.length; i++) {
      const on = i < d.t.length && hi[i] > 2.5;
      if (on && start === null) start = d.t[i];
      if (!on && start !== null) {
        const x0 = tx(g, start), x1 = tx(g, d.t[i - 1]);
        for (const p of panels) ctx.fillRect(x0, p.y0, Math.max(1, x1 - x0), p.y1 - p.y0);
        start = null;
      }
    }
  }
  ctx.restore();

  // axes
  let yr1 = [-1, 1], yr2 = [0, 1];
  if (pd) { yr1 = yrange(pd.d.A1, true); if (pd.d.RMS) yr2 = yrange(pd.d.RMS, false); }
  drawAxisY(ctx, g, g.p1, yr1, "A1 acoustic [V]", "#4ea1ff");
  if (g.p2 && $("#showRms").checked) drawAxisY(ctx, g, g.p2, yr2, "RMS [V]", "#c792ea");
  // x ticks
  ctx.fillStyle = "#8892a0"; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = "11px system-ui";
  for (const v of niceTicks(S.view[0], S.view[1], Math.max(4, Math.floor((g.x1 - g.x0) / 90)))) {
    const x = Math.round(tx(g, v)) + 0.5;
    ctx.strokeStyle = "#232a33";
    for (const p of panels) { ctx.beginPath(); ctx.moveTo(x, p.y0); ctx.lineTo(x, p.y1); ctx.stroke(); }
    ctx.fillText(+v.toPrecision(10), x, H - PAD.b + 6);
  }
  ctx.textAlign = "right"; ctx.fillText("t [ms]", g.x1, H - 12);

  // series
  ctx.save(); ctx.beginPath(); ctx.rect(g.x0, 0, g.x1 - g.x0, H); ctx.clip();
  if (pd && pd.d.A1) drawSeries(ctx, g, g.p1, pd.d.A1, yr1, "#4ea1ff");
  const pi = S.info.process;
  if (pi && pi.profile && pi.velocity_mm_s && pi.laser_ms) {
    const pmax = Math.max(...pi.profile.P_W) * 1.15 || 1, p = g.p1;
    const ty = (P) => p.y1 - (P / pmax) * (p.y1 - p.y0);
    const [t0, t1] = pi.laser_ms, v = pi.velocity_mm_s;
    const pts = [[S.view[0], 0], [t0, 0]];
    for (const x of pi.profile.x_mm) { const t = t0 + (x / v) * 1e3; if (t <= t1) pts.push([t, powerAt(t).P]); }
    pts.push([t1, powerAt(t1).P], [t1, 0], [S.view[1], 0]);
    ctx.strokeStyle = "#ffb74d"; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach(([t, P], i) => { const x = tx(g, t), y = ty(P); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
    ctx.restore(); ctx.save();
    ctx.fillStyle = "#ffb74d"; ctx.font = "11px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    for (const P of niceTicks(0, pmax, 5)) ctx.fillText(P, g.x1 + 4, ty(P));
    ctx.textAlign = "right"; ctx.textBaseline = "top"; ctx.fillText("laser power [W]", g.x1 - 4, p.y0 + 2);
    ctx.beginPath(); ctx.rect(g.x0, 0, g.x1 - g.x0, H); ctx.clip();
  }
  if (g.p2 && pd && pd.d.RMS && $("#showRms").checked) drawSeries(ctx, g, g.p2, pd.d.RMS, yr2, "#c792ea");
  const rs = rightSeries();
  if (g.p2 && rs) {
    const p = g.p2, ys = rs.y;
    let lo = rs.lo, hi = rs.hi;
    let mn = Infinity, mx = -Infinity;
    for (const y of ys) if (y != null) { if (y < mn) mn = y; if (y > mx) mx = y; }
    if (lo == null) lo = mn;
    if (hi == null) hi = mx * 1.05;
    if (!(hi > lo)) hi = lo + 1;
    const ty = (y) => p.y1 - ((y - lo) / (hi - lo)) * (p.y1 - p.y0);
    ctx.strokeStyle = "#4dd0e1"; ctx.fillStyle = "#4dd0e1"; ctx.lineWidth = 1.1;
    if (rs.dots) {
      for (let i = 0; i < ys.length; i++) if (ys[i] != null) { const x = tx(g, rs.t[i] + syncOff()); if (x >= g.x0 && x <= g.x1) ctx.fillRect(x - 1, ty(ys[i]) - 1, 2, 2); }
    } else {
      ctx.beginPath(); let pen = false;
      for (let i = 0; i < ys.length; i++) {
        if (ys[i] == null) { pen = false; continue; }
        const x = tx(g, rs.t[i] + syncOff()), y = ty(ys[i]);
        pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
      }
      ctx.stroke();
    }
    ctx.restore(); ctx.save();
    // right axis
    ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "11px system-ui";
    for (const v of niceTicks(lo, hi, 6)) ctx.fillText(+v.toPrecision(3), g.x1 + 4, ty(v));
    ctx.textBaseline = "top"; ctx.textAlign = "right"; ctx.fillText(rs.label, g.x1 - 4, p.y0 + 2);
    ctx.beginPath(); ctx.rect(g.x0, 0, g.x1 - g.x0, H); ctx.clip();
  }
  // auto-detected (not yet accepted) suggestions: thin dotted
  for (const e of suggestions()) {
    if (e.accepted) continue;
    const x = Math.round(tx(g, e.t_ms)) + 0.5;
    ctx.strokeStyle = e.type === "opening" ? "rgba(61,220,132,.55)" : "rgba(255,93,108,.55)";
    ctx.setLineDash([1, 3]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, g.p1.y0); ctx.lineTo(x, (g.p2 || g.p1).y1); ctx.stroke(); ctx.setLineDash([]);
  }
  // frame exposure window when zoomed in
  if (S.info.video && (S.view[1] - S.view[0]) < 2) {
    const a = tx(g, S.cursorT - 0.5e3 / S.fps), b = tx(g, S.cursorT + 0.5e3 / S.fps);
    ctx.fillStyle = "rgba(255,255,255,.07)";
    for (const p of panels) ctx.fillRect(a, p.y0, b - a, p.y1 - p.y0);
  }
  // events
  for (const e of (S.ann ? S.ann.events : [])) {
    const x = Math.round(tx(g, e.t_ms)) + 0.5;
    ctx.strokeStyle = e.type === "opening" ? "#3ddc84" : "#ff5d6c";
    ctx.setLineDash(e.source === "acoustic" ? [5, 4] : []); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, g.p1.y0); ctx.lineTo(x, (g.p2 || g.p1).y1); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle; ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "10px system-ui";
    ctx.fillText((e.type === "opening" ? "open" : "collapse") + (e.source === "acoustic" ? " (A)" : " (V)"), x + 3, g.p1.y0 + 1);
  }
  // cursor
  const cx = Math.round(tx(g, S.cursorT)) + 0.5;
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, g.p1.y0 - 4); ctx.lineTo(cx, (g.p2 || g.p1).y1); ctx.stroke();
  ctx.restore();
  // selection rectangle
  if (drag && drag.mode === "zoom" && Math.abs(drag.x1 - drag.x0) > 3) {
    ctx.fillStyle = "rgba(78,161,255,.15)";
    ctx.fillRect(Math.min(drag.x0, drag.x1), g.p1.y0, Math.abs(drag.x1 - drag.x0), (g.p2 || g.p1).y1 - g.p1.y0);
  }
  // laser label
  if (S.laser) {
    ctx.fillStyle = "#ffb74d"; ctx.font = "10px system-ui"; ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(`laser on ${S.laser[0].toFixed(3)} → ${S.laser[1].toFixed(3)} ms`, g.x1, 2);
  }
}

function rightSeries() {
  const sel = $("#rightAxis").value, an = S.det && S.det.video;
  if (sel === "trace" && S.trace && S.trace.t.length) return { t: S.trace.t, y: S.trace.y, label: "video trace" };
  if (sel === "depth" && an) {
    const um = umPerPx();
    return { t: an.t, y: um ? an.depth.map((d) => d * um) : an.depth, label: um ? "depth [µm]" : "depth [px]", lo: 0, hi: null };
  }
  if (sel === "angle" && an) return { t: an.t, y: an.angle, label: "front-wall angle [°]", dots: true, lo: -45, hi: 45 };
  return null;
}

let drag = null;
function setupPlotEvents() {
  const cv = $("#plot");
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("mousedown", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
    drag = { mode: e.button === 2 || e.shiftKey ? "pan" : "zoom", x0: x, x1: x, v0: S.view.slice() };
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const r = cv.getBoundingClientRect(), g = plotGeom();
    drag.x1 = e.clientX - r.left;
    if (drag.mode === "pan") {
      const dt = ((drag.x1 - drag.x0) / (g.x1 - g.x0)) * (drag.v0[1] - drag.v0[0]);
      S.view = [drag.v0[0] - dt, drag.v0[1] - dt]; requestPlot();
    } else drawPlot();
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    const g = plotGeom(), d = drag; drag = null;
    if (d.mode === "zoom") {
      if (Math.abs(d.x1 - d.x0) > 4) {
        const a = xt(g, Math.min(d.x0, d.x1)), b = xt(g, Math.max(d.x0, d.x1));
        S.view = [a, b]; requestPlot();
      } else if (d.x0 >= g.x0 && d.x0 <= g.x1) {
        seekTime(xt(g, d.x0));
      }
    }
  });
  cv.addEventListener("dblclick", () => { S.view = S.laser ? [S.laser[0] - 5, S.laser[1] + 15] : S.full.slice(); requestPlot(); });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = cv.getBoundingClientRect(), g = plotGeom();
    const tm = xt(g, e.clientX - r.left), f = Math.exp(e.deltaY * 0.0015);
    S.view = [tm - (tm - S.view[0]) * f, tm + (S.view[1] - tm) * f];
    requestPlot();
  }, { passive: false });
  new ResizeObserver(() => requestPlot()).observe($("#plotWrap"));
}

// --------------------------------------------------------------------------- readout
let roTimer = null;
function updateReadout() {
  const lt = S.laser ? `  ·  t − laser on = <b>${(S.cursorT - S.laser[0]).toFixed(3)} ms</b>` : "";
  const d = detFrame(S.frame);
  const pa = powerAt(S.cursorT);
  const pw = pa.P == null ? "" : `  ·  P = <b style="color:var(--laser)">${pa.P.toFixed(1)} W</b>` + (pa.x == null ? "" : ` at x = <b>${pa.x.toFixed(3)} mm</b>`);
  const kh = d ? `\nkeyhole depth <b>${fmtDepth(d.depth)}</b>  ·  front-wall angle <b>${d.angle == null ? "–" : d.angle.toFixed(1) + "°"}</b>` : "";
  $("#readout").innerHTML = `t = <b>${S.cursorT.toFixed(4)} ms</b>  ·  frame <b>${S.frame}</b>${lt}${pw}\n<span id="roVals"></span>${kh}`;
  clearTimeout(roTimer);
  if (S.info && S.info.acoustic && !S.info.acoustic.error && !S.playing) {
    roTimer = setTimeout(async () => {
      const r = await (await api(`/api/acoustic_value/${S.key}?t=${S.cursorT}`)).json();
      const el = $("#roVals"); if (!el) return;
      el.innerHTML = `A1 = <b>${r.A1 != null ? r.A1.toFixed(3) : "–"} V</b>  ·  RMS = <b>${r.RMS != null ? r.RMS.toFixed(3) : "–"} V</b>  ·  laser ${r.A2 > 2.5 ? "<b style='color:var(--laser)'>ON</b>" : "off"}`;
    }, 80);
  }
}

// ------------------------------------------------------------------------------ annotations
let saveTimer = null;
function saveAnn() {
  clearTimeout(saveTimer);
  $("#saveState").textContent = "saving…";
  saveTimer = setTimeout(async () => {
    await api(`/api/annotations/${S.key}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S.ann) });
    $("#saveState").textContent = "saved to KeyholeViewer/annotations/" + S.key + ".json";
    const s = S.samples.find((x) => x.key === S.key); if (s) { s.events = S.ann.events.length; renderList(); }
  }, 300);
}

function addEvent(type, source) {
  if (!S.key) return;
  S.ann.events.push({ id: Date.now(), type, source, t_ms: +S.cursorT.toFixed(5), frame: S.frame, note: "" });
  S.ann.events.sort((a, b) => a.t_ms - b.t_ms);
  renderEvents(); drawPlot(); saveAnn();
}

function renderEvents() {
  const tb = $("#events tbody"); tb.innerHTML = "";
  S.ann.events.forEach((e, i) => {
    const tr = document.createElement("tr"); tr.className = "ev";
    tr.innerHTML = `<td>${i + 1}</td><td><span class="tag ${e.type}">${e.type}</span></td><td>${e.source}</td>
      <td>${e.t_ms.toFixed(4)}</td><td>${e.frame}</td><td>${fmtP(e.t_ms)}</td><td><input value="${(e.note || "").replace(/"/g, "&quot;")}"></td>
      <td><button class="del" title="delete">✕</button></td>`;
    tr.onclick = (ev) => { if (ev.target.tagName !== "INPUT" && ev.target.tagName !== "BUTTON") seekTime(e.t_ms); };
    tr.querySelector("input").onchange = (ev) => { e.note = ev.target.value; saveAnn(); };
    tr.querySelector(".del").onclick = () => { S.ann.events.splice(i, 1); renderEvents(); drawPlot(); saveAnn(); };
    tb.appendChild(tr);
  });
  // comparison video vs acoustic (k-th event of each type)
  const by = (t, s) => S.ann.events.filter((e) => e.type === t && e.source === s);
  let rows = "";
  for (const t of ["opening", "collapse"]) {
    const v = by(t, "video"), a = by(t, "acoustic");
    for (let k = 0; k < Math.max(v.length, a.length); k++) {
      const dv = v[k] ? v[k].t_ms.toFixed(4) : "–", da = a[k] ? a[k].t_ms.toFixed(4) : "–";
      const dd = v[k] && a[k] ? (a[k].t_ms - v[k].t_ms).toFixed(4) : "–";
      const rel = S.laser && v[k] ? (v[k].t_ms - S.laser[0]).toFixed(3) : "–";
      const pv = v[k] ? fmtP(v[k].t_ms) : "–", pa = a[k] ? fmtP(a[k].t_ms) : "–";
      rows += `<tr><td>${t} #${k + 1}</td><td>${dv}</td><td>${da}</td><td>${dd}</td><td>${pv}</td><td>${pa}</td><td>${rel}</td></tr>`;
    }
  }
  if (S.det) renderDetection();
  $("#compare").innerHTML = rows ? `<table><thead><tr><th></th><th>video [ms]</th><th>acoustic [ms]</th><th>Δ acoustic−video [ms]</th><th>P video [W]</th><th>P acoustic [W]</th><th>video t − laser on [ms]</th></tr></thead><tbody>${rows}</tbody></table>` : "";
}

// ------------------------------------------------------------------------------ video trace / ROI
function setupRoi() {
  const cv = $("#video"), box = $("#roiBox");
  let st = null;
  // ROI is kept in raw image coordinates; mouse positions are converted through the flip
  const toImg = (e) => { const r = cv.getBoundingClientRect(); return flipXY((e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height); };
  const drawBox = () => {
    if (!S.roi) { box.style.display = "none"; return; }
    const r = cv.getBoundingClientRect(), sx = r.width / cv.width, sy = r.height / cv.height;
    const [ax, ay] = flipXY(S.roi[0], S.roi[1]), [bx, by] = flipXY(S.roi[2], S.roi[3]);
    const [x0, y0, x1, y1] = [Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)];
    Object.assign(box.style, { display: "block", left: cv.offsetLeft + x0 * sx + "px", top: cv.offsetTop + y0 * sy + "px", width: (x1 - x0) * sx + "px", height: (y1 - y0) * sy + "px" });
  };
  cv.addEventListener("mousedown", (e) => { if (e.shiftKey) { st = toImg(e); e.preventDefault(); } });
  window.addEventListener("mousemove", (e) => {
    if (!st) return; const p = toImg(e);
    S.roi = [Math.min(st[0], p[0]), Math.min(st[1], p[1]), Math.max(st[0], p[0]), Math.max(st[1], p[1])].map((v, i) => Math.round(Math.max(0, Math.min(i % 2 ? cv.height : cv.width, v))));
    drawBox();
  });
  window.addEventListener("mouseup", () => { st = null; });
  new ResizeObserver(drawBox).observe(cv);
  S.redrawRoi = drawBox;
}

async function startTrace() {
  if (!S.info || !S.info.video) return;
  const key = S.key;
  const roi = S.roi && S.roi[2] - S.roi[0] > 3 ? S.roi : [0, 0, S.info.video.width, S.info.video.height];
  await api(`/api/videotrace/${key}`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: +$("#refFrame").value, roi, step: 1 }) });
  const poll = async () => {
    if (S.key !== key) return;
    const j = await (await api(`/api/videotrace/${key}`)).json();
    $("#traceStatus").textContent = j.state === "running" ? `computing ${j.done}/${j.total}…` : j.state === "error" ? "error: " + j.error : `trace: ROI ${j.roi.join(",")} ÷ frame ${j.ref}`;
    S.trace = { t: j.t, y: j.y };
    drawPlot();
    if (j.state === "running") setTimeout(poll, 700);
  };
  poll();
}

// ---------------------------------------------------------------------------- auto-detection
function detCfg(force) {
  return { thr_V: +$("#thrV").value, thr_px: +$("#thrPx").value, dwell_ms: +$("#dwell").value,
    video_thr: +$("#vthr").value, step: 2, force: !!force };
}

async function runDetect(force) {
  if (!S.key) return;
  const key = S.key;
  $("#detStatus").textContent = "running…";
  await api(`/api/detect/${key}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(detCfg(force)) });
  const poll = async () => {
    if (S.key !== key) return;
    const j = await (await api(`/api/detect/${key}`)).json();
    if (j.state === "running") {
      $("#detStatus").textContent = j.total ? `analysing video ${j.done}/${j.total} frames…` : "analysing video…";
      if (j.acoustic && !(S.det && S.det.acoustic)) { S.det = { acoustic: j.acoustic, laser: j.laser }; renderDetection(); drawPlot(); }
      setTimeout(poll, 800);
      return;
    }
    if (j.state === "error") { $("#detStatus").textContent = "error: " + j.error; return; }
    S.det = j;
    $("#detStatus").textContent = "done";
    if (j.video && $("#rightAxis").value === "trace" && !S.trace) $("#rightAxis").value = "depth";
    renderDetection(); drawPlot(); showFrame(S.frame); updateReadout();
  };
  poll();
}

function suggestions() {
  if (!S.det) return [];
  const out = [];
  const add = (list, source) => (list || []).forEach((e) => out.push({ ...e, source,
    frame: e.frame != null ? e.frame : frameAt(e.t_ms) }));
  add(S.det.acoustic && S.det.acoustic.events, "acoustic");
  add(S.det.video && S.det.video.events.map((e) => ({ ...e, t_ms: +(e.t_ms + syncOff()).toFixed(5) })), "video");
  out.sort((a, b) => a.t_ms - b.t_ms);
  const ev = S.ann ? S.ann.events : [];
  for (const s of out) s.accepted = ev.some((e) => e.type === s.type && e.source === s.source && Math.abs(e.t_ms - s.t_ms) < 1e-3);
  return out;
}

function acceptSuggestion(s) {
  if (s.accepted) return;
  S.ann.events.push({ id: Date.now() + Math.random(), type: s.type, source: s.source, t_ms: s.t_ms, frame: s.frame,
    note: "auto" + (s.note ? ": " + s.note : "") });
  S.ann.events.sort((a, b) => a.t_ms - b.t_ms);
}

function renderDetection() {
  const tb = $("#sugg tbody"), st = $("#stats tbody");
  tb.innerHTML = ""; st.innerHTML = "";
  $("#detCsv").classList.toggle("hidden", !(S.det && S.det.video));
  $("#detInfo").textContent = "";
  if (!S.det) return;
  for (const s of suggestions()) {
    const tr = document.createElement("tr"); tr.className = "ev" + (s.accepted ? " done" : "");
    tr.innerHTML = `<td><span class="tag ${s.type}">${s.type}</span></td><td>${s.source}</td><td>${s.t_ms.toFixed(4)}</td>
      <td>${s.frame}</td><td>${fmtP(s.t_ms)}</td><td>${s.note || ""}</td><td>${s.accepted ? "✓" : '<button>Accept</button>'}</td>`;
    tr.onclick = (e) => {
      if (e.target.tagName === "BUTTON") { acceptSuggestion(s); renderEvents(); renderDetection(); drawPlot(); saveAnn(); }
      else seekTime(s.t_ms);
    };
    tb.appendChild(tr);
  }
  if (!tb.children.length) tb.innerHTML = `<tr><td colspan="6" class="hint">no keyhole found${S.det.acoustic ? ` (RMS never above ${S.det.acoustic.thr_V} V)` : ""}</td></tr>`;
  const an = S.det.video;
  if (an) {
    for (const p of an.stats || []) {
      st.insertAdjacentHTML("beforeend", `<tr><td>${p.t0.toFixed(2)} – ${p.t1.toFixed(2)} ms</td><td>${fmtDepth(p.depth_mean.toFixed(1))}</td>
        <td>${fmtDepth(p.depth_max)}</td><td>${p.angle_mean == null ? "–" : `${p.angle_mean.toFixed(1)}° ± ${p.angle_std.toFixed(1)}°`}</td></tr>`);
    }
    $("#detCsv").href = `/api/detect/${S.key}/video.csv?um_per_px=${umPerPx()}`;
    const sp = an.speed, pi2 = S.info.process;
    let spd = "";
    if (sp) {
      spd = `keyhole speed ${sp.px_per_ms.toFixed(2)} px/ms`;
      if (umPerPx()) spd += ` = ${(sp.px_per_ms * umPerPx()).toFixed(1)} mm/s at ${umPerPx()} µm/px`;
      if (pi2 && pi2.velocity_mm_s) spd += ` · with v = ${+pi2.velocity_mm_s.toFixed(1)} mm/s → ${(pi2.velocity_mm_s / sp.px_per_ms).toFixed(2)} µm/px`;
      spd += " · ";
    }
    $("#detInfo").textContent = spd + `surface row ${an.surface_row}${an.surface_profile ? ` (local ${Math.min(...an.surface_profile)}–${Math.max(...an.surface_profile)})` : ""} (raw) · scan direction ${an.scan_dir > 0 ? "+x" : "−x"} (raw) · ` +
      `references: pre-laser frames ${an.pre_frames[0]}…${an.pre_frames[1]}${an.post_ref ? " + post-laser frames" : " only (video ends before laser off)"} · ` +
      `front-wall angle measured from the vertical, positive = tip trails behind the opening`;
  }
  if (S.det && S.det.acoustic_note) $("#detInfo").textContent = `Acoustic: ${S.det.acoustic_note}. ` + $("#detInfo").textContent;
}

// --------------------------------------------------------------------------------- hysteresis
function showView(which) {
  $("#sampleView").classList.toggle("hidden", which !== "sample");
  $("#hystView").classList.toggle("hidden", which !== "hyst");
  $("#navHyst").classList.toggle("sel", which === "hyst");
}

async function showHysteresis() {
  stop();
  S.key = null; renderList();
  showView("hyst");
  if (location.hash !== "#hysteresis") history.replaceState(null, "", "#hysteresis");
  pollBatch();
  await loadHysteresis();
}

async function loadHysteresis() {
  const q = `basis=${$("#hystBasis").value}&merge_ms=${+$("#hystMerge").value || 0}&group=${$("#hystGroup").value}`;
  S.hyst = await (await api("/api/hysteresis?" + q)).json();
  renderHysteresis();
}

let batchTimer = null;
async function pollBatch() {
  clearTimeout(batchTimer);
  const b = await (await api("/api/batch")).json();
  const el = $("#batchStatus");
  if (b.state === "running") {
    el.textContent = `analysing… ${b.done}/${b.total} samples` + (b.current.length ? ` (now: ${b.current.slice(0, 3).join(", ")}${b.current.length > 3 ? "…" : ""})` : "");
    $("#batchBtn").disabled = true;
    if (!$("#hystView").classList.contains("hidden")) { if (b.done !== S.batchDone) { S.batchDone = b.done; loadHysteresis(); } }
    batchTimer = setTimeout(pollBatch, 2500);
  } else {
    $("#batchBtn").disabled = false;
    el.textContent = b.state === "done" ? `last run: ${b.total} samples in ${b.elapsed_s} s` + (b.errors.length ? ` · ${b.errors.length} problems (hover)` : "") :
      b.state === "error" ? "error: " + b.errors.join("; ") : "";
    el.title = (b.errors || []).join("\n");
    if (b.state === "done" && S.batchDone !== -1) { S.batchDone = -1; loadHysteresis(); loadSamples(false); }
  }
}

async function startBatch() {
  await api("/api/batch", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...detCfg(!!$("#batchForce").checked), workers: 3 }) });
  S.batchDone = 0;
  pollBatch();
}

const fmtS = (s) => (s ? `${s.mean.toFixed(1)}${s.sd != null ? ` ± ${s.sd.toFixed(1)}` : ""} W <span class="hint">(n=${s.n})</span>` : "–");

function renderHysteresis() {
  const d = S.hyst; if (!d) return;
  $("#hystMeta").innerHTML = `${d.n_samples} samples · ${d.n_analysed} analysed · ${d.n_usable} with power profile, velocity and laser timing`;
  // cards
  const cards = d.groups.map((g) => {
    const yes = /^hysteresis/.test(g.verdict), inv = /^inverse/.test(g.verdict);
    const lineDP = g.line_dP.length ? `<tr><td>ΔP within lines</td><td>${g.line_dP.map((l) => `${l.id}: ${l.dP > 0 ? "+" : ""}${l.dP.toFixed(1)}`).join(", ")} W</td></tr>` : "";
    const cst = g.const_keyhole.length || g.const_no_keyhole.length ?
      `<tr><td>constant P</td><td>keyhole at ${g.const_keyhole.join(", ") || "–"} W · none at ${g.const_no_keyhole.join(", ") || "–"} W</td></tr>` : "";
    return `<div class="card"><h3>${g.material} · ${g.v} mm/s${g.wall ? ` · wall ${g.wall}` : ""} · ${g.source}</h3>
      <div class="verdict ${yes || inv ? "yes" : "no"}">${g.verdict}</div>
      <table>
        <tr><td class="c-on">P<sub>on</sub> (rising)</td><td>${fmtS(g.P_on)}</td></tr>
        <tr><td class="c-off">P<sub>off</sub> (falling)</td><td>${fmtS(g.P_off)}</td></tr>
        <tr><td>ΔP</td><td>${g.dP == null ? "–" : `<b>${g.dP > 0 ? "+" : ""}${g.dP.toFixed(1)} W</b>${g.se == null ? "" : ` ± ${g.se.toFixed(1)}`}`}</td></tr>
        <tr><td>ramp rate</td><td>${g.mean_rate_W_ms == null ? "–" : g.mean_rate_W_ms.toFixed(2) + " W/ms"}</td></tr>
        ${g.survived_to_P.length ? `<tr><td>open until</td><td>${g.survived_to_P.map((p) => p.toFixed(0)).join(", ")} W</td></tr>` : ""}
        ${lineDP}${cst}
      </table>
      ${g.why ? `<div class="hint">${g.why}</div>` : ""}</div>`;
  });
  $("#hystCards").innerHTML = cards.join("") || `<div class="hint">No analysed samples yet – press “Rescan + auto-detect all samples”.</div>`;
  renderHystChart();
  // table
  const tb = $("#hystTable tbody"); tb.innerHTML = "";
  for (const r of d.rows) {
    const srcs = Object.entries(r.sources || {});
    const base = `<td>${r.material} ${r.id}</td><td>${r.profile ? r.profile.text : "–"}</td><td>${r.v ?? "–"}</td>`;
    if (!srcs.length) {
      tb.insertAdjacentHTML("beforeend", `<tr class="ev" data-key="${r.key}">${base}<td>–</td><td class="hint">${(r.notes || []).join("; ")}</td><td></td></tr>`);
      continue;
    }
    for (const [src, res] of srcs) {
      const trs = res.transitions.map((t) => t.cls === "none" ? `<span class="tr-cls none">no keyhole (≤ ${t.P_W} W)</span>` :
        `<span class="tr-cls ${t.cls}">${t.cls} · ${t.P_W == null ? "?" : t.P_W.toFixed(1)} W · ${t.x_mm == null ? "?" : t.x_mm.toFixed(2)} mm</span>`).join("");
      const on = res.transitions.find((t) => t.cls === "on"), off = res.transitions.find((t) => t.cls === "off");
      const dP = on && off ? `${(on.P_W - off.P_W) > 0 ? "+" : ""}${(on.P_W - off.P_W).toFixed(1)} W` : "";
      tb.insertAdjacentHTML("beforeend", `<tr class="ev" data-key="${r.key}">${base}<td>${src}${res.basis === "manual" ? " (marked)" : ""}</td><td>${trs}${r.laser_assumed ? ' <span class="warn" title="no acoustic gate">t₀ assumed</span>' : ""}</td><td>${dP}</td></tr>`);
    }
  }
  tb.querySelectorAll("tr").forEach((tr) => (tr.onclick = () => { location.hash = tr.dataset.key; }));
}

function renderHystChart() {
  const d = S.hyst; if (!d) return;
  const showA = $("#hystShowA").checked, showV = $("#hystShowV").checked;
  const groups = d.groups.filter((g) => (g.source === "acoustic" ? showA : showV));
  if (!groups.length) { $("#hystChart").innerHTML = '<div class="hint">nothing to show</div>'; return; }
  const all = [];
  for (const g of groups) all.push(...g.raw_on, ...g.raw_off, ...g.initial_P, ...g.survived_to_P, ...g.never_formed_max_P, ...g.const_keyhole, ...g.const_no_keyhole, ...g.flat_on_P, ...g.flat_off_P);
  const lo = Math.floor((Math.min(...all) - 5) / 10) * 10, hi = Math.ceil((Math.max(...all) + 5) / 10) * 10;
  const W = 1000, L = 210, R = 20, rowH = 46, top = 10, H = top + groups.length * rowH + 34;
  const X = (p) => L + ((p - lo) / (hi - lo)) * (W - L - R);
  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui" font-size="12">`;
  for (const t of niceTicks(lo, hi, 10)) {
    svg += `<line x1="${X(t)}" x2="${X(t)}" y1="${top}" y2="${H - 30}" stroke="#2b313b"/>` +
      `<text x="${X(t)}" y="${H - 14}" fill="#8892a0" text-anchor="middle">${t}</text>`;
  }
  svg += `<text x="${W - R}" y="${H - 2}" fill="#8892a0" text-anchor="end">commanded laser power [W]</text>`;
  const jit = (i) => ((i * 7919) % 11 - 5) * 1.6;  // deterministic vertical jitter
  groups.forEach((g, gi) => {
    const y = top + gi * rowH + rowH / 2;
    if (gi % 2 === 0) svg += `<rect x="0" y="${y - rowH / 2}" width="${W}" height="${rowH}" fill="#1b2027"/>`;
    svg += `<text x="8" y="${y - 4}" fill="#dfe4ea">${g.material} · ${g.v} mm/s${g.wall ? ` · W${g.wall}` : ""}</text><text x="8" y="${y + 12}" fill="#8892a0">${g.source}${g.dP != null ? ` · ΔP ${g.dP > 0 ? "+" : ""}${g.dP.toFixed(1)} W` : ""}</text>`;
    const tri = (p, up, fill, i) => { const x = X(p), yy = y + jit(i); return up ?
      `<path d="M${x} ${yy - 6} L${x + 5.5} ${yy + 4} L${x - 5.5} ${yy + 4} Z" fill="${fill}" stroke="${fill === "none" ? "#3ddc84" : "none"}" stroke-width="1.5"><title>${p.toFixed(1)} W</title></path>` :
      `<path d="M${x} ${yy + 6} L${x + 5.5} ${yy - 4} L${x - 5.5} ${yy - 4} Z" fill="${fill}"><title>${p.toFixed(1)} W</title></path>`; };
    const side = (p, left, color, i) => { const x = X(p), yy = y + jit(i); return left ?
      `<path d="M${x - 6} ${yy} L${x + 4} ${yy - 5.5} L${x + 4} ${yy + 5.5} Z" fill="none" stroke="${color}" stroke-width="1.5"><title>${p.toFixed(1)} W</title></path>` :
      `<path d="M${x + 6} ${yy} L${x - 4} ${yy - 5.5} L${x - 4} ${yy + 5.5} Z" fill="none" stroke="${color}" stroke-width="1.5"><title>${p.toFixed(1)} W</title></path>`; };
    g.const_keyhole.forEach((p, i) => (svg += `<circle cx="${X(p)}" cy="${y + jit(i + 3)}" r="4.5" fill="#b0bac6"><title>constant ${p} W: keyhole</title></circle>`));
    g.const_no_keyhole.forEach((p, i) => (svg += `<circle cx="${X(p)}" cy="${y + jit(i + 3)}" r="4.5" fill="none" stroke="#b0bac6" stroke-width="1.5"><title>constant ${p} W: no keyhole</title></circle>`));
    g.survived_to_P.forEach((p, i) => (svg += side(p, true, "#ff5d6c", i + 1)));
    g.never_formed_max_P.forEach((p, i) => (svg += side(p, false, "#3ddc84", i + 2)));
    g.initial_P.forEach((p, i) => (svg += tri(p, true, "none", i + 4)));
    g.raw_off.forEach((p, i) => (svg += tri(p, false, "#ff5d6c", i)));
    g.raw_on.forEach((p, i) => (svg += tri(p, true, "#3ddc84", i + 5)));
    if (g.P_on) svg += `<line x1="${X(g.P_on.mean)}" x2="${X(g.P_on.mean)}" y1="${y - rowH / 2 + 4}" y2="${y + rowH / 2 - 4}" stroke="#3ddc84" stroke-dasharray="3 3"/>`;
    if (g.P_off) svg += `<line x1="${X(g.P_off.mean)}" x2="${X(g.P_off.mean)}" y1="${y - rowH / 2 + 4}" y2="${y + rowH / 2 - 4}" stroke="#ff5d6c" stroke-dasharray="3 3"/>`;
  });
  svg += "</svg>";
  $("#hystChart").innerHTML = svg;
}

// ------------------------------------------------------------------------------------- wiring
function wire() {
  $("#search").oninput = renderList;
  $("#onlyBoth").onchange = renderList;
  $("#rescan").onclick = async () => { await loadSamples(true); if (S.key) selectSample(S.key); };
  $("#slider").oninput = (e) => seekFrame(+e.target.value);
  $("#frameNo").onchange = (e) => seekFrame(+e.target.value);
  $("#bStart").onclick = () => seekFrame(S.first);
  $("#bBack").onclick = () => seekFrame(S.frame - 1);
  $("#bFwd").onclick = () => seekFrame(S.frame + 1);
  $("#bBack10").onclick = () => seekFrame(S.frame - 10);
  $("#bFwd10").onclick = () => seekFrame(S.frame + 10);
  $("#bPlay").onclick = play;
  $("#mode").value = LS.get("mode", "ratio");
  $("#mode").onchange = () => { LS.set("mode", $("#mode").value); S.lo = null; showFrame(S.frame); };
  $("#rawSwitch").checked = LS.get("rawSwitch", false);
  $("#rawSwitch").onchange = applyRawSwitch;
  $("#rawRange").value = LS.get("rawRange", "sample");
  $("#rawRange").onchange = (e) => { LS.set("rawRange", e.target.value); showFrame(S.frame); };
  applyRawSwitch();
  $("#refFrame").onchange = () => { S.refN = null; if ($("#mode").value === "ratio") autoContrast(); };
  $("#lo").onchange = (e) => { S.lo = +e.target.value; showFrame(S.frame); };
  $("#hi").onchange = (e) => { S.hi = +e.target.value; showFrame(S.frame); };
  $("#auto").onclick = autoContrast;
  $("#invert").onchange = () => showFrame(S.frame);
  $("#autoEach").onchange = () => showFrame(S.frame);
  $("#showRms").onchange = () => drawPlot();
  $("#fitLaser").onclick = () => { if (S.laser) { S.view = [S.laser[0] - 5, S.laser[1] + 15]; requestPlot(); } };
  $("#fitVideo").onclick = () => { S.view = [frameTime(S.first), frameTime(S.last)]; requestPlot(); };
  $("#fitAll").onclick = () => { S.view = S.full.slice(); requestPlot(); };
  $("#traceBtn").onclick = () => { $("#rightAxis").value = "trace"; startTrace(); };
  $("#detBtn").onclick = () => runDetect(false);
  $("#navHyst").onclick = () => { location.hash = "hysteresis"; showHysteresis(); };
  $("#batchBtn").onclick = startBatch;
  $("#hystBasis").onchange = loadHysteresis;
  $("#hystMerge").onchange = loadHysteresis;
  $("#hystGroup").onchange = loadHysteresis;
  $("#hystShowA").onchange = renderHystChart;
  $("#hystShowV").onchange = renderHystChart;
  $("#acceptAll").onclick = () => { suggestions().forEach(acceptSuggestion); renderEvents(); renderDetection(); drawPlot(); saveAnn(); };
  $("#rightAxis").onchange = () => drawPlot();
  for (const id of ["flipV", "flipH", "overlay"]) {
    $("#" + id).checked = LS.get(id, $("#" + id).checked);
    $("#" + id).onchange = (e) => { LS.set(id, e.target.checked); showFrame(S.frame); if (S.redrawRoi) S.redrawRoi(); };
  }
  if (["", "4.2"].includes(String(LS.get("umpx", "")))) { try { localStorage.removeItem("kv_umpx"); } catch (e) { /* ignore */ } }
  for (const id of ["thrV", "thrPx", "dwell", "vthr", "umpx"]) {
    $("#" + id).value = LS.get(id, $("#" + id).value);
    $("#" + id).onchange = (e) => {
      LS.set(id, e.target.value);
      if (id === "umpx") { renderDetection(); drawPlot(); showFrame(S.frame); updateReadout(); }
      else if (S.det) runDetect(false);   // re-threshold (video analysis is cached unless pixel thr changes)
    };
  }
  $("#syncOffset").onchange = (e) => { S.ann.sync_offset_ms = +e.target.value || 0; saveAnn(); drawSliderPower(); seekFrame(S.frame); };
  new ResizeObserver(() => drawSliderPower()).observe($("#sliderWrap"));
  $("#notes").oninput = (e) => { S.ann.notes = e.target.value; saveAnn(); };
  document.querySelectorAll(".mark").forEach((b) => (b.onclick = () => addEvent(b.dataset.type, b.dataset.source)));
  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === "arrowright") { seekFrame(S.frame + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
    else if (k === "arrowleft") { seekFrame(S.frame - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
    else if (k === " ") { play(); e.preventDefault(); }
    else if (k === "r") { $("#rawSwitch").checked = !$("#rawSwitch").checked; applyRawSwitch(); }
    else if (k === "q") addEvent("opening", "video");
    else if (k === "w") addEvent("collapse", "video");
    else if (k === "a") addEvent("opening", "acoustic");
    else if (k === "s") addEvent("collapse", "acoustic");
  });
  setupPlotEvents();
  setupRoi();
}

(async function init() {
  wire();
  await loadSamples(false);
  const want = decodeURIComponent(location.hash.slice(1));
  if (want === "hysteresis") { showHysteresis(); return; }
  const first = S.samples.find((s) => s.key === want) || S.samples.find((s) => s.hasVideo && s.hasAcoustic) || S.samples[0];
  if (first) selectSample(first.key);
  window.addEventListener("hashchange", () => {
    const k = decodeURIComponent(location.hash.slice(1));
    if (k === "hysteresis") { showHysteresis(); return; }
    if (k && k !== S.key && S.samples.some((s) => s.key === k)) selectSample(k);
  });
})();
