/**
 * Camera heat map: where tracked people have stood, per camera, this session.
 *
 * Built from the tracker boxes the wall shows (overlay.boxes: live detections
 * with the backend, the recorded demo track in Playback): each update adds a soft
 * blob at every person's feet (bottom-centre of the box). Old activity fades
 * slowly, so the map shows where people gathered and walked, not a guess.
 * Counts only: no identities are stored.
 */

const GW = 64; // grid columns
const GH = 36; // grid rows
const DECAY = 0.996; // per update: older activity fades
const RADIUS = 2.2; // blob radius in grid cells

/** @type {Map<string, {grid: Float32Array, peak: number, canvas: HTMLCanvasElement|null, stale: boolean}>} */
const maps = new Map();
let enabled = readPref();

function readPref() {
  try {
    return localStorage.getItem("fw.heatmap") === "1";
  } catch {
    return false;
  }
}

export function isEnabled() {
  return enabled;
}

export function setEnabled(on) {
  enabled = Boolean(on);
  try {
    localStorage.setItem("fw.heatmap", enabled ? "1" : "0");
  } catch {
    /* per-viewer convenience only */
  }
}

function entry(cameraId) {
  let m = maps.get(cameraId);
  if (!m) {
    m = { grid: new Float32Array(GW * GH), peak: 0, canvas: null, stale: true };
    maps.set(cameraId, m);
  }
  return m;
}

/** Add one overlay update. Boxes are normalised 0..1 (x, y, w, h). */
export function addBoxes(cameraId, boxes) {
  if (!Array.isArray(boxes)) return;
  const m = entry(cameraId);
  const g = m.grid;
  for (let i = 0; i < g.length; i++) g[i] *= DECAY;
  const r = Math.ceil(RADIUS * 2);
  for (const b of boxes) {
    if (!b || b.label === "weapon") continue; // people only
    const fx = (Number(b.x) + Number(b.w) / 2) * GW;
    const fy = (Number(b.y) + Number(b.h)) * GH;
    if (!Number.isFinite(fx) || !Number.isFinite(fy)) continue;
    const cx = Math.floor(fx);
    const cy = Math.floor(fy);
    for (let y = cy - r; y <= cy + r; y++) {
      if (y < 0 || y >= GH) continue;
      for (let x = cx - r; x <= cx + r; x++) {
        if (x < 0 || x >= GW) continue;
        const d2 = (x + 0.5 - fx) ** 2 + (y + 0.5 - fy) ** 2;
        g[y * GW + x] += Math.exp(-d2 / (2 * RADIUS * RADIUS));
      }
    }
  }
  let peak = 0;
  for (let i = 0; i < g.length; i++) if (g[i] > peak) peak = g[i];
  m.peak = peak;
  m.stale = true;
}

export function reset() {
  maps.clear();
}

/** Blue → yellow → red ramp; alpha rises with heat. */
function ramp(t) {
  const stops = [
    [0.0, [42, 125, 225]],
    [0.45, [64, 200, 170]],
    [0.7, [250, 204, 21]],
    [1.0, [220, 38, 38]],
  ];
  let i = 1;
  while (i < stops.length - 1 && t > stops[i][0]) i++;
  const [t0, c0] = stops[i - 1];
  const [t1, c1] = stops[i];
  const k = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
  return c0.map((v, j) => Math.round(v + (c1[j] - v) * k));
}

function render(m) {
  if (!m.canvas) {
    m.canvas = document.createElement("canvas");
    m.canvas.width = GW;
    m.canvas.height = GH;
  }
  const ctx = m.canvas.getContext("2d");
  const img = ctx.createImageData(GW, GH);
  const peak = m.peak || 1;
  for (let i = 0; i < GW * GH; i++) {
    const t = Math.sqrt(m.grid[i] / peak); // sqrt: faint paths stay visible
    const o = i * 4;
    if (t < 0.04) {
      img.data[o + 3] = 0;
      continue;
    }
    const [r, g, b] = ramp(t);
    img.data[o] = r;
    img.data[o + 1] = g;
    img.data[o + 2] = b;
    img.data[o + 3] = Math.round(40 + 175 * t);
  }
  ctx.putImageData(img, 0, 0);
  m.stale = false;
}

/** Paint this camera's heat over `fit` (the picture's rect inside the tile). */
export function draw(ctx, cameraId, fit) {
  if (!enabled) return;
  const m = maps.get(cameraId);
  if (!m || m.peak <= 0) return;
  if (m.stale) render(m);
  ctx.save();
  ctx.globalAlpha = 0.65;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(m.canvas, fit.x, fit.y, fit.w, fit.h);
  ctx.restore();
}
