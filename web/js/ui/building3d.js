/**
 * Building tab: a 3D model of the monitored building with a heat map per floor.
 *
 * Floor 1: heat from the camera tracker boxes (where people have been on each
 * camera, mapped into that camera's zone), plus the open incident (red beam)
 * and its path between cameras.
 * Floors 2+: no cameras. Their heat comes from a simulated badge tap-in feed
 * and is labelled "simulated" on the model and in the panel. Never names.
 * The floor plan is an illustrative layout, not surveyed geometry.
 */

import * as THREE from "../../vendor/three/three.module.min.js";
import { OrbitControls } from "../../vendor/three/OrbitControls.js";
import { pathHops, followedIncident } from "../tracking.js?v=pro9";

const FLOORS = 4;
const FLOOR_H = 4;
const W = 40; // building width (x)
const D = 22; // building depth (z)
const GW = 120; // heat grid columns (x)
const GD = 66; // heat grid rows (z)

// Ground-floor camera zones, in walking order (lobby → east → west).
const ZONES = [
  { cam: "cam-01", name: "Lobby", x0: 9, x1: 19, z0: -2, z1: 10 },
  { cam: "cam-02", name: "East corridor", x0: -3, x1: 9, z0: -2, z1: 2.5 },
  { cam: "cam-03", name: "West corridor", x0: -19, x1: -3, z0: -2, z1: 2.5 },
];

// Simulated badge activity on upper floors: stairwells, elevator, classrooms.
const HOTSPOTS = [
  [-17, -8], [17, -8], [0, -8],
  [-10, 4], [10, 4], [-2, 7], [14, 6], [-15, 6],
];

const SEVERE = 0xdc2626;

function isOpen(inc) {
  return inc && !["RESOLVED", "DISMISSED"].includes(inc.state);
}

function labelSprite(text, { color = "#0f172a", bg = "rgba(255,255,255,0.92)", size = 24 } = {}) {
  const pad = 12;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = `600 ${size}px system-ui, -apple-system, Segoe UI, sans-serif`;
  ctx.font = font;
  const lines = String(text).split("\n");
  const tw = Math.max(...lines.map((l) => ctx.measureText(l).width));
  c.width = Math.ceil(tw + pad * 2);
  c.height = Math.ceil(lines.length * size * 1.25 + pad * 1.2);
  ctx.font = font;
  ctx.fillStyle = bg;
  ctx.beginPath();
  ctx.roundRect(0, 0, c.width, c.height, 10);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, pad, pad * 0.6 + i * size * 1.25));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  const k = 0.034;
  s.scale.set(c.width * k, c.height * k, 1);
  s.renderOrder = 10;
  s.userData.key = `${text}|${color}|${bg}`;
  return s;
}

function setLabel(sprite, text, opts = {}) {
  const key = `${text}|${opts.color || "#0f172a"}|${opts.bg || "rgba(255,255,255,0.92)"}`;
  if (sprite.userData.key === key) return;
  const fresh = labelSprite(text, opts);
  sprite.material.map.dispose();
  sprite.material.dispose();
  sprite.material = fresh.material;
  sprite.scale.copy(fresh.scale);
  sprite.userData.key = key;
}

/** Blue → teal → yellow → red. */
function ramp(t) {
  const stops = [
    [0.0, [42, 125, 225]],
    [0.4, [45, 212, 191]],
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

/** A floor-sized heat layer: grid → canvas → texture on a plane. */
function makeHeatLayer() {
  const grid = new Float32Array(GW * GD);
  const canvas = document.createElement("canvas");
  canvas.width = GW;
  canvas.height = GD;
  const ctx = canvas.getContext("2d");
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(W, D),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.06;
  mesh.renderOrder = 2;
  return {
    grid,
    mesh,
    splat(x, z, weight = 1, radius = 2.4) {
      const fx = ((x + W / 2) / W) * GW;
      const fz = ((z + D / 2) / D) * GD;
      const r = Math.ceil(radius * 2);
      for (let gz = Math.floor(fz) - r; gz <= Math.floor(fz) + r; gz++) {
        if (gz < 0 || gz >= GD) continue;
        for (let gx = Math.floor(fx) - r; gx <= Math.floor(fx) + r; gx++) {
          if (gx < 0 || gx >= GW) continue;
          const d2 = (gx + 0.5 - fx) ** 2 + (gz + 0.5 - fz) ** 2;
          grid[gz * GW + gx] += weight * Math.exp(-d2 / (2 * radius * radius));
        }
      }
    },
    decay(k) {
      for (let i = 0; i < grid.length; i++) grid[i] *= k;
    },
    paint(peakFloor = 0) {
      let peak = peakFloor;
      for (let i = 0; i < grid.length; i++) if (grid[i] > peak) peak = grid[i];
      const img = ctx.createImageData(GW, GD);
      for (let i = 0; i < grid.length; i++) {
        const t = peak > 0 ? Math.sqrt(grid[i] / peak) : 0;
        if (t < 0.05) continue;
        const o = i * 4;
        const [r, g, b] = ramp(t);
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = Math.round(30 + 200 * t);
      }
      ctx.putImageData(img, 0, 0);
      tex.needsUpdate = true;
    },
  };
}

export function mountBuilding(root, store) {
  root.innerHTML = `
    <div class="b3d">
      <div class="b3d__stage" data-stage></div>
      <aside class="b3d__panel card">
        <h2 class="b3d__title">MacQuarrie Hall · activity heat map</h2>
        <p class="b3d__note">Where people have been, floor by floor. <b>Floor 1</b>: from the 3 cameras' person tracks. <b>Floors 2–${FLOORS}</b>: no cameras, so heat comes from badge tap-ins, <span class="b3d__sim">simulated</span> in this demo. No names, ever.</p>
        <div class="b3d__ramp" aria-hidden="true"></div>
        <p class="b3d__rampkey"><span>quiet</span><span>busy</span></p>
        <div class="b3d__incident" data-incident hidden></div>
        <div class="b3d__btns">
          <button type="button" class="btn btn--secondary btn--sm" data-explode aria-pressed="false">Separate floors</button>
          <button type="button" class="btn btn--secondary btn--sm" data-reset>Reset view</button>
        </div>
        <p class="b3d__hint">Drag to orbit · scroll to zoom · right-drag to pan</p>
      </aside>
    </div>
  `;
  const stage = root.querySelector("[data-stage]");
  const incEl = root.querySelector("[data-incident]");
  const explodeBtn = root.querySelector("[data-explode]");
  const resetBtn = root.querySelector("[data-reset]");

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch {
    stage.innerHTML = `<p class="b3d__nogl">3D needs WebGL, which this browser has turned off.</p>`;
    return () => {};
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 500);
  const HOME = new THREE.Vector3(40, 26, 36);
  const TARGET = new THREE.Vector3(7, 3, 2);
  camera.position.copy(HOME);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(TARGET);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 15;
  controls.maxDistance = 140;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x94a3b8, 1.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(30, 50, 20);
  scene.add(sun);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(160, 160),
    new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.4;
  scene.add(ground);
  const gridHelper = new THREE.GridHelper(160, 64, 0xcbd5e1, 0xe2e8f0);
  gridHelper.position.y = -0.39;
  scene.add(gridHelper);

  // Floors: a slab, an outline, a heat layer, a small label.
  const floors = [];
  for (let f = 0; f < FLOORS; f++) {
    const g = new THREE.Group();
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(W, 0.3, D),
      new THREE.MeshStandardMaterial({
        color: f === 0 ? 0xe2e8f0 : 0xcbd5e1, roughness: 0.9, transparent: true, opacity: f === 0 ? 0.95 : 0.35, depthWrite: f === 0,
      }),
    );
    slab.position.y = -0.15;
    g.add(slab);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(W, FLOOR_H - 0.3, D)),
      new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.4 }),
    );
    edges.position.y = (FLOOR_H - 0.3) / 2;
    g.add(edges);
    const heat = makeHeatLayer();
    g.add(heat.mesh);
    const tag = labelSprite(f === 0 ? "Floor 1 · cameras" : `Floor ${f + 1} · badge activity (simulated)`, {
      color: f === 0 ? "#0f172a" : "#5b21b6",
      size: 20,
    });
    tag.position.set(-W / 2 - 1, 0.8, D / 2);
    g.add(tag);
    g.position.y = f * FLOOR_H;
    g.userData = { baseY: f * FLOOR_H, heat };
    scene.add(g);
    floors.push(g);
  }
  const g0 = floors[0];

  // Camera zones on floor 1: thin outlines + a name, red when the incident is there.
  const zones = ZONES.map((z) => {
    const cx = (z.x0 + z.x1) / 2;
    const cz = (z.z0 + z.z1) / 2;
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(z.x1 - z.x0, 0.05, z.z1 - z.z0)),
      new THREE.LineBasicMaterial({ color: 0x1e3a8a, transparent: true, opacity: 0.6 }),
    );
    outline.position.set(cx, 0.08, cz);
    g0.add(outline);
    const label = labelSprite(z.name, { size: 22 });
    label.position.set(cx, 3.2, cz);
    g0.add(label);
    return { ...z, cx, cz, outline, label, seen: null };
  });
  const zoneByCam = new Map(zones.map((z) => [z.cam, z]));

  // Incident beam, ring and path.
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.5, 0.5, FLOOR_H * FLOORS + 6, 24, 1, true),
    new THREE.MeshBasicMaterial({ color: SEVERE, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }),
  );
  beam.visible = false;
  g0.add(beam);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(1.2, 1.6, 40),
    new THREE.MeshBasicMaterial({ color: SEVERE, transparent: true, side: THREE.DoubleSide, depthWrite: false }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.visible = false;
  g0.add(ring);
  let pathLine = null;
  let pathSig = "";

  // Floor 1 heat: each new set of boxes on a camera adds heat in its zone.
  function addCameraHeat(state) {
    const heat = g0.userData.heat;
    let changed = false;
    for (const z of zones) {
      const boxes = state.cameras?.[z.cam]?.boxes;
      if (!Array.isArray(boxes) || boxes === z.seen) continue;
      z.seen = boxes;
      changed = true;
      for (const b of boxes) {
        if (!b || b.label === "weapon") continue;
        const u = Math.min(0.97, Math.max(0.03, Number(b.x) + Number(b.w) / 2));
        const v = Math.min(0.97, Math.max(0.03, Number(b.y) + Number(b.h)));
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        heat.splat(z.x0 + u * (z.x1 - z.x0), z.z0 + v * (z.z1 - z.z0), 1, 2.6);
      }
    }
    if (changed) {
      heat.decay(0.992);
      heat.paint();
    }
  }

  // Upper floors: simulated badge activity around stairwells, elevator, rooms.
  function paintBadgeHeat(t) {
    for (let f = 1; f < FLOORS; f++) {
      const heat = floors[f].userData.heat;
      heat.grid.fill(0);
      HOTSPOTS.forEach(([x, z], i) => {
        const w = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * 0.25 + i * 1.7 + f * 2.3));
        heat.splat(x, z, w, 3.2 + ((i + f) % 3));
      });
      heat.paint(1.2);
    }
  }

  function update(state) {
    const inc = followedIncident(state) || (state.selectedId && state.incidents?.[state.selectedId]) || null;
    const openInc = isOpen(inc) ? inc : null;
    const incCam = openInc?.camera_id || null;
    addCameraHeat(state);
    for (const z of zones) {
      const hot = incCam === z.cam && openInc?.severity === "SEVERE";
      z.outline.material.color.setHex(hot ? SEVERE : 0x1e3a8a);
      z.outline.material.opacity = hot ? 1 : 0.6;
      setLabel(z.label, z.name, hot ? { color: "#ffffff", bg: "rgba(220,38,38,0.95)", size: 22 } : { size: 22 });
    }
    const hz = incCam ? zoneByCam.get(incCam) : null;
    beam.visible = ring.visible = Boolean(hz);
    if (hz) {
      beam.position.set(hz.cx, (FLOOR_H * FLOORS + 6) / 2, hz.cz);
      ring.position.set(hz.cx, 0.12, hz.cz);
    }
    const hops = openInc ? pathHops(state, openInc, Infinity).hops : [];
    const pts = hops.map((h) => zoneByCam.get(h.cameraId)).filter(Boolean).map((z) => new THREE.Vector3(z.cx, 0.3, z.cz));
    const sig = pts.map((p) => `${p.x},${p.z}`).join("|");
    if (sig !== pathSig) {
      pathSig = sig;
      if (pathLine) {
        g0.remove(pathLine);
        pathLine.geometry.dispose();
        pathLine = null;
      }
      if (pts.length >= 2) {
        pathLine = new THREE.Mesh(
          new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 48, 0.16, 8, false),
          new THREE.MeshBasicMaterial({ color: SEVERE }),
        );
        g0.add(pathLine);
      }
    }
    if (openInc) {
      const where = zoneByCam.get(openInc.camera_id)?.name || openInc.camera_id;
      const route = hops.map((h) => zoneByCam.get(h.cameraId)?.name || h.cameraId).join(" → ");
      incEl.textContent = `${String(openInc.class_token || "Incident").toLowerCase()} · now in ${where}${route.includes("→") ? ` · path: ${route}` : ""}`;
      incEl.hidden = false;
    } else {
      incEl.hidden = true;
    }
  }

  let exploded = false;
  explodeBtn.addEventListener("click", () => {
    exploded = !exploded;
    explodeBtn.setAttribute("aria-pressed", String(exploded));
    explodeBtn.textContent = exploded ? "Stack floors" : "Separate floors";
  });
  resetBtn.addEventListener("click", () => {
    camera.position.copy(HOME);
    controls.target.copy(TARGET);
  });

  function resize() {
    const r = stage.getBoundingClientRect();
    if (!r.width || !r.height) return;
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);

  let raf = 0;
  let active = false;
  let lastBadge = -1;
  const clock = new THREE.Clock();
  function frame() {
    raf = requestAnimationFrame(frame);
    const t = clock.getElapsedTime();
    floors.forEach((g, f) => {
      const want = g.userData.baseY + (exploded ? f * 4 : 0);
      g.position.y += (want - g.position.y) * 0.12;
    });
    if (lastBadge < 0 || t - lastBadge > 0.5) {
      lastBadge = t;
      paintBadgeHeat(t);
    }
    if (ring.visible) {
      const p = (t * 0.8) % 1;
      ring.scale.setScalar(1 + 0.8 * p);
      ring.material.opacity = 0.9 * (1 - p);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  function setActive(on) {
    if (on === active) return;
    active = on;
    if (on) {
      resize();
      clock.start();
      lastBadge = -1;
      frame();
    } else {
      cancelAnimationFrame(raf);
    }
  }

  update(store.getState());
  const unsub = store.subscribe((state) => {
    // Floor 1 heat accumulates on every tab; drawing only runs on this one.
    update(state);
    setActive(state.route === "building");
  });
  setActive(store.getState().route === "building");
  return () => {
    unsub();
    setActive(false);
    renderer.dispose();
  };
}
