/**
 * Building tab: a 3D model of the monitored building, driven by the live store.
 *
 * Honesty rules (same as the call agent): only the ground floor has cameras,
 * so only its three camera zones show live data (people on screen per camera,
 * armed people, the open incident and its path between cameras). Upper floors
 * have no cameras in this demo and say so; they never show invented counts.
 * The floor plan is an illustrative layout, not surveyed geometry.
 */

import * as THREE from "../../vendor/three/three.module.min.js";
import { OrbitControls } from "../../vendor/three/OrbitControls.js";
import { pathHops, followedIncident } from "../tracking.js?v=pro9";

const FLOORS = 4;
const FLOOR_H = 4;
const W = 40; // building width (x)
const D = 22; // building depth (z)

// Ground-floor camera zones, in walking order (lobby → east → west).
const ZONES = [
  { cam: "cam-01", name: "Ground-floor lobby", x0: 9, x1: 19, z0: -2, z1: 10, cam3: [18, 3.4, 9.5] },
  { cam: "cam-02", name: "East corridor", x0: -3, x1: 9, z0: -2, z1: 2.5, cam3: [8.5, 3.4, 2.2] },
  { cam: "cam-03", name: "West corridor + stairwell", x0: -19, x1: -3, z0: -2, z1: 2.5, cam3: [-3.5, 3.4, 2.2] },
];

const C = {
  idle: 0x2a7de1,
  people: 0x2dd4bf,
  severe: 0xdc2626,
  glass: 0x93c5fd,
  slab: 0xe5e7eb,
  person: 0x0ea5e9,
};

function isOpen(inc) {
  return inc && !["RESOLVED", "DISMISSED"].includes(inc.state);
}

function labelSprite(text, { color = "#111827", bg = "rgba(255,255,255,0.92)", size = 28 } = {}) {
  const pad = 14;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = `600 ${size}px system-ui, -apple-system, Segoe UI, sans-serif`;
  const lines = String(text).split("\n");
  const tw = Math.max(...lines.map((l) => ctx.measureText(l).width));
  c.width = Math.ceil(tw + pad * 2);
  c.height = Math.ceil(lines.length * size * 1.25 + pad * 1.4);
  ctx.font = `600 ${size}px system-ui, -apple-system, Segoe UI, sans-serif`;
  ctx.fillStyle = bg;
  const r = 12;
  ctx.beginPath();
  ctx.roundRect(0, 0, c.width, c.height, r);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, pad, pad * 0.7 + i * size * 1.25));
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true });
  const s = new THREE.Sprite(mat);
  const k = 0.022;
  s.scale.set(c.width * k, c.height * k, 1);
  s.renderOrder = 10;
  s.userData.text = text;
  return s;
}

function setLabel(sprite, text, opts) {
  if (sprite.userData.text === text) return sprite;
  const fresh = labelSprite(text, opts);
  sprite.material.map.dispose();
  sprite.material.dispose();
  sprite.material = fresh.material;
  sprite.scale.copy(fresh.scale);
  sprite.userData.text = text;
  return sprite;
}

export function mountBuilding(root, store) {
  root.innerHTML = `
    <div class="b3d">
      <div class="b3d__stage" data-stage></div>
      <aside class="b3d__panel card">
        <h2 class="b3d__title">MacQuarrie Hall · 3D</h2>
        <p class="b3d__note">Illustrative layout. The ground floor is live from 3 cameras. Floors 2–${FLOORS} have no cameras in this demo, so they show no counts.</p>
        <ul class="b3d__zones" data-zones></ul>
        <div class="b3d__incident" data-incident></div>
        <div class="b3d__btns">
          <button type="button" class="btn btn--secondary btn--sm" data-explode aria-pressed="false">Separate floors</button>
          <button type="button" class="btn btn--secondary btn--sm" data-reset>Reset view</button>
        </div>
        <p class="b3d__legend"><span class="b3d__dot b3d__dot--idle"></span>No one on camera <span class="b3d__dot b3d__dot--people"></span>People on camera <span class="b3d__dot b3d__dot--severe"></span>Incident / armed</p>
        <p class="b3d__hint">Drag to orbit · scroll to zoom · right-drag to pan</p>
      </aside>
    </div>
  `;
  const stage = root.querySelector("[data-stage]");
  const zonesEl = root.querySelector("[data-zones]");
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
  renderer.shadowMap.enabled = true;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 500);
  const HOME = new THREE.Vector3(38, 34, 42);
  camera.position.copy(HOME);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 4, 2);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 15;
  controls.maxDistance = 140;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x94a3b8, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(30, 50, 20);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -40, right: 40, top: 40, bottom: -40 });
  scene.add(sun);

  // Ground + grid
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(160, 160),
    new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 1 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(160, 64, 0xcbd5e1, 0xe2e8f0);
  grid.position.y = -0.01;
  scene.add(grid);

  // Floors
  const floors = [];
  for (let f = 0; f < FLOORS; f++) {
    const g = new THREE.Group();
    const slab = new THREE.Mesh(
      new THREE.BoxGeometry(W, 0.35, D),
      new THREE.MeshStandardMaterial({ color: C.slab, roughness: 0.9, transparent: f > 0, opacity: f > 0 ? 0.55 : 1 }),
    );
    slab.position.y = -0.17;
    slab.receiveShadow = true;
    g.add(slab);
    const walls = new THREE.Mesh(
      new THREE.BoxGeometry(W, FLOOR_H - 0.35, D),
      new THREE.MeshPhysicalMaterial({
        color: C.glass, transparent: true, opacity: f === 0 ? 0.06 : 0.12, roughness: 0.15, depthWrite: false,
      }),
    );
    walls.position.y = (FLOOR_H - 0.35) / 2;
    g.add(walls);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(walls.geometry),
      new THREE.LineBasicMaterial({ color: 0x64748b, transparent: true, opacity: 0.45 }),
    );
    edges.position.copy(walls.position);
    g.add(edges);
    if (f > 0) {
      const tag = labelSprite(`Floor ${f + 1} · no camera coverage`, { color: "#475569", size: 22 });
      tag.position.set(-W / 2 + 7, 1.6, D / 2 + 0.5);
      g.add(tag);
    } else {
      const tag = labelSprite("Floor 1 · live from 3 cameras", { color: "#0f172a", size: 22 });
      tag.position.set(-W / 2 + 7, 0.9, D / 2 + 1.5);
      g.add(tag);
    }
    g.position.y = f * FLOOR_H;
    g.userData.baseY = f * FLOOR_H;
    scene.add(g);
    floors.push(g);
  }
  const ground0 = floors[0];

  // Zones on the ground floor
  const zones = ZONES.map((z) => {
    const w = z.x1 - z.x0;
    const d = z.z1 - z.z0;
    const mat = new THREE.MeshStandardMaterial({
      color: C.idle, emissive: C.idle, emissiveIntensity: 0.15, transparent: true, opacity: 0.55,
    });
    const tile = new THREE.Mesh(new THREE.BoxGeometry(w, 0.08, d), mat);
    tile.position.set((z.x0 + z.x1) / 2, 0.06, (z.z0 + z.z1) / 2);
    ground0.add(tile);
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(w, 2.6, d)),
      new THREE.LineBasicMaterial({ color: 0x1e3a8a, transparent: true, opacity: 0.35 }),
    );
    outline.position.set(tile.position.x, 1.3, tile.position.z);
    ground0.add(outline);
    // Camera marker
    const camMesh = new THREE.Mesh(
      new THREE.ConeGeometry(0.45, 1.1, 16),
      new THREE.MeshStandardMaterial({ color: 0x111827 }),
    );
    camMesh.position.set(...z.cam3);
    camMesh.rotation.z = Math.PI / 2.6;
    ground0.add(camMesh);
    const label = labelSprite(`${z.name}\n0 people`, { size: 24 });
    label.position.set(tile.position.x, 4.6, tile.position.z);
    ground0.add(label);
    const people = new THREE.Group();
    ground0.add(people);
    return { ...z, tile, mat, label, people, cx: tile.position.x, cz: tile.position.z };
  });
  const zoneByCam = new Map(zones.map((z) => [z.cam, z]));

  // Incident beam + path
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.5, 0.5, FLOOR_H * FLOORS + 6, 24, 1, true),
    new THREE.MeshBasicMaterial({ color: C.severe, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }),
  );
  beam.visible = false;
  ground0.add(beam);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(1.2, 1.6, 40),
    new THREE.MeshBasicMaterial({ color: C.severe, transparent: true, opacity: 0.8, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.visible = false;
  ground0.add(ring);
  let pathLine = null;

  const personGeo = new THREE.CapsuleGeometry(0.32, 1.0, 4, 10);
  const personMat = new THREE.MeshStandardMaterial({ color: C.person, roughness: 0.4 });
  const armedMat = new THREE.MeshStandardMaterial({ color: C.severe, emissive: C.severe, emissiveIntensity: 0.4 });

  function placePeople(z, boxes) {
    while (z.people.children.length) z.people.remove(z.people.children[0]);
    const persons = boxes.filter((b) => b && b.label !== "weapon");
    const armed = boxes.filter((b) => b && b.label === "weapon").length;
    persons.forEach((b, i) => {
      const u = Math.min(0.95, Math.max(0.05, Number(b.x) + Number(b.w) / 2));
      const v = Math.min(0.95, Math.max(0.05, Number(b.y) + Number(b.h)));
      const m = new THREE.Mesh(personGeo, i < armed ? armedMat : personMat);
      m.castShadow = true;
      m.position.set(z.x0 + u * (z.x1 - z.x0), 0.85, z.z0 + v * (z.z1 - z.z0));
      z.people.add(m);
    });
    return { count: persons.length, armed };
  }

  let last = "";
  function update(state) {
    const inc = followedIncident(state) || (state.selectedId && state.incidents?.[state.selectedId]) || null;
    const openInc = isOpen(inc) ? inc : null;
    const incCam = openInc?.camera_id || null;
    const rows = [];
    for (const z of zones) {
      const cam = state.cameras?.[z.cam] || {};
      const boxes = Array.isArray(cam.boxes) ? cam.boxes : [];
      const { count, armed } = placePeople(z, boxes);
      const hot = incCam === z.cam && openInc?.severity === "SEVERE";
      const color = hot || armed ? C.severe : count ? C.people : C.idle;
      z.mat.color.setHex(color);
      z.mat.emissive.setHex(color);
      z.mat.emissiveIntensity = hot ? 0.55 : count ? 0.3 : 0.12;
      z.hot = hot;
      const status = armed ? ` · ${armed} armed` : "";
      setLabel(z.label, `${z.name}\n${count} ${count === 1 ? "person" : "people"}${status}`, {
        color: hot || armed ? "#ffffff" : "#0f172a",
        bg: hot || armed ? "rgba(220,38,38,0.95)" : "rgba(255,255,255,0.92)",
        size: 24,
      });
      rows.push({ name: z.name, count, armed, hot, online: cam.online !== false });
    }
    // Incident beam on its camera zone
    const hz = incCam ? zoneByCam.get(incCam) : null;
    beam.visible = ring.visible = Boolean(hz && openInc);
    if (hz) {
      beam.position.set(hz.cx, (FLOOR_H * FLOORS + 6) / 2, hz.cz);
      ring.position.set(hz.cx, 0.15, hz.cz);
    }
    // Path between camera zones
    const hops = openInc ? pathHops(state, openInc, Infinity).hops : [];
    const pts = hops.map((h) => zoneByCam.get(h.cameraId)).filter(Boolean).map((z) => new THREE.Vector3(z.cx, 0.4, z.cz));
    const sig = pts.map((p) => `${p.x},${p.z}`).join("|");
    if (sig !== last) {
      last = sig;
      if (pathLine) {
        ground0.remove(pathLine);
        pathLine.geometry.dispose();
      }
      pathLine = null;
      if (pts.length >= 2) {
        const curve = new THREE.CatmullRomCurve3(pts);
        pathLine = new THREE.Mesh(
          new THREE.TubeGeometry(curve, 48, 0.18, 8, false),
          new THREE.MeshBasicMaterial({ color: C.severe }),
        );
        ground0.add(pathLine);
      }
    }
    // Side panel
    zonesEl.replaceChildren(
      ...rows.map((r) => {
        const li = document.createElement("li");
        li.className = `b3d__zone${r.hot || r.armed ? " is-hot" : r.count ? " is-busy" : ""}`;
        li.innerHTML = `<span class="b3d__zname"></span><span class="b3d__zcount mono"></span>`;
        li.firstChild.textContent = r.name;
        li.lastChild.textContent = r.online ? `${r.count}${r.armed ? ` · ${r.armed} armed` : ""}` : "offline";
        return li;
      }),
    );
    if (openInc) {
      const where = zoneByCam.get(openInc.camera_id)?.name || openInc.camera_id;
      const route = hops.map((h) => zoneByCam.get(h.cameraId)?.name || h.cameraId).join(" → ");
      incEl.textContent = `${String(openInc.class_token || "Incident").toLowerCase()} · ${String(openInc.severity || "").toLowerCase()} · now: ${where}${route.includes("→") ? ` · path: ${route}` : ""}`;
      incEl.hidden = false;
    } else {
      incEl.textContent = "";
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
    controls.target.set(0, 4, 2);
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
  const clock = new THREE.Clock();
  function frame() {
    raf = requestAnimationFrame(frame);
    const t = clock.getElapsedTime();
    floors.forEach((g, f) => {
      const want = g.userData.baseY + (exploded ? f * 3.5 : 0);
      g.position.y += (want - g.position.y) * 0.12;
    });
    for (const z of zones) if (z.hot) z.mat.emissiveIntensity = 0.35 + 0.3 * (0.5 + 0.5 * Math.sin(t * 5));
    if (ring.visible) {
      const s = 1 + 0.6 * ((t * 0.8) % 1);
      ring.scale.set(s, s, s);
      ring.material.opacity = 0.9 * (1 - ((t * 0.8) % 1));
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
      frame();
    } else {
      cancelAnimationFrame(raf);
    }
  }

  update(store.getState());
  const unsub = store.subscribe((state) => {
    setActive(state.route === "building");
    if (state.route === "building") update(state);
  });
  setActive(store.getState().route === "building");
  return () => {
    unsub();
    setActive(false);
    renderer.dispose();
  };
}
