import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { trackCardRows } from "./js/models.js";
import { cameraStream } from "./js/transport.js";
import { restartAll } from "./js/cameraSources.js";
import { createStore } from "./js/store.js";
import { startAutoFollow } from "./js/ui/autofollow.js";

globalThis.requestAnimationFrame = (fn) => {
  fn();
  return 1;
};

let played = 0;
const video = { readyState: 1, currentTime: 12, play: () => (played += 1) };
const stream = {
  src: "http://zgx-b505:8080/mjpeg/cam-01",
  removeAttribute() {
    this.src = "";
  },
  setAttribute(_name, value) {
    this.src = value;
  },
};
globalThis.window = {
  __cameraVideos: new Map([["cam-01", video]]),
  __cameraStreams: new Map([["cam-02", stream]]),
};
restartAll();
assert.equal(video.currentTime, 0);
assert.equal(played, 1);
assert.equal(stream.src, "http://zgx-b505:8080/mjpeg/cam-01");

const store = createStore();
store.handle({ type: "camera.online", camera_id: "cam-01", online: true });
assert.equal(store.getState().cameras["cam-01"]?.online, true);

let onState = null;
const opened = [];
startAutoFollow(
  {
    subscribe(fn) {
      onState = fn;
      return () => {};
    },
    getActiveSevere() {
      return null;
    },
  },
  { open: (tab) => opened.push(tab) },
  null,
);
onState({
  demo: { autoFollow: false },
  call: { incidentId: "inc-1" },
  incidents: { "inc-1": { incident_id: "inc-1", state: "DISPATCHED" } },
  order: ["inc-1"],
});
// Live shows incidents, map and call in its own columns: auto follow no
// longer opens panels.
assert.deepEqual(opened, []);

assert.deepEqual(cameraStream("cam-01", "ws://zgx-b505:8080/ws"), {
  kind: "mjpeg",
  url: "http://zgx-b505:8080/mjpeg/cam-01",
});
assert.equal(cameraStream("cam-04", "wss://demo.example/ws"), null);
assert.deepEqual(cameraStream("cam-03", "wss://demo.example/ws"), {
  kind: "mjpeg",
  url: "https://demo.example/mjpeg/cam-03",
});
const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
assert.match(html, /img-src[^;]*http:/);
assert.match(html, /media-src[^;]*http:/);

// The live call panel is a transcript, not a subtitle strip: a demo viewer
// must be able to scroll back through every line from the current call.
const liveCallSource = readFileSync(new URL("./js/ui/call.js", import.meta.url), "utf8");
assert.doesNotMatch(liveCallSource, /subtitleLines:\s*5/);

// The live page must fetch the full-transcript renderer by a fresh module URL.
// Without this chain, a browser can retain the former five-line subtitle module.
const mainSource = readFileSync(new URL("./js/main.js", import.meta.url), "utf8");
const liveSource = readFileSync(new URL("./js/ui/live.js", import.meta.url), "utf8");
assert.match(html, /main\.js\?v=pro16/);
assert.match(mainSource, /live\.js\?v=pro16/);
assert.match(liveSource, /call\.js\?v=pro10/);

const stack = ["YOLO", "VADCLIP", "ByteTrack"];
assert.deepEqual(
  trackCardRows().map((r) => [r.name, r.detail, r.active]),
  stack.map((name) => [name, "", true]),
);
assert.deepEqual(
  trackCardRows(null).map((r) => r.active),
  stack.map(() => true),
);
assert.equal(
  trackCardRows({ guardian: { state: "failed", http_status: 400 }, beeai: { state: "pending" } }).every((r) => r.active && r.detail === ""),
  true,
);

console.log("web self-check OK");
