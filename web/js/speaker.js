/**
 * Call speaker: plays new call lines out loud with ElevenLabs voices, through
 * the backend's GET /voice/say (the API key never reaches the browser).
 * A simulated campus police line: nothing is dialed.
 *
 * Off by default. Only lines that arrive after it is switched on are spoken,
 * one at a time, in order.
 */

import { routes } from "./config.js?v=pro9";

let enabled = false;
let seen = -1;
const queue = [];
let playing = false;
let lastError = "";
const listeners = new Set();

function base() {
  const u = routes.dispatch("x");
  return u ? u.replace(/\/api\/incidents\/.*$/, "") : null;
}

export function available() {
  return Boolean(base());
}

export function isOn() {
  return enabled;
}

export function error() {
  return lastError;
}

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit() {
  for (const fn of listeners) fn();
}

export function setOn(on, transcriptLength = 0) {
  enabled = Boolean(on) && available();
  queue.length = 0;
  seen = transcriptLength; // speak only what comes next
  lastError = "";
  emit();
}

function playNext() {
  if (playing || !queue.length || !enabled) return;
  const line = queue.shift();
  const who = line.speaker === "dispatcher" ? "dispatcher" : "fw";
  const url = `${base()}/voice/say?who=${who}&text=${encodeURIComponent(line.text)}`;
  const audio = new Audio(url);
  playing = true;
  const done = () => {
    playing = false;
    playNext();
  };
  audio.addEventListener("ended", done, { once: true });
  audio.addEventListener(
    "error",
    () => {
      if (who === "fw") {
        lastError = "Voice unavailable: check ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env";
        emit();
      }
      done();
    },
    { once: true },
  );
  audio.play().catch(done);
}

/** Call with the store's transcript on every update. */
export function feed(transcript) {
  const lines = Array.isArray(transcript) ? transcript : [];
  if (seen < 0 || lines.length < seen) seen = lines.length; // first look / reset
  if (!enabled) {
    seen = lines.length;
    return;
  }
  for (let i = seen; i < lines.length; i++) {
    const l = lines[i];
    if (l && String(l.text || "").trim()) queue.push(l);
  }
  seen = lines.length;
  playNext();
}
