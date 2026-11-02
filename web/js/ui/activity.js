/**
 * The live console has one model stack: the status card. This footer used
 * to repeat a second inventory (decode, weapons, Qwen, phone, viewers,
 * tokens). It stays empty so those chips cannot show idle or Active.
 */

export function mountActivity(root) {
  if (!root) return () => {};
  root.hidden = true;
  root.style.display = "none";
  root.replaceChildren();
  root.removeAttribute("aria-label");
  return () => {};
}
