// A step-by-step walkthrough: each step dims the page except one element (its target) and shows
// a card next to it. A step whose target isn't on screen (e.g. no paper open yet) is shown as a
// card in the middle. ← / → step through it, Esc ends it.

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
};

const GAP = 12;  // px between the target and the card

function visibleRect(node) {
  if (!node) return null;
  const r = node.getBoundingClientRect();
  if (r.width < 4 || r.height < 4 || r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) return null;
  return r;
}

// steps: [{target: () => Element | null, title, body: string | Node, before?: () => void}]
// Returns {hide, resume}: hide() takes the tour off screen (e.g. while a panel it opened is in use)
// and resume() brings it back at the next step. onClose runs once, when it ends.
export function startTour(steps, { onClose } = {}) {
  let index = 0;
  const spot = el("div", { class: "tour-spot" });
  const card = el("div", { class: "tour-card", role: "dialog", "aria-modal": "true", "aria-labelledby": "tour-title" });
  const root = el("div", { class: "tour" }, spot, card);
  document.body.append(root);

  const close = () => {
    root.remove();
    removeEventListener("resize", place);
    document.removeEventListener("keydown", onKey, true);
    onClose?.();
  };
  const go = (i) => {
    if (i < 0) return;
    if (i >= steps.length) return close();
    index = i;
    steps[i].before?.();
    render();
  };

  function render() {
    const s = steps[index];
    const last = index === steps.length - 1;
    card.replaceChildren(
      el("div", { class: "tour-top" },
        el("span", { class: "tour-count muted" }, `${index + 1} of ${steps.length}`),
        el("button", { type: "button", class: "modal-close", title: "End the walkthrough (Esc)", onclick: close }, "×")),
      el("h3", { id: "tour-title" }, s.title),
      el("div", { class: "tour-body" }, s.body),
      el("div", { class: "tour-actions" },
        index ? el("button", { type: "button", class: "secondary", onclick: () => go(index - 1) }, "Back")
          : el("button", { type: "button", class: "secondary", onclick: close }, "Skip"),
        el("button", { type: "button", onclick: () => go(index + 1) }, last ? "Done" : "Next")),
    );
    // Wait a frame: before() may have opened a drawer or panel that the target is in. And again
    // once a drawer has slid into place (styles.css: 0.2s).
    requestAnimationFrame(() => { place(); card.querySelector(".tour-actions button:last-child").focus(); });
    setTimeout(place, 260);
  }

  function place() {
    const target = steps[index].target?.();
    target?.scrollIntoView?.({ block: "nearest" });
    const r = visibleRect(target);
    root.classList.toggle("centered", !r);
    if (!r) return;
    const pad = 6;
    Object.assign(spot.style, {
      left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px`,
    });
    if (matchMedia("(max-width: 640px)").matches) return; // the card is docked at the bottom (styles.css)
    // Beside the target where there's room (right, left, below, above), else over its middle.
    const cw = card.offsetWidth, ch = card.offsetHeight;
    const fits = {
      right: innerWidth - r.right - GAP >= cw, left: r.left - GAP >= cw,
      below: innerHeight - r.bottom - GAP >= ch, above: r.top - GAP >= ch,
    };
    let x, y;
    if (fits.right || fits.left) {
      x = fits.right ? r.right + GAP : r.left - GAP - cw;
      y = r.top + r.height / 2 - ch / 2;
    } else if (fits.below || fits.above) {
      y = fits.below ? r.bottom + GAP : r.top - GAP - ch;
      x = r.left + r.width / 2 - cw / 2;
    } else {
      x = r.left + r.width / 2 - cw / 2;
      y = r.top + r.height / 2 - ch / 2;
    }
    card.style.left = `${Math.max(8, Math.min(x, innerWidth - cw - 8))}px`;
    card.style.top = `${Math.max(8, Math.min(y, innerHeight - ch - 8))}px`;
  }

  const onKey = (e) => {
    if (root.hidden) return;
    if (e.key === "Escape") { e.stopPropagation(); close(); }
    else if (e.key === "ArrowRight") go(index + 1);
    else if (e.key === "ArrowLeft") go(index - 1);
  };
  addEventListener("resize", place);
  document.addEventListener("keydown", onKey, true);
  go(0);
  return {
    hide() { root.hidden = true; },
    resume() { root.hidden = false; go(index + 1); },
  };
}
