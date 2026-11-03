import * as pdfjsLib from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
import { marked } from "https://cdn.jsdelivr.net/npm/marked@15.0.12/lib/marked.esm.js";
import DOMPurify from "https://cdn.jsdelivr.net/npm/dompurify@3.2.6/dist/purify.es.mjs";
import katex from "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.mjs";

// Browsers refuse cross-origin module workers, so start the CDN worker from a same-origin blob.
const PDF_WORKER = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
// One long-lived worker shared by all documents (destroying a document leaves it running).
const pdfWorker = new pdfjsLib.PDFWorker({
  port: new Worker(
    URL.createObjectURL(new Blob([`import "${PDF_WORKER}";`], { type: "text/javascript" })),
    { type: "module" },
  ),
});

const $ = (id) => document.getElementById(id);
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

// A trash-can glyph that takes the text color, for delete buttons.
const trashIcon = () => {
  const t = document.createElement("template");
  t.innerHTML = `<svg class="icon" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor"
    stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M2.5 4h11M6.5 4V2.5h3V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5"/></svg>`;
  return t.content.firstElementChild;
};

const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

const state = {
  providers: [],
  paper: null,          // current paper metadata
  pdf: null,            // pdfjs document
  conv: null,           // current conversation {id, title, provider, model, ...} or null
  pendingSelection: null,
  glossary: null,       // {exact, folded} maps of normalized term -> {term, description}, entries
  refs: null,           // {"theorem 3.1": {kind, number, label, content, page}}
  bib: null,            // {entries: {id: {number, authors, title, year, ..., arxiv_id, lookup}}, indexes}
  streaming: false,
  abort: null,          // AbortController of the answer being streamed
  searchMode: false,    // home page: no paper open, the chat is a search session
  scale: store.get("pdfScale", null),  // chosen with the zoom buttons; null fits the page to the viewer
  layoutScale: null,    // the scale the pages are laid out at
};

// ---------------------------------------------------------------- API

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

// A failed response's error text. FastAPI's validation errors (422) carry a list of {loc, msg}
// rather than a string.
async function errorMessage(res) {
  try {
    const { detail } = await res.json();
    if (Array.isArray(detail)) return detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join("; ");
    return detail || res.statusText;
  } catch { return res.statusText; }
}

// ---------------------------------------------------------------- Routing
// #/                      -> home: a fresh search session (a chat with no paper)
// #/search/<convId>       -> an existing search session
// #/paper/<id>            -> paper with a fresh conversation
// #/paper/<id>/c/<convId> -> paper with an existing conversation
// #/open/<source>         -> load a paper by arXiv ID/URL or PDF URL, then a fresh conversation

// #/open/<source> loads any arXiv ID/URL or PDF URL (downloading it if new), then becomes #/paper/<id>.
function openHash(source) {
  return `#/open/${encodeURIComponent(source)}`;
}

function paperHash(paperId, convId) {
  return `#/paper/${encodeURIComponent(paperId)}` + (convId ? `/c/${convId}` : "");
}

function searchHash(convId) {
  return convId ? `#/search/${convId}` : "#/";
}

async function route() {
  const o = location.hash.match(/^#\/open\/(.+)$/);
  if (o) {
    try {
      const paper = await openPaper(decodeURIComponent(o[1]));
      resetChat();
      history.replaceState(null, "", paperHash(paper.id));
    } catch (e) {
      setStatus(`Error: ${e.message}`);
    }
    renderHistory();
    return;
  }
  const m = location.hash.match(/^#\/paper\/([^/]+)(?:\/c\/([\w]+))?/);
  if (!m) {
    state.paper = null; state.conv = null;
    showPaper(null);
    const s = location.hash.match(/^#\/search\/(\w+)/);
    try {
      if (s) await openConversation(s[1]);
      else resetChat();
    } catch (e) {
      resetChat();
      setStatus(`Error: ${e.message}`);
    }
    renderHistory();
    return;
  }
  const paperId = decodeURIComponent(m[1]);
  const convId = m[2] || null;
  try {
    if (!state.paper || state.paper.id !== paperId) await openPaper(paperId);
    if (convId) await openConversation(convId);
    else resetChat();
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
  renderHistory();
}

// ---------------------------------------------------------------- Papers

function setStatus(text) { $("paper-status").textContent = text || ""; }

// known: {title, authors, published} for a non-arXiv PDF found by search.
async function openPaper(source, known = {}) {
  setStatus("");
  setSearchMode(false); // shows the viewer, so the loading card is visible
  const loading = startLoading("Opening paper…", known.title || source);
  try {
    // Idempotent: returns the stored copy if already ingested, otherwise downloads it.
    const paper = await api("/api/papers", { method: "POST", body: { url: source, ...known } });
    state.paper = paper;
    showPaper(paper);
    loading.step("Rendering pages…", paper.title);
    await renderPdf(paper);
    loadAnalysis(paper);
    checkNewerVersion(paper);
    return paper;
  } finally {
    loading.done();
  }
}

// ---------------------------------------------------------------- arXiv versions

// A paper is pinned to the arXiv version its text came from. An earlier version kept when a paper
// was updated is a paper of its own, its id carrying the version ("2210.02747v2").
const isArchived = (p) => /v\d+$/.test(p.id);
const paperLabel = (p) => (p.arxiv_version && !isArchived(p) ? `${p.id}v${p.arxiv_version}` : p.id);

// When arXiv has a newer version of the open paper, a notice offers it: getting it processes it
// like a new paper, in this one's place, and moves this version (and its conversations) to the
// Archive folder.
async function checkNewerVersion(paper) {
  showVersionNotice(null);
  if (!paper.arxiv_version || isArchived(paper)) return;
  let version = null;
  try { ({ version } = await api(`/api/papers/${encodeURIComponent(paper.id)}/newer`)); } catch { return; }
  if (version && state.paper?.id === paper.id) showVersionNotice(paper, version);
}

function showVersionNotice(paper, version) {
  const box = $("version-notice");
  box.hidden = !paper;
  if (!paper) return box.replaceChildren();
  const current = paper.arxiv_version;
  box.replaceChildren(
    el("span", {}, `A newer version of this paper is on arXiv: v${version} (you're reading v${current}). `,
      el("span", { class: "muted" }, `Getting it adds v${version} in this paper's place and moves v${current}, `
        + "with its conversations, to the Archive folder.")),
    el("button", { type: "button", onclick: (e) => updatePaper(paper, version, e.currentTarget) }, `Get v${version}`));
}

async function updatePaper(paper, version, btn) {
  btn.disabled = true;
  const loading = startLoading(`Getting v${version}…`, paper.title);
  try {
    const updated = await api(`/api/papers/${encodeURIComponent(paper.id)}/update`, { method: "POST" });
    state.paper = null;  // same id, new paper: open it afresh
    await openPaper(updated.id);
    resetChat();  // the open conversation went to the archive with the old version
    history.replaceState(null, "", paperHash(updated.id));
    setStatus(`v${paper.arxiv_version} is in the Archive folder, with its conversations.`);
  } catch (e) {
    btn.disabled = false;
    setStatus(`Error: ${e.message}`);
  } finally {
    loading.done();
    renderHistory();
  }
}

// Loading card over the viewer while a paper downloads and renders. Only the latest load
// may update or hide it.
let loadingToken = 0;
function startLoading(title, detail) {
  const token = ++loadingToken;
  const step = (t, d) => {
    if (token !== loadingToken) return;
    $("paper-loading-title").textContent = t;
    $("paper-loading-detail").textContent = d || "";
  };
  step(title, detail);
  $("paper-loading").hidden = false;
  return { step, done: () => { if (token === loadingToken) $("paper-loading").hidden = true; } };
}

// The three lookups, as listed in the analysis pill's popover.
const ANALYSIS_STAGES = [
  { key: "glossary", label: "Notation glossary", desc: "Symbols, acronyms and defined terms", unit: "terms" },
  { key: "refs", label: "Numbered elements", desc: "Theorems, equations, figures, sections and where proofs start", unit: "elements" },
  { key: "bib", label: "Bibliography", desc: "The reference list, so citations can be opened", unit: "references" },
];

// a: the /analysis response, or null. The pill (and its popover) shows while any stage is
// pending, and stays (as "incomplete") while any has failed, so it can be retried.
function showAnalysis(a) {
  const statuses = ANALYSIS_STAGES.map((s) => a?.[s.key]?.status || "");
  const pending = statuses.filter((x) => x === "pending").length;
  const failed = statuses.some((x) => x.startsWith("failed"));
  $("analysis-wrap").hidden = !pending && !failed;
  $("analysis-bar").hidden = !pending;
  $("analysis-pill").classList.toggle("failed", !pending && failed);
  if (!pending && !failed) return toggleAnalysisPop(false);
  $("analysis-label").textContent = pending
    ? `Analyzing paper · ${ANALYSIS_STAGES.length - pending}/${ANALYSIS_STAGES.length}` : "Analysis incomplete";
  $("analysis-stages").replaceChildren(...ANALYSIS_STAGES.map((s, i) => {
    const status = statuses[i];
    const n = Object.keys(a[s.key]?.entries || {}).length;
    let kind = "waiting", text = "Not started", reason = null, retry = null;
    if (status === "pending") { kind = "running"; text = n ? `${n} ${s.unit} so far…` : "Running…"; }
    else if (status === "ready") { kind = "done"; text = `${n} ${s.unit}`; }
    else if (status.startsWith("failed")) {
      kind = "failed";
      const resume = a[s.key].resumable;
      text = resume ? `${n} ${s.unit} kept` : "Failed";
      reason = el("span", { class: "stage-reason" }, status.replace(/^failed:\s*/, ""));
      retry = el("button", {
        type: "button", class: "stage-retry",
        title: resume ? "Ask the model to continue after the items it already produced" : "Run this lookup again",
        onclick: async () => {
          retry.disabled = true;
          try {
            await api(`/api/papers/${encodeURIComponent(state.paper.id)}/analysis/${s.key}/retry`, { method: "POST" });
          } catch (e) { retry.disabled = false; return alert(`Could not retry: ${e.message}`); }
          pollAnalysis(state.paper);
        },
      }, resume ? "Continue" : "Retry");
    }
    return el("li", { class: `stage ${kind}` },
      el("span", { class: "stage-icon", "aria-hidden": "true" }, { done: "✓", failed: "!", waiting: "·" }[kind] || ""),
      el("span", { class: "stage-text" }, el("strong", {}, s.label), el("span", { class: "muted" }, s.desc), reason),
      el("span", { class: "stage-side" }, el("span", { class: "stage-state" }, text), retry));
  }));
}

function toggleAnalysisPop(open = $("analysis-pop").hidden) {
  $("analysis-pop").hidden = !open;
  $("analysis-pill").setAttribute("aria-expanded", String(open));
}

$("analysis-pill").onclick = () => toggleAnalysisPop();
document.addEventListener("pointerdown", (e) => {
  if (!$("analysis-pop").hidden && !$("analysis-wrap").contains(e.target)) toggleAnalysisPop(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("analysis-pop").hidden) toggleAnalysisPop(false);
});

// ---------------------------------------------------------------- Numbered elements panel

// Generated by the model after ingest: {"theorem 3.1": {kind, number, label, content, page, proof}}.
function setRefs(entries) {
  state.refs = entries;
  const n = Object.keys(entries).length;
  $("elements-btn").hidden = !n;
  $("elements-btn").textContent = `Elements (${n})`;
}

const KIND_PLURALS = { hypothesis: "hypotheses" };
let elementsKind = ""; // the kind chip selected in the panel ("" = all)

// "Elements" panel: every numbered element in reading order (by page, then as the model listed them).
function openElements() {
  const items = Object.values(state.refs || {})
    .map((e, i) => ({ e, i }))
    .sort((a, b) => (a.e.page ?? Infinity) - (b.e.page ?? Infinity) || a.i - b.i)
    .map(({ e }) => e);
  const counts = {};
  for (const e of items) counts[e.kind] = (counts[e.kind] || 0) + 1;
  if (!counts[elementsKind]) elementsKind = "";
  const chip = (kind, label) => el("button", {
    type: "button", class: "kind-chip" + (kind === elementsKind ? " active" : ""), "data-kind": kind,
    onclick: () => {
      elementsKind = kind;
      for (const c of $("elements-kinds").children) c.classList.toggle("active", c.dataset.kind === kind);
      filterElements();
    },
  }, label);
  $("elements-kinds").replaceChildren(chip("", `All ${items.length}`),
    ...Object.entries(counts).map(([kind, n]) => {
      const name = n === 1 ? kind : KIND_PLURALS[kind] || `${kind}s`;
      return chip(kind, `${name[0].toUpperCase()}${name.slice(1)} ${n}`);
    }));
  const paperId = state.paper.id;
  $("elements-list").replaceChildren(...items.map((ref) => {
    const gotos = el("span", { class: "ref-gotos" });
    if (ref.page) gotos.append(el("button", { type: "button", class: "ref-goto",
      onclick: () => { closeElements(); scrollToPage(ref.page); } }, `Go to page ${ref.page}`));
    if (ref.proof) gotos.append(el("button", { type: "button", class: "ref-goto", title: `Proof on page ${ref.proof.page}`,
      onclick: () => { closeElements(); scrollToPassage(ref.proof.page, ref.proof.start); } }, "Go to proof"));
    const body = el("div", { class: "element-content" });
    body.innerHTML = renderMarkdown(ref.content);
    const item = el("article", { class: "element", "data-kind": ref.kind },
      el("div", { class: "ref-head" }, el("span", { class: "ref-label" }, ref.label), gotos));
    if (ref.kind === "figure" || ref.kind === "table") {
      // Cropped from the PDF on first request, so only once scrolled into view.
      const src = `/api/papers/${encodeURIComponent(paperId)}/crop?kind=${ref.kind}&number=${encodeURIComponent(ref.number)}`;
      const img = el("img", { class: "ref-img", src, alt: ref.label, loading: "lazy" });
      const link = el("a", { class: "ref-img-link", href: src, target: "_blank", title: "Open full size" }, img);
      img.onerror = () => link.remove();
      item.append(link);
    }
    item.append(body);
    item.dataset.search = `${ref.label} ${ref.content}`.toLowerCase();
    return item;
  }));
  $("elements-filter").value = "";
  filterElements();
  $("elements-modal").hidden = false;
  $("elements-list").scrollTop = 0;
  $("elements-filter").focus();
}

function filterElements() {
  const q = $("elements-filter").value.trim().toLowerCase();
  let shown = 0;
  for (const node of $("elements-list").children) {
    const match = (!elementsKind || node.dataset.kind === elementsKind) && (!q || node.dataset.search.includes(q));
    node.hidden = !match;
    if (match) shown++;
  }
  $("elements-empty").hidden = shown > 0;
}

function closeElements() { $("elements-modal").hidden = true; }

$("elements-btn").onclick = openElements;
$("elements-close").onclick = closeElements;
$("elements-filter").addEventListener("input", filterElements);
$("elements-modal").addEventListener("pointerdown", (e) => { if (e.target === $("elements-modal")) closeElements(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("elements-modal").hidden) closeElements();
});

// Element links in answers: "#ref:<kind>-<number>" -> scroll the paper to that element.
function refFromLink(a) {
  const m = a.getAttribute("href").match(/^#ref:([a-z]+)-(.+)$/i);
  return m && state.refs?.[`${m[1].toLowerCase()} ${m[2].toUpperCase()}`];
}

document.addEventListener("click", (e) => {
  const a = e.target.closest?.('a[href^="#ref:"]');
  if (!a) return;
  e.preventDefault();
  const ref = refFromLink(a);
  // The label's words find the element on its page (and flash it); otherwise the page top.
  if (ref?.page) scrollToPassage(ref.page, ref.label);
});
document.addEventListener("mouseover", (e) => {
  const a = e.target.closest?.('a[href^="#ref:"]');
  if (!a || a.title) return;
  const ref = refFromLink(a);
  a.title = ref ? `${ref.label}${ref.page ? ` (page ${ref.page})` : ""}` : "Not found in this paper's index";
});

// ---------------------------------------------------------------- Notation glossary

// Generated by the model after ingest: {term: description}. Indexed by normalized term so a
// PDF selection matches regardless of how the text layer spaced it ("d k" vs "dk").
const normTerm = (s) => s.normalize("NFKC").replace(/\s+/g, "");
const EDGE_PUNCT = /^[\s.,;:!?()\[\]"'“”‘’]+|[\s.,;:!?()\[\]"'“”‘’]+$/g;

// The lookups are generated by the model after ingest, and saved item by item as the reply
// streams, so poll while any is pending and use whatever entries exist so far.
async function loadAnalysis(paper) {
  state.glossary = state.refs = state.bib = null;
  $("notation-btn").hidden = $("elements-btn").hidden = true;
  showAnalysis(null);
  pollAnalysis(paper);
}

let analysisPolling = null; // id of the paper being polled, so a retry doesn't start a second loop
async function pollAnalysis(paper) {
  if (analysisPolling === paper.id) return;
  analysisPolling = paper.id;
  const applied = {}; // "status:count" per lookup, to re-index only when something changed
  try {
    for (;;) {
      if (state.paper?.id !== paper.id) return; // another paper was opened meanwhile
      let a;
      try { a = await api(`/api/papers/${encodeURIComponent(paper.id)}/analysis`); } catch { return showAnalysis(null); }
      if (state.paper?.id !== paper.id) return;
      let changed = false;
      for (const [key, set] of [["glossary", setGlossary], ["refs", setRefs], ["bib", setBib]]) {
        const { status, entries } = a[key];
        const sig = `${status}:${Object.keys(entries).length}`;
        if (applied[key] === sig) continue;
        applied[key] = sig;
        if (Object.keys(entries).length) { set(entries); changed = true; }
      }
      if (changed) indexesChanged();
      showAnalysis(a);
      for (const [name, x] of Object.entries(a)) if (x.status?.startsWith("failed")) console.warn(`${name}:`, x.status);
      if (!Object.values(a).some((x) => x.status === "pending")) return;
      await new Promise((res) => setTimeout(res, 2000));
    }
  } finally {
    if (analysisPolling === paper.id) analysisPolling = null;
  }
}

function setGlossary(entries) {
  const exact = new Map(), folded = new Map();
  for (const [term, description] of Object.entries(entries)) {
    const k = normTerm(term);
    if (!exact.has(k)) exact.set(k, { term, description });
    if (!folded.has(k.toLowerCase())) folded.set(k.toLowerCase(), { term, description });
  }
  state.glossary = { exact, folded, entries: Object.entries(entries) };
  $("notation-btn").hidden = !state.glossary.entries.length;
  $("notation-btn").textContent = `Notation (${state.glossary.entries.length})`;
}

// "Notation" panel: every glossary entry, in the order the model listed them (roughly paper order).
function openNotation() {
  const list = $("notation-list");
  list.replaceChildren();
  for (const [term, description] of state.glossary?.entries || []) {
    const dd = el("dd");
    dd.innerHTML = renderMarkdown(description);
    const dt = el("dt", {}, term);
    dt.dataset.search = dd.dataset.search = normTerm(`${term} ${description}`).toLowerCase();
    list.append(dt, dd);
  }
  $("notation-filter").value = "";
  filterNotation();
  $("notation-modal").hidden = false;
  $("notation-filter").focus();
}

function filterNotation() {
  const q = normTerm($("notation-filter").value).toLowerCase();
  let shown = 0;
  for (const node of $("notation-list").children) {
    const match = !q || node.dataset.search.includes(q);
    node.hidden = !match;
    if (match && node.tagName === "DT") shown++;
  }
  $("notation-empty").hidden = shown > 0;
}

function closeNotation() { $("notation-modal").hidden = true; }

$("notation-btn").onclick = openNotation;
$("notation-close").onclick = closeNotation;
$("notation-filter").addEventListener("input", filterNotation);
$("notation-modal").addEventListener("pointerdown", (e) => { if (e.target === $("notation-modal")) closeNotation(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("notation-modal").hidden) closeNotation();
});

function lookupGlossary(text) {
  const g = state.glossary;
  if (!g || text.length > 80) return null;
  // Exact (whitespace-insensitive) first, then without edge punctuation, then case-insensitive.
  for (const candidate of [text, text.replace(EDGE_PUNCT, "")]) {
    const k = normTerm(candidate);
    if (!k) continue;
    const hit = g.exact.get(k) || g.folded.get(k.toLowerCase());
    if (hit) return hit;
  }
  return null;
}

// ---------------------------------------------------------------- Automatic links in the PDF

// References the indexes know about become clickable in the text layer: numbered elements
// ("Theorem 3.1", "Table 2", "§4"), citations ("[18]", "Kalchbrenner et al., 2016") and
// notation-like glossary terms. Clicking one opens the same popover a selection would.
// Matching is per text-layer span, so a reference split across spans/lines isn't linked
// (selecting it still works).
let linkVersion = 0;   // bumped whenever an index arrives, so rendered pages get re-scanned
let termRegex = null;  // notation-like glossary terms
let linkCounter = 0;   // ids tying together the marks of one link split across spans

let REF_SCAN_RE = null; // built on first use: REF_NUM is defined further down
const NUM_CITE_RE = /\[\s*\d+(?:\s*[-–—,;]\s*\d+)*\s*\]/g;
const AUTHOR_YEAR_RE = /[A-Z\u00C0-\u017F][\p{L}'’-]+(?:\s+(?:et\s+al\.?|(?:and|&)\s+[\p{L}'’-]+))?,?\s*\(?(?:19|20)\d{2}[a-z]?\)?/gu;

// Symbols, acronyms and short variables; plain words ("attention") would underline half the page.
function notationLike(term) {
  if (term.length > 16 || /\s{2}/.test(term)) return false;
  // Hyphens/apostrophes alone don't make notation ("encoder-decoder"), nor do capitalized
  // phrases ("Concept Space") or two-letter words ("In").
  if (/[^\p{L}\s'’-]/u.test(term)) return true;
  if (/\s/.test(term)) return false;
  return /\p{Lu}.*\p{Lu}/u.test(term) || /^\p{L}$/u.test(term) || /^\p{Ll}{2}$/u.test(term);
}

function buildTermRegex() {
  const terms = (state.glossary?.entries || []).map(([t]) => t).filter(notationLike).filter((t) => t.trim().length > 1)
    .sort((a, b) => b.length - a.length).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s*"));
  termRegex = terms.length ? new RegExp(String.raw`(?<![\p{L}\p{N}])(?:` + terms.join("|") + String.raw`)(?![\p{L}\p{N}])`, "gu") : null;
}

function indexesChanged() {
  linkVersion++;
  buildTermRegex();
  for (const t of document.querySelectorAll("#pages .textLayer")) linkifyPage(t);
}

function findLinks(text, singleLetters = []) {
  const found = [];
  const take = (start, end, kind) => {
    if (end <= start || found.some((f) => start < f.end && end > f.start)) return;
    found.push({ start, end, kind });
  };
  const trimmed = (m) => {  // drop trailing punctuation/space (and an unopened ")") the patterns may swallow
    let s = m[0].replace(/[\s,.;:]+$/, "");
    if (s.endsWith(")") && !s.includes("(")) s = s.slice(0, -1);
    return [m.index, m.index + s.length, s];
  };
  if (state.refs) {
    REF_SCAN_RE ||= new RegExp(String.raw`\b[A-Za-z]+\.?\s*~?\(?\s*` + REF_NUM + String.raw`\s*\)?|§+\s*` + REF_NUM, "g");
    for (const m of text.matchAll(REF_SCAN_RE)) {
      const [a, b, s] = trimmed(m);
      if (lookupReference(s)) take(a, b, "ref");
    }
  }
  if (state.bib) {
    for (const re of [NUM_CITE_RE, AUTHOR_YEAR_RE]) {
      for (const m of text.matchAll(re)) {
        const [a, b, s] = trimmed(m);
        if (parseCitations(s).length) take(a, b, "cite");
      }
    }
  }
  if (state.glossary) {
    if (termRegex) for (const m of text.matchAll(termRegex)) take(m.index, m.index + m[0].length, "term");
    for (const [a, b] of singleLetters) take(a, b, "term");
  }
  return found;
}

// Matches run on the page's joined text, because the text layer splits runs freely (a citation's
// number is often its own span: "[", "13", "]"); each match is then marked in every span it covers.
function linkifyPage(textDiv) {
  if (!state.refs && !state.bib && !state.glossary) return;
  // A page still rendering its text is linked when it finishes (renderPage); marking it linked now
  // would skip it then.
  if (!textDiv.dataset.ready) return;
  if (Number(textDiv.dataset.linked || 0) === linkVersion + 1) return;
  textDiv.dataset.linked = linkVersion + 1;
  const pieces = [];
  let text = "";
  const singleLetters = [];
  for (const node of textDiv.querySelectorAll("span[role='presentation'], br")) {
    if (node.tagName === "BR") { text += "\n"; continue; }
    if (node.dataset.text !== undefined) node.textContent = node.dataset.text; // undo an earlier pass
    const t = node.textContent;
    // A one-letter variable counts only as a span of its own (math), never inside prose.
    if (t.trim().length === 1 && state.glossary && lookupGlossary(t.trim())) {
      const at = text.length + t.indexOf(t.trim());
      singleLetters.push([at, at + 1]);
    }
    pieces.push({ span: node, start: text.length, end: text.length + t.length });
    text += t;
  }
  const perSpan = new Map();
  for (const link of findLinks(text, singleLetters)) {
    const xref = text.slice(link.start, link.end).replace(/\s+/g, " ");
    const id = String(++linkCounter);
    for (const p of pieces) {
      if (p.end <= link.start || p.start >= link.end) continue;
      const list = perSpan.get(p) || [];
      list.push({ from: Math.max(link.start, p.start) - p.start, to: Math.min(link.end, p.end) - p.start, kind: link.kind, xref, id });
      perSpan.set(p, list);
    }
  }
  for (const [p, list] of perSpan) {
    const original = p.span.textContent;
    p.span.dataset.text = original;
    const parts = [];
    let at = 0;
    for (const l of list.sort((x, y) => x.from - y.from)) {
      if (l.from > at) parts.push(original.slice(at, l.from));
      const mark = el("mark", { class: `xref xref-${l.kind}` }, original.slice(l.from, l.to));
      mark.dataset.xref = l.xref;
      mark.dataset.link = l.id;
      parts.push(mark);
      at = l.to;
    }
    if (at < original.length) parts.push(original.slice(at));
    p.span.replaceChildren(...parts);
  }
}

// Links are invisible until hovered (or pressed, on a touch screen); a link split across spans
// lights up as a whole.
for (const [type, on] of [["pointerover", true], ["pointerout", false]]) {
  $("pages").addEventListener(type, (e) => {
    const mark = e.target.closest?.("mark.xref");
    if (!mark) return;
    for (const m of mark.closest(".textLayer").querySelectorAll(`mark[data-link="${mark.dataset.link}"]`)) m.classList.toggle("hover", on);
  });
}

function openLink(mark) {
  const pageDiv = mark.closest(".page");
  popoverSelection = { text: mark.dataset.xref, page: pageDiv ? Number(pageDiv.dataset.page) : null };
  showPopover(mark.getBoundingClientRect());
}

$("pages").addEventListener("click", (e) => {
  if (pointerType !== "mouse") return;  // taps are handled on pointerup, below
  const mark = e.target.closest?.("mark.xref");
  if (!mark || window.getSelection()?.toString().trim()) return; // a drag-selection, not a click
  e.stopPropagation();
  openLink(mark);
});

// A finger is far less precise than a mouse, and a link like "[18]" is smaller than a fingertip,
// so a tap just beside a link opens it too. Handled on pointerup rather than click, which iOS
// doesn't always send for a tap on plain text.
const TAP_SLOP = 14;  // px from a link's box
function linkNear(x, y) {
  const page = document.elementFromPoint(x, y)?.closest(".page");
  let best = null, bestDist = TAP_SLOP;
  for (const m of page?.querySelectorAll("mark.xref") || []) {
    const r = m.getBoundingClientRect();
    const d = Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom));
    if (d <= bestDist) { best = m; bestDist = d; }
  }
  return best;
}
let tapStart = null;
$("pages").addEventListener("pointerdown", (e) => {
  // A second finger means a pinch, not a tap.
  tapStart = e.pointerType === "mouse" || !e.isPrimary ? null : { x: e.clientX, y: e.clientY, t: e.timeStamp };
});
$("pages").addEventListener("pointerup", (e) => {
  const tap = tapStart;
  tapStart = null;
  // A scroll or a long-press (text selection) isn't a tap.
  if (!tap || Math.hypot(e.clientX - tap.x, e.clientY - tap.y) > 10 || e.timeStamp - tap.t > 500) return;
  const mark = e.target.closest?.("mark.xref") || linkNear(e.clientX, e.clientY);
  if (mark) openLink(mark);
});

// ---------------------------------------------------------------- Citations -> bibliography

const foldName = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z]/g, "");

function setBib(entries) {
  const byNumber = new Map(), byAuthorYear = new Map();
  for (const [id, e] of Object.entries(entries)) {
    if (e.number) byNumber.set(e.number, id);
    const surname = foldName(e.first_author || "");
    const year = (e.year || "").match(/\d{4}[a-z]?/)?.[0];
    if (surname && year) {
      for (const y of new Set([year, year.slice(0, 4)])) {
        const k = `${surname} ${y}`;
        byAuthorYear.set(k, [...(byAuthorYear.get(k) || []), id]);
      }
    }
  }
  state.bib = { entries, byNumber, byAuthorYear };
}

// Entry ids cited by a short selection: "[18]", "[9, 18]", "[16–18]", a bare "18", or author-year
// forms such as "Kalchbrenner et al., 2016" and "(Bahdanau et al. 2014; Luong et al. 2015)".
function parseCitations(text) {
  const b = state.bib;
  const t = text.replace(/\s+/g, " ").trim();
  if (!b || !t || t.length > 160) return [];
  const ids = [];
  const add = (id) => { if (id && !ids.includes(id)) ids.push(id); };
  if (b.byNumber.size) {
    const numberList = (s) => {
      for (const part of s.split(/[,;]/)) {
        const [lo, hi] = part.split(/[-–—]/).map((x) => parseInt(x, 10));
        if (Number.isNaN(lo)) continue;
        for (let n = lo; n <= (Number.isNaN(hi) || hi === undefined ? lo : Math.min(hi, lo + 30)); n++) add(b.byNumber.get(String(n)));
      }
    };
    const groups = [...t.matchAll(/\[\s*(\d+(?:\s*[-–—,;]\s*\d+)*)\s*\]/g)];
    if (groups.length) groups.forEach((m) => numberList(m[1]));
    else if (/^\[?\s*\d+(?:\s*[-–—,;]\s*\d+)*\s*\]?[.,;:]?$/.test(t)) numberList(t.replace(/[\[\].:]/g, ""));
  }
  if (b.byAuthorYear.size) {
    const re = /([A-Z\u00C0-\u017F][\p{L}'’-]+)(?:\s+(?:et\s+al\.?|(?:and|&)\s+[\p{L}'’-]+))?,?\s*\(?((?:19|20)\d{2}[a-z]?)\)?/gu;
    for (const m of t.matchAll(re)) (b.byAuthorYear.get(`${foldName(m[1])} ${m[2]}`) || []).forEach(add);
  }
  return ids.slice(0, 8);
}

function citeAuthors(authors) {
  if (!authors?.length) return "";
  return authors.length > 4 ? `${authors.slice(0, 3).join(", ")}, et al.` : authors.join(", ");
}

function renderCitations(ids) {
  const box = $("ask-cite");
  box.hidden = !ids.length;
  if (!ids.length) return;
  box.replaceChildren(el("div", { class: "cite-head" }, ids.length > 1 ? "Cited papers" : "Cited paper"));
  for (const id of ids) {
    const e = state.bib.entries[id];
    const source = e.arxiv_id || (e.url && /\.pdf($|\?)/i.test(e.url) ? e.url : null);
    const title = source
      ? el("a", { class: "cite-title", href: openHash(source), title: "Open this paper in the reader (new session)",
          onclick: () => hidePopover() }, e.title)
      : el("span", { class: "cite-title" }, e.title);
    const meta = [e.year, e.venue].filter(Boolean).join(" · ");
    const status = el("div", { class: "cite-status" });
    if (source) {
      status.append(e.arxiv_id ? `arXiv ${e.arxiv_id} ` : "PDF ",
        el("a", { href: openHash(source), target: "_blank", title: "Open in a new browser tab" }, "↗"));
    } else if (!e.lookup) {
      status.append("Looking up on arXiv…");
    } else {
      status.append("Not found on arXiv · ",
        el("a", { href: `https://scholar.google.com/scholar?q=${encodeURIComponent(e.title)}`, target: "_blank", rel: "noopener" },
          "Search Google Scholar ↗"));
      if (e.url) status.append(" · ", el("a", { href: e.url, target: "_blank", rel: "noopener" }, "Link ↗"));
    }
    box.append(el("div", { class: "cite-entry" },
      e.number ? el("span", { class: "cite-num" }, `[${e.number}]`) : null,
      el("div", { class: "cite-body" }, title,
        el("div", { class: "cite-authors" }, citeAuthors(e.authors)),
        meta ? el("div", { class: "cite-meta" }, meta) : null, status)));
  }
}

async function showCitations(ids, selectionText) {
  renderCitations(ids);
  const pending = ids.filter((id) => !state.bib.entries[id].lookup);
  if (!pending.length) return;
  const paperId = state.paper.id;
  let resolved = {};
  try {
    resolved = await api(`/api/papers/${encodeURIComponent(paperId)}/bib/resolve`, { method: "POST", body: { ids: pending } });
  } catch { return; }
  if (state.paper?.id !== paperId) return;
  Object.assign(state.bib.entries, resolved);
  // Re-render only if the popover still shows the same selection.
  if (!$("ask-popover").hidden && popoverSelection?.text === selectionText) renderCitations(ids);
}

// ---------------------------------------------------------------- Cross-references

// Words (and abbreviations) that introduce a reference, mapped to the kinds the server indexes.
const REF_WORDS = [
  [/^def(?:inition|n)?$/, "definition"], [/^(?:thm|theorem)$/, "theorem"], [/^lem(?:ma)?$/, "lemma"],
  [/^prop(?:osition)?$/, "proposition"], [/^cor(?:ollary)?$/, "corollary"], [/^conj(?:ecture)?$/, "conjecture"],
  [/^claim$/, "claim"], [/^(?:ass(?:umption)?|asm)$/, "assumption"], [/^hyp(?:othesis)?$/, "hypothesis"],
  [/^rem(?:ark)?$/, "remark"], [/^ex(?:ample)?$/, "example"], [/^prob(?:lem)?$/, "problem"],
  [/^alg(?:orithm)?$/, "algorithm"], [/^(?:eqn?|equation)$/, "equation"], [/^tab(?:le)?$/, "table"],
  [/^fig(?:ure)?$/, "figure"], [/^(?:sec(?:t|tion)?|subsection|appendix|app)$/, "section"],
];
const REF_NUM = String.raw`([A-Za-z]?\d+(?:\.\d+)*[a-z]?|[A-Z](?:\.\d+)*\b)`;
const REF_RE = new RegExp(String.raw`\b([A-Za-z]+)\.?\s*~?\(?\s*` + REF_NUM + String.raw`\s*\)?`, "g");
const SECTION_SIGN_RE = new RegExp(String.raw`§+\s*` + REF_NUM);
const BARE_EQ_RE = /^\(\s*(\d+(?:\.\d+)*[a-z]?)\s*\)$/; // a selected "(5)" is an equation number
const THEOREM_LIKE = ["theorem", "lemma", "proposition", "corollary", "definition", "conjecture", "claim",
  "assumption", "hypothesis", "remark", "example", "problem"];

const refNumber = (n) => n.replace(/[\s()]/g, "").replace(/^\.+|\.+$/g, "").toUpperCase();

// The first cross-reference in a short selection, as {kind, number} candidates to look up.
function parseReference(text) {
  const t = text.replace(/\s+/g, " ").trim();
  if (!t || t.length > 60) return null;
  const bare = t.replace(/[.,;:]+$/, "").match(BARE_EQ_RE);
  if (bare) return { kind: "equation", number: refNumber(bare[1]) };
  const sign = t.match(SECTION_SIGN_RE);
  if (sign) return { kind: "section", number: refNumber(sign[1]) };
  for (const m of t.matchAll(REF_RE)) {
    const word = m[1].toLowerCase(), singular = word.replace(/s$/, ""); // "Lemmas 4 and 5"
    const kind = REF_WORDS.find(([re]) => re.test(word) || re.test(singular))?.[1];
    if (kind) return { kind, number: refNumber(m[2]) };
  }
  return null;
}

function lookupReference(text) {
  const ref = parseReference(text);
  if (!ref || !state.refs) return null;
  const hit = state.refs[`${ref.kind} ${ref.number}`]
    || state.refs[`${ref.kind} ${ref.number.replace(/(\d)[A-Z]$/, "$1")}`]; // "Fig. 3a" -> Figure 3
  if (hit) return hit;
  // Papers often number theorems, lemmas, definitions... with one shared counter, and the text
  // may call an element by a different word than its heading; accept a unique match.
  if (THEOREM_LIKE.includes(ref.kind)) {
    const same = THEOREM_LIKE.map((k) => state.refs[`${k} ${ref.number}`]).filter(Boolean);
    if (same.length === 1) return same[0];
  }
  return null;
}

function showPaper(paper) {
  setSearchMode(!paper);
  $("empty-state").hidden = !!paper;
  const title = $("paper-title");
  title.replaceChildren();
  $("summarize-btn").hidden = !paper;
  if (paper) {
    const href = paper.arxiv_version ? `https://arxiv.org/abs/${paperLabel(paper)}` : paper.source_url;
    title.append(paper.title, el("a", { href, target: "_blank" }, paperLabel(paper)));
    document.title = `${paper.title} — Paper Reader`;
  } else {
    $("pages").replaceChildren();
    $("notation-btn").hidden = $("elements-btn").hidden = true; // the last paper's lookups
    document.title = "Paper Reader";
    setStatus("");
    showAnalysis(null);
    showVersionNotice(null);
  }
}

async function openFromTopbar(source, known) {
  hideSearchResults();
  try {
    const paper = await openPaper(source, known);
    $("paper-url").value = "";
    location.hash = paperHash(paper.id);
  } catch (err) {
    setStatus(`Error: ${err.message}`);
  }
}

// A link or arXiv ID opens directly; anything else is searched as a paper name.
const SOURCE_RE = /^(?:https?:\/\/|arxiv:)|arxiv\.org\/|^\d{4}\.\d{4,5}(?:v\d+)?$|^[a-z-]+(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?$/i;

$("paper-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("paper-url").value.trim();
  if (!text) return;
  if (SOURCE_RE.test(text)) openFromTopbar(text);
  else searchPapers(text);
});

// ---------------------------------------------------------------- Paper search

let searchToken = 0;

async function searchPapers(query) {
  const token = ++searchToken;
  setStatus("Searching…");
  let results;
  try {
    ({ results } = await api(`/api/search?q=${encodeURIComponent(query)}`));
  } catch (err) {
    if (token === searchToken) setStatus(`Error: ${err.message}`);
    return;
  }
  if (token !== searchToken) return;  // a newer search (or opening a paper) superseded this one
  setStatus("");
  const box = $("search-results");
  box.replaceChildren();
  if (!results.length) box.append(el("div", { class: "search-empty muted" }, `No papers found for “${query}”.`));
  for (const r of results) box.append(searchResult(r));
  box.hidden = false;
  box.querySelector(".search-result.openable")?.classList.add("active");
}

function searchResult(r) {
  const authors = r.authors.length > 3 ? `${r.authors.slice(0, 3).join(", ")} et al.` : r.authors.join(", ");
  const meta = [authors, r.year, r.venue].filter(Boolean).join(" · ");
  const badge = r.arxiv_id ? el("span", { class: "badge" }, `arXiv ${r.arxiv_id}`)
    : r.open ? el("span", { class: "badge", title: "Not on arXiv; opens the PDF" }, "PDF")
    : el("span", { class: "badge unsupported", title: "No arXiv version or downloadable PDF found" }, "unsupported");
  const external = r.url && el("a", {
    class: "ext", href: r.url, target: "_blank", rel: "noopener", title: "Open the paper's page in a new tab",
    onclick: (e) => e.stopPropagation(),
  }, "↗");
  const row = el("div", { class: "search-result" + (r.open ? " openable" : "") },
    el("div", { class: "body" }, el("div", { class: "t" }, r.title), meta && el("div", { class: "meta" }, meta)),
    badge, external);
  if (r.open) {
    const known = r.arxiv_id ? undefined : { title: r.title, authors: r.authors, published: r.year };
    row.onclick = () => openFromTopbar(r.open, known);
    row.onmouseenter = () => setActiveResult(row);
  }
  return row;
}

function hideSearchResults() {
  searchToken++;
  $("search-results").hidden = true;
}

function setActiveResult(row) {
  $("search-results").querySelectorAll(".active").forEach((n) => n.classList.remove("active"));
  row?.classList.add("active");
  row?.scrollIntoView({ block: "nearest" });
}

// Arrow keys move between openable results; Enter opens the highlighted one; Esc closes the list.
$("paper-url").addEventListener("keydown", (e) => {
  const box = $("search-results");
  if (box.hidden) return;
  if (e.key === "Escape") return hideSearchResults();
  const rows = [...box.querySelectorAll(".search-result.openable")];
  const i = rows.findIndex((n) => n.classList.contains("active"));
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (rows.length) setActiveResult(rows[(i + (e.key === "ArrowDown" ? 1 : rows.length - 1)) % rows.length]);
  } else if (e.key === "Enter" && i >= 0 && !e.isComposing) {
    e.preventDefault();
    rows[i].click();
  }
});
// Editing the query hides results for the old one until Enter searches again.
$("paper-url").addEventListener("input", () => { if (!$("search-results").hidden) hideSearchResults(); });
document.addEventListener("pointerdown", (e) => {
  if (!$("paper-form").contains(e.target)) $("search-results").hidden = true;
});

// ---------------------------------------------------------------- PDF viewer

let pageObserver = null;
let renderToken = 0;

async function renderPdf(paper) {
  const token = ++renderToken;
  const pages = $("pages");
  pages.replaceChildren();
  clearBackSpots();  // the last paper's places
  const old = state.pdf;
  state.pdf = null;  // not used while it's being destroyed (a resize re-fits from it)
  if (old) await old.destroy();
  // The file's hash in the URL: a paper updated to a newer version isn't shown from the cache.
  const url = `/api/papers/${encodeURIComponent(paper.id)}/pdf?sha=${paper.pdf_sha256 || ""}`;
  const cached = await cachedPdf(url);
  state.pdf = await pdfjsLib.getDocument(cached ? { data: cached, worker: pdfWorker } : { url, worker: pdfWorker }).promise;
  if (token !== renderToken) return;
  layoutPages(token);
  if (!cached && paper.pdf_sha256) cachePdf(url, paper.pdf_sha256, state.pdf);
}

// PDFs kept in the browser (Cache Storage), so a paper opened again doesn't wait for its download.
// On Vercel the file comes from arXiv, which doesn't let the browser's HTTP cache keep it. Keyed by
// the file's hash, so they never go stale; the least recently opened go first past PDF_CACHE_SIZE.
const PDF_CACHE = "pdfs-v1";
const PDF_CACHE_SIZE = 40;

async function cachedPdf(url) {
  try {
    const cache = await caches.open(PDF_CACHE);
    const hit = await cache.match(url);
    if (!hit) return null;
    const data = await hit.arrayBuffer();
    await cache.delete(url);  // put back last: the most recently opened
    await cache.put(url, new Response(data, { headers: { "Content-Type": "application/pdf" } }));
    return new Uint8Array(data);
  } catch { return null; }  // no Cache Storage (e.g. not https), or it was cleared
}

// Once pdf.js has the whole file (it streams it in ranges), stores it if it's the pinned version.
async function cachePdf(url, sha, pdf) {
  try {
    const data = await pdf.getData();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", data)),
      (b) => b.toString(16).padStart(2, "0")).join("");
    if (hash !== sha) return;
    const cache = await caches.open(PDF_CACHE);
    await cache.put(url, new Response(data, { headers: { "Content-Type": "application/pdf" } }));
    const keys = await cache.keys();  // oldest first
    for (const old of keys.slice(0, Math.max(0, keys.length - PDF_CACHE_SIZE))) await cache.delete(old);
  } catch {}  // the paper was closed (pdf destroyed) or storage is full: not cached this time
}

async function layoutPages(token) {
  const pdf = state.pdf;
  const pages = $("pages");
  pages.replaceChildren();
  const first = await pdf.getPage(1);
  const base = first.getViewport({ scale: 1 });
  const scale = state.scale || fitScale(base);
  state.layoutScale = scale;
  // At "fit page" zoom on a touch screen, scrolling snaps from page to page (see styles.css).
  $("viewer").classList.toggle("paged", !state.scale && touchScreen.matches);

  if (pageObserver) pageObserver.disconnect();
  pageObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting && !entry.target.dataset.rendered) {
        entry.target.dataset.rendered = "1";
        renderPage(entry.target, scale, token);
      }
    }
  }, { root: $("viewer"), rootMargin: "800px 0px" });

  for (let i = 1; i <= pdf.numPages; i++) {
    const div = el("div", { class: "page", "data-page": i });
    div.style.width = `${base.width * scale}px`;
    div.style.height = `${base.height * scale}px`;
    div.append(el("span", { class: "page-num" }, String(i)));
    pages.append(div);
    pageObserver.observe(div);
  }
}

async function renderPage(div, scale, token) {
  if (!state.pdf) return;
  const page = await state.pdf.getPage(Number(div.dataset.page));
  if (token !== renderToken) return;
  const viewport = page.getViewport({ scale });
  const dpr = window.devicePixelRatio || 1;
  div.style.width = `${viewport.width}px`;
  div.style.height = `${viewport.height}px`;
  div.style.setProperty("--scale-factor", scale);
  div.style.setProperty("--user-unit", 1);
  div.style.setProperty("--total-scale-factor", scale);

  const canvas = el("canvas");
  canvas.width = Math.floor(viewport.width * dpr);
  canvas.height = Math.floor(viewport.height * dpr);
  canvas.style.width = `${viewport.width}px`;
  canvas.style.height = `${viewport.height}px`;
  const textDiv = el("div", { class: "textLayer" });
  div.prepend(canvas, textDiv);

  await page.render({
    canvasContext: canvas.getContext("2d"),
    viewport,
    transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null,
  }).promise;
  const textLayer = new pdfjsLib.TextLayer({
    textContentSource: page.streamTextContent(),
    container: textDiv,
    viewport,
  });
  await textLayer.render();
  textDiv.dataset.ready = "1";
  linkifyPage(textDiv);
}

// With a mouse, the page fills the viewer's width. A touch screen shows one whole page at a time.
const touchScreen = matchMedia("(hover: none)");
function fitScale(base) {
  const viewer = $("viewer");
  if (!touchScreen.matches) return Math.min(2, (viewer.clientWidth - 100) / base.width);
  // Room for the page number beside the page, and the gap above it.
  return Math.min((viewer.clientWidth - 72) / base.width, (viewer.clientHeight - 24) / base.height);
}

// The spot of the paper at screen point (x, y), by default the top of the viewer: its page and
// where on that page, as fractions of its size, so it can be found again at another zoom.
function paperSpot(x, y) {
  const box = $("viewer").getBoundingClientRect();
  x ??= box.left; y ??= box.top + 1;
  const pages = [...$("pages").children];
  const page = pages.find((d) => d.getBoundingClientRect().bottom > y) || pages.at(-1);
  if (!page) return null;
  const r = page.getBoundingClientRect();
  return { n: page.dataset.page, fx: (x - r.left) / r.width, fy: Math.max(0, (y - r.top) / r.height), x: x - box.left, y: y - box.top };
}

// Lays the pages out again at the current zoom, keeping the place being read in view: that spot
// of the paper stays at the same point on screen.
async function relayout(spot = paperSpot()) {
  await layoutPages(++renderToken);
  scrollToSpot(spot);
}

// Puts a spot from paperSpot() back at the point on screen it was at.
function scrollToSpot(spot, behavior = "auto") {
  const div = spot && document.querySelector(`.page[data-page="${spot.n}"]`);
  if (!div) return;
  $("viewer").scrollTo({
    top: div.offsetTop + spot.fy * div.offsetHeight - spot.y,
    left: div.offsetLeft + spot.fx * div.offsetWidth - spot.x,
    behavior,
  });
}

// A zoom close to the fitted size snaps back to it (null: re-fits on resize, pages one at a time).
async function setScale(scale, spot) {
  if (!state.pdf) return;
  const fit = fitScale((await state.pdf.getPage(1)).getViewport({ scale: 1 }));
  state.scale = Math.abs(scale / fit - 1) < 0.05 ? null : scale;
  store.set("pdfScale", state.scale);
  relayout(spot);
}

function zoom(factor) {
  if (!state.pdf) return;
  setScale(Math.max(0.5, Math.min(4, (state.scale || state.layoutScale || 1.3) * factor)));
}
$("zoom-in").onclick = () => zoom(1.15);
$("zoom-out").onclick = () => zoom(1 / 1.15);

// Pinch to zoom on a touch screen. While the fingers move, the rendered pages are only stretched
// (cheap); on release they're laid out again, sharp, at the new scale. The browser's own pinch,
// which would zoom the whole app, is turned off over the paper (styles.css and below).
let pinch = null;
const fingerGap = (t) => Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
$("viewer").addEventListener("touchstart", (e) => {
  if (e.touches.length !== 2 || !state.pdf) return;
  tapStart = null;  // two fingers down isn't a tap on a link
  const [a, b] = e.touches;
  const x = (a.clientX + b.clientX) / 2, y = (a.clientY + b.clientY) / 2;
  const box = $("pages").getBoundingClientRect();
  pinch = { gap: fingerGap(e.touches), ratio: 1, spot: paperSpot(x, y) };
  $("pages").style.transformOrigin = `${x - box.left}px ${y - box.top}px`;
}, { passive: true });
$("viewer").addEventListener("touchmove", (e) => {
  if (!pinch || e.touches.length !== 2) return;
  e.preventDefault();
  const scale = state.layoutScale * fingerGap(e.touches) / pinch.gap;
  pinch.ratio = Math.max(0.5, Math.min(4, scale)) / state.layoutScale;
  $("pages").style.transform = `scale(${pinch.ratio})`;
}, { passive: false });
const endPinch = (e) => {
  if (!pinch || e.touches.length >= 2) return;
  const { ratio, spot } = pinch;
  pinch = null;
  $("pages").style.transform = "";
  if (Math.abs(ratio - 1) < 0.02) return;
  setScale(state.layoutScale * ratio, spot);
};
$("viewer").addEventListener("touchend", endPinch);
$("viewer").addEventListener("touchcancel", endPinch);
// Safari's own gesture events: cancelling them keeps its page zoom out of the way.
$("viewer").addEventListener("gesturestart", (e) => e.preventDefault());

// Re-fit when the viewer changes size: rotating the iPad, showing or hiding the chat panel.
let refitTimer = null;
new ResizeObserver(() => {
  clearTimeout(refitTimer);
  refitTimer = setTimeout(async () => {
    if (!state.pdf || state.scale || !state.layoutScale) return;
    const base = (await state.pdf.getPage(1)).getViewport({ scale: 1 });
    if (Math.abs(fitScale(base) / state.layoutScale - 1) > 0.01) relayout();
  }, 150);
}).observe($("viewer"));

// Every jump to another part of the paper (a reference, Go to page/proof) first remembers where
// the reader was, kept as fractions of the page so it survives zooming. Back returns there; a
// stack, so a chain of references unwinds one step at a time.
let backSpots = [];

function updateBackButton() {
  const btn = $("paper-back");
  btn.hidden = !backSpots.length;
  if (backSpots.length) btn.title = `Back to page ${backSpots.at(-1).n}`;
}

function clearBackSpots() {
  backSpots = [];
  updateBackButton();
}

$("paper-back").onclick = () => {
  const spot = backSpots.pop();
  updateBackButton();
  scrollToSpot(spot, "smooth");
};

function scrollToPage(n) {
  const div = document.querySelector(`.page[data-page="${n}"]`);
  if (!div) return;
  const here = paperSpot();
  if (here) { backSpots.push(here); updateBackButton(); }
  div.scrollIntoView({ behavior: "smooth", block: "start" });
}

const passageWords = (s) => s.toLowerCase().match(/[a-z0-9]+/g) || [];

// Scroll to where a passage (e.g. a proof's first words) starts on page n and flash it there;
// the page top if the text layer doesn't contain it (garbled extraction, or still rendering).
async function scrollToPassage(n, passage) {
  scrollToPage(n);
  const words = passageWords(passage || "").slice(0, 6);
  const div = document.querySelector(`.page[data-page="${n}"]`);
  if (!div || words.length < 3) return;
  let spans = [];
  for (let i = 0; i < 40 && !spans.length; i++) {  // the page renders once it scrolls into range
    spans = [...div.querySelectorAll(".textLayer span[role='presentation']")];
    if (!spans.length) await new Promise((res) => setTimeout(res, 100));
  }
  // Matched with spaces ignored: the text layer splits some words, e.g. small-caps headings
  // ("R UBRIC -C ONDITIONED").
  const owner = [];  // span of each character of `text`
  let text = "";
  for (const span of spans) {
    const chars = passageWords(span.textContent).join("");
    text += chars;
    for (let k = 0; k < chars.length; k++) owner.push(span);
  }
  const target = words.join("");
  const at = text.indexOf(target);
  if (at < 0) return;
  const hit = [...new Set(owner.slice(at, at + target.length))];
  hit[0].scrollIntoView({ behavior: "smooth", block: "center" });
  for (const span of hit) {
    span.classList.remove("flash");
    void span.offsetWidth;  // restart the animation when jumping to the same place again
    span.classList.add("flash");
  }
}

// ---------------------------------------------------------------- Selection -> ask popover

let popoverSelection = null;

// The selected passage of the paper, or null when nothing in the PDF is selected.
function paperSelection() {
  const sel = window.getSelection();
  const text = sel?.toString().trim();
  if (!text || !sel.rangeCount || !$("pages").contains(sel.anchorNode)) return null;
  const pageDiv = sel.anchorNode.parentElement?.closest(".page");
  return { text, page: pageDiv ? Number(pageDiv.dataset.page) : null, rect: sel.getRangeAt(0).getBoundingClientRect() };
}

// With a mouse the selection is final on mouseup. A touch screen selects by long-press and drag
// handles, with no mouseup, so there the popover follows the selection once it stops changing,
// and a tap anywhere outside it dismisses it.
let pointerType = "mouse";
document.addEventListener("pointerdown", (e) => {
  pointerType = e.pointerType;
  // Tapping into the chat box attaches the selection instead (see the #question focus handler).
  if (pointerType !== "mouse" && !$("ask-popover").contains(e.target) && e.target !== $("question")) hidePopover();
}, true);

document.addEventListener("pointerup", (e) => {
  if (e.pointerType !== "mouse" || $("ask-popover").contains(e.target)) return;
  setTimeout(() => {
    const sel = paperSelection();
    if (!sel) {
      if (!$("ask-popover").contains(document.activeElement)) hidePopover();
      return;
    }
    popoverSelection = { text: sel.text, page: sel.page };
    showPopover(sel.rect);
  }, 0);
});

let selectionTimer = null;
document.addEventListener("selectionchange", () => {
  if (pointerType === "mouse") return;
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(() => {
    const sel = paperSelection();
    if (!sel) return;
    popoverSelection = { text: sel.text, page: sel.page };
    showPopover(sel.rect);
  }, 400);
});

function showPopover(rect) {
  const pop = $("ask-popover");
  $("ask-quote").textContent = popoverSelection.text;
  let wide = false;
  const ref = lookupReference(popoverSelection.text);
  const refBox = $("ask-ref");
  refBox.hidden = !ref;
  if (ref) {
    const gotos = el("span", { class: "ref-gotos" });
    const head = el("div", { class: "ref-head" }, el("span", { class: "ref-label" }, ref.label), gotos);
    if (ref.page) {
      gotos.append(el("button", { type: "button", class: "ref-goto", title: "Scroll the paper to this page",
        onclick: () => { hidePopover(); scrollToPage(ref.page); } }, `Go to page ${ref.page}`));
    }
    if (ref.proof) {
      gotos.append(el("button", { type: "button", class: "ref-goto", title: `Scroll to the proof (page ${ref.proof.page})`,
        onclick: () => { hidePopover(); scrollToPassage(ref.proof.page, ref.proof.start); } }, "Go to proof"));
    }
    const body = el("div", { class: "ref-content" });
    body.innerHTML = renderMarkdown(ref.content);
    refBox.replaceChildren(head);
    if (ref.kind === "figure" || ref.kind === "table") {
      // The figure/table itself, cropped from the PDF; its caption text stays as the fallback.
      const src = `/api/papers/${encodeURIComponent(state.paper.id)}/crop?kind=${ref.kind}&number=${encodeURIComponent(ref.number)}`;
      const img = el("img", { class: "ref-img", src, alt: ref.label });
      const link = el("a", { class: "ref-img-link", href: src, target: "_blank", title: "Open full size" }, img);
      img.onload = () => { body.classList.add("ref-caption"); positionPopover(); };
      img.onerror = () => { link.remove(); $("ask-popover").classList.remove("wide"); positionPopover(); };
      refBox.append(link);
      wide = true;
    }
    refBox.append(body);
  }
  const cites = ref ? [] : parseCitations(popoverSelection.text);
  if (cites.length) showCitations(cites, popoverSelection.text); else $("ask-cite").hidden = true;
  const hit = !ref && !cites.length && lookupGlossary(popoverSelection.text);
  const gloss = $("ask-gloss");
  gloss.hidden = !hit;
  if (hit) {
    gloss.replaceChildren(el("div", { class: "gloss-term" }, hit.term));
    const body = el("div", { class: "gloss-desc" });
    body.innerHTML = renderMarkdown(hit.description);
    gloss.append(body);
  }
  $("ask-current").hidden = !state.conv;
  pop.classList.toggle("wide", wide);
  pop.hidden = false;
  positionPopover(rect);
  $("ask-input").value = "";
  // On a touch screen, focusing would pop up the keyboard over the passage and drop its drag handles.
  if (pointerType === "mouse") $("ask-input").focus({ preventScroll: true });
}

// Below the anchor if it fits, else above; re-run when content (e.g. a figure image) changes size.
// Kept within the visual viewport, which an on-screen keyboard shrinks.
let popoverAnchor = null;
function positionPopover(rect = popoverAnchor) {
  const pop = $("ask-popover");
  if (!rect || pop.hidden) return;
  popoverAnchor = rect;
  const vv = window.visualViewport;
  const minTop = (vv?.offsetTop ?? 0) + 8, maxBottom = (vv ? vv.offsetTop + vv.height : window.innerHeight) - 8;
  const w = pop.offsetWidth, h = pop.offsetHeight;
  const left = Math.min(Math.max(8, rect.left + rect.width / 2 - w / 2), window.innerWidth - w - 8);
  let top = rect.bottom + 8;
  if (top + h > maxBottom) top = rect.top - h - 8;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(minTop, Math.min(top, maxBottom - h))}px`;
}
window.visualViewport?.addEventListener("resize", () => positionPopover());
window.visualViewport?.addEventListener("scroll", () => positionPopover());

let popoverMic = null;
function hidePopover() {
  $("ask-popover").hidden = true;
  popoverMic?.cancel();  // a question being recorded there has nowhere to go
}

function submitPopover(inCurrent) {
  const question = $("ask-input").value.trim();
  if (!question || !popoverSelection) return;
  hidePopover();
  window.getSelection()?.removeAllRanges();
  if (!inCurrent) startNewChat(false);
  sendQuestion(question, popoverSelection);
}

$("ask-new").onclick = () => submitPopover(false);

// "Summarize": a new chat whose first message asks for a summary; the answer streams in.
$("summarize-btn").onclick = () => {
  if (state.streaming || !state.paper) return;
  startNewChat(false);
  sendQuestion("Write a summary of the paper.", null);
};
$("ask-current").onclick = () => submitPopover(true);
$("ask-input").addEventListener("keydown", (e) => {
  if (e.key === "Escape") return hidePopover();
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitPopover((e.metaKey || e.ctrlKey) && !!state.conv);
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("ask-popover").hidden) hidePopover();
});

// "Attach to composer" path: selecting text and then typing in the chat box uses it as context.
function setPendingSelection(sel) {
  state.pendingSelection = sel;
  const box = $("pending-selection");
  box.hidden = !sel;
  box.replaceChildren();
  if (sel) {
    box.append(
      el("div", { class: "txt" }, (sel.page ? `p.${sel.page}: ` : "") + sel.text),
      el("button", { type: "button", title: "Remove", onclick: () => setPendingSelection(null) }, "×"),
    );
  }
}

// ---------------------------------------------------------------- Providers / models

async function loadConfig() {
  const cfg = await api("/api/config");
  state.providers = cfg.providers;
  const select = $("provider");
  select.replaceChildren(...cfg.providers.map((p) =>
    el("option", { value: p.id, disabled: !p.available },
      p.label + (p.available ? "" : " (no API key)"))));
  // Always start from the configured default (.env), not a choice remembered from an earlier visit.
  const chosen = cfg.providers.find((p) => p.id === cfg.default_provider);
  if (chosen) setProvider(chosen.id, chosen.default_model);
  if (cfg.voice) addVoice(cfg.voice);
}

async function setProvider(providerId, model) {
  $("provider").value = providerId;
  const p = state.providers.find((x) => x.id === providerId);
  $("model").value = model || p?.default_model || "";
  showModelChoice();
  const list = $("model-list");
  list.replaceChildren();
  try {
    const { models } = await api(`/api/providers/${providerId}/models`);
    list.replaceChildren(...models.map((m) => el("option", { value: m })));
  } catch { /* the model field still accepts free text */ }
}

$("provider").addEventListener("change", () => {
  const id = $("provider").value;
  setProvider(id, state.providers.find((x) => x.id === id).default_model);
});

// Provider and model live in a popover behind the gear; its tooltip names the current choice.
function showModelChoice() {
  const model = $("model").value.trim();
  $("settings-btn").title = `Model settings: ${providerLabel($("provider").value)}${model ? ` · ${model}` : ""}`;
}
$("model").addEventListener("input", showModelChoice);

function toggleSettings(open = $("settings-pop").hidden) {
  $("settings-pop").hidden = !open;
  $("settings-btn").setAttribute("aria-expanded", String(open));
  $("settings-btn").classList.toggle("open", open);
}
$("settings-btn").onclick = () => toggleSettings();
document.addEventListener("pointerdown", (e) => {
  if (!$("settings-pop").hidden && !$("settings-wrap").contains(e.target)) toggleSettings(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("settings-pop").hidden) toggleSettings(false);
});

// ---------------------------------------------------------------- Usage (spending)

// What the model calls have cost (answers and paper analyses), per day, week or month, with any
// budgets. The server buckets by the browser's time zone; the button shows a dot near a limit.
const usage = { data: null, period: store.get("usagePeriod", "month") };
const PERIOD_NOUN = { day: "Daily", week: "Weekly", month: "Monthly" };
const RESET_NOUN = { daily: "daily", weekly: "weekly", monthly: "monthly", yearly: "yearly" };

function usd(v) {
  if (!v) return "$0.00";
  return "$" + (v >= 1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toFixed(4));
}

const isoDate = (s) => { const [y, m, d] = s.split("-").map(Number); return new Date(y, m - 1, d); };

function periodName(start, period, current) {
  if (current) return { day: "Today", week: "This week", month: "This month" }[period];
  const d = isoDate(start);
  if (period === "month") return d.toLocaleDateString(undefined, { month: "long", year: "numeric" });
  const day = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return period === "week" ? `Week of ${day}` : day;
}

// Under 80% is fine; from 80% it's a warning, from 100% critical. Always said in words too.
function budgetLevel(spent, limit) {
  const pct = limit > 0 ? (100 * spent) / limit : 0;
  return { pct, level: pct >= 100 ? "critical" : pct >= 80 ? "warning" : "good" };
}

function budgetMeter(title, spent, limit, note) {
  const { pct, level } = budgetLevel(spent, limit);
  const status = level === "critical" ? "Limit reached" : level === "warning" ? `${Math.round(pct)}% used, nearing the limit`
    : `${pct < 1 && spent > 0 ? "<1" : Math.round(pct)}% used`;
  return el("div", { class: `budget ${level}` },
    el("div", { class: "budget-row" }, el("span", {}, title), el("span", {}, `${usd(spent)} of ${usd(limit)}`)),
    el("div", { class: "budget-track", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(limit),
      "aria-valuenow": String(Math.min(spent, limit)), "aria-label": title },
      el("div", { class: "budget-fill", style: `width: ${Math.min(100, pct)}%` })),
    el("div", { class: "budget-status" }, status + (note ? ` · ${note}` : "")));
}

function renderUsage() {
  for (const b of document.querySelectorAll(".usage-periods button")) {
    b.setAttribute("aria-selected", String(b.dataset.period === usage.period));
  }
  const body = $("usage-body");
  const data = usage.data;
  if (!data) return;
  const p = data.periods[usage.period];

  const calls = `${p.calls} model call${p.calls === 1 ? "" : "s"}` +
    (p.unpriced ? ` · ${p.unpriced} not priced (model not in the price list)` : "");
  const hero = el("div", {},
    el("div", { class: "usage-label" }, periodName(p.start, usage.period, true)),
    el("div", { class: "usage-amount" }, usd(p.cost)),
    el("div", { class: "usage-label" }, calls));

  // Recent periods. Hovering (or focusing) a column shows its amount in the readout above.
  const max = Math.max(...p.history.map((h) => h.cost), 0);
  const readout = el("div", { class: "usage-readout" });
  const show = (h, current) => readout.replaceChildren(
    `${periodName(h.start, usage.period, current)}: `, el("strong", {}, usd(h.cost)));
  const last = p.history.at(-1);
  show(last, true);
  const chart = el("div", { class: "usage-chart" }, p.history.map((h, i) => {
    const current = i === p.history.length - 1;
    const label = `${periodName(h.start, usage.period, current)}: ${usd(h.cost)}`;
    return el("div", { class: "usage-col" + (current ? " current" : ""), tabindex: "0", role: "img", "aria-label": label,
      onmouseenter: () => show(h, current), onfocus: () => show(h, current) },
      el("div", { class: "usage-bar", style: `height: ${max ? Math.max(h.cost ? 2 : 0, (100 * h.cost) / max) : 0}%` }));
  }));
  chart.addEventListener("mouseleave", () => show(last, true));
  const first = p.history[0];
  const history = el("div", {},
    el("div", { class: "usage-section-head" }, `Last ${p.history.length} ${usage.period}s`),
    readout, chart,
    el("div", { class: "usage-axis" },
      el("span", {}, periodName(first.start, usage.period, false)), el("span", {}, periodName(last.start, usage.period, true))));

  const sections = [hero, history];

  if (p.models.length) {
    const shown = p.models.slice(0, 6);
    const rest = p.models.length - shown.length;
    sections.push(el("div", {},
      el("div", { class: "usage-section-head" }, "By model"),
      el("ul", { class: "usage-models" }, shown.map((m) => el("li", { title: `${providerLabel(m.provider)} · ${m.model}` },
        el("span", { class: "name" }, m.model),
        el("span", { class: "calls" }, `${m.calls}×`),
        el("span", { class: "cost" }, m.unpriced === m.calls ? "—" : usd(m.cost)))),
        rest ? el("li", { class: "muted" }, `and ${rest} more`) : null)));
  }

  const budgets = [];
  if (p.budget) budgets.push(budgetMeter(`${PERIOD_NOUN[usage.period]} budget`, p.cost, p.budget));
  for (const b of data.provider_budgets) {
    const kind = b.reset ? `${RESET_NOUN[b.reset] || b.reset} limit` : "credit limit";
    budgets.push(budgetMeter(`${b.source} ${kind}`, b.spent, b.limit, "all use of the key, not only this app"));
  }
  if (budgets.length) sections.push(el("div", {}, el("div", { class: "usage-section-head" }, "Budget"), budgets));

  sections.push(el("p", { class: "muted settings-note" },
    `All time: ${usd(data.total)}. Estimated from each model's token prices (cache discounts included); ` +
    "OpenRouter calls show what OpenRouter billed."));
  body.replaceChildren(...sections);
}

// The dot on the Usage button: the most urgent budget, in any period.
function showUsageAlert() {
  const data = usage.data;
  const levels = [
    ...Object.values(data.periods).filter((p) => p.budget).map((p) => budgetLevel(p.cost, p.budget).level),
    ...data.provider_budgets.map((b) => budgetLevel(b.spent, b.limit).level),
  ];
  const level = levels.includes("critical") ? "critical" : levels.includes("warning") ? "warning" : null;
  $("usage-alert").hidden = !level;
  $("usage-alert").className = `usage-alert ${level || ""}`;
  const month = data.periods.month;
  $("usage-btn").title = `Spending on model calls: ${usd(month.cost)} this month` +
    (level ? ` · ${level === "critical" ? "a budget limit is reached" : "nearing a budget limit"}` : "");
}

let usageLoading = null;
function loadUsage() {
  usageLoading ||= api(`/api/usage?tz=${encodeURIComponent(Intl.DateTimeFormat().resolvedOptions().timeZone)}`)
    .then((data) => { usage.data = data; showUsageAlert(); if (!$("usage-pop").hidden) renderUsage(); })
    .catch((e) => { if (!usage.data) $("usage-body").replaceChildren(el("p", { class: "error-msg" }, `Could not load usage: ${e.message}`)); })
    .finally(() => { usageLoading = null; });
  return usageLoading;
}

function toggleUsage(open = $("usage-pop").hidden) {
  $("usage-pop").hidden = !open;
  $("usage-btn").setAttribute("aria-expanded", String(open));
  $("usage-btn").classList.toggle("open", open);
  if (open) {
    renderUsage();
    loadUsage(); // refresh: paper analyses spend in the background
  }
}
$("usage-btn").onclick = () => toggleUsage();
for (const b of document.querySelectorAll(".usage-periods button")) {
  b.onclick = () => { usage.period = b.dataset.period; store.set("usagePeriod", usage.period); renderUsage(); };
}
document.addEventListener("pointerdown", (e) => {
  if (!$("usage-pop").hidden && !$("usage-wrap").contains(e.target)) toggleUsage(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("usage-pop").hidden) toggleUsage(false);
});

// ---------------------------------------------------------------- History sidebar

// Sidebar: folders (server-side, nestable) holding papers, each paper with its conversations.
// Folded folders/papers are a per-browser view preference.
let subfolderDraft = null;  // {parentId, value}: a subfolder being named, kept across re-renders
async function renderHistory() {
  let papers = [], convs = [], folders = [];
  try {
    [papers, convs, folders] = await Promise.all([api("/api/papers"), api("/api/conversations"), api("/api/folders")]);
  } catch { return; }
  renderSearches(convs.filter((c) => !c.paper_id));
  const byPaper = {};
  for (const c of convs) if (c.paper_id) (byPaper[c.paper_id] ||= []).push(c);
  const inFolder = {};
  for (const p of papers) (inFolder[p.folder_id || ""] ||= []).push(p);
  const folderIds = new Set(folders.map((f) => f.id));
  const childFolders = {};
  for (const f of folders) (childFolders[folderIds.has(f.parent_id) ? f.parent_id : ""] ||= []).push(f);
  const subtree = (id) => [id, ...(childFolders[id] || []).flatMap((c) => subtree(c.id))];
  const foldedPapers = new Set(store.get("foldedPapers", []));
  const foldedFolders = new Set(store.get("foldedFolders", []));
  const ctx = { byPaper, folders, foldedPapers };
  // Folding is toggled in place rather than re-rendered, so a double-click (rename) still
  // lands on the same element.
  const folder = (id, body) => {
    const arrow = el("button", { type: "button", class: "fold" });
    const setFolded = (folded) => {
      if (folded) foldedFolders.add(id); else foldedFolders.delete(id);
      store.set("foldedFolders", [...foldedFolders]);
      body.hidden = folded;
      arrow.textContent = folded ? "▸" : "▾";
      arrow.title = folded ? "Expand folder" : "Collapse folder";
    };
    const toggle = () => setFolded(!foldedFolders.has(id));
    setFolded(foldedFolders.has(id));
    arrow.onclick = toggle;
    return { arrow, toggle, setFolded };
  };

  const renderFolder = (f) => {
    const list = inFolder[f.id] || [];
    const subs = childFolders[f.id] || [];
    const body = el("div", { class: "folder-body" }, subs.map(renderFolder), list.map((p) => paperGroup(p, ctx)));
    const { arrow, toggle, setFolded } = folder(f.id, body);
    const name = el("span", { class: "folder-name", title: "Click to fold, double-click to rename" }, f.name);
    name.onclick = (e) => { if (e.detail < 2) toggle(); };
    const rename = () => {
      head.draggable = false;  // so the mouse selects text in the field instead of dragging the folder
      name.replaceWith(folderNameInput(f.name, async (newName) => {
        if (newName && newName !== f.name) await api(`/api/folders/${f.id}`, { method: "PATCH", body: { name: newName } });
        renderHistory();
      }));
    };
    name.ondblclick = () => {
      toggle(); // undo the fold from the double-click's first click
      rename();
    };
    // The new subfolder's name field opens at the top of this folder's contents.
    const nameSubfolder = () => {
      const input = folderNameInput(subfolderDraft.value, async (newName) => {
        subfolderDraft = null;
        if (newName) await api("/api/folders", { method: "POST", body: { name: newName, parent_id: f.id } });
        renderHistory();
      });
      input.addEventListener("input", () => { subfolderDraft.value = input.value; });
      body.querySelector(".folder-empty")?.remove();
      body.prepend(input);
    };
    const addSubfolder = () => {
      if (subfolderDraft?.parentId === f.id) return body.querySelector(".folder-input")?.focus();
      subfolderDraft = { parentId: f.id, value: "" };
      setFolded(false);
      nameSubfolder();
    };
    const remove = async () => {
      const tree = subtree(f.id);
      const nested = tree.flatMap((id) => inFolder[id] || []);
      const parent = folders.find((p) => p.id === f.parent_id);
      const choice = nested.length || tree.length > 1
        ? await confirmFolderDelete(f.name, nested.length, tree.length - 1, parent?.name)
        : confirm(`Delete folder "${f.name}"?`) ? "keep" : null;
      if (!choice) return;
      try {
        await api(`/api/folders/${f.id}` + (choice === "all" ? "?with_papers=true" : ""), { method: "DELETE" });
      } catch (e) { return alert(`Could not delete the folder: ${e.message}`); }
      if (choice === "all") papersDeleted(nested.map((p) => p.id)); else renderHistory();
    };
    const head = el("div", { class: "folder-head", draggable: "true" }, arrow,
      el("span", { class: "folder-icon" }, "📁"), name,
      el("button", { type: "button", class: "add", title: "New subfolder", onclick: addSubfolder }, "+"),
      // Touch screens don't reliably send double-clicks.
      el("button", { type: "button", class: "rename", title: "Rename folder", onclick: rename }, "✎"),
      el("button", { type: "button", class: "del", title: "Delete folder", onclick: remove }, trashIcon()));
    const node = dropTarget(el("div", { class: "folder" }, head, body), f.id);
    // Folders are dragged by their header into another folder, or to the top level.
    head.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("application/x-folder-id", f.id);
      e.dataTransfer.effectAllowed = "move";
      draggedFolder = node;
      node.classList.add("dragging");
    });
    head.addEventListener("dragend", () => {
      draggedFolder = null;
      node.classList.remove("dragging");
      setDropOver(null);
    });
    touchDrag(head, node, (parentId) => moveFolder(f.id, parentId));
    if (subfolderDraft?.parentId === f.id) nameSubfolder();
    else if (!list.length && !subs.length) body.append(el("div", { class: "folder-empty muted" }, "Drag papers here"));
    return node;
  };

  const root = $("history");
  root.replaceChildren(...(childFolders[""] || []).map(renderFolder));
  const unfiled = inFolder[""] || [];
  if (folders.length && (unfiled.length || papers.length)) {
    const body = el("div", { class: "folder-body" }, unfiled.map((p) => paperGroup(p, ctx)));
    const { arrow, toggle } = folder("unfiled", body);
    const name = el("span", { class: "folder-name muted", title: "Click to fold", onclick: toggle }, "Unfiled");
    const section = el("div", { class: "folder unfiled" }, el("div", { class: "folder-head" }, arrow, name), body);
    root.append(dropTarget(section, null));
  } else {
    for (const p of unfiled) root.append(paperGroup(p, ctx));
  }
  if (!papers.length) root.append(el("div", { class: "muted", style: "padding:8px" }, "No papers yet."));
}

// "Searches" tab: search sessions (chats with no paper), most recent first.
function renderSearches(convs) {
  const root = $("searches");
  if (!convs.length) {
    root.replaceChildren(el("div", { class: "muted", style: "padding:8px" },
      "No searches yet. Ask the assistant on the home page to find papers for you."));
    return;
  }
  const date = (t) => new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  root.replaceChildren(...convs.map((c) => el("a", {
    class: "search-link" + (state.conv?.id === c.id ? " active" : ""), href: searchHash(c.id), title: c.title,
  },
    el("span", { class: "body" }, el("span", { class: "t" }, c.title),
      el("span", { class: "meta" }, `${date(c.updated_at)} · ${providerLabel(c.provider)}`)),
    el("button", {
      type: "button", class: "del", title: "Delete search",
      onclick: async (e) => {
        e.preventDefault(); e.stopPropagation();
        if (!confirm("Delete this search?")) return;
        await api(`/api/conversations/${c.id}`, { method: "DELETE" });
        if (state.conv?.id === c.id) location.hash = searchHash();
        else renderHistory();
      },
    }, "×"))));
}

// The sidebar's two libraries. Entering or leaving search mode shows the matching one.
function setSidebarTab(tab) {
  const searches = tab === "searches";
  $("tab-papers").setAttribute("aria-selected", String(!searches));
  $("tab-searches").setAttribute("aria-selected", String(searches));
  $("history").hidden = $("new-folder").hidden = $("folder-create").hidden = searches;
  $("searches").hidden = $("new-search").hidden = !searches;
}
$("tab-papers").onclick = () => setSidebarTab("papers");
$("tab-searches").onclick = () => setSidebarTab("searches");
setSidebarTab("papers");

// The home page (#/) is a fresh search session; hashchange doesn't fire when already there.
$("new-search").onclick = () => {
  if (location.hash === searchHash() || !location.hash) startNewChat(true);
  else location.hash = searchHash();
};

// Resolves "keep" (contents move up to the parent folder, or become unfiled), "all" (subfolders
// and papers deleted too) or null (cancelled).
function confirmFolderDelete(name, papers, subfolders, parentName) {
  const dialog = $("folder-delete");
  const count = (n, what) => n === 1 ? `1 ${what}` : `${n} ${what}s`;
  const holds = [subfolders && count(subfolders, "subfolder"), papers && count(papers, "paper")].filter(Boolean);
  $("folder-delete-text").textContent = `Delete folder "${name}"? It holds ${holds.join(" and ")}.`
    + (parentName ? ` Kept items move to "${parentName}".` : "");
  $("folder-delete-keep").textContent = subfolders ? "Delete folder, keep its contents" : "Delete folder, keep papers";
  $("folder-delete-all").textContent = !papers ? "Delete folder and its subfolders"
    : subfolders ? `Delete folder, its subfolders and ${count(papers, "paper")}`
    : papers === 1 ? "Delete folder and its paper" : `Delete folder and its ${papers} papers`;
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise((resolve) => dialog.addEventListener("close", () => {
    resolve(["keep", "all"].includes(dialog.returnValue) ? dialog.returnValue : null);
  }, { once: true }));
}

// Leaves the reader if the open paper was among the deleted ones.
// The sidebar stays on whichever tab it shows.
function papersDeleted(ids) {
  if (state.paper && ids.includes(state.paper.id)) {
    keepSidebarTab = true;
    location.hash = ""; // route() re-renders the sidebar
  } else renderHistory();
}

// The new-folder field lives outside #history, so a sidebar refresh can't drop it mid-typing.
$("new-folder").onclick = () => {
  const box = $("folder-create");
  if (box.firstChild) return box.firstChild.focus();
  box.append(folderNameInput("", async (name) => {
    box.replaceChildren();
    if (name) await api("/api/folders", { method: "POST", body: { name } });
    renderHistory();
  }));
};

function folderNameInput(value, done) {
  const input = el("input", { class: "folder-input", value, placeholder: "Folder name", maxlength: "80" });
  let finished = false;
  const finish = (v) => { if (!finished) { finished = true; done(v); } };
  input.onkeydown = (e) => {
    if (e.key === "Enter") finish(input.value.trim());
    if (e.key === "Escape") finish(null);
  };
  // Clicking elsewhere in the page commits; switching to another window keeps the field open.
  input.onblur = () => { if (input.isConnected && document.hasFocus()) finish(input.value.trim()); };
  queueMicrotask(() => { input.focus(); input.select(); });
  return input;
}

let draggedFolder = null;  // the .folder element being dragged with the mouse

function setDropOver(node) {
  for (const n of document.querySelectorAll(".drop-over")) if (n !== node) n.classList.remove("drop-over");
  node?.classList.add("drop-over");
}

// Papers and folders are dragged onto a folder (or "Unfiled") to move them into it. The innermost
// folder under the pointer takes the drop; a folder can't go into itself or its own subfolders.
// With foldersOnly, the target takes only folders (the sidebar's empty space: the top level).
function dropTarget(node, folderId, { foldersOnly = false } = {}) {
  node.dataset.dropFolder = folderId ?? "";  // found by touchDrag()
  node.addEventListener("dragover", (e) => {
    const { types } = e.dataTransfer;
    if (!types.includes("application/x-folder-id") && (foldersOnly || !types.includes("application/x-paper-id"))) return;
    e.stopPropagation();
    if (draggedFolder?.contains(node)) return setDropOver(null);
    e.preventDefault();
    setDropOver(node);
  });
  node.addEventListener("dragleave", (e) => { if (!node.contains(e.relatedTarget)) node.classList.remove("drop-over"); });
  node.addEventListener("drop", async (e) => {
    setDropOver(null);
    const paperId = e.dataTransfer.getData("application/x-paper-id");
    const movedFolderId = e.dataTransfer.getData("application/x-folder-id");
    if (!paperId && !movedFolderId) return;
    e.preventDefault();
    e.stopPropagation();
    if (paperId) await movePaper(paperId, folderId);
    else await moveFolder(movedFolderId, folderId);
  });
  return node;
}
dropTarget($("history"), null, { foldersOnly: true });

// Touch screens: long-press a paper or a folder's header, then drag it onto a folder. Touch
// events rather than HTML5 drag and drop, which iOS turns into dragging the link's URL.
function touchDrag(handle, item, onDrop) {
  handle.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1) return;
    const { clientX: x0, clientY: y0 } = e.touches[0];
    let dragging = false, target = null, finger = null, scroller = null;
    const timer = setTimeout(() => { dragging = true; item.classList.add("dragging"); }, 450);
    const hover = () => {
      let over = document.elementFromPoint(finger.clientX, finger.clientY)?.closest("[data-drop-folder]") || null;
      if (over && item.contains(over)) over = null;  // not into itself or its own subfolders
      if (over !== target) {
        target?.classList.remove("drop-over");
        target = over;
        target?.classList.add("drop-over");
      }
    };
    // Near the sidebar's top or bottom edge, scroll it so folders out of view can be reached.
    const autoScroll = () => {
      const box = $("history").getBoundingClientRect();
      const dy = finger.clientY < box.top + 40 ? -12 : finger.clientY > box.bottom - 40 ? 12 : 0;
      if (dy) { $("history").scrollTop += dy; hover(); }
    };
    const move = (ev) => {
      finger = ev.touches[0];
      if (!dragging) {
        if (Math.hypot(finger.clientX - x0, finger.clientY - y0) > 8) done();  // a scroll, not a long-press
        return;
      }
      ev.preventDefault();  // the finger moves the paper, not the sidebar
      scroller ??= setInterval(autoScroll, 30);
      hover();
    };
    const done = (ev) => {
      clearTimeout(timer);
      clearInterval(scroller);
      handle.removeEventListener("touchmove", move);
      handle.removeEventListener("touchend", done);
      handle.removeEventListener("touchcancel", done);
      item.classList.remove("dragging");
      target?.classList.remove("drop-over");
      if (!dragging) return;
      ev?.preventDefault();  // no click on the link under the finger
      if (ev?.type === "touchend" && target) onDrop(target.dataset.dropFolder || null);
    };
    handle.addEventListener("touchmove", move, { passive: false });
    handle.addEventListener("touchend", done);
    handle.addEventListener("touchcancel", done);
  }, { passive: true });
}

async function movePaper(paperId, folderId) {
  await api(`/api/papers/${encodeURIComponent(paperId)}`, { method: "PATCH", body: { folder_id: folderId } });
  renderHistory();
}

async function moveFolder(folderId, parentId) {
  if (folderId === parentId) return;
  try {
    await api(`/api/folders/${folderId}`, { method: "PATCH", body: { parent_id: parentId } });
  } catch (e) { return alert(`Could not move the folder: ${e.message}`); }
  renderHistory();
}

function paperGroup(p, { byPaper, folders, foldedPapers }) {
  const convList = byPaper[p.id] || [];
  const isFolded = convList.length > 0 && foldedPapers.has(p.id);
  const row = el("div", { class: "paper-row" });
  if (convList.length) {
    row.append(el("button", {
      type: "button", class: "fold", "aria-expanded": String(!isFolded),
      title: isFolded ? "Show this paper's chats" : "Collapse this paper's chats",
      onclick: () => {
        if (foldedPapers.has(p.id)) foldedPapers.delete(p.id); else foldedPapers.add(p.id);
        store.set("foldedPapers", [...foldedPapers]);
        renderHistory();
      },
    }, isFolded ? "▸" : "▾"));
  } else {
    row.append(el("span", { class: "fold-spacer" })); // keeps titles aligned with foldable papers
  }
  const chats = convList.length === 1 ? "1 chat" : `${convList.length} chats`;
  row.append(el("a", {
    class: "paper-link" + (state.paper?.id === p.id && !state.conv ? " active" : ""),
    href: paperHash(p.id), title: p.title,
    // Clicking the open paper folds/unfolds its chats; clicking another paper opens it unfolded.
    onclick: (e) => {
      if (!convList.length || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
      const isOpen = state.paper?.id === p.id;
      if (isOpen && !isFolded) foldedPapers.add(p.id); else foldedPapers.delete(p.id);
      store.set("foldedPapers", [...foldedPapers]);
      // Navigating re-renders the sidebar via route(); staying on the paper doesn't.
      if (isOpen) { e.preventDefault(); renderHistory(); }
    },
  }, p.title, el("span", { class: "pid" }, isFolded ? `${paperLabel(p)} · ${chats}` : paperLabel(p))));
  row.append(el("button", {
    type: "button", class: "del", title: "Delete paper",
    onclick: async () => {
      const withChats = convList.length ? ` and its ${chats}` : "";
      if (!confirm(`Delete "${p.title}"${withChats}? This can't be undone.`)) return;
      try {
        await api(`/api/papers/${encodeURIComponent(p.id)}`, { method: "DELETE" });
      } catch (e) { return alert(`Could not delete the paper: ${e.message}`); }
      papersDeleted([p.id]);
    },
  }, trashIcon()));
  const convBox = el("div", { class: "paper-convs" });
  convBox.hidden = isFolded;
  for (const c of convList) {
    convBox.append(el("a", {
      class: "conv-link" + (state.conv?.id === c.id ? " active" : ""),
      href: paperHash(p.id, c.id), title: c.title,
      // Opening a conversation shows it, even if the chat panel is minimized (not for ⌘-click: new tab).
      onclick: (e) => { if (!e.metaKey && !e.ctrlKey && !e.shiftKey && e.button === 0) setChatCollapsed(false); },
    },
      el("span", { class: "t" }, c.title),
      el("span", { class: "badge" }, providerLabel(c.provider)),
      el("button", {
        class: "del", title: "Delete conversation",
        onclick: async (e) => {
          e.preventDefault(); e.stopPropagation();
          if (!confirm("Delete this conversation?")) return;
          await api(`/api/conversations/${c.id}`, { method: "DELETE" });
          if (state.conv?.id === c.id) location.hash = paperHash(p.id);
          else renderHistory();
        },
      }, "×")));
  }
  const group = el("div", { class: "paper-group", draggable: "true" }, row, convBox);
  group.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("application/x-paper-id", p.id);
    e.dataTransfer.effectAllowed = "move";
    group.classList.add("dragging");
  });
  group.addEventListener("dragend", () => group.classList.remove("dragging"));
  if (folders.length) touchDrag(group, group, (folderId) => movePaper(p.id, folderId));
  return group;
}

function providerLabel(id) {
  return state.providers.find((p) => p.id === id)?.label || id;
}

// ---------------------------------------------------------------- Markdown + math

function renderMarkdown(src) {
  const stash = [];
  const hold = (html) => `\uE000${stash.push(html) - 1}\uE000`;
  // Protect code, then math, from the markdown parser.
  let s = src.replace(/```[\s\S]*?(```|$)|`[^`\n]+`/g, (m) => hold({ code: m }));
  s = s.replace(/\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]/g, (_, a, b) => hold({ math: a ?? b, display: true }));
  s = s.replace(/\\\(([\s\S]+?)\\\)|(?<![\\$\w])\$(?!\s)([^$\n]+?)(?<!\s)\$(?![\w$])/g,
    (_, a, b) => hold({ math: a ?? b, display: false }));
  // [Title](paper:<arXiv ID or PDF URL>) links open the paper in this app (see SYSTEM_PROMPT).
  // DOMPurify drops unknown schemes, so turn them into in-app routes first.
  s = s.replace(/\]\(\s*<?paper:([^)\s>]+)>?\s*\)/g, (_, src) => `](${openHash(src)})`);
  // [Theorem 3.1](ref:theorem-3.1) links jump to a numbered element of the open paper (see
  // elements_text); "#ref:" fragments are intercepted on click, so they never reach the router.
  s = s.replace(/\]\(\s*<?ref:([\w.-]+)>?\s*\)/g, (_, id) => `](#ref:${id})`);
  // Code goes back before parsing (marked handles it); math after.
  s = s.replace(/\uE000(\d+)\uE000/g, (m, i) => (stash[i].code !== undefined ? stash[i].code : m));
  let html = DOMPurify.sanitize(marked.parse(s));
  html = html.replace(/\uE000(\d+)\uE000/g, (m, i) => {
    const item = stash[i];
    if (!item || item.math === undefined) return m;
    try {
      return katex.renderToString(item.math, { displayMode: item.display, throwOnError: false });
    } catch { return m; }
  });
  return html;
}

const ARXIV_LINK = /arxiv\.org\/(?:abs|pdf|html)\/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?\/\d{7}(?:v\d+)?)/i;

function decorateLinks(container) {
  for (const a of container.querySelectorAll("a[href]:not(.paper-ref-tab)")) {
    if (a.getAttribute("href").startsWith("#/open/")) {
      // Paper link: opens in the reader with a fresh session; ↗ opens it in a new browser tab.
      a.classList.add("paper-ref");
      a.title = "Open this paper in the reader (new session)";
      if (!a.nextSibling?.classList?.contains("paper-ref-tab")) {
        a.after(el("a", { class: "paper-ref-tab", href: a.getAttribute("href"), target: "_blank",
          title: "Open in a new browser tab" }, "↗"));
      }
      showLinkedTitle(a, container);
      continue;
    }
    a.target = "_blank";
    a.rel = "noopener";
    const m = a.getAttribute("href").match(ARXIV_LINK);
    if (m && !a.nextSibling?.classList?.contains("open-paper")) {
      const id = m[1].replace(/\.pdf$/, "");
      a.after(el("button", {
        class: "open-paper", title: "Open this paper in the reader",
        onclick: () => openReferencedPaper(id),
      }, "open"));
    }
  }
}

// Models sometimes misremember arXiv IDs, so each paper link shows the title arXiv has for its ID
// (fetched in batches, cached): a wrong ID is obvious at a glance, a nonexistent one is flagged.
const ARXIV_SOURCE = /^(?:https?:\/\/)?(?:www\.)?(?:arxiv\.org\/(?:abs|pdf|html)\/)?(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?\/\d{7})(?:v\d+)?(?:\.pdf)?$/i;
const linkedTitles = new Map(); // arXiv ID -> title | null (not found) | undefined (pending)
const titleQueue = new Set();
const titleWaiters = new Set();
let titleTimer = null;

function showLinkedTitle(a, container) {
  const id = decodeURIComponent(a.getAttribute("href").slice("#/open/".length)).match(ARXIV_SOURCE)?.[1];
  if (!id) return;
  if (a.dataset.checked) return;
  if (!linkedTitles.has(id)) {
    linkedTitles.set(id, undefined);
    titleQueue.add(id);
    clearTimeout(titleTimer);
    titleTimer = setTimeout(fetchLinkedTitles, 400);
  }
  const title = linkedTitles.get(id);
  if (title === undefined) { titleWaiters.add(container); return; }
  a.dataset.checked = "1";
  const tab = a.nextSibling; // the ↗ link
  if (title === null) {
    tab.after(el("span", { class: "paper-ref-real missing", title: "This arXiv ID does not exist" }, ` (arXiv ${id} not found)`));
  } else if (sameTitle(a.textContent, title)) {
    a.title = `${title} (arXiv ${id}) — open in the reader`;
  } else {
    tab.after(el("span", { class: "paper-ref-real", title: `arXiv ${id}` }, ` (${title})`));
  }
}

// Link text that is the paper's title (give or take case and punctuation) needs no repeat.
function sameTitle(text, title) {
  const n = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const a = n(text), b = n(title);
  return a.length > 10 && (a.includes(b) || b.includes(a));
}

async function fetchLinkedTitles() {
  const ids = [...titleQueue];
  titleQueue.clear();
  let titles = {};
  try { titles = await api(`/api/arxiv/titles?ids=${encodeURIComponent(ids.join(","))}`); } catch {
    for (const id of ids) linkedTitles.delete(id); // retry on the next render
    return;
  }
  for (const id of ids) linkedTitles.set(id, titles[id] ?? null);
  const waiting = [...titleWaiters];
  titleWaiters.clear();
  for (const c of waiting) if (c.isConnected) decorateLinks(c);
}

function openReferencedPaper(arxivId) {
  location.hash = paperHash(arxivId);
}

// ---------------------------------------------------------------- Chat rendering

function resetChat() {
  state.conv = null;
  $("chat-title").textContent = state.searchMode ? "New search" : "New conversation";
  $("messages").replaceChildren(el("div", { class: "chat-empty muted" },
    state.paper ? "Highlight text in the paper to ask about it, or ask about the whole paper below."
      : state.searchMode ? searchIntro()
      : "Open a paper to start."));
  setPendingSelection(null);
}

const SEARCH_EXAMPLES = [
  "Recent work on state-space models for long-context language modeling",
  "The paper that introduced RLHF with PPO for instruction following",
  "Surveys of mechanistic interpretability",
];

function searchIntro() {
  return el("div", { class: "search-intro" },
    el("h2", {}, "Find papers"),
    el("p", {}, "Ask the assistant to look up papers for you: a topic, a half-remembered title, related work. "
      + "It searches arXiv and the web and links every paper it finds; click a link to open the paper in the reader."),
    el("div", { class: "search-examples" }, SEARCH_EXAMPLES.map((q) => el("button", {
      type: "button", class: "search-example", onclick: () => { $("question").value = q; $("question").focus(); },
    }, q))),
    el("p", { class: "muted" }, "To open a paper directly, paste its link above. Past searches are under Searches in the sidebar."));
}

function startNewChat(navigate = true) {
  resetChat();
  if (navigate && (state.paper || state.searchMode)) {
    history.replaceState(null, "", state.paper ? paperHash(state.paper.id) : searchHash());
    renderHistory();
  }
}
$("new-chat").onclick = () => startNewChat(true);

async function openConversation(convId) {
  const conv = await api(`/api/conversations/${convId}`);
  state.conv = conv;
  $("chat-title").textContent = conv.title;
  // Keep the current provider/model selection; the next message switches the conversation to it.
  renderMessages(conv.messages);
}

function renderMessages(messages) {
  const box = $("messages");
  box.replaceChildren();
  let current = null; // assistant view spanning assistant/tool messages until the next user message
  for (const m of messages) {
    if (m.role === "user") {
      current = null;
      box.append(userBubble(m.display?.question ?? m.content.find((b) => b.type === "text")?.text, m.display?.selection));
    } else if (m.role === "assistant") {
      if (!current) { current = new AssistantView(); box.append(current.root); }
      if (m.display?.thinking) current.thinking(m.display.thinking);
      for (const b of m.content) {
        if (b.type === "text") { current.newText(); current.text(b.text); }
        else if (b.type === "tool_use") current.toolCall(b);
      }
      if (m.display?.usage) current.usage(m.display.usage, m.display.model, m.display.stop_reason);
      if (m.display?.stopped) current.stopped();
    } else if (m.role === "tool") {
      if (!current) { current = new AssistantView(); box.append(current.root); }
      for (const b of m.content) {
        current.toolResult({ id: b.tool_use_id, name: b.name, is_error: b.is_error, preview: b.content.slice(0, 600), papers: b.papers });
      }
    }
  }
  box.scrollTop = box.scrollHeight;
}

function userBubble(question, selection) {
  const node = el("div", { class: "msg user" });
  if (selection?.text) {
    const quote = el("div", { class: "quote" });
    if (selection.page) {
      quote.append(el("span", { class: "pg", onclick: () => scrollToPage(selection.page) }, `page ${selection.page} ↗`));
    }
    quote.append(selection.text);
    node.append(quote);
  }
  node.append(el("div", { class: "q" }, question));
  return node;
}

class AssistantView {
  constructor() {
    this.root = el("div", { class: "msg assistant" });
    this.textNode = null;
    this.textSrc = "";
    this.thinkingNode = null;
    this.tools = {};
    this.totals = null;
    this.pending = false;
  }

  thinking(t) {
    if (!this.thinkingNode) {
      this.thinkingNode = el("details", { class: "thinking" }, el("summary", {}, "Thinking"), el("div", { class: "body" }));
      this.root.append(this.thinkingNode);
      this.textNode = null;
    }
    this.thinkingNode.querySelector(".body").textContent += t;
  }

  newText() { this.textNode = null; }

  text(t) {
    if (!this.textNode) {
      this.textNode = el("div", { class: "md" });
      this.textSrc = "";
      this.root.append(this.textNode);
      this.thinkingNode = null;
    }
    this.textSrc += t;
    if (!this.pending) {
      this.pending = true;
      requestAnimationFrame(() => {
        this.pending = false;
        this.textNode.innerHTML = renderMarkdown(this.textSrc);
        decorateLinks(this.textNode);
      });
    }
  }

  toolCall({ id, name, input }) {
    this.textNode = null;
    this.thinkingNode = null;
    const args = Object.values(input || {}).map((v) => String(v)).join(", ");
    const node = el("details", { class: "tool running" },
      el("summary", {}, `${name}(${args.length > 80 ? args.slice(0, 80) + "…" : args})`),
      el("div", { class: "body" }, JSON.stringify(input, null, 1)));
    const cards = el("div", { class: "paper-cards" });
    this.tools[id] = { node, cards };
    this.root.append(node, cards);
  }

  toolResult({ id, is_error, preview, papers }) {
    const t = this.tools[id];
    if (!t) return;
    t.node.classList.remove("running");
    if (is_error) t.node.classList.add("error");
    t.node.querySelector(".body").textContent += `\n\n→ ${preview}${preview.length >= 600 ? "…" : ""}`;
    const seen = new Set();
    for (const p of papers || []) {
      if (seen.has(p.arxiv_id)) continue;
      seen.add(p.arxiv_id);
      t.cards.append(el("div", { class: "paper-card" },
        el("span", { class: "t" }, p.title, " ", el("span", { class: "id" }, p.arxiv_id)),
        el("a", { href: paperHash(p.arxiv_id), title: "Open in the reader (new session)" }, "Open"),
        el("a", { href: paperHash(p.arxiv_id), target: "_blank", title: "Open in a new browser tab" }, "↗")));
    }
  }

  usage(u, model, stopReason) {
    this.totals ||= { input: 0, cached_input: 0, output: 0, steps: 0 };
    this.totals.input = u.input; // the last step's prompt size is the meaningful one
    this.totals.cached_input = u.cached_input;
    this.totals.output += u.output;
    this.totals.steps += 1;
    if (!this.usageNode) { this.usageNode = el("div", { class: "usage" }); }
    this.root.append(this.usageNode); // keep it last
    const k = (n) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));
    const pct = this.totals.input ? Math.round((100 * this.totals.cached_input) / this.totals.input) : 0;
    this.usageNode.textContent =
      `${model ? model + " · " : ""}prompt ${k(this.totals.input)} tokens (${pct}% cached) · output ${k(this.totals.output)}` +
      (this.totals.steps > 1 ? ` · ${this.totals.steps} steps` : "");
    const cut = CUT_OFF[stopReason];
    if (cut) this.root.insertBefore(el("div", { class: "error-msg" }, `Answer cut off: ${cut}.`), this.usageNode);
  }

  error(msg) { this.root.append(el("div", { class: "error-msg" }, msg)); }

  stopped() { this.root.append(el("div", { class: "stopped-msg muted" }, "Stopped.")); }
}

// Stop reasons (per provider) that mean the answer did not finish.
const CUT_OFF = {
  length: "output token limit reached",
  max_tokens: "output token limit reached",
  MAX_TOKENS: "output token limit reached",
  incomplete: "the response was incomplete",
  interrupted: "the provider ended the stream early",
};

// ---------------------------------------------------------------- Voice questions

// A mic button that records a question, transcribed while it's spoken: the text grows in `input`
// as you talk, and the recording stops by itself when you go quiet (or on a second click). The
// text isn't sent, so it can be checked first. The browser listens: a Listener tells speech from
// pauses in the microphone's 16 kHz PCM, each phrase goes to the server once you pause after it,
// for its final text, and the phrase still being spoken now and then for a quick draft. The server
// has Whisper on OpenRouter transcribe them, primed with the open paper's terms (app/transcribe.py).
const MIC_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
  stroke-linejoin="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/>
  <path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>`;
const MAX_RECORDING_MS = 2 * 60 * 1000;
const DRAFT_TIMEOUT_MS = 4000;
let recording = null;  // {stop} of the one recording in progress
let voiceEndSilence = 1.5;  // s of silence that ends a question (VOICE_END_SILENCE, from /api/config)

// Splits the microphone's 100 ms frames (static/mic-worklet.js) into phrases. A frame is speech when
// it's well above the background noise, taken as the quietest frame of the last few seconds.
// push() returns what's due: {phrase} (the PCM of a phrase followed by a pause: its text is final),
// {draft} (the phrase still being spoken, every DRAFT_EVERY s), {end} (after a long silence, or none
// spoken). finish() is the PCM of a phrase cut off by stopping the recording, if any.
class Listener {
  static FRAME = 0.1;  // s per frame
  static PAUSE = 0.6;  // s of silence that closes a phrase
  static NO_SPEECH = 8;  // s with no speech at all before giving up
  static MIN_DRAFT = 0.4;  // s of speech before a phrase gets a draft (Whisper guesses wildly from less)
  static DRAFT_EVERY = 1;  // s between drafts
  static PAD = 2;  // frames kept around speech, so word edges aren't clipped
  static NOISE_WINDOW = 50;  // frames over which the background noise is the quietest

  constructor(endSilence) {
    this.endSilence = endSilence;
    this.frames = [];
    this.levels = [];  // dBFS per frame
    this.run = 0;  // consecutive speech frames
    this.heard = false;
    this.start = null;  // first speech frame of the open phrase (null: none yet)
    this.last = -1;  // last speech frame
    this.drafted = 0;  // frame of the last draft
  }

  static level(frame) {
    let sum = 0;
    for (const v of frame) sum += v * v;
    return 10 * Math.log10(sum / frame.length / 32768 ** 2 + 1e-10);
  }

  push(frame) {
    const i = this.frames.push(frame) - 1, F = Listener.FRAME;
    const db = Listener.level(frame);
    this.levels.push(db);
    const noise = Math.min(...this.levels.slice(-Listener.NOISE_WINDOW));
    // 15 dB over the noise, but always heard at -35 dBFS (normal speech) and never below -55.
    const speech = db > Math.min(-35, Math.max(-55, noise + 15));
    this.run = speech ? this.run + 1 : 0;
    if (speech && (this.run >= 2 || this.start !== null)) {  // a lone loud frame is a click
      this.start ??= i - this.run + 1;
      this.last = i;
      this.heard = true;
    }
    if (!this.heard) return (i + 1) * F >= Listener.NO_SPEECH ? { end: true } : {};
    const out = {}, silence = (i - this.last) * F;
    if (this.start !== null && silence >= Listener.PAUSE) {
      out.phrase = this.#audio(this.last + Listener.PAD + 1);
      this.start = null;
    } else if (this.start !== null && (this.last - this.start + 1) * F >= Listener.MIN_DRAFT
               && (i - this.drafted) * F >= Listener.DRAFT_EVERY) {
      out.draft = this.#audio(i + 1);
      this.drafted = i;
    }
    if (silence >= this.endSilence) out.end = true;
    return out;
  }

  finish() {
    return this.start === null ? null : this.#audio(this.frames.length);
  }

  #audio(end) {  // the open phrase's frames, up to `end`, as one Int16Array
    const frames = this.frames.slice(Math.max(0, this.start - Listener.PAD), end);
    const pcm = new Int16Array(frames.reduce((n, f) => n + f.length, 0));
    let at = 0;
    for (const f of frames) { pcm.set(f, at); at += f.length; }
    return pcm;
  }
}

async function transcribePhrase(pcm, params, signal) {
  const res = await fetch(`/api/transcribe?${new URLSearchParams(params)}`, {
    method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: pcm, signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()).text;
}

function micButton(input) {
  const time = el("span", { class: "mic-time" });
  const btn = el("button", { type: "button", class: "mic", title: "Ask by voice: stops when you stop talking (or click again)" }, time);
  btn.insertAdjacentHTML("afterbegin", MIC_SVG);
  let rec = null;

  btn.onclick = async () => {
    if (rec) return rec.stop();
    recording?.stop();
    // Created in the click itself: iOS starts audio only from a user gesture.
    const ctx = new AudioContext();
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      await ctx.audioWorklet.addModule("/static/mic-worklet.js");
    } catch (e) {
      stream?.getTracks().forEach((t) => t.stop());
      ctx.close();
      return setStatus(`Microphone unavailable: ${e.message}`);
    }
    const source = ctx.createMediaStreamSource(stream);
    const pcm = new AudioWorkletNode(ctx, "pcm16k");
    source.connect(pcm);
    pcm.connect(ctx.destination);  // it outputs silence, but a node must reach the output to run in every browser
    ctx.resume();

    // The transcript replaces the selection (or goes at the end), spaced from the text around it.
    const original = input.value;
    const focused = document.activeElement === input;
    const a = focused ? input.selectionStart : original.length, b = focused ? input.selectionEnd : a;
    const before = original.slice(0, a), after = original.slice(b);
    let head = before;
    const show = (text) => {
      head = before + (before && text && !/\s$/.test(before) ? " " : "") + text;
      input.value = head + (after && text && !/^\s/.test(after) ? " " : "") + after;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.scrollTop = input.scrollHeight;
    };
    input.readOnly = true;  // the transcript is being written into it

    // Each phrase's text: its latest draft until its final text arrives. Finals are requested one
    // after another, so each is primed with the question's text before it.
    const paper = state.paper && !state.searchMode ? { paper_id: state.paper.id } : {};
    const aborter = new AbortController();
    const phrases = [];  // {text, final}
    let draft = "", drafting = false, phraseNo = 0;
    let finals = Promise.resolve();
    const render = () => show([...phrases.map((p) => p.text), draft].filter(Boolean).join(" "));
    const closePhrase = (audio) => {
      const phrase = { text: draft, final: false };
      phrases.push(phrase);
      draft = "";
      phraseNo++;
      finals = finals.then(async () => {
        const context = phrases.filter((p) => p.final).map((p) => p.text).join(" ");
        phrase.text = await transcribePhrase(audio, { ...paper, context }, aborter.signal);
        phrase.final = true;
        if (!finished) render();
      });
    };
    const requestDraft = (audio) => {
      if (drafting) return;  // one at a time: a slow one is skipped, not queued
      drafting = true;
      const no = phraseNo;
      // Given up on after a few seconds: it would be stale, and it holds up the next one.
      transcribePhrase(audio, { ...paper, draft: 1 }, AbortSignal.any([aborter.signal, AbortSignal.timeout(DRAFT_TIMEOUT_MS)]))
        .then((text) => { if (no === phraseNo && !finished) { draft = text; render(); } })
        .catch(() => {})  // only a draft: the phrase's final text follows
        .finally(() => { drafting = false; });
    };

    const listener = new Listener(voiceEndSilence);
    pcm.port.onmessage = (e) => {
      const due = listener.push(new Int16Array(e.data));
      if (due.phrase) closePhrase(due.phrase);
      if (due.draft) requestDraft(due.draft);
      if (due.end) end();
    };

    const started = Date.now();
    const tick = setInterval(() => {
      const s = Math.floor((Date.now() - started) / 1000);
      time.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 250);
    const limit = setTimeout(() => rec?.stop(), MAX_RECORDING_MS);
    btn.classList.add("recording");

    let capturing = true;
    const stopCapture = () => {  // the mic, not the transcription: its last words are still coming
      if (!capturing) return;
      capturing = false;
      clearInterval(tick);
      clearTimeout(limit);
      pcm.port.onmessage = null;
      source.disconnect();
      pcm.disconnect();
      stream.getTracks().forEach((t) => t.stop());  // releases the mic (and the recording indicator)
      ctx.close();
      btn.classList.replace("recording", "busy");
      time.textContent = "…";
    };
    let finished = false;
    const finish = (error) => {
      if (finished) return;
      finished = true;
      stopCapture();
      aborter.abort();
      rec = null;
      if (recording === handle) recording = null;
      btn.classList.remove("recording", "busy");
      time.textContent = "";
      input.readOnly = false;
      if (error) setStatus(error);
      if (pointerType === "mouse" || focused) {  // on a touch screen, focusing pops up the keyboard
        input.focus();
        input.setSelectionRange(head.length, head.length);
      }
    };
    // The question is over (a long silence, or stopped): wait for the last phrases' final text.
    const end = () => {
      if (!capturing) return;
      stopCapture();
      const rest = listener.finish();
      if (rest) closePhrase(rest);
      finals.then(
        () => {
          if (finished) return;
          const text = phrases.map((p) => p.text).filter(Boolean).join(" ");
          show(text);
          finish(text ? null : "No speech heard");
        },
        (e) => finish(e.name === "AbortError" ? null : e.message),
      );
    };

    const handle = recording = rec = {
      stop: end,
      cancel: () => { input.value = original; input.dispatchEvent(new Event("input", { bubbles: true })); finish(); },
    };
  };
  btn.cancel = () => rec?.cancel();
  return btn;
}

function addVoice(voice) {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) return;  // needs https (or localhost)
  voiceEndSilence = voice.end_silence;
  $("send").before(micButton($("question")));
  $("ask-new").parentNode.prepend(popoverMic = micButton($("ask-input")));
}

// ---------------------------------------------------------------- Sending

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (!q || state.streaming) return; // keep a question typed during an answer for later
  $("question").value = "";
  const sel = state.pendingSelection;
  setPendingSelection(null);
  sendQuestion(q, sel);
});
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.streaming) state.abort?.abort();
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
// Selecting text and then clicking into the composer (instead of the popover) attaches it.
$("question").addEventListener("focus", () => {
  if (!$("ask-popover").hidden && popoverSelection) {
    setPendingSelection(popoverSelection);
    hidePopover();
  }
});

$("stop").onclick = () => state.abort?.abort();

async function sendQuestion(question, selection) {
  if (state.streaming) return;
  if (!state.paper && !state.searchMode) { setStatus("Open a paper first"); return; }
  if (!state.searchMode) setChatCollapsed(false); // show the answer, e.g. when asking from the highlight popover
  state.streaming = true;
  state.abort = new AbortController();
  $("send").hidden = true;
  $("stop").hidden = false;

  const box = $("messages");
  box.querySelector(".chat-empty")?.remove();
  box.append(userBubble(question, selection));
  const view = new AssistantView();
  view.root.classList.add("cursor");
  box.append(view.root);
  box.scrollTop = box.scrollHeight;

  const body = {
    conversation_id: state.conv?.id || null,
    paper_id: state.paper?.id || null, // none: a search session
    provider: $("provider").value,
    model: $("model").value.trim(),
    question,
    selection: selection || null,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      signal: state.abort.signal, // Stop closes the stream; the server keeps what was said so far
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (line.startsWith("data: ")) handleEvent(JSON.parse(line.slice(6)), view);
      }
      const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
      if (nearBottom) box.scrollTop = box.scrollHeight;
    }
  } catch (e) {
    if (e.name === "AbortError") view.stopped();
    else view.error(`Error: ${e.message}`);
  } finally {
    view.root.classList.remove("cursor");
    state.streaming = false;
    $("send").hidden = false;
    $("stop").hidden = true;
    renderHistory();
  }
}

function handleEvent(ev, view) {
  switch (ev.type) {
    case "conversation": {
      const isNew = !state.conv;
      state.conv = ev.conversation;
      $("chat-title").textContent = ev.conversation.title;
      if (isNew) {
        history.replaceState(null, "", state.paper ? paperHash(state.paper.id, ev.conversation.id) : searchHash(ev.conversation.id));
      }
      break;
    }
    case "thinking": view.thinking(ev.text); break;
    case "text": view.text(ev.text); break;
    case "tool_call": view.toolCall(ev); break;
    case "tool_result": view.toolResult(ev); break;
    case "usage": view.usage(ev.usage, $("model").value.trim(), ev.stop_reason); loadUsage(); break;
    case "step": view.newText(); break;
    case "error": view.error(ev.message); break;
  }
}

// ---------------------------------------------------------------- Resizable, minimizable chat panel

// The bar between paper and chat: drag it to resize the chat, drag it most of the way shut to
// hide the chat, or click it. The minimized strip works the same way: click it, or drag it open.
(function setupResizer() {
  const app = $("app");
  const saved = store.get("chatWidth", null);
  if (saved) app.style.setProperty("--chat-width", `${saved}px`);
  const r = $("resizer");
  const SNAP_SHUT = 200; // a chat narrower than this while dragging hides it
  const dragHandle = (handle, onClick) => {
    let dragged = false;
    handle.addEventListener("pointerdown", (e) => {
      // Narrow screens stack the panels and hide the resizer, so there's no width to drag.
      if (!e.isPrimary || e.button !== 0 || !r.offsetParent) return;
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      const x0 = e.clientX, w0 = app.style.getPropertyValue("--chat-width");
      dragged = false;
      const move = (ev) => {
        if (!dragged && Math.abs(ev.clientX - x0) < 4) return; // still a click
        dragged = true;
        r.classList.add("dragging");
        const w = window.innerWidth - ev.clientX;
        const shut = w < SNAP_SHUT;
        if (shut !== app.classList.contains("chat-collapsed")) setChatCollapsed(shut);
        if (shut) return;
        const cw = Math.max(320, Math.min(window.innerWidth - 600, w));
        app.style.setProperty("--chat-width", `${cw}px`);
        store.set("chatWidth", cw);
      };
      const up = () => {
        r.classList.remove("dragging");
        // Dragged shut: reopen at the width it had, not the minimum it passed on the way.
        if (dragged && app.classList.contains("chat-collapsed") && w0) {
          app.style.setProperty("--chat-width", w0);
          store.set("chatWidth", parseInt(w0, 10));
        }
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        handle.removeEventListener("pointercancel", up);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
      handle.addEventListener("pointercancel", up);
    });
    // A drag still ends in a click; only a press without moving (or a keyboard press) counts.
    handle.addEventListener("click", (e) => {
      if (e.detail && dragged) return;
      onClick();
    });
  };
  dragHandle(r, () => setChatCollapsed(true));
  dragHandle($("chat-expand"), () => { setChatCollapsed(false); $("question").focus(); });
})();

function setChatCollapsed(collapsed) {
  $("app").classList.toggle("chat-collapsed", collapsed && !state.searchMode);
  store.set("chatCollapsed", collapsed);
}

// Set to leave search mode's next change of sidebar tab undone, e.g. after deleting the open paper.
let keepSidebarTab = false;

// Search mode (home page): no paper is open, and the chat takes the reader's place.
function setSearchMode(on) {
  const app = $("app");
  app.classList.toggle("search-mode", on);
  app.classList.toggle("chat-collapsed", !on && store.get("chatCollapsed", false));
  $("new-chat").textContent = on ? "New search" : "New chat";
  $("new-chat").title = on ? "Start a new search session" : "Start a new conversation on this paper";
  $("question").placeholder = on
    ? "Ask for papers on a topic, or describe one you're looking for… (Enter to send)"
    : "Ask a question… (Enter to send, Shift+Enter for newline)";
  if (on !== state.searchMode && !keepSidebarTab) setSidebarTab(on ? "searches" : "papers");
  keepSidebarTab = false;
  state.searchMode = on;
}
setChatCollapsed(store.get("chatCollapsed", false));
$("chat-collapse").onclick = () => setChatCollapsed(true);
document.addEventListener("keydown", (e) => {
  if (e.key === "\\" && (e.metaKey || e.ctrlKey) && !state.searchMode) {
    e.preventDefault();
    setChatCollapsed(!$("app").classList.contains("chat-collapsed"));
  }
});

// ---------------------------------------------------------------- Boot

window.addEventListener("hashchange", route);
await loadConfig().catch((e) => setStatus(`Config error: ${e.message}`));
loadUsage();
await route();
