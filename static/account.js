// Accounts: the sign-in page (username and password, or Google), the links sent by email
// (#verify=…, #reset=…), and the Account panel where users add their own API keys.

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

// A JSON request; a failure throws an Error with the server's message (and its JSON as .body).
async function post(path, body, method = "POST") {
  const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
    throw Object.assign(new Error(detail || res.statusText), { body: data, status: res.status });
  }
  return data;
}

const GOOGLE_G = `<svg viewBox="0 0 48 48" width="18" height="18" aria-hidden="true">
  <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
  <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
  <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
  <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
</svg>`;

// ---------------------------------------------------------------- Signing in

// The signed-in user. Until there is one, the sign-in page is shown and this never resolves:
// signing in reloads the page. Links from emails are handled first.
export async function signIn() {
  const hash = new URLSearchParams(location.hash.slice(1));
  let notice = null;
  if (hash.has("verify")) {
    history.replaceState(null, "", "#/");
    try {
      await post("/api/auth/verify", { token: hash.get("verify") });
    } catch (e) { notice = { error: e.message }; }
  }
  if (hash.has("signin-error")) {
    notice = { error: hash.get("signin-error") };
    history.replaceState(null, "", "#/");
  }
  const reset = hash.get("reset");
  if (!reset) {
    const res = await fetch("/api/auth/me");
    if (res.ok) {
      document.body.classList.remove("signed-out");
      $("auth").hidden = true;
      return res.json();
    }
  }
  const options = await fetch("/api/auth/options").then((r) => r.json()).catch(() => ({}));
  showAuthPage(options, reset ? "reset" : "signin", { notice, resetToken: reset });
  return new Promise(() => {});
}

function showAuthPage(options, view, extra = {}) {
  document.body.classList.add("signed-out");
  const root = $("auth");
  root.hidden = false;
  const card = el("div", { class: "auth-card" });
  root.replaceChildren(card);
  const show = (v, more = {}) => showAuthPage(options, v, { ...extra, ...more, notice: more.notice ?? null });

  card.append(
    el("div", { class: "auth-brand" }, el("img", { src: "/static/icon.svg", alt: "", width: 36, height: 36 }),
      el("div", {}, el("h1", {}, "Paper Reader"), el("p", { class: "muted" }, "Read research papers with an AI assistant at your side."))),
  );
  const notice = extra.notice;
  if (notice) card.append(el("p", { class: notice.error ? "auth-msg error" : "auth-msg" }, notice.error || notice.text));

  const form = (fields, submit, onsubmit) => {
    const msg = el("p", { class: "auth-msg error", hidden: true });
    const button = el("button", { type: "submit", class: "auth-submit" }, submit);
    const f = el("form", {
      class: "auth-form", novalidate: true,
      onsubmit: async (e) => {
        e.preventDefault();
        msg.hidden = true;
        button.disabled = true;
        try { await onsubmit(Object.fromEntries(new FormData(f))); } catch (err) {
          msg.textContent = err.message;
          msg.hidden = false;
        } finally { button.disabled = false; }
      },
    }, fields, msg, button);
    return f;
  };
  const field = (label, name, type = "text", attrs = {}) =>
    el("label", { class: "auth-field" }, label, el("input", { name, type, required: true, ...attrs }));

  if (view === "signin" || view === "signup") {
    if (options.google) {
      const g = el("a", { class: "google-btn", href: "/auth/google" });
      g.innerHTML = `${GOOGLE_G}<span>Continue with Google</span>`;
      card.append(g, el("div", { class: "auth-or" }, el("span", {}, "or")));
    }
    card.append(el("div", { class: "auth-tabs", role: "tablist" },
      el("button", { type: "button", role: "tab", "aria-selected": String(view === "signin"), onclick: () => show("signin") }, "Sign in"),
      el("button", { type: "button", role: "tab", "aria-selected": String(view === "signup"), onclick: () => show("signup") }, "Create account"),
    ));
  }

  if (view === "signin") {
    card.append(form([
      field("Username or email", "login", "text", { autocomplete: "username", autofocus: true }),
      field("Password", "password", "password", { autocomplete: "current-password" }),
    ], "Sign in", async ({ login, password }) => {
      try {
        await post("/api/auth/login", { login, password });
      } catch (e) {
        if (e.body?.unverified) return show("check", { login, password, email: null, notice: { error: e.message } });
        throw e;
      }
      location.reload();
    }), el("button", { type: "button", class: "auth-link", onclick: () => show("forgot") }, "Forgot password?"));
  } else if (view === "signup") {
    card.append(form([
      field("Username", "username", "text", { autocomplete: "username", autofocus: true, minlength: 3, maxlength: 32 }),
      field("Email", "email", "email", { autocomplete: "email" }),
      field("Password", "password", "password", { autocomplete: "new-password", minlength: 8, placeholder: "At least 8 characters" }),
    ], "Create account", async ({ username, email, password }) => {
      const r = await post("/api/auth/signup", { username, email, password });
      show("check", { login: username, password, email: r.email });
    }), el("p", { class: "muted auth-note" }, "We'll email you a link to confirm your address before you can sign in.",
      options.email === false ? " (This server has no email service set up yet: the link is written to its log.)" : ""));
  } else if (view === "check") {
    const sent = el("p", { class: "auth-msg", hidden: true }, "Sent again. Check your spam folder too.");
    card.append(
      el("h2", {}, "Check your inbox"),
      el("p", {}, extra.email ? `We sent a confirmation link to ${extra.email}. Open it to finish creating your account.`
        : "Open the confirmation link we emailed you to finish creating your account."),
      sent,
      el("div", { class: "auth-row" },
        el("button", { type: "button", class: "auth-link", onclick: async () => {
          await post("/api/auth/resend", { login: extra.login, password: extra.password }).catch(() => {});
          sent.hidden = false;
        } }, "Send the link again"),
        el("button", { type: "button", class: "auth-link", onclick: () => show("signin") }, "Back to sign in")),
    );
  } else if (view === "forgot") {
    card.append(el("h2", {}, "Reset your password"), form([
      field("Email", "email", "email", { autocomplete: "email", autofocus: true }),
    ], "Email me a link", async ({ email }) => {
      await post("/api/auth/forgot", { email });
      show("signin", { notice: { text: `If there's an account for ${email}, we've sent it a link to choose a new password.` } });
    }), el("button", { type: "button", class: "auth-link", onclick: () => show("signin") }, "Back to sign in"));
  } else if (view === "reset") {
    card.append(el("h2", {}, "Choose a new password"), form([
      field("New password", "password", "password", { autocomplete: "new-password", minlength: 8, autofocus: true }),
      field("Repeat it", "again", "password", { autocomplete: "new-password" }),
    ], "Set password", async ({ password, again }) => {
      if (password !== again) throw new Error("The two passwords differ");
      await post("/api/auth/reset", { token: extra.resetToken, password });
      location.replace("/#/");
      location.reload();
    }));
  }
  card.querySelector("[autofocus]")?.focus();
}

export async function signOut() {
  await post("/api/auth/logout", {}).catch(() => {});
  location.replace("/#/");
  location.reload();
}

// ---------------------------------------------------------------- Account panel (API keys)

export const KEYS = [
  { id: "openrouter", label: "OpenRouter", url: "https://openrouter.ai/settings/keys",
    note: "Recommended: one key for models from every vendor (Claude, GPT, Gemini, DeepSeek…). Also turns on voice questions." },
  { id: "anthropic", label: "Claude (Anthropic)", url: "https://console.anthropic.com/settings/keys" },
  { id: "openai", label: "OpenAI", url: "https://platform.openai.com/api-keys" },
  { id: "gemini", label: "Gemini (Google AI Studio)", url: "https://aistudio.google.com/apikey" },
  { id: "tavily", label: "Tavily", url: "https://app.tavily.com", optional: true,
    note: "Optional: better web search for the assistant. Without it, searches go to DuckDuckGo." },
];

// onChange(user) runs after every save, with the updated user; onClose when the panel closes.
export function openAccount(user, { onChange, onClose } = {}) {
  const modal = $("account-modal");
  const body = $("account-body");
  const close = () => {
    modal.hidden = true;
    document.removeEventListener("keydown", onKey);
    onClose?.();
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  $("account-close").onclick = close;
  modal.onclick = (e) => { if (e.target === modal) close(); };

  const render = () => {
    const how = [user.google && "Google", user.has_password && "password"].filter(Boolean).join(" or ");
    const rows = KEYS.map((k) => keyRow(k, user, async (value) => {
      user = await post("/api/account", { keys: { [k.id]: value } }, "PATCH");
      onChange?.(user);
      render();
    }));
    body.replaceChildren(
      el("div", { class: "account-who" },
        el("div", {}, el("strong", {}, user.name || user.username || user.email),
          user.is_admin ? el("span", { class: "badge" }, "admin") : null),
        el("div", { class: "muted" }, user.email, how ? ` · signs in with ${how}` : "")),
      el("h3", {}, "API keys"),
      el("p", { class: "muted account-intro" }, user.is_admin
        ? "As an admin you use the keys this server was deployed with (marked “server”). A key you add here is used only for a provider the server has no key for."
        : "Paper Reader runs on your own API keys: the model calls of your questions and paper analyses are billed to them. You need at least one model provider. Keys are stored encrypted and used only for your requests."),
      ...rows,
    );
  };
  render();
  modal.hidden = false;
  body.querySelector("input")?.focus();
}

function keyRow(k, user, save) {
  const current = user.keys[k.id];
  const input = el("input", {
    type: "password", autocomplete: "off", spellcheck: "false",
    placeholder: current ? `Replace the saved key (${current.hint || "…"})` : "Paste a key",
  });
  const msg = el("p", { class: "key-msg", hidden: true });
  const saveBtn = el("button", { type: "submit" }, "Save");
  const status = current?.source === "server" ? el("span", { class: "badge" }, "server")
    : current ? el("span", { class: "badge good" }, `saved ${current.hint}`) : null;
  const run = async (value) => {
    msg.hidden = true;
    saveBtn.disabled = true;
    saveBtn.textContent = value ? "Checking…" : "Save";
    try { await save(value); } catch (e) {
      msg.textContent = e.message;
      msg.hidden = false;
      saveBtn.disabled = false;
      saveBtn.textContent = "Save";
    }
  };
  return el("form", {
    class: "key-row",
    onsubmit: (e) => { e.preventDefault(); if (input.value.trim()) run(input.value.trim()); },
  },
  el("div", { class: "key-head" },
    el("span", { class: "key-label" }, k.label), status,
    el("a", { href: k.url, target: "_blank", rel: "noopener", class: "key-get" }, "Get a key ↗")),
  k.note ? el("p", { class: "muted key-note" }, k.note) : null,
  current?.source === "server" ? null : el("div", { class: "key-input" }, input, saveBtn,
    current ? el("button", { type: "button", class: "secondary", onclick: () => run(null) }, "Remove") : null),
  msg);
}
