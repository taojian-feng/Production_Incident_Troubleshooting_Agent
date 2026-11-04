"use strict";
// Reports hold log lines and model output, so all data is rendered through textContent.
(() => {
  const config = window.IA_CONFIG || {};
  const main = document.getElementById("main");
  const TOKEN_KEY = "ia.tokens";
  const VERIFIER_KEY = "ia.verifier";
  const STATE_KEY = "ia.state";
  const HEADINGS = [
    ["summary", "Summary"],
    ["precedent", "Similar past incidents"],
    ["recommendation", "Recommended actions"],
  ];

  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== null && value !== undefined) node.setAttribute(key, value);
    }
    node.append(...children.filter(Boolean));
    return node;
  }

  function show(...nodes) {
    main.replaceChildren(...nodes);
    window.scrollTo(0, 0);
  }

  function showError(message, retry) {
    show(
      el("p", { class: "error", role: "alert", text: message }),
      retry ? el("button", { type: "button", onclick: retry, text: "Try again" }) : null
    );
  }

  // ---- sign-in: authorization code with PKCE

  function randomString() {
    const bytes = crypto.getRandomValues(new Uint8Array(32));
    return base64Url(bytes);
  }

  function base64Url(bytes) {
    return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  async function signIn() {
    const verifier = randomString();
    const state = randomString();
    sessionStorage.setItem(VERIFIER_KEY, verifier);
    sessionStorage.setItem(STATE_KEY, state);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
    const query = new URLSearchParams({
      response_type: "code",
      client_id: config.clientId,
      redirect_uri: config.redirectUri,
      scope: "openid email",
      state,
      code_challenge: base64Url(new Uint8Array(digest)),
      code_challenge_method: "S256",
    });
    window.location.assign(`${config.authDomain}/oauth2/authorize?${query}`);
  }

  function signOut() {
    sessionStorage.clear();
    if (config.preview) return window.location.assign("/");
    const query = new URLSearchParams({ client_id: config.clientId, logout_uri: config.redirectUri });
    window.location.assign(`${config.authDomain}/logout?${query}`);
  }

  async function finishSignIn(params) {
    const expected = sessionStorage.getItem(STATE_KEY);
    const verifier = sessionStorage.getItem(VERIFIER_KEY);
    sessionStorage.removeItem(STATE_KEY);
    sessionStorage.removeItem(VERIFIER_KEY);
    window.history.replaceState(null, "", window.location.pathname);
    if (!verifier || params.get("state") !== expected) throw new Error("Sign-in could not be verified.");
    const response = await fetch(`${config.authDomain}/oauth2/token`, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        grant_type: "authorization_code",
        client_id: config.clientId,
        redirect_uri: config.redirectUri,
        code: params.get("code"),
        code_verifier: verifier,
      }),
    });
    if (!response.ok) throw new Error("Sign-in did not complete.");
    const tokens = await response.json();
    sessionStorage.setItem(TOKEN_KEY, tokens.id_token);
  }

  function claims(token) {
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(escape(atob(payload))));
  }

  function currentToken() {
    const token = sessionStorage.getItem(TOKEN_KEY);
    if (!token) return null;
    return claims(token).exp * 1000 > Date.now() + 5000 ? token : null;
  }

  // ---- API

  async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (!config.preview) {
      const token = currentToken();
      if (!token) {
        await signIn();
        return new Promise(() => {});
      }
      headers.authorization = `Bearer ${token}`;
    }
    const response = await fetch(`/api${path}`, { ...options, headers });
    if (response.status === 401) {
      sessionStorage.removeItem(TOKEN_KEY);
      await signIn();
      return new Promise(() => {});
    }
    if (!response.ok) throw new Error(`The server returned ${response.status}.`);
    return response.json();
  }

  // ---- shared pieces

  function statusPill(status) {
    const label = status === "validated" ? "Validated" : "Needs review";
    return el("span", { class: `pill ${status}`, text: label });
  }

  function formatTime(value) {
    return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  // ---- report list

  async function renderList() {
    document.title = "Incident Reports";
    const { reports } = await api("/reports");
    if (!reports.length) {
      return show(
        el("h1", { text: "Reports" }),
        el("p", { class: "muted", text: "No reports yet. They appear here when an alert is triaged." })
      );
    }
    const rows = reports.map((report) =>
      el(
        "tr",
        {},
        el("td", { class: "when", text: formatTime(report.created_at) }),
        el("td", { class: `sev ${report.severity}`, text: report.severity }),
        el("td", { text: report.service }),
        el("td", {}, el("a", { href: `#report/${encodeURIComponent(report.report_id)}`, text: report.title })),
        el("td", {}, statusPill(report.status))
      )
    );
    const header = el(
      "tr",
      {},
      ...["Time", "Severity", "Service", "Alert", "Status"].map((name) => el("th", { scope: "col", text: name }))
    );
    show(
      el("h1", { text: "Reports" }),
      el("p", { class: "muted", text: `${reports.length} most recent, newest first.` }),
      el("div", { class: "scroll" }, el("table", {}, el("thead", {}, header), el("tbody", {}, ...rows)))
    );
  }

  // ---- report detail

  function renderStep(step, reportId) {
    const links = step.citations.map((id) =>
      el("a", {
        href: `#report/${encodeURIComponent(reportId)}/${id}`,
        text: id,
        "aria-label": `Evidence ${id}`,
      })
    );
    return el(
      "article",
      { class: step.verified ? "step" : "step flagged" },
      el("p", { text: step.content }),
      links.length ? el("div", { class: "sources" }, "Sources:", ...links) : null,
      step.verified ? null : el("p", { class: "flag", text: `Needs review: ${step.review_reason || "not tied to verified evidence"}` })
    );
  }

  function renderEvidence(item) {
    return el(
      "details",
      { class: "evidence", id: `evidence-${item.evidence_id}` },
      el("summary", { text: `${item.evidence_id} (${item.source}): ${item.title}` }),
      el("pre", { text: item.snippet })
    );
  }

  function renderFeedback(report, past) {
    let rating = null;
    const up = el("button", { type: "button", "aria-pressed": "false", text: "Helpful" });
    const down = el("button", { type: "button", "aria-pressed": "false", text: "Not helpful" });
    const correction = el("textarea", {
      id: "correction",
      placeholder: "What was the actual cause or fix? Corrections can become evaluation cases.",
    });
    const send = el("button", { type: "submit", class: "primary", disabled: "", text: "Send feedback" });
    const status = el("p", { class: "status", role: "status" });

    function choose(value) {
      rating = value;
      up.setAttribute("aria-pressed", String(value === "up"));
      down.setAttribute("aria-pressed", String(value === "down"));
      send.disabled = false;
    }
    up.addEventListener("click", () => choose("up"));
    down.addEventListener("click", () => choose("down"));

    const form = el(
      "form",
      {
        class: "feedback",
        onsubmit: async (event) => {
          event.preventDefault();
          send.disabled = true;
          status.className = "status";
          status.textContent = "Sending…";
          try {
            await api(`/reports/${encodeURIComponent(report.report_id)}/feedback`, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ rating, correction: correction.value.trim() || null }),
            });
            status.textContent = "Feedback saved. Thank you.";
            correction.value = "";
          } catch (error) {
            status.className = "status error";
            status.textContent = `Feedback was not saved. ${error.message}`;
            send.disabled = false;
          }
        },
      },
      past.length
        ? el(
            "ul",
            { class: "past" },
            ...past.map((item) =>
              el("li", {
                text: `${item.user}: ${item.rating === "up" ? "helpful" : "not helpful"}${item.correction ? `. ${item.correction}` : ""}`,
              })
            )
          )
        : null,
      el("div", { class: "row" }, up, down),
      el("label", { for: "correction", text: "Correction (optional)" }),
      correction,
      el("div", { class: "row spaced" }, send),
      status
    );
    return form;
  }

  async function renderReport(reportId, focus) {
    const { report, feedback } = await api(`/reports/${encodeURIComponent(reportId)}`);
    document.title = `${report.alert.title} | Incident Reports`;
    const sections = HEADINGS.flatMap(([kind, heading]) => {
      const steps = report.steps.filter((step) => step.kind === kind);
      return steps.length ? [el("h2", { text: heading }), ...steps.map((step) => renderStep(step, reportId))] : [];
    });
    show(
      el("p", {}, el("a", { href: "#", text: "All reports" })),
      el("h1", { text: report.alert.title }),
      el(
        "div",
        { class: "meta" },
        statusPill(report.status),
        el("span", { class: `sev ${report.alert.severity}`, text: report.alert.severity }),
        el("span", { text: report.alert.service }),
        el("span", { text: formatTime(report.created_at) }),
        el("span", { text: `${(report.latency_ms / 1000).toFixed(0)}s, $${report.cost.usd.toFixed(3)}` })
      ),
      report.alert.description ? el("p", { class: "muted spaced", text: report.alert.description }) : null,
      ...sections,
      el("h2", { text: "Evidence" }),
      ...report.evidence.map(renderEvidence),
      el("h2", { text: "Feedback" }),
      renderFeedback(report, feedback)
    );
    if (focus) {
      const target = document.getElementById(`evidence-${focus}`);
      if (target) {
        target.open = true;
        target.scrollIntoView({ block: "center" });
      }
    }
  }

  // ---- routing: #report/<id> and #report/<id>/<evidence id>

  async function route() {
    const [view, id, focus] = window.location.hash.slice(1).split("/");
    try {
      if (view === "report" && id) await renderReport(decodeURIComponent(id), focus);
      else await renderList();
    } catch (error) {
      showError(`Could not load this page. ${error.message}`, route);
    }
  }

  async function start() {
    const params = new URLSearchParams(window.location.search);
    try {
      if (params.has("error")) throw new Error("Sign-in was not completed.");
      if (params.has("code")) await finishSignIn(params);
    } catch (error) {
      return show(
        el("p", { class: "error", role: "alert", text: error.message }),
        el("button", { type: "button", class: "primary", onclick: signIn, text: "Sign in" })
      );
    }
    const token = currentToken();
    if (token) document.getElementById("who").textContent = claims(token).email || "";
    const out = document.getElementById("sign-out");
    out.hidden = !token;
    out.addEventListener("click", signOut);
    window.addEventListener("hashchange", route);
    route();
  }

  start();
})();
