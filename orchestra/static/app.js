"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const esc = (s) => String(s ?? "");

// When the server runs with ORCHESTRA_TOKEN, the phone gets it by opening
// http://host:8000/?t=<token>. Stash it so it survives navigation, and strip it
// from the visible URL so it doesn't end up in a screenshot or shared link.
const token = (() => {
  const fromUrl = new URLSearchParams(location.search).get("t");
  if (fromUrl) {
    sessionStorage.setItem("orchestra_token", fromUrl);
    history.replaceState({}, "", location.pathname);
  }
  return sessionStorage.getItem("orchestra_token") || "";
})();

const authHeaders = () => (token ? { "X-Orchestra-Token": token } : {});
// EventSource cannot send headers, so the stream carries the token as a query
// param — the same value the middleware already accepts.
const withToken = (url) => (token ? `${url}?t=${encodeURIComponent(token)}` : url);

const state = {
  models: [],
  chair: null,
  panel: new Set(),
  sessionId: null,
  source: null,
  convergence: [],
  activity: new Map(),
};

function showAuthError() {
  const badge = $("#live-badge");
  badge.textContent = "token required";
  badge.className = "badge warn";
  const err = $("#setup-error");
  err.textContent =
    "This server requires an access token. Open the URL the server printed at " +
    "startup, including the ?t=... part.";
  err.hidden = false;
  $("#run").disabled = true;
}

// ── boot ────────────────────────────────────────────────────────────

async function boot() {
  const res = await fetch("/api/models", { headers: authHeaders() });
  if (res.status === 401) return showAuthError();
  const data = await res.json();
  state.models = data.models;

  const badge = $("#live-badge");
  badge.textContent = data.simulation_forced
    ? "simulation forced"
    : `${data.live_count}/${data.models.length} models live`;
  badge.className = "badge " + (data.live_count > 0 && !data.simulation_forced ? "good" : "warn");

  renderChairs();
  renderPanel();

  // Default: the best available chair, everyone else on the panel.
  const preferred = ["claude-opus-5", "claude-fable-5", "gpt-5"].find((k) =>
    data.orchestrators.includes(k)
  );
  selectChair(preferred || data.orchestrators[0]);
  state.models.forEach((m) => {
    if (m.key !== state.chair) state.panel.add(m.key);
  });
  syncSelection();
}

function renderChairs() {
  const host = $("#orchestrators");
  host.innerHTML = "";
  state.models
    .filter((m) => m.can_orchestrate)
    .forEach((m) => {
      const card = el("label", "model-card");
      card.dataset.key = m.key;
      card.dataset.role = "chair";
      card.innerHTML = `
        <div class="row">
          <span class="name"><i class="dot ${m.live ? "live" : "sim"}"></i>${esc(m.name)}</span>
          <span class="vendor">${esc(m.vendor)}</span>
        </div>
        <div class="lens">${esc(m.lens)} · w${m.weight}</div>`;
      card.addEventListener("click", () => selectChair(m.key));
      host.appendChild(card);
    });
}

function renderPanel() {
  const host = $("#panel");
  host.innerHTML = "";
  state.models.forEach((m) => {
    const card = el("label", "model-card");
    card.dataset.key = m.key;
    card.dataset.role = "panel";
    card.innerHTML = `
      <div class="row">
        <span class="name"><i class="dot ${m.live ? "live" : "sim"}"></i>${esc(m.name)}</span>
        <span class="vendor">${esc(m.vendor)}</span>
      </div>
      <div class="lens">${esc(m.lens)} · ${esc((m.strengths || []).slice(0, 2).join(", "))}</div>`;
    card.addEventListener("click", () => {
      if (m.key === state.chair) return;
      state.panel.has(m.key) ? state.panel.delete(m.key) : state.panel.add(m.key);
      syncSelection();
    });
    host.appendChild(card);
  });
}

function selectChair(key) {
  state.chair = key;
  state.panel.delete(key);
  syncSelection();
}

function syncSelection() {
  document.querySelectorAll('[data-role="chair"]').forEach((c) => {
    c.classList.toggle("selected", c.dataset.key === state.chair);
  });
  document.querySelectorAll('[data-role="panel"]').forEach((c) => {
    const isChair = c.dataset.key === state.chair;
    c.classList.toggle("selected", state.panel.has(c.dataset.key));
    c.style.opacity = isChair ? 0.4 : 1;
    c.style.pointerEvents = isChair ? "none" : "auto";
  });
  $("#panel-count").textContent = `${state.panel.size} panelists selected`;
  renderEstimate();
}

// Mirrors CouncilConfig.max_calls(): the chair spends 2 + rounds, and so does
// each panelist. Shown before the run so a 15-model, 5-round council isn't a
// surprise on someone's billing page.
function renderEstimate() {
  const rounds = Number($("#rounds").value);
  const calls = (2 + rounds) * (1 + state.panel.size);
  const live = [...state.panel].filter(
    (k) => (state.models.find((m) => m.key === k) || {}).live
  ).length;
  const chairLive = (state.models.find((m) => m.key === state.chair) || {}).live;
  const billable = (2 + rounds) * (live + (chairLive ? 1 : 0));
  $("#estimate").textContent =
    `Up to ${calls} model calls (${billable} against live APIs, the rest simulated). ` +
    `Negotiation stops early once convergence is reached, so this is a ceiling.`;
}

// ── controls ────────────────────────────────────────────────────────

$("#rounds").addEventListener("input", (e) => {
  $("#rounds-out").value = e.target.value;
  renderEstimate();
});
$("#target").addEventListener("input", (e) => {
  $("#target-out").value = (e.target.value / 100).toFixed(2);
});
$("#theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
});
document.querySelectorAll("[data-select]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const mode = btn.dataset.select;
    state.panel.clear();
    if (mode !== "none") {
      state.models.forEach((m) => {
        if (m.key === state.chair) return;
        if (mode === "all" || m.live) state.panel.add(m.key);
      });
    }
    syncSelection();
  });
});

$("#run").addEventListener("click", start);
$("#reset").addEventListener("click", () => location.reload());

// ── run ─────────────────────────────────────────────────────────────

async function start() {
  const task = $("#task").value.trim();
  const errBox = $("#setup-error");
  errBox.hidden = true;

  if (!task) return fail("Describe the task first.");
  if (!state.chair) return fail("Pick a chair model.");
  if (state.panel.size === 0) return fail("Select at least one panelist.");

  $("#run").disabled = true;
  $("#run").textContent = "Convening…";

  const body = {
    task,
    context: $("#context").value,
    orchestrator: state.chair,
    panel: [...state.panel],
    rounds: Number($("#rounds").value),
    convergence_target: Number($("#target").value) / 100,
  };

  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    $("#run").disabled = false;
    $("#run").textContent = "Convene the council";
    return fail(detail.detail || "Could not start the run.");
  }

  const { session_id } = await res.json();
  state.sessionId = session_id;
  $("#setup").hidden = true;
  $("#run-view").hidden = false;
  $("#raw-events").hidden = false;
  initActivity([state.chair, ...state.panel]);
  listen(session_id);

  function fail(msg) {
    errBox.textContent = msg;
    errBox.hidden = false;
  }
}

function listen(sessionId) {
  const source = new EventSource(withToken(`/api/sessions/${sessionId}/events`));
  state.source = source;
  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    logRaw(event);
    handle(event);
    if (event.type === "closed") source.close();
  };
  source.onerror = () => {
    // EventSource retries on its own; only surface a hard close.
    if (source.readyState === EventSource.CLOSED) {
      $("#chair-log").prepend(el("p", "error", "Connection to the run was lost."));
    }
  };
}

// ── event handling ──────────────────────────────────────────────────

function handle(event) {
  switch (event.type) {
    case "start":
      break;
    case "phase":
      markPhase(event.phase);
      break;
    case "model_start":
      setActivity(event.model, "working", event.phase);
      break;
    case "model_done":
      setActivity(
        event.model,
        event.ok ? (event.simulated ? "sim" : "ok") : "failed",
        event.phase,
        event
      );
      break;
    case "framing":
      if (event.simulated) $("#sim-warning").hidden = false;
      renderFraming(event.data);
      break;
    case "proposal":
      if (event.data.simulated) $("#sim-warning").hidden = false;
      break;
    case "ledger":
      renderLedger(event.data);
      break;
    case "round":
      state.convergence.push(event.data.convergence);
      renderSpark();
      break;
    case "chair_note":
      $("#chair-log").appendChild(el("div", "chair-note", event.text));
      break;
    case "note":
    case "warning":
      $("#chair-log").appendChild(el("div", "chair-note", event.message));
      break;
    case "error":
      $("#report").hidden = false;
      $("#report").appendChild(el("p", "error", event.message));
      break;
    case "done":
      renderReport(event.report);
      $("#reset").hidden = false;
      break;
  }
}

function markPhase(phase) {
  const items = [...document.querySelectorAll("#stepper li")];
  let seen = false;
  items.forEach((li) => {
    if (li.dataset.phase === phase) {
      seen = true;
      li.className = "active";
    } else {
      li.className = seen ? "" : "done";
    }
  });
  if (phase === "done") items.forEach((li) => (li.className = "done"));
}

function initActivity(keys) {
  const host = $("#activity");
  host.innerHTML = "";
  state.activity.clear();
  [...new Set(keys)].forEach((key) => {
    const model = state.models.find((m) => m.key === key);
    const row = el("div", "act-row");
    row.innerHTML = `
      <i class="dot ${model && model.live ? "live" : "sim"}"></i>
      <span class="key">${esc(model ? model.name : key)}</span>
      <span class="meta">idle</span>`;
    host.appendChild(row);
    state.activity.set(key, row);
  });
}

function setActivity(key, status, phase, event) {
  const row = state.activity.get(key);
  if (!row) return;
  row.classList.toggle("working", status === "working");
  row.classList.toggle("failed", status === "failed");
  const meta = row.querySelector(".meta");
  const dot = row.querySelector(".dot, .spinner");

  if (status === "working") {
    if (dot) dot.className = "spinner";
    meta.textContent = phase.replace(/_/g, " ");
  } else {
    const model = state.models.find((m) => m.key === key);
    if (dot) dot.className = `dot ${status === "sim" || !(model && model.live) ? "sim" : "live"}`;
    if (status === "failed") {
      meta.textContent = (event && event.error ? event.error : "failed").slice(0, 40);
      meta.title = (event && event.error) || "";
    } else {
      meta.textContent = `${event ? event.latency_s : 0}s`;
    }
  }
}

// ── renderers ───────────────────────────────────────────────────────

function renderFraming(framing) {
  const host = $("#chair-log");
  host.innerHTML = "";
  const box = el("div", "chair-note");
  box.appendChild(el("strong", null, "Objective"));
  box.appendChild(el("p", null, framing.objective));
  if ((framing.key_uncertainties || []).length) {
    box.appendChild(el("strong", null, "Key uncertainties"));
    const ul = el("ul");
    framing.key_uncertainties.forEach((u) => ul.appendChild(el("li", null, u)));
    box.appendChild(ul);
  }
  host.appendChild(box);
}

function renderLedger(snapshot) {
  $("#m-convergence").textContent = snapshot.convergence.toFixed(2);
  $("#m-accepted").textContent = snapshot.counts.accepted;
  $("#m-contested").textContent = snapshot.counts.contested;
  $("#m-rejected").textContent = snapshot.counts.rejected;
  $("#ledger-hint").textContent = `${snapshot.claims.length} claims`;

  const host = $("#ledger");
  host.innerHTML = "";
  const order = { contested: 0, accepted: 1, rejected: 2 };
  const claims = [...snapshot.claims].sort(
    (a, b) => order[a.status] - order[b.status] || b.support - a.support
  );

  claims.forEach((c) => {
    const node = el("div", `claim ${c.status}`);

    const head = el("div", "claim-head");
    head.appendChild(el("span", null, c.id));
    head.appendChild(el("span", null, c.kind));
    head.appendChild(el("span", null, `by ${c.author}`));
    head.appendChild(el("span", null, `support ${c.support >= 0 ? "+" : ""}${c.support.toFixed(2)}`));
    head.appendChild(el("span", null, `engaged ${(c.participation * 100).toFixed(0)}%`));
    node.appendChild(head);

    node.appendChild(el("div", "claim-text", c.text));

    const bar = el("div", "support-bar");
    const fill = el("i", c.support >= 0 ? "pos" : "neg");
    fill.style.width = `${Math.abs(c.support) * 50}%`;
    bar.appendChild(fill);
    node.appendChild(bar);

    const votes = el("div", "votes");
    Object.entries(c.votes).forEach(([model, v]) => {
      const chip = el("span", `vote ${v.stance}`, `${model} ${v.stance[0]}${v.confidence.toFixed(1)}`);
      chip.title = v.rationale || "";
      votes.appendChild(chip);
    });
    node.appendChild(votes);

    const disputes = Object.entries(c.votes).filter(([, v]) => v.stance === "dispute");
    if (disputes.length || (c.amendments || []).length) {
      const det = el("details");
      det.appendChild(el("summary", null, "objections & amendments"));
      disputes.forEach(([model, v]) => {
        det.appendChild(el("p", "rationale", `${model}: ${v.rationale}`));
      });
      (c.amendments || []).forEach((a) => det.appendChild(el("p", "rationale", `↪ ${a}`)));
      node.appendChild(det);
    }
    host.appendChild(node);
  });
}

function renderSpark() {
  const svg = $("#spark");
  svg.innerHTML = "";
  const points = state.convergence;
  const W = 320, H = 90, pad = 8;

  const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
  grid.setAttribute("x1", 0); grid.setAttribute("x2", W);
  grid.setAttribute("y1", H / 2); grid.setAttribute("y2", H / 2);
  grid.setAttribute("stroke", "var(--border)"); grid.setAttribute("stroke-dasharray", "3 4");
  svg.appendChild(grid);

  if (!points.length) return;
  const x = (i) => pad + (i * (W - pad * 2)) / Math.max(1, points.length - 1);
  const y = (v) => H - pad - v * (H - pad * 2);

  const path = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  path.setAttribute("points", points.map((v, i) => `${x(i)},${y(v)}`).join(" "));
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "var(--accent)");
  path.setAttribute("stroke-width", "2");
  svg.appendChild(path);

  points.forEach((v, i) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(v)); dot.setAttribute("r", 3);
    dot.setAttribute("fill", "var(--accent)");
    dot.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "title")).textContent =
      `round ${i + 1}: ${v.toFixed(2)}`;
    svg.appendChild(dot);
  });
}

function renderReport(report) {
  const host = $("#report");
  host.hidden = false;
  host.innerHTML = "";
  const s = report.synthesis || {};
  const stats = report.stats || {};

  if (report.any_simulated) $("#sim-warning").hidden = false;
  $("#m-calls").textContent = stats.calls ?? "—";

  host.appendChild(el("h2", null, s.headline || "Council report"));

  const meta = el("p", "hint",
    `confidence ${(s.confidence ?? 0).toFixed(2)} · convergence ${report.ledger.convergence.toFixed(2)} · ` +
    `${stats.calls} calls (${stats.failures} failed, ${stats.simulated_calls} simulated) · ` +
    `${stats.output_tokens.toLocaleString()} output tokens · ${stats.wall_s}s`);
  host.appendChild(meta);

  if (s.analysis) {
    host.appendChild(el("h4", null, "Analysis"));
    s.analysis.split(/\n\n+/).forEach((p) => host.appendChild(el("p", null, p)));
  }

  if ((s.findings || []).length) {
    host.appendChild(el("h4", null, "Findings"));
    const ol = el("ol");
    s.findings.forEach((f) => ol.appendChild(el("li", null, f)));
    host.appendChild(ol);
  }

  if ((s.action_plan || []).length) {
    host.appendChild(el("h4", null, "Action plan"));
    const wrap = el("div", "actions");
    s.action_plan.forEach((a) => {
      const node = el("div", "action");
      node.appendChild(el("div", "what", a.action));
      if (a.first_step) node.appendChild(el("div", "step", `First step: ${a.first_step}`));
      const tags = el("div", "tags");
      if (a.owner_hint) tags.appendChild(el("span", "tag", a.owner_hint));
      tags.appendChild(el("span", "tag", `effort ${a.effort}`));
      tags.appendChild(el("span", "tag", `impact ${a.impact}`));
      node.appendChild(tags);
      wrap.appendChild(node);
    });
    host.appendChild(wrap);
  }

  if ((s.dissent || []).length) {
    host.appendChild(el("h4", null, "Dissent register"));
    const ul = el("ul", "dissent");
    s.dissent.forEach((d) => ul.appendChild(el("li", null, d)));
    host.appendChild(ul);
  }

  if ((s.what_would_change_our_mind || []).length) {
    host.appendChild(el("h4", null, "What would change this conclusion"));
    const ul = el("ul");
    s.what_would_change_our_mind.forEach((d) => ul.appendChild(el("li", null, d)));
    host.appendChild(ul);
  }

  const actions = el("div", "report-actions");
  const md = el("button", "ghost", "Download markdown");
  md.addEventListener("click", () => {
    window.open(withToken(`/api/sessions/${state.sessionId}/report.md`), "_blank");
  });
  const json = el("button", "ghost", "Download JSON");
  json.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = `council-${state.sessionId}.json`;
    a.click();
  });
  actions.append(md, json);
  host.appendChild(actions);

  host.scrollIntoView({ behavior: "smooth", block: "start" });
}

function logRaw(event) {
  const pre = $("#raw-log");
  const line = `${String(event.seq).padStart(3, "0")} ${event.type}${event.model ? " " + event.model : ""}${event.phase ? " [" + event.phase + "]" : ""}\n`;
  pre.textContent += line;
  pre.scrollTop = pre.scrollHeight;
}

boot();
