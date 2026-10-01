/* ORION shared frontend utilities. Vanilla JS, no framework.
   All rendering reflects backend data only — never fabricate metrics. */
(function () {
  "use strict";

  // ---- Demo mode (?demo=1) ----
  const params = new URLSearchParams(window.location.search);
  const demoMode = params.get("demo") === "1";
  if (demoMode) document.body.classList.add("demo");

  const Orion = {};

  // ---- CRT effects (on by default; off in demo; remembered per browser) ----
  (function initCRT() {
    let crtOn = true;
    try {
      const saved = localStorage.getItem("orion.crt");
      if (saved !== null) crtOn = saved === "1";
    } catch (e) { /* ignore */ }
    if (demoMode) crtOn = false; // conference mode: no decorative flicker
    function apply() { document.body.classList.toggle("crt", crtOn); }
    apply();
    document.addEventListener("DOMContentLoaded", function () {
      const btn = document.getElementById("crt-toggle");
      if (!btn) return;
      btn.textContent = "CRT: " + (crtOn ? "ON" : "OFF");
      btn.addEventListener("click", function () {
        crtOn = !crtOn;
        apply();
        btn.textContent = "CRT: " + (crtOn ? "ON" : "OFF");
        try { localStorage.setItem("orion.crt", crtOn ? "1" : "0"); } catch (e) { /* ignore */ }
      });
    });
  })();

  // ---- fetch helpers ----
  Orion.getJSON = async function (url) {
    const r = await fetch(url, { headers: { Accept: "application/json" } });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  };

  Orion.postJSON = async function (url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  };

  Orion.postForm = async function (url, formData) {
    const r = await fetch(url, { method: "POST", body: formData });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  };

  // ---- formatting ----
  Orion.esc = function (s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  };

  Orion.shortId = function (id) { return id ? String(id).slice(0, 8) : "?"; };

  Orion.fmtTime = function (iso) {
    if (!iso) return "N/A";
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  };

  // ---- reusable renderers (mirror the Jinja components) ----
  Orion.statusBadge = function (status) {
    const s = (status || "neutral").toLowerCase();
    return `<span class="badge badge-${s}">${Orion.esc(status || "N/A")}</span>`;
  };

  // Offensive statuses are "bad" for the defender (red); resisted = good (green).
  Orion.statusTone = function (status) {
    const s = (status || "").toUpperCase();
    if (s === "ATTACK_SUCCESS" || s === "ERROR") return "bad";
    if (s === "ATTACK_BLOCKED" || s === "NO_ATTACK") return "good";
    if (s === "PARTIALLY_MITIGATED") return "warn";
    return "";
  };

  Orion.metricCard = function (key, value, unit, tone) {
    const v = value === null || value === undefined ? "N/A" : value;
    return `<div class="metric ${tone || ""}"><div class="k">${Orion.esc(key.replace(/_/g, " "))}</div>` +
           `<div class="v">${Orion.esc(v)}${unit ? " " + Orion.esc(unit) : ""}</div></div>`;
  };

  Orion.atlasItem = function (m) {
    return `<div class="atlas-item">` +
      `<div><span class="tid">${Orion.esc(m.technique_id)}</span> &mdash; <span class="tname">${Orion.esc(m.technique || "N/A")}</span></div>` +
      `<div class="tac">Tactic: ${Orion.esc(m.tactic || "N/A")} &middot; confidence: ${Orion.esc(m.confidence || "N/A")}</div>` +
      (m.rationale ? `<div class="rat">${Orion.esc(m.rationale)}</div>` : "") +
      (!m.known ? `<div class="unverified">⚠ unverified: not in Orion's curated ATLAS index</div>` : "") +
      `</div>`;
  };

  Orion.toolItem = function (t) {
    return `<div class="tool-item"><div class="tname">${Orion.esc(t.name)}</div>` +
      (t.description ? `<div class="tdesc">${Orion.esc(t.description)}</div>` : "") +
      (t.input_schema ? `<pre>${Orion.esc(JSON.stringify(t.input_schema))}</pre>` : "") +
      `</div>`;
  };

  // ---- markdown (marked loaded via CDN where needed) ----
  Orion.markdown = function (md) {
    if (!md) return "";
    if (window.marked && typeof window.marked.parse === "function") return window.marked.parse(md);
    return "<pre>" + Orion.esc(md) + "</pre>";
  };

  // ---- state helpers ----
  Orion.setState = function (el, kind, msg) {
    if (!el) return;
    el.classList.remove("hidden");
    const cls = kind === "error" ? "state error" : "state";
    if (kind === "running") {
      el.innerHTML = `<div class="${cls}"><span class="loading-line">ORION IS RUNNING... ${Orion.esc(msg || "")}</span></div>`;
    } else {
      el.innerHTML = `<div class="${cls}">${Orion.esc(msg || "")}</div>`;
    }
  };

  window.Orion = Orion;
})();
