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

  // ---- shared plan review / edit / approve flow (EDIT PLAN) ----
  Orion.planReview = async function (container, sourceType, analysis) {
    Orion.setState(container, "running", "building draft plan for review…");
    container.classList.remove("hidden");
    let plan;
    try {
      const d = await Orion.postJSON("/api/plans/draft", { source_type: sourceType, analysis: analysis });
      plan = d.plan;
    } catch (e) { Orion.setState(container, "error", "[x] " + e.message); return; }

    const editable = plan.proposals.filter(p => ["READY", "NEEDS_INPUT"].includes(p.status));
    const excluded = plan.proposals.filter(p => p.status === "EXCLUDED");

    function paramInputs(p) {
      if (p.status !== "READY" || !p.scenario) return "";
      let h = "";
      ["epsilon", "iterations"].forEach(k => {
        if (k in (p.parameters || {}))
          h += `<label class="sub" style="margin-right:0.6rem;">${k} <input type="number" step="any" style="width:90px;" data-eid="${Orion.esc(p.experiment_id)}" data-param="${k}" value="${Orion.esc(p.parameters[k])}"></label>`;
      });
      return `<div style="margin-top:0.3rem;">${h}</div>`;
    }
    function row(p) {
      const removable = `<label class="toggle" style="margin:0;"><input type="checkbox" class="keep" data-eid="${Orion.esc(p.experiment_id)}" checked> keep</label>`;
      return `<div class="exp"><span class="eid">${Orion.esc(p.experiment_id)}</span><span class="ename">${Orion.esc(p.name)}</span>
        <span class="risk ${p.status === "READY" ? "LOW" : "MEDIUM"}">${Orion.esc(p.status)}</span>
        <span class="risk ${Orion.esc(p.risk || "LOCAL")}" style="margin-left:0.3rem;">${Orion.esc(p.risk || "LOCAL")}</span>
        <div class="sub">${Orion.esc(p.reason || "")}${p.missing && p.missing.length ? " · missing: " + Orion.esc(p.missing.join(", ")) : ""}</div>
        ${paramInputs(p)}
        <div style="margin-top:0.3rem;">${removable}
          <input type="text" class="note" data-eid="${Orion.esc(p.experiment_id)}" placeholder="analyst note (optional)" style="width:60%;margin-left:0.6rem;"></div>
      </div>`;
    }

    let html = `<div class="assessment-box"><div class="term-title">ORION // PLAN REVIEW — ${Orion.esc(plan.plan_id)}</div>
      <div class="sub">STATUS: UNDER_REVIEW · remove experiments, tune parameters, add notes, then approve.</div>
      ${editable.map(row).join("") || '<div class="state">No editable experiments.</div>'}`;
    if (excluded.length) {
      html += `<button class="btn btn-ghost" id="pr-excluded">[ SHOW EXCLUDED (${excluded.length}) ]</button>
        <div id="pr-excluded-list" class="hidden" style="margin-top:0.4rem;">`
        + excluded.map(p => `<div class="exp na"><span class="ename">${Orion.esc(p.name)}</span> <span class="appl NOT_APPLICABLE">NOT_APPLICABLE</span><div class="sub">${Orion.esc(p.reason || "")}</div></div>`).join("")
        + `</div>`;
    }
    html += `<div class="btn-row" style="margin-top:0.6rem;">
        <button class="btn btn-red" id="pr-approve">[ APPROVE PLAN ]</button></div>
      <div id="pr-handoff" class="hidden" style="margin-top:0.6rem;"></div></div>`;
    container.innerHTML = html;

    const exBtn = document.getElementById("pr-excluded");
    if (exBtn) exBtn.addEventListener("click", () => document.getElementById("pr-excluded-list").classList.toggle("hidden"));

    document.getElementById("pr-approve").addEventListener("click", async function () {
      this.disabled = true;
      const exclude = [];
      container.querySelectorAll(".keep").forEach(cb => { if (!cb.checked) exclude.push(cb.dataset.eid); });
      const overrides = {};
      container.querySelectorAll("input[data-param]").forEach(inp => {
        if (inp.value !== "") { (overrides[inp.dataset.eid] = overrides[inp.dataset.eid] || {})[inp.dataset.param] = inp.value; }
      });
      const notes = {};
      container.querySelectorAll(".note").forEach(inp => { if (inp.value.trim()) notes[inp.dataset.eid] = inp.value.trim(); });
      const out = document.getElementById("pr-handoff");
      Orion.setState(out, "running", "approving plan & preparing handoff…");
      out.classList.remove("hidden");
      try {
        const res = await Orion.postJSON("/api/plans/" + encodeURIComponent(plan.plan_id) + "/approve",
          { exclude: exclude, overrides: overrides, notes: notes });
        out.innerHTML = Orion.handoffSummary(res.plan);
      } catch (e) { Orion.setState(out, "error", "[x] approval failed: " + e.message); this.disabled = false; }
    });
  };

  Orion.handoffSummary = function (plan) {
    const c = plan.counts || {};
    return `<div class="assessment-box"><div class="term-title">ORION // PLAN HANDOFF</div>
      <div class="statusline">
        <div>PLAN ID: <strong>${Orion.esc(plan.plan_id)}</strong> · STATUS: <span class="ok">APPROVED</span></div>
        <div><span class="ok">[+]</span> human approval recorded</div>
        <div><span class="ok">[+]</span> threat model attached</div>
        <div><span class="ok">[+]</span> evidence package attached</div>
        <div><span class="ok">[+]</span> executable experiments: ${c.approved || 0}</div>
        <div><span class="warn">[!]</span> conditional experiments: ${c.conditional || 0}</div>
        <div><span class="off">[-]</span> excluded experiments: ${c.excluded || 0}</div>
      </div>
      <div class="btn-row" style="margin-top:0.6rem;">
        <a class="btn btn-red" href="/experiment?plan_id=${encodeURIComponent(plan.plan_id)}">[ OPEN EXPERIMENT ]</a>
        <a class="btn btn-ghost" href="/api/plans/${encodeURIComponent(plan.plan_id)}" target="_blank">[ VIEW APPROVED PLAN ]</a>
      </div>
      <p class="sub" style="color:var(--warning);margin-top:0.4rem;">Approval prepared execution. Nothing has run — open the Experiment to Attack → Measure → Defend → Retest explicitly.</p>
      </div>`;
  };

  // ---- reusable image-attack comparison (Original / Adversarial / Perturbation) ----
  // Type-aware: returns "" when the run has no image artifacts (non-image attacks).
  Orion.imageComparison = function (rec, traceId, opts) {
    opts = opts || {};
    const art = (rec && rec.artifacts) || {};
    if (!art.original && !art.adversarial) return "";
    const base = "/api/artifacts/" + encodeURIComponent(traceId) + "/";
    const br = rec.baseline_result || {}, ar = rec.adversarial_result || {};
    const met = rec.metrics || {};
    const mv = (k) => (met[k] && typeof met[k].value !== "undefined") ? met[k].value : null;
    const linf = mv("perturbation_linf"), l2 = mv("perturbation_l2");
    const origPred = br.prediction || br.decision || "—";
    const advPred = ar.prediction || ar.decision || "—";
    const changed = String(origPred) !== String(advPred);
    const success = (met.attack_success && met.attack_success.value) || rec.status === "ATTACK_SUCCESS";

    function panel(cls, title, file, sub) {
      const img = file ? `<a href="${base + encodeURIComponent(file)}" target="_blank" title="click to inspect">
        <img src="${base + encodeURIComponent(file)}" alt="${Orion.esc(title)}"></a>` :
        '<div class="state">NO IMAGE ARTIFACT</div>';
      return `<div class="img-panel ${cls}"><h4>${Orion.esc(title)}</h4>${img}<div class="sub">${sub}</div></div>`;
    }

    const summary = `<pre class="term-pre">ATTACK: ${Orion.esc(rec.attack_technique || rec.scenario_name || "—")}
STATUS: ${Orion.esc(rec.status || "—")}
ORIGINAL:    ${Orion.esc(origPred)}${br.confidence != null ? " / " + br.confidence : ""}
ADVERSARIAL: ${Orion.esc(advPred)}${ar.confidence != null ? " / " + ar.confidence : ""}
RESULT: ${changed ? "CLASSIFICATION CHANGED" : (success ? "ATTACK SUCCESS" : "ATTACK FAILED / UNCHANGED")}${linf != null ? "\nPERTURBATION: L∞ = " + linf + (l2 != null ? " · L2 = " + l2 : "") : ""}</pre>`;

    const pertSub = (linf != null ? `L∞: ${linf}` : "") + (l2 != null ? ` · L2: ${l2}` : "") +
      `<br><span class="sub">absolute difference</span> <button class="btn-link img-amp" data-img="${base + encodeURIComponent(art.difference || "")}">[ amplify ]</button>`;

    return `<div class="term"><div class="term-title">IMAGE ATTACK RESULT ${success ? '<span class="badge badge-attack_success">ATTACK_SUCCESS</span>' : ''}</div>
      ${summary}
      <div class="img-triptych">
        ${panel("", "Original", art.original, "Prediction: " + Orion.esc(origPred) + (br.confidence != null ? "<br>Confidence: " + br.confidence : ""))}
        ${panel("adv", "Adversarial", art.adversarial, "Prediction: " + Orion.esc(advPred) + (ar.confidence != null ? "<br>Confidence: " + ar.confidence : ""))}
        ${panel("", "Perturbation / Difference map", art.difference, pertSub)}
      </div></div>`;
  };

  // Amplify a difference image in-place (canvas) — display-only, never metrics.
  Orion.wireImageAmplify = function (container) {
    (container || document).querySelectorAll(".img-amp").forEach(function (btn) {
      if (btn.dataset.wired) return; btn.dataset.wired = "1";
      btn.addEventListener("click", function () {
        const panel = btn.closest(".img-panel");
        const imgEl = panel && panel.querySelector("img");
        if (!imgEl) return;
        if (btn.dataset.on === "1") { imgEl.style.display = ""; const cv = panel.querySelector("canvas"); if (cv) cv.remove(); btn.textContent = "[ amplify ]"; btn.dataset.on = "0"; return; }
        const im = new Image(); im.crossOrigin = "anonymous";
        im.onload = function () {
          const cv = document.createElement("canvas"); cv.width = im.naturalWidth; cv.height = im.naturalHeight;
          cv.style.cssText = imgEl.style.cssText; cv.className = "amp-canvas";
          const ctx = cv.getContext("2d"); ctx.drawImage(im, 0, 0);
          try {
            const d = ctx.getImageData(0, 0, cv.width, cv.height); const f = 8;
            for (let i = 0; i < d.data.length; i += 4) { d.data[i] = Math.min(255, d.data[i]*f); d.data[i+1] = Math.min(255, d.data[i+1]*f); d.data[i+2] = Math.min(255, d.data[i+2]*f); }
            ctx.putImageData(d, 0, 0);
          } catch (e) { /* cross-origin taint — leave as-is */ }
          imgEl.style.display = "none"; imgEl.parentNode.appendChild(cv);
          btn.textContent = "[ amplified ×8 · display only ]"; btn.dataset.on = "1";
        };
        im.src = btn.dataset.img;
      });
    });
  };

  window.Orion = Orion;
})();
