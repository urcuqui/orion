/* ORION shared frontend utilities. Vanilla JS, no framework.
   All rendering reflects backend data only — never fabricate metrics. */
(function () {
  "use strict";

  // ---- Demo mode (?demo=1) ----
  const params = new URLSearchParams(window.location.search);
  const demoMode = params.get("demo") === "1";
  if (demoMode) document.body.classList.add("demo");

  const Orion = {};

  // Appearance affects presentation only. Analyst is the default.
  (function initAppearance() {
    let appearance = "analyst";
    try { appearance = localStorage.getItem("orion.appearance") === "classic" ? "classic" : "analyst"; } catch (_) {}
    if (demoMode) appearance = "classic";
    function apply() {
      document.body.classList.toggle("classic", appearance === "classic");
      document.body.classList.toggle("crt", appearance === "classic");
    }
    apply();
    const select = document.getElementById("orion-appearance");
    if (select) {
      select.value = appearance;
      select.addEventListener("change", () => {
        appearance = select.value; apply();
        try { localStorage.setItem("orion.appearance", appearance); } catch (_) {}
      });
    }
  })();

  // Native dialog: explicit external-execution confirmation, cancel/Escape,
  // focus containment and restoration. Callers capture the exact payload first.
  Orion.confirmExecution = function (details) {
    return new Promise(resolve => {
      const previous = document.activeElement;
      const dialog = document.createElement("dialog");
      dialog.className = "security-dialog";
      dialog.setAttribute("aria-labelledby", "execution-confirm-title");
      dialog.setAttribute("aria-describedby", "execution-confirm-description");
      dialog.innerHTML = `<h2 id="execution-confirm-title">${Orion.esc(details.title || "RUN LIVE ATTACK")}</h2>
        <dl>${Object.entries(details.fields || {}).map(([k,v]) => `<dt>${Orion.esc(k)}</dt><dd>${Orion.esc(v == null || v === "" ? "UNKNOWN" : v)}</dd>`).join("")}</dl>
        <p id="execution-confirm-description">This operation sends real requests to the target. Continue only if you are authorized to test it. Isolation is UNKNOWN unless explicitly recorded by the backend.</p>
        <form method="dialog" class="btn-row"><button class="btn btn-ghost" value="cancel" autofocus>Cancel</button><button class="btn btn-red" value="run">Confirm and run</button></form>`;
      document.body.appendChild(dialog);
      dialog.addEventListener("close", () => {
        const approved = dialog.returnValue === "run";
        dialog.remove(); if (previous && previous.isConnected) previous.focus(); resolve(approved);
      }, { once: true });
      dialog.showModal();
    });
  };

  Orion.nextActionCard = function (label, description) {
    return `<section class="next-action-card" aria-label="Next action"><div class="eyebrow">NEXT ACTION</div><h2>${Orion.esc(label || "UNKNOWN")}</h2><p>${Orion.esc(description)}</p><a class="btn btn-ghost" href="#stage-detail">Review current stage</a></section>`;
  };

  Orion.executionBoundary = function (fields) {
    return `<div class="term-title space-above-lg">EXECUTION BOUNDARY</div><dl class="kv">${Object.entries(fields).map(([k,v]) => `<dt class="k">${Orion.esc(k)}</dt><dd class="v">${Orion.esc(v)}</dd>`).join("")}</dl>`;
  };

  Orion.provenanceChain = function (ws) {
    const nodes = [["System",ws.self_profile_id], ["Target",ws.target_profile_id], ["Environment",ws.environment_profile_id],
      ["Analysis context",ws.analysis_context_id], ["Threat model",ws.threat_model_id], ["Plan",ws.plan_id],
      ["Experiment",ws.experiment_workspace_id,"/experiment/"], ["Attack run",ws.attack_run_id,"/runs/"],
      ["Finding",ws.finding_id,"/findings/"], ["Control",ws.defense_id], ["Retest",ws.retest_run_id,"/runs/"]];
    return `<div class="prov-chain">${nodes.map(([label,id,route]) => `<div class="prov-node ${id ? "on" : "off"}"><span class="prov-label">${label}</span>${id && route ? `<a class="prov-val" href="${route + encodeURIComponent(id)}">${Orion.esc(id)}</a>` : `<span class="prov-val">${Orion.esc(id || "UNKNOWN / NOT RECORDED")}</span>`}</div>`).join('<div class="prov-arrow" aria-hidden="true">↓</div>')}</div>`;
  };

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

  Orion.shortId = function (id) { if (!id) return "?"; const value = String(id); return value.length > 16 ? value.slice(0, 8) + "…" + value.slice(-6) : value; };

  Orion.fmtTime = function (iso) {
    if (!iso) return "N/A";
    try { return new Date(iso).toLocaleString(); } catch (e) { return iso; }
  };

  // ---- reusable renderers (mirror the Jinja components) ----
  Orion.statusBadge = function (status) {
    const s = String(status || "UNKNOWN").toLowerCase().replace(/[\s-]+/g, "_");
    const symbols = {"confirmed": "✓", "observed": "●", "hypothesis": "◇", "inferred": "≈", "unknown": "?", "not_applicable": "—", "running": "●", "executing": "●", "queued": "○", "pending": "○", "ready": "○", "proposed": "◇", "blocked": "×", "failed": "×", "error": "×", "mitigated": "✓", "partial": "!", "partially_mitigated": "!", "not_mitigated": "×", "approved": "✓", "applied": "!", "complete": "✓", "completed": "✓", "effective": "✓", "ineffective": "×", "partially_effective": "!", "not_retested": "○", "eligible": "✓", "available": "●", "attack_success": "!", "attack_blocked": "✓", "no_attack": "—"};
    return `<span class="badge badge-${Orion.esc(s.replace(/[^a-z0-9_]/g, ""))}"><span aria-hidden="true">${symbols[s] || "·"}</span> ${Orion.esc(status || "UNKNOWN")}</span>`;
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
    const v = value === null || value === undefined ? "UNKNOWN" : value;
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
    el.setAttribute("role", kind === "error" ? "alert" : "status");
    el.setAttribute("aria-live", kind === "error" ? "assertive" : "polite");
    const cls = kind === "error" ? "state error" : "state";
    if (kind === "running") {
      el.innerHTML = `<div class="${cls}"><span class="loading-line">● ${Orion.esc(msg || "")}</span></div>`;
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
      <div class="sub">STATUS: UNDER_REVIEW · remove experiments, tune parameters, add notes, then approve. Approval makes eligible experiments executable. Approval does not run them.</div>
      <p class="sub">${editable.filter(p => p.status === "READY").length} READY · ${editable.filter(p => p.status === "NEEDS_INPUT").length} NEEDS INPUT · ${excluded.length} EXCLUDED</p>
      ${editable.map(row).join("") || '<div class="state">No editable experiments.</div>'}`;
    if (excluded.length) {
      html += `<button class="btn btn-ghost" id="pr-excluded">Show Excluded (${excluded.length})</button>
        <div id="pr-excluded-list" class="hidden space-above-sm">`
        + excluded.map(p => `<div class="exp na"><span class="ename">${Orion.esc(p.name)}</span> <span class="appl NOT_APPLICABLE">NOT_APPLICABLE</span><div class="sub">${Orion.esc(p.reason || "")}</div></div>`).join("")
        + `</div>`;
    }
    html += `<div class="btn-row space-above">
        <button class="btn btn-primary" id="pr-approve">Approve Plan</button></div>
      <div id="pr-handoff" class="hidden space-above"></div></div>`;
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
      <div class="btn-row space-above">
        <a class="btn btn-ghost" href="/experiment?plan_id=${encodeURIComponent(plan.plan_id)}">Open Experiment</a>
        <a class="btn btn-ghost" href="/api/plans/${encodeURIComponent(plan.plan_id)}" target="_blank">View Approved Plan</a>
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
