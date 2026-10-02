/* Know Your Target: recon → agent interpretation → threat model → experiments.
   Recon collects · Orion interprets · humans decide. No offensive auto-execution. */
(function () {
  "use strict";
  const assessmentEl = document.getElementById("assessment");
  const targetStatus = document.getElementById("target-status");
  let current = null; // current assessment

  // ---- capability switching ----
  const caps = ["recon", "agent", "threat", "guided"];
  function showCap(cap) {
    caps.forEach(c => {
      const el = document.getElementById("cap-" + c);
      if (el) el.classList.toggle("hidden", c !== cap);
    });
    document.querySelectorAll("#capability-menu li").forEach(li =>
      li.classList.toggle("active", li.dataset.cap === cap));
    if (cap === "recon") loadReconRuns();
    if (cap === "threat") renderThreatOnly();
    const el = document.getElementById("cap-" + cap);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  document.querySelectorAll("[data-cap]").forEach(b =>
    b.addEventListener("click", () => showCap(b.dataset.cap)));

  // ---- recon runs list ----
  async function loadReconRuns() {
    const el = document.getElementById("recon-runs");
    Orion.setState(el, "running", "querying recon runs…");
    try {
      const data = await Orion.getJSON("/api/recon/runs");
      const runs = data.runs || [];
      if (!runs.length) {
        el.innerHTML = '<div class="state">[-] No recon runs in memory. <a href="/know-environment.html">Launch one</a>, then refresh.</div>';
        return;
      }
      el.innerHTML = runs.map(r => `
        <div class="recon-run-row">
          <div class="meta"><span class="rid">${Orion.esc(r.display_id)}</span>
            &middot; ${Orion.esc(r.target)} &middot; ${Orion.statusBadge(r.status)}</div>
          <button class="btn btn-red btn-analyze" data-run="${Orion.esc(r.run_id)}">[ ANALYZE WITH ORION ]</button>
        </div>`).join("");
      el.querySelectorAll(".btn-analyze").forEach(b =>
        b.addEventListener("click", () => analyzeRecon(b.dataset.run)));
    } catch (e) {
      Orion.setState(el, "error", "[x] recon registry unavailable: " + e.message);
    }
  }
  const refresh = document.getElementById("refresh-recon");
  if (refresh) refresh.addEventListener("click", loadReconRuns);

  // ---- direct URL probe (real GET requests to a running service) ----
  async function probeUrl() {
    const url = (document.getElementById("probe-url").value || "").trim();
    if (!url) { Orion.setState(assessmentEl, "error", "Enter a URL, e.g. http://127.0.0.1:5001"); assessmentEl.classList.remove("hidden"); return; }
    const activeEl = document.getElementById("probe-active");
    const active = activeEl ? activeEl.checked : false;
    Orion.setState(assessmentEl, "running", "probing " + url + (active ? " (GET + active POST)…" : " (GET-only)…"));
    assessmentEl.classList.remove("hidden");
    try {
      current = await Orion.postJSON("/api/target-analysis/probe", { url: url, active: active });
      renderAssessment(current);
      assessmentEl.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      Orion.setState(assessmentEl, "error", "[x] probe failed: " + e.message);
    }
  }
  const probeBtn = document.getElementById("btn-probe");
  if (probeBtn) probeBtn.addEventListener("click", probeUrl);

  // ---- analysis actions ----
  async function analyzeRecon(runId) {
    Orion.setState(assessmentEl, "running", "interpreting recon evidence…");
    assessmentEl.classList.remove("hidden");
    try {
      current = await Orion.postJSON("/api/agent/analyze-recon", { recon_run_id: runId });
      renderAssessment(current);
    } catch (e) {
      Orion.setState(assessmentEl, "error", "[x] agent analysis unavailable: " + e.message);
    }
  }

  async function analyzeContext() {
    const ctx = document.getElementById("ctx-input").value.trim();
    if (!ctx) { Orion.setState(assessmentEl, "error", "Enter target context first."); assessmentEl.classList.remove("hidden"); return; }
    Orion.setState(assessmentEl, "running", "interpreting target context…");
    assessmentEl.classList.remove("hidden");
    try {
      current = await Orion.postJSON("/api/agent/analyze-context", { context: ctx });
      renderAssessment(current);
    } catch (e) {
      Orion.setState(assessmentEl, "error", "[x] agent analysis unavailable: " + e.message);
    }
  }
  const ctxBtn = document.getElementById("btn-analyze-ctx");
  if (ctxBtn) ctxBtn.addEventListener("click", analyzeContext);

  // ---- render assessment ----
  function clsTag(c) { return `<span class="cls cls-${c}">${c.replace("_"," ")}</span>`; }
  function evRefs(ids) {
    if (!ids || !ids.length) return '<span class="evref">evidence: 0</span>';
    return `<span class="evref">evidence: ${ids.length} · ${ids.map(Orion.esc).join(", ")}</span>`;
  }
  function renderItem(it, whyExtra) {
    const c = it.classification || "HYPOTHESIS";
    const why = (it.rationale || "") + (whyExtra || "");
    return `<div class="analysis-item cls-${c}">
      ${clsTag(c)}${Orion.esc(it.statement)}
      <button class="why-btn">[ WHY? ]</button>
      <div class="meta">confidence: ${Orion.esc(it.confidence||"NONE")} · ${evRefs(it.evidence)}</div>
      <div class="why-box">${Orion.esc(why)}</div>
    </div>`;
  }

  function renderAssessment(a) {
    if (targetStatus) targetStatus.outerHTML = '<span class="ok" id="target-status">[+] TARGET STATUS: ANALYZED</span>';
    const tm = a.threat_model || {};
    const adv = tm.adversary || {};
    const prov = tm.provenance || {};
    const acc = tm.access_detail || {};
    const ais = a.ai_surface || {};
    const types = a.target_types || [];
    const key = a.recon_run_id || ("ctx:" + a.target);
    const reconHref = "/know-environment.html";
    const primary = types[0] ? types[0].type : "unknown";
    const secondary = types.slice(1).map(t => t.type);
    const scope = ais.status === "CONFIRMED" ? "ESTABLISHED" :
                  (ais.status === "POSSIBLE" ? "POSSIBLE — REQUIRES CORROBORATION" : "NOT ESTABLISHED");
    const focus = ais.status === "CONFIRMED" ? "AI/ML security analysis" :
                  (ais.status === "POSSIBLE" ? "Confirm AI surface, then AI analysis" : "Web/application analysis");

    // --- Target classification / scope summary ---
    let html = `<div class="assessment-box">
      <div class="term-title">TARGET CLASSIFICATION</div>
      <div class="kv">
        <div class="k">Target</div><div class="v">${Orion.esc(a.target)}</div>
        <div class="k">Primary type</div><div class="v">${Orion.esc(primary)}</div>
        <div class="k">Secondary types</div><div class="v">${Orion.esc(secondary.join(", ") || "—")}</div>
        <div class="k">AI surface</div><div class="v">${Orion.esc(ais.status || "N/A")}</div>
        <div class="k">AI security scope</div><div class="v">${Orion.esc(scope)}</div>
        <div class="k">Recommended focus</div><div class="v">${Orion.esc(focus)}</div>
      </div>
      <div class="ttypes">${types.map(t => `<span class="ttype">${Orion.esc(t.type)} <span class="c">(${Orion.esc(t.confidence)})</span></span>`).join("")}</div>
      <div class="ai-surface ${Orion.esc(ais.status)}">
        <div class="st">AI SURFACE: ${Orion.esc(ais.status)} · confidence ${Orion.esc(ais.confidence)} · score ${ais.score != null ? ais.score : "?"}</div>
        <div class="sub">${Orion.esc(ais.rationale || "")}</div>
        ${(ais.signals && ais.signals.length) ? `<div class="signals">${ais.signals.map(s =>
          `<span class="sig sig-${Orion.esc(s.strength)}">${Orion.esc(s.strength)}: ${Orion.esc(s.value)}${s.evidence_id ? " <span class=\"evref\">(" + Orion.esc(s.evidence_id) + ")</span>" : ""}</span>`).join("")}</div>`
          : `<div class="sub">No AI signals matched.</div>`}
      </div>
      <div class="next-action">NEXT ACTION: ${Orion.esc(a.next_action || "Human review required")}</div></div>`;

    // --- Abstention UX (useful result, not an empty screen) ---
    if (a.abstention) {
      html += `<div class="term"><div class="term-title">INSUFFICIENT AI EVIDENCE</div>
        <div class="statusline"><span class="off">[-] ${Orion.esc(a.abstention)}</span></div>
        <div class="btn-row" style="margin-top:0.6rem;">
          <a class="btn btn-red" href="${reconHref}">[ RUN DEEPER RECON ]</a>
          <button class="btn" data-cap="agent">[ PROVIDE TARGET CONTEXT ]</button>
          <a class="btn btn-ghost" href="${reconHref}">[ SELECT AN AI APPLICATION ]</a>
        </div></div>`;
    }

    // --- Classified sections (never mixed) ---
    function section(title, items) {
      if (!items || !items.length) return `<div class="term"><div class="term-title">${title}</div><div class="state">None.</div></div>`;
      return `<div class="term"><div class="term-title">${title}</div>${items.map(i => renderItem(i)).join("")}</div>`;
    }
    html += section("OBSERVATIONS", a.observations);
    html += section("INFERENCES", a.inferences);
    html += section("THREAT HYPOTHESES", a.threat_hypotheses);
    html += section("VALIDATED FINDINGS", a.validated_findings);

    // --- MITRE ATLAS (gated) ---
    const atlas = a.mitre_atlas || { applicable: false, mappings: [], message: "" };
    html += `<div class="term"><div class="term-title">MITRE ATLAS</div>`;
    html += (atlas.applicable && atlas.mappings.length)
      ? atlas.mappings.map(m => Orion.atlasItem(m)).join("")
      : `<div class="state">${Orion.esc(atlas.message || "Not applicable with current evidence.")}</div>`;
    html += `</div>`;

    // --- Suggested experiments (APPLICABLE first; NOT_APPLICABLE hidden) ---
    const exps = a.suggested_experiments || [];
    const primaryExps = exps.filter(e => e.applicability !== "NOT_APPLICABLE");
    const naExps = exps.filter(e => e.applicability === "NOT_APPLICABLE");
    function expHtml(e) {
      return `<div class="exp ${e.applicability === "APPLICABLE" ? "" : "na"}">
        <span class="eid">${Orion.esc(e.id)}</span><span class="ename">${Orion.esc(e.name)}</span>
        <span class="risk ${Orion.esc(e.risk)}">${Orion.esc(e.risk)}</span>
        <span class="appl ${Orion.esc(e.applicability)}">${Orion.esc(e.applicability.replace("_"," "))}</span>
        <button class="why-btn">[ WHY? ]</button>
        <div class="why-box">${Orion.esc(e.rationale || "")}${e.missing && e.missing.length ? " · missing prerequisites: " + Orion.esc(e.missing.join(", ")) : ""}${e.atlas && e.atlas.length ? " · ATLAS: " + e.atlas.map(m=>Orion.esc(m.technique_id)).join(", ") : ""}</div>
        <div class="exp-launch hidden" style="margin-top:0.4rem;">
          ${(e.scenario && e.applicability === "APPLICABLE") ? `<a class="btn btn-red" href="/adversarial">[ LAUNCH EXPERIMENT ]</a>` : `<span class="sub">${e.applicability === "APPLICABLE" ? "Manual experiment — review required." : "Not applicable to this target with current evidence."}</span>`}
        </div></div>`;
    }
    html += `<div class="term"><div class="term-title">SUGGESTED EXPERIMENTS <span class="proposed">PROPOSED</span></div>
      ${primaryExps.length ? primaryExps.map(expHtml).join("") : '<div class="state">No applicable experiments for this target with current evidence.</div>'}
      ${naExps.length ? `<div style="margin-top:0.6rem;">
        <button class="btn btn-ghost" id="btn-show-na">[ SHOW NON-APPLICABLE EXPERIMENTS (${naExps.length}) ]</button>
        <div id="na-exps" class="hidden" style="margin-top:0.5rem;">${naExps.map(expHtml).join("")}</div></div>` : ""}
      <div class="btn-row" style="margin-top:0.6rem;">
        <button class="btn" id="btn-review">[ REVIEW PLAN ]</button>
        <button class="btn btn-red hidden" id="btn-approve-plan">[ REVIEW &amp; EDIT PLAN ]</button>
      </div>
      <div id="plan-note" class="sub" style="margin-top:0.4rem;"></div>
      <div id="plan-handoff" class="hidden" style="margin-top:0.6rem;"></div></div>`;

    // --- Threat model (honest: OBSERVED / UNKNOWN / UNDEFINED) ---
    const tmStatus = tm.status || "PROPOSED";
    const tagClass = tmStatus === "APPROVED" ? "approved-tag" : (tmStatus === "REJECTED" ? "rejected-tag" : "");
    function field(label, value, p) {
      const badge = p === "OBSERVED" ? '<span class="prov obs">[OBSERVED]</span>'
        : (p === "UNKNOWN" ? '<span class="prov unk">[UNKNOWN]</span>'
        : (p === "UNDEFINED" ? '<span class="prov unk">[UNDEFINED]</span>' : ""));
      return `<div class="k">${label}</div><div class="v">${Orion.esc(value || "N/A")} ${badge}</div>`;
    }
    html += `<div class="tm-box">
      <div class="term-title">THREAT MODEL <span class="proposed ${tagClass}" id="tm-status">${Orion.esc(tmStatus)}</span></div>
      <div class="kv">
        ${field("Goal", adv.goal, prov.goal)}
        ${field("Knowledge", adv.knowledge, prov.knowledge)}
        ${field("Network access", acc.network_access || adv.access, prov.network_access)}
        ${field("Model knowledge", acc.model_knowledge, prov.model_knowledge)}
        ${field("Credential access", acc.credential_access, prov.credential_access)}
        ${field("Budget", adv.budget, prov.budget)}
        <div class="k">Assets</div><div class="v">${Orion.esc((tm.assets||[]).join(", ")||"N/A")}</div>
        <div class="k">Surfaces</div><div class="v">${Orion.esc((tm.surfaces||[]).join(", ")||"N/A")}</div>
      </div>
      ${tm.note ? `<div class="sub" style="color:var(--warning);margin-top:0.4rem;">${Orion.esc(tm.note)}</div>` : ""}
      <div class="sub" style="margin-top:0.3rem;">${Orion.esc(tm.rationale||"")}</div>
      <div class="tm-approve">
        <button class="btn" id="tm-accept">[ ACCEPT ]</button>
        <button class="btn btn-ghost" id="tm-edit">[ EDIT ]</button>
        <button class="btn btn-red" id="tm-reject">[ REJECT ]</button>
      </div></div>`;

    assessmentEl.innerHTML = html;

    // delegation + wiring
    assessmentEl.querySelectorAll(".why-btn").forEach(b =>
      b.addEventListener("click", () => {
        const box = b.parentElement.querySelector(".why-box");
        if (box) box.classList.toggle("open");
      }));
    assessmentEl.querySelectorAll("[data-cap]").forEach(b =>
      b.addEventListener("click", () => showCap(b.dataset.cap)));
    const naBtn = document.getElementById("btn-show-na");
    if (naBtn) naBtn.addEventListener("click", () => {
      document.getElementById("na-exps").classList.toggle("hidden");
    });
    document.getElementById("tm-accept").addEventListener("click", () => approveTM(key, true));
    document.getElementById("tm-reject").addEventListener("click", () => approveTM(key, false));
    document.getElementById("tm-edit").addEventListener("click", () => {
      document.getElementById("plan-note").textContent = "Edit mode: adjust the scenario YAML before running (threat model editing is manual in this build).";
    });
    document.getElementById("btn-review").addEventListener("click", () => {
      document.getElementById("btn-approve-plan").classList.remove("hidden");
      document.getElementById("plan-note").textContent = "[!] Review complete. Approving the plan prepares the handoff — it never auto-executes offensive tests.";
    });
    document.getElementById("btn-approve-plan").addEventListener("click", approvePlan);

    renderThreatOnly();
  }

  function approvePlan() {
    if (!current) return;
    Orion.planReview(document.getElementById("plan-handoff"), "know_your_target", current);
  }

  async function approveTM(runId, approved) {
    const note = document.getElementById("plan-note");
    try {
      const res = await Orion.postJSON("/api/threat-model/" + encodeURIComponent(runId) + "/approve", { approved });
      const tag = document.getElementById("tm-status");
      tag.textContent = res.threat_model.status;
      tag.className = "proposed " + (approved ? "approved-tag" : "rejected-tag");
      if (note) note.innerHTML = approved
        ? '<span style="color:var(--green)">[+] Threat model approved by human.</span>'
        : '<span style="color:var(--red-bright)">[x] Threat model rejected.</span>';
      current.threat_model.status = res.threat_model.status;
      renderThreatOnly();
    } catch (e) {
      if (note) note.textContent = "[x] approval failed: " + e.message;
    }
  }

  function renderThreatOnly() {
    const el = document.getElementById("threat-only");
    if (!el) return;
    if (!current) { el.innerHTML = '<div class="state">[-] NO THREAT MODEL — analyze a target first.</div>'; return; }
    const tm = current.threat_model || {};
    const adv = tm.adversary || {};
    el.innerHTML = `<pre class="term-pre">THREAT MODEL  [${Orion.esc(tm.status||"PROPOSED")}]
---------------------------------
ASSETS
${(tm.assets||[]).map(a=>"[+] "+a).join("\n") || "  (none)"}

ADVERSARY
  goal:      ${Orion.esc(adv.goal||"N/A")}
  knowledge: ${Orion.esc(adv.knowledge||"N/A")}
  access:    ${Orion.esc(adv.access||"N/A")}
  budget:    ${Orion.esc(adv.budget||"N/A")}

SURFACES
${(tm.surfaces||[]).map(s=>"[+] "+s).join("\n") || "  (none)"}</pre>`;
  }

  // ---- guided mode ----
  function setPhase(p, state) {
    const li = document.querySelector(`#guided-phases li[data-p="${p}"]`);
    if (!li) return;
    li.classList.remove("done", "active");
    if (state) li.classList.add(state);
    li.querySelector(".box").textContent = state === "done" ? "[✓]" : (state === "active" ? "[>]" : "[ ]");
  }
  function gstatus(msg, cls) {
    document.getElementById("guided-status").innerHTML = `<span class="${cls||'work'}">${Orion.esc(msg)}</span>`;
  }

  async function startGuided() {
    const objective = document.getElementById("g-objective").value.trim() || "Map the attack surface";
    const target = document.getElementById("g-target").value.trim() || "demo.thm.local";
    ["recon","evidence","analysis","threat","plan","review"].forEach(p => setPhase(p, null));
    document.getElementById("btn-guided").disabled = true;
    setPhase("recon", "active");
    gstatus("[*] launching mock reconnaissance…");

    // 1. start a mock recon run
    let runId;
    try {
      const fd = new FormData();
      fd.append("objective", objective); fd.append("target", target);
      fd.append("mock", "1"); fd.append("max_iterations", "4");
      const r = await fetch("/know-environment/run", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "recon start failed");
      runId = d.run_id;
    } catch (e) {
      setPhase("recon", null); gstatus("[x] recon failed: " + e.message, "err");
      document.getElementById("btn-guided").disabled = false; return;
    }

    // 2. consume recon SSE until done
    gstatus("[*] reconnaissance running…");
    await new Promise((resolve) => {
      const es = new EventSource("/know-environment/events/" + encodeURIComponent(runId));
      es.onmessage = (ev) => {
        let data; try { data = JSON.parse(ev.data); } catch (_) { return; }
        if (data.type === "evaluate") setPhase("evidence", "active");
        if (data.type === "report" || data.type === "done") { es.close(); resolve(); }
        if (data.type === "error") { es.close(); resolve(); }
      };
      es.onerror = () => { es.close(); resolve(); };
    });
    setPhase("recon", "done"); setPhase("evidence", "done");

    // 3. agent analysis
    setPhase("analysis", "active"); gstatus("[*] interpreting evidence…");
    try {
      current = await Orion.postJSON("/api/agent/analyze-recon", { recon_run_id: runId });
    } catch (e) {
      setPhase("analysis", null); gstatus("[x] analysis failed: " + e.message, "err");
      document.getElementById("btn-guided").disabled = false; return;
    }
    setPhase("analysis", "done");
    setPhase("threat", "done");
    setPhase("plan", "done");
    setPhase("review", "active");
    gstatus("[!] REVIEW PLAN — human review required.", "warn");

    assessmentEl.classList.remove("hidden");
    renderAssessment(current);
    assessmentEl.scrollIntoView({ behavior: "smooth", block: "start" });
    document.getElementById("btn-guided").disabled = false;
  }
  const gbtn = document.getElementById("btn-guided");
  if (gbtn) gbtn.addEventListener("click", startGuided);

  // ---- deep links ----
  const p = new URLSearchParams(window.location.search);
  const cap = p.get("cap");
  const reconId = p.get("recon");
  if (reconId) { showCap("recon"); analyzeRecon(reconId); }
  else if (cap && caps.includes(cap)) showCap(cap);
  else showCap("recon");
})();
