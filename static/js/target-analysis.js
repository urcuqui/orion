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
  function renderAssessment(a) {
    if (targetStatus) targetStatus.outerHTML = '<span class="ok" id="target-status">[+] TARGET STATUS: ANALYZED</span>';
    const s = a.summary || {};
    const tm = a.threat_model || {};
    const adv = tm.adversary || {};
    const hyps = a.threat_hypotheses || [];
    const exps = a.suggested_experiments || [];
    const key = a.recon_run_id || ("ctx:" + a.target);

    let html = `<div class="assessment-box">
      <div class="term-title">ORION TARGET ASSESSMENT</div>
      <div class="kv">
        <div class="k">Target</div><div class="v">${Orion.esc(a.target)}</div>
        <div class="k">Source</div><div class="v">${Orion.esc(a.source)}${a.recon_run ? " · " + Orion.esc(a.recon_run) : ""}</div>
      </div>
      <hr class="term-rule">
      <div class="term-title" style="color:var(--muted)">Observed</div>
      <ul class="observed-list">
        <li>${s.counts ? s.counts.endpoints : 0} endpoint(s)</li>
        <li>${s.counts ? s.counts.auth_flows : 0} authentication indicator(s)</li>
        <li>${s.counts ? s.counts.findings : 0} recon finding(s)</li>
        <li>${s.counts ? s.counts.screenshots : 0} screenshot(s)</li>
      </ul>
      <div class="term-title" style="color:var(--muted)">Threat hypotheses</div>
      ${hyps.length ? hyps.map(h => `<div class="hyp sev-${(h.severity||'info').toLowerCase()}">
          <span class="sev">[${Orion.esc(h.severity)}]</span>${Orion.esc(h.name)}
          <div class="sub">${Orion.esc(h.rationale||"")}</div></div>`).join("")
        : '<div class="state">No threat hypotheses derived.</div>'}
      <div class="sub" style="margin-top:0.5rem;">Candidate MITRE ATLAS mappings: ${a.atlas_count || 0}</div>
      <div class="next-action">NEXT ACTION: ${Orion.esc(a.next_action || "Human review required")}</div>
    </div>`;

    // Threat model (PROPOSED → approve)
    const tmStatus = tm.status || "PROPOSED";
    const tagClass = tmStatus === "APPROVED" ? "approved-tag" : (tmStatus === "REJECTED" ? "rejected-tag" : "");
    html += `<div class="tm-box">
      <div class="term-title">THREAT MODEL <span class="proposed ${tagClass}" id="tm-status">${Orion.esc(tmStatus)}</span></div>
      <div class="kv">
        <div class="k">Goal</div><div class="v">${Orion.esc(adv.goal||"N/A")}</div>
        <div class="k">Knowledge</div><div class="v">${Orion.esc(adv.knowledge||"N/A")}</div>
        <div class="k">Access</div><div class="v">${Orion.esc(adv.access||"N/A")}</div>
        <div class="k">Budget</div><div class="v">${Orion.esc(adv.budget||"N/A")}</div>
        <div class="k">Assets</div><div class="v">${Orion.esc((tm.assets||[]).join(", ")||"N/A")}</div>
        <div class="k">Surfaces</div><div class="v">${Orion.esc((tm.surfaces||[]).join(", ")||"N/A")}</div>
      </div>
      <div class="sub" style="margin-top:0.4rem;">${Orion.esc(tm.rationale||"")}</div>
      <div class="tm-approve">
        <button class="btn" id="tm-accept">[ ACCEPT ]</button>
        <button class="btn btn-ghost" id="tm-edit">[ EDIT ]</button>
        <button class="btn btn-red" id="tm-reject">[ REJECT ]</button>
      </div>
    </div>`;

    // Experiment plan (PROPOSED → review → approve, no auto-exec)
    html += `<div class="term">
      <div class="term-title">PROPOSED EXPERIMENT PLAN <span class="proposed">PROPOSED</span></div>
      ${exps.map(e => `<div class="exp">
        <span class="eid">${Orion.esc(e.id)}</span><span class="ename">${Orion.esc(e.name)}</span>
        <span class="risk ${Orion.esc(e.risk)}">${Orion.esc(e.risk)}</span>
        <div class="sub">${Orion.esc(e.rationale||"")} ${e.atlas && e.atlas.length ? "· ATLAS: " + e.atlas.map(m=>Orion.esc(m.technique_id)).join(", ") : ""}</div>
        <div class="exp-launch hidden" style="margin-top:0.4rem;">
          ${e.scenario ? `<a class="btn btn-red" href="/adversarial">[ LAUNCH EXPERIMENT ]</a>` : `<span class="sub">Manual experiment — review required.</span>`}
        </div></div>`).join("")}
      <div class="btn-row" style="margin-top:0.6rem;">
        <button class="btn" id="btn-review">[ REVIEW PLAN ]</button>
        <button class="btn btn-red hidden" id="btn-approve-exp">[ APPROVE EXPERIMENT ]</button>
      </div>
      <div id="plan-note" class="sub" style="margin-top:0.4rem;"></div>
    </div>`;

    assessmentEl.innerHTML = html;

    // wire threat-model approval
    document.getElementById("tm-accept").addEventListener("click", () => approveTM(key, true));
    document.getElementById("tm-reject").addEventListener("click", () => approveTM(key, false));
    document.getElementById("tm-edit").addEventListener("click", () => {
      document.getElementById("plan-note").textContent = "Edit mode: adjust the scenario YAML before running (threat model editing is manual in this build).";
    });
    // wire experiment plan review
    document.getElementById("btn-review").addEventListener("click", () => {
      document.getElementById("btn-approve-exp").classList.remove("hidden");
      document.getElementById("plan-note").textContent = "[!] Review complete. Approving reveals launch actions. Orion never auto-executes offensive tests.";
    });
    document.getElementById("btn-approve-exp").addEventListener("click", () => {
      assessmentEl.querySelectorAll(".exp-launch").forEach(x => x.classList.remove("hidden"));
      document.getElementById("plan-note").innerHTML = '<span style="color:var(--green)">[+] Experiment plan approved by human. Launch individual tests explicitly.</span>';
    });

    renderThreatOnly();
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
