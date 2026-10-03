/* Experiment workspace console: one stateful lifecycle over the existing
   plan / run / replay / compare services. Deterministic transitions. */
(function () {
  "use strict";
  const root = document.getElementById("exp-root");
  const q = new URLSearchParams(location.search);
  const wsId = root.dataset.wsId || "";
  const compatStage = root.dataset.compatStage || "";
  const traceId = root.dataset.traceId || q.get("trace_id") || "";
  const planId = q.get("plan_id") || "";

  const STAGES = ["plan", "attack", "measure", "defend", "retest"];
  const BAR = { COMPLETE: "[████████]", APPROVED: "[████████]", APPLIED: "[████████]",
                RUNNING: "[>>>>>>..]", READY: "[>>>>....]", PENDING: "[........]",
                PROPOSED: "[>>......]", FAILED: "[xxxx....]" };

  function progress(stages) {
    return `<pre class="term-pre">` + STAGES.map(s => {
      const st = (stages[s] || "PENDING");
      return `${s.toUpperCase().padEnd(8)} ${BAR[st] || "[........]"} ${st}`;
    }).join("\n") + `</pre>`;
  }

  function ctxLine(ws) {
    const ids = [["plan", ws.plan_id], ["context", ws.analysis_context_id], ["self", ws.self_profile_id],
                 ["target", ws.target_profile_id], ["env", ws.environment_profile_id],
                 ["tm", ws.threat_model_id], ["attack_run", ws.attack_run_id],
                 ["defense", ws.defense_id], ["retest_run", ws.retest_run_id]];
    return `<div class="sub">` + ids.filter(([, v]) => v).map(([k, v]) => `${k}: ${Orion.esc(v)}`).join(" · ") + `</div>`;
  }

  // ---------- workspace mode ----------
  async function loadWorkspace(id) {
    let data;
    try { data = await Orion.getJSON("/api/experiment/" + encodeURIComponent(id)); }
    catch (e) { Orion.setState(root, "error", "[x] " + e.message); return; }
    renderWorkspace(data);
  }

  function renderWorkspace(data) {
    const ws = data;                 // to_dict of lifecycle (+ plan)
    const plan = data.plan || {};
    const na = ws.next_action || {};
    let stage = na.stage && na.stage !== "evidence" ? na.stage : ws.current_stage;
    // If the plan is approved but nothing is runnable, surface the ATTACK stage
    // so the user sees *why* there is no attack (and how to fix it).
    if (!ws.active_experiment_id && (ws.stages || {}).plan === "APPROVED") stage = "attack";

    let html = `<div class="assessment-box"><div class="term-title">ORION // EXPERIMENT ${Orion.esc(ws.experiment_workspace_id)}</div>
      ${progress(ws.stages || {})}
      ${ctxLine(ws)}
      <div class="next-action">NEXT: [ ${Orion.esc((na.label || "—"))} ]</div></div>`;

    // Experiment queue (plan proposals)
    const props = (plan.proposals || []);
    if (props.length) {
      html += `<div class="term"><div class="term-title">EXPERIMENT QUEUE</div>`;
      props.forEach(p => {
        const active = p.experiment_id === ws.active_experiment_id;
        const runnable = p.status === "READY" && p.scenario;
        html += `<div class="exp ${active ? "" : "na"}">
          <span class="eid">${Orion.esc(p.experiment_id)}</span><span class="ename">${Orion.esc(p.name)}</span>
          <span class="appl ${p.status === "READY" ? "APPLICABLE" : "INSUFFICIENT_EVIDENCE"}">${Orion.esc(p.status)}</span>
          ${active ? '<span class="risk LOW">ACTIVE</span>' : (runnable ? `<button class="btn btn-ghost btn-select" data-eid="${Orion.esc(p.experiment_id)}">[ OPEN ]</button>` : "")}
        </div>`;
      });
      html += `</div>`;
    }

    // Active stage detail
    html += `<div class="term" id="stage-detail"><div class="term-title">EXPERIMENT / ${stage.toUpperCase()}</div><div class="state">…</div></div>`;
    root.innerHTML = html;

    root.querySelectorAll(".btn-select").forEach(b => b.addEventListener("click", async () => {
      await Orion.postJSON(`/api/experiment/${ws.experiment_workspace_id}/select/${b.dataset.eid}`, {});
      loadWorkspace(ws.experiment_workspace_id);
    }));

    renderStage(ws, plan, stage);
  }

  async function renderStage(ws, plan, stage) {
    const el = document.getElementById("stage-detail");
    const wsid = ws.experiment_workspace_id;
    const prop = (plan.proposals || []).find(p => p.experiment_id === ws.active_experiment_id) || {};

    function wrap(title, inner) { el.innerHTML = `<div class="term-title">EXPERIMENT / ${title}</div>${inner}`; }

    if (stage === "plan") {
      const c = plan.counts || {};
      wrap("PLAN", `<div class="kv">
        <div class="k">Plan</div><div class="v">${Orion.esc(plan.plan_id || "")}</div>
        <div class="k">Approved</div><div class="v">${plan.approved_by_human ? '<span class="ok">YES</span>' : "NO"}</div>
        <div class="k">Ready</div><div class="v">${c.approved || 0}</div>
        <div class="k">Needs input</div><div class="v">${c.conditional || 0}</div>
        <div class="k">Excluded</div><div class="v">${c.excluded || 0}</div></div>
        <div class="btn-row" style="margin-top:0.5rem;"><button class="btn btn-red" id="go-attack">[ OPEN ATTACK ]</button></div>`);
      const g = document.getElementById("go-attack");
      if (g) g.addEventListener("click", () => { ws.current_stage = "attack"; renderStage(ws, plan, "attack"); });
      return;
    }

    if (stage === "attack" && !ws.active_experiment_id) {
      // No runnable experiment — explain why and point back to analysis.
      const na = (plan.proposals || []).filter(p => p.applicability === "NOT_APPLICABLE");
      wrap("ATTACK", `<div class="state error">[x] No runnable attack in this plan.</div>
        <p class="sub">All AI-specific experiments were <strong>EXCLUDED</strong> because the AI surface was
        not <strong>CONFIRMED</strong> for this target (it was POSSIBLE or NOT_OBSERVED). Adversarial /
        prompt-injection / RAG / tool experiments require confirmed evidence.</p>
        <pre class="term-pre">${(na.map(p => "- " + p.name + "  (NOT_APPLICABLE)").join("\n")) || "(none)"}</pre>
        <p class="sub">Re-analyze the live service with the <strong>Active Probe</strong> so Orion confirms
        the AI/ML surface, then approve a new plan.</p>
        <div class="btn-row">
          <a class="btn btn-red" href="/environment">[ KNOW THE ENVIRONMENT · LIVE RECON ]</a>
          <a class="btn btn-blue" href="/target-analysis">[ KNOW YOUR TARGET ]</a>
        </div>`);
      return;
    }

    if (stage === "attack") {
      const params = prop.parameters || {};
      const plines = Object.keys(params).filter(k => k !== "weights_path" && k !== "num_outputs")
        .map(k => `  ${k} ${".".repeat(Math.max(2, 14 - k.length))} ${Orion.esc(params[k])}`).join("\n");
      const label = prop.sensitive ? "[ CONFIRM &amp; RUN ]" : "[ RUN EXPERIMENT ]";
      const done = ws.stages.attack === "COMPLETE";
      const tgt = (plan.target || {});
      const liveUrl = (tgt.target && /^https?:\/\//.test(tgt.target)) ? tgt.target : "";
      const bbox = (!done && liveUrl) ? `
        <hr class="term-rule">
        <div class="term-title" style="color:var(--red)">BLACK-BOX ATTACK (live, real queries)</div>
        <p class="sub">Sends perturbed inputs to the live endpoint and measures decision change. Authorised targets only.</p>
        <div class="kv">
          <div class="k">Target URL</div><div class="v"><input type="text" id="bb-url" value="${Orion.esc(liveUrl)}" style="width:100%"></div>
          <div class="k">Endpoint path</div><div class="v"><input type="text" id="bb-path" value="/" style="width:100%"></div>
          <div class="k">File field</div><div class="v"><input type="text" id="bb-field" value="image" style="width:120px"></div>
          <div class="k">epsilon</div><div class="v"><input type="number" step="any" id="bb-eps" value="0.05" style="width:120px"></div>
          <div class="k">max queries</div><div class="v"><input type="number" id="bb-q" value="20" style="width:120px"></div>
        </div>
        <div class="btn-row" style="margin-top:0.4rem;"><button class="btn btn-red" id="run-bbox">[ CONFIRM & RUN BLACK-BOX ATTACK ]</button></div>` : "";
      wrap("ATTACK", `<div class="kv">
        <div class="k">Experiment</div><div class="v">${Orion.esc(prop.name || "—")}</div>
        <div class="k">Scenario</div><div class="v">${Orion.esc(prop.scenario || "manual")}</div>
        <div class="k">Status</div><div class="v">${Orion.esc(ws.stages.attack)}</div></div>
        ${plines ? `<pre class="term-pre">PARAMETERS\n${plines}</pre>` : ""}
        <div class="btn-row">${done ? `<button class="btn" id="to-measure">[ VIEW MEASUREMENTS ]</button>`
          : `<button class="btn btn-red" id="run-attack" data-sensitive="${!!prop.sensitive}">${label}</button>
             <span class="sub" style="align-self:center">demo runner (synthetic robustness)</span>`}</div>
        ${bbox}
        <div id="attack-out" class="sub" style="margin-top:0.4rem;"></div>`);
      const bb = document.getElementById("run-bbox");
      if (bb) bb.addEventListener("click", async () => {
        if (!bb.dataset.ok) { bb.dataset.ok = "1"; bb.innerHTML = "[ CONFIRM: real queries to " + Orion.esc(document.getElementById("bb-url").value) + " ]"; return; }
        Orion.setState(document.getElementById("attack-out"), "running", "running live black-box evasion…");
        try {
          await Orion.postJSON(`/api/experiment/${wsid}/attack-blackbox`, {
            url: document.getElementById("bb-url").value, path: document.getElementById("bb-path").value,
            field: document.getElementById("bb-field").value,
            epsilon: document.getElementById("bb-eps").value, max_queries: document.getElementById("bb-q").value });
          loadWorkspace(wsid);
        } catch (e) { Orion.setState(document.getElementById("attack-out"), "error", "[x] " + e.message); }
      });
      const rb = document.getElementById("run-attack");
      if (rb) rb.addEventListener("click", async () => {
        if (rb.dataset.sensitive === "true" && !rb.dataset.ok) { rb.dataset.ok = "1"; rb.innerHTML = "[ CONFIRM: RUN SENSITIVE ]"; return; }
        Orion.setState(document.getElementById("attack-out"), "running", "executing attack…");
        try { await Orion.postJSON(`/api/experiment/${wsid}/attack`, {}); loadWorkspace(wsid); }
        catch (e) { Orion.setState(document.getElementById("attack-out"), "error", "[x] " + e.message); }
      });
      const tm = document.getElementById("to-measure");
      if (tm) tm.addEventListener("click", () => renderStage(ws, plan, "measure"));
      return;
    }

    if (stage === "measure") {
      Orion.setState(el, "running", "loading measurements…");
      try {
        const m = await Orion.postJSON(`/api/experiment/${wsid}/measure`, {});
        const met = m.metrics || {};
        const rows = ["clean_accuracy", "robust_accuracy", "attack_success_rate", "perturbation_linf", "confidence_shift"]
          .filter(k => met[k] && typeof met[k].value !== "undefined")
          .map(k => `<tr><td>${k.replace(/_/g, " ")}</td><td>${Orion.esc(met[k].value)}</td></tr>`).join("");
        wrap("MEASURE", `<div class="sub">status: ${Orion.statusBadge(m.status)}</div>
          <table class="orion"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${rows || '<tr><td colspan=2 class="sub">no numeric metrics</td></tr>'}</tbody></table>
          <div class="btn-row" style="margin-top:0.5rem;">
            <button class="btn btn-blue" id="to-defend">[ OPEN DEFEND ]</button>
            <a class="btn btn-ghost" href="/runs/${encodeURIComponent(ws.attack_run_id)}">[ VIEW EVIDENCE ]</a>
            <a class="btn btn-ghost" href="/agent?mission=${encodeURIComponent('Explain these measurement results and the trade-off.')}&context=${encodeURIComponent('trace=' + ws.attack_run_id)}">[ EXPLAIN RESULTS ]</a>
          </div>`);
        document.getElementById("to-defend").addEventListener("click", () => { loadWorkspace(wsid); });
      } catch (e) { Orion.setState(el, "error", "[x] " + e.message); }
      return;
    }

    if (stage === "defend") {
      const hardening = (plan.hardening || []).map(d => d.name || d).filter(Boolean);
      wrap("DEFEND", `<div class="sub">A defense is not validated until the attack is replayed (Retest).</div>
        <pre class="term-pre">FAILURE ......... ${Orion.esc(prop.name || "attack")} (see Measure)
PROPOSED DEFENSE  ${hardening.length ? Orion.esc(hardening.join(", ")) : "scenario hardening (input preprocessing / confidence threshold)"}
STATUS .......... ${Orion.esc(ws.stages.defend)}</pre>
        <div class="btn-row"><button class="btn btn-blue" id="apply-defense">[ APPLY DEFENSE ]</button>
          <a class="btn btn-ghost" href="/agent?mission=${encodeURIComponent('Suggest candidate controls for this result.')}">[ SUGGEST CONTROLS ]</a></div>
        <div id="defend-out" class="sub" style="margin-top:0.4rem;"></div>`);
      document.getElementById("apply-defense").addEventListener("click", async () => {
        Orion.setState(document.getElementById("defend-out"), "running", "applying defense…");
        try { await Orion.postJSON(`/api/experiment/${wsid}/defend`, {}); loadWorkspace(wsid); }
        catch (e) { Orion.setState(document.getElementById("defend-out"), "error", "[x] " + e.message); }
      });
      return;
    }

    if (stage === "retest") {
      const done = ws.stages.retest === "COMPLETE";
      wrap("RETEST", `<div class="sub">SAME ATTACK · DIFFERENT SECURITY POSTURE</div>
        <pre class="term-pre">ORIGINAL RUN .... ${Orion.esc(ws.attack_run_id || "—")}
DEFENSE ......... ${Orion.esc(ws.defense_id || "—")}
ATTACK CONFIG ... UNCHANGED
STATUS .......... ${Orion.esc(ws.stages.retest)}</pre>
        <div class="btn-row">${done ? "" : `<button class="btn btn-red" id="run-retest">[ RUN RETEST ]</button>`}
          <a class="btn btn-ghost" href="/agent?mission=${encodeURIComponent('Analyze the defense trade-off from this retest.')}">[ ANALYZE TRADE-OFF ]</a></div>
        <div id="retest-out" style="margin-top:0.4rem;"></div>`);
      const rb = document.getElementById("run-retest");
      if (rb) rb.addEventListener("click", async () => {
        Orion.setState(document.getElementById("retest-out"), "running", "replaying same attack with new posture…");
        try {
          const r = await Orion.postJSON(`/api/experiment/${wsid}/retest`, {});
          document.getElementById("retest-out").innerHTML = comparison(r.comparison, ws.attack_run_id, r.retest_run_id);
        } catch (e) { Orion.setState(document.getElementById("retest-out"), "error", "[x] " + e.message); }
      });
      if (done) { try {
        const cmp = await Orion.getJSON(`/orion/compare?before=${encodeURIComponent(ws.attack_run_id)}&after=${encodeURIComponent(ws.retest_run_id)}`);
        document.getElementById("retest-out").innerHTML = comparison(cmp, ws.attack_run_id, ws.retest_run_id);
      } catch (e) {} }
      return;
    }
  }

  function comparison(cmp, beforeId, afterId) {
    const m = cmp.metrics || {};
    const keys = Object.keys(m);
    if (!keys.length) return '<div class="state">No comparable metrics.</div>';
    const rows = keys.map(k => `<tr><td>${k.replace(/_/g, " ")}</td><td class="col-attack">${m[k].before}</td><td class="col-hardened">${m[k].after}</td><td>${m[k].delta > 0 ? "+" : ""}${m[k].delta}</td></tr>`).join("");
    return `<div class="term-title" style="color:var(--green)">BEFORE vs AFTER</div>
      <table class="orion compare"><thead><tr><th>Metric</th><th>BEFORE</th><th>AFTER</th><th>Δ</th></tr></thead><tbody>${rows}</tbody></table>
      <div class="btn-row" style="margin-top:0.4rem;">
        <a class="btn btn-ghost" href="/runs/${encodeURIComponent(afterId)}">[ OPEN EVIDENCE ]</a></div>`;
  }

  // ---------- compat single-run mode (/measure/<trace>, /defend/<trace>) ----------
  async function renderTraceCompat(trace, stage) {
    try {
      const rec = await Orion.getJSON("/api/runs/" + encodeURIComponent(trace));
      let html = `<div class="assessment-box"><div class="term-title">EXPERIMENT / ${stage.toUpperCase()} (run ${Orion.esc(trace.slice(0, 8))})</div>
        <div class="sub">status: ${Orion.statusBadge(rec.status)} · attack: ${Orion.esc(rec.attack_technique || rec.scenario_name || "—")}</div>`;
      if (stage === "measure") {
        const met = rec.metrics || {};
        const rows = Object.keys(met).filter(k => met[k] && typeof met[k].value !== "undefined")
          .map(k => `<tr><td>${k.replace(/_/g, " ")}</td><td>${Orion.esc(met[k].value)}</td></tr>`).join("");
        html += `<table class="orion"><tbody>${rows || '<tr><td class=sub>no metrics</td></tr>'}</tbody></table>
          <div class="btn-row"><a class="btn btn-blue" href="/defend/${encodeURIComponent(trace)}">[ OPEN DEFEND ]</a>
          <a class="btn btn-ghost" href="/runs/${encodeURIComponent(trace)}">[ VIEW EVIDENCE ]</a></div>`;
      } else {
        const can = (rec.hardening || []).length > 0;
        html += `<div class="sub">A defense is not validated until the attack is replayed.</div>
          <div class="btn-row">${can ? `<button class="btn btn-blue" id="compat-retest">[ RETEST SAME EXPERIMENT ]</button>` : '<span class="sub">This run declares no hardening to retest.</span>'}
          <a class="btn btn-ghost" href="/runs/${encodeURIComponent(trace)}">[ VIEW EVIDENCE ]</a></div><div id="cr-out" style="margin-top:0.4rem;"></div>`;
      }
      html += `</div>`;
      root.innerHTML = html;
      const cr = document.getElementById("compat-retest");
      if (cr) cr.addEventListener("click", async () => {
        Orion.setState(document.getElementById("cr-out"), "running", "retesting…");
        try {
          const h = await Orion.postJSON("/api/replay/" + encodeURIComponent(trace), { mode: "hardened" });
          const cmp = await Orion.getJSON(`/orion/compare?before=${encodeURIComponent(trace)}&after=${encodeURIComponent(h.trace_id)}`);
          document.getElementById("cr-out").innerHTML = comparison(cmp, trace, h.trace_id);
        } catch (e) { Orion.setState(document.getElementById("cr-out"), "error", "[x] " + e.message); }
      });
    } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
  }

  // ---------- landing ----------
  async function landing() {
    let list = [];
    try { list = (await Orion.getJSON("/api/experiment")).workspaces || []; } catch (e) {}
    let html = `<div class="term"><div class="term-title">Open an experiment</div>
      <p class="sub">Experiments are created when you approve a plan (Know Yourself / Target / Analysis Context).</p>
      <div class="btn-row"><input type="text" id="open-plan" placeholder="paste a plan id (ORN-PLAN-…)" style="flex:1;min-width:220px;">
        <button class="btn btn-red" id="open-plan-btn">[ OPEN FROM PLAN ]</button></div></div>`;
    html += `<div class="term"><div class="term-title">Recent experiments</div>`;
    if (list.length) {
      html += list.map(w => `<div class="recon-run-row"><div class="meta"><span class="rid">${Orion.esc(w.experiment_workspace_id)}</span>
        · ${Orion.esc(w.active_experiment_id || "—")} · stage: ${Orion.esc(w.current_stage)}</div>
        <a class="btn btn-ghost" href="/experiment/${encodeURIComponent(w.experiment_workspace_id)}">[ RESUME ]</a></div>`).join("");
    } else { html += '<div class="state">No experiments yet.</div>'; }
    html += `</div>`;
    root.innerHTML = html;
    document.getElementById("open-plan-btn").addEventListener("click", async () => {
      const pid = document.getElementById("open-plan").value.trim();
      if (!pid) return;
      try {
        const ws = await Orion.postJSON("/api/experiment/from-plan/" + encodeURIComponent(pid), {});
        location.href = "/experiment/" + encodeURIComponent(ws.experiment_workspace_id);
      } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
    });
  }

  // ---------- route ----------
  if (wsId) loadWorkspace(wsId);
  else if (planId) {
    Orion.postJSON("/api/experiment/from-plan/" + encodeURIComponent(planId), {})
      .then(ws => location.href = "/experiment/" + encodeURIComponent(ws.experiment_workspace_id))
      .catch(e => Orion.setState(root, "error", "[x] " + e.message));
  } else if (traceId && (compatStage === "measure" || compatStage === "defend")) {
    renderTraceCompat(traceId, compatStage);
  } else landing();
})();
