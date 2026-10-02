/* Attack workspace: load an approved plan handoff and run experiments explicitly.
   Approve Plan != Run Attack — the human presses RUN here. */
(function () {
  "use strict";
  const root = document.getElementById("attack-root");
  const planId = new URLSearchParams(window.location.search).get("plan_id");

  if (!planId) {
    root.innerHTML = '<div class="state">No plan selected. Approve a plan in Know Yourself or Know Your Target, then open the Attack workspace.</div>';
    return;
  }

  function paramsText(p) {
    const keys = Object.keys(p || {}).filter(k => k !== "weights_path" && k !== "num_outputs");
    if (!keys.length && !p.weights_path) return "";
    let lines = keys.map(k => `  ${k} ${".".repeat(Math.max(2, 16 - k.length))} ${Orion.esc(p[k])}`);
    if (p.weights_path) lines.unshift(`  model ............ ${Orion.esc(p.weights_path)} (white-box)`);
    return `<pre class="term-pre">${lines.join("\n")}</pre>`;
  }

  function expRow(p) {
    const runnable = p.status === "READY" && p.scenario;
    const btnLabel = p.sensitive ? "[ CONFIRM &amp; RUN ]" : "[ RUN EXPERIMENT ]";
    let actions = "";
    if (runnable) {
      actions = `<button class="btn btn-red btn-run" data-eid="${Orion.esc(p.experiment_id)}" data-sensitive="${p.sensitive}">${btnLabel}</button>`;
    } else if (p.status === "NEEDS_INPUT") {
      actions = `<button class="btn btn-ghost" disabled>[ PROVIDE INPUT ]</button>`;
    } else if (p.status === "READY" && !p.scenario) {
      actions = `<span class="sub">manual experiment — no automated runner</span>`;
    }
    const sens = p.sensitive ? ` <span class="proposed">SENSITIVE</span>` : "";
    return `<div class="exp">
      <span class="eid">${Orion.esc(p.experiment_id)}</span><span class="ename">${Orion.esc(p.name)}</span>
      <span class="appl ${p.status === "READY" ? "APPLICABLE" : "INSUFFICIENT_EVIDENCE"}">${Orion.esc(p.status)}</span>${sens}
      <div class="sub">${Orion.esc(p.reason || "")}${p.missing && p.missing.length ? " · missing: " + Orion.esc(p.missing.join(", ")) : ""}</div>
      ${paramsText(p.parameters)}
      <div class="btn-row" style="margin-top:0.3rem;" id="act-${Orion.esc(p.experiment_id)}">${actions}</div>
    </div>`;
  }

  function render(data) {
    const plan = data.plan, h = data.handoff;
    const t = plan.target || {};
    const approved = plan.proposals.filter(p => ["READY","RUNNING","COMPLETED","FAILED"].includes(p.status));
    const conditional = plan.proposals.filter(p => p.status === "NEEDS_INPUT");
    const excluded = plan.proposals.filter(p => p.status === "EXCLUDED");

    let html = `<div class="assessment-box"><div class="term-title">PLAN ${Orion.esc(plan.plan_id)}</div>
      <div class="kv">
        <div class="k">Source</div><div class="v">${Orion.esc((plan.source_type||"").replace("_"," ").toUpperCase())}</div>
        <div class="k">Target</div><div class="v">${Orion.esc(t.model || t.target || t.system_type || "unknown")}</div>
        <div class="k">System type</div><div class="v">${Orion.esc(t.system_type || "unknown")}</div>
        <div class="k">Access</div><div class="v">${Orion.esc(t.access || "unknown")}</div>
        <div class="k">Threat model</div><div class="v">${h.threat_model && Object.keys(h.threat_model).length ? '<span class="ok">[ LOADED ]</span>' : 'N/A'}</div>
        <div class="k">Evidence</div><div class="v">${(plan.evidence_ids||[]).length} item(s) ${plan.evidence_ids && plan.evidence_ids.length ? '<span class="ok">[ LOADED ]</span>' : ''}</div>
        <div class="k">Approved by human</div><div class="v">${plan.approved_by_human ? '<span class="ok">✓ '+Orion.esc(plan.approved_at||"")+'</span>' : 'NO'}</div>
      </div></div>`;

    html += `<div class="term"><div class="term-title">EXPERIMENT QUEUE</div>`;
    html += approved.length ? approved.map(expRow).join("") : '<div class="state">No executable experiments in this plan.</div>';
    if (conditional.length) {
      html += `<div class="term-title" style="margin-top:0.8rem;">CONDITIONAL — NEEDS INPUT (${conditional.length})</div>`;
      html += conditional.map(expRow).join("");
    }
    html += `</div>`;

    if (excluded.length) {
      html += `<div class="term"><button class="btn btn-ghost" id="btn-excluded">[ SHOW EXCLUDED EXPERIMENTS (${excluded.length}) ]</button>
        <div id="excluded" class="hidden" style="margin-top:0.5rem;">`
        + excluded.map(p => `<div class="exp na"><span class="ename">${Orion.esc(p.name)}</span> <span class="appl NOT_APPLICABLE">NOT_APPLICABLE</span><div class="sub">${Orion.esc(p.reason||"")}</div></div>`).join("")
        + `</div></div>`;
    }

    root.innerHTML = html;

    const exBtn = document.getElementById("btn-excluded");
    if (exBtn) exBtn.addEventListener("click", () => document.getElementById("excluded").classList.toggle("hidden"));

    root.querySelectorAll(".btn-run").forEach(b => b.addEventListener("click", () => runExperiment(plan.plan_id, b)));
  }

  async function runExperiment(pid, btn) {
    const eid = btn.dataset.eid;
    if (btn.dataset.sensitive === "true" && !btn.dataset.confirmed) {
      btn.dataset.confirmed = "1";
      btn.textContent = "[ CONFIRM: RUN SENSITIVE EXPERIMENT ]";
      return;
    }
    const act = document.getElementById("act-" + eid);
    act.innerHTML = '<span class="loading-line">ORION IS RUNNING...</span>';
    try {
      const res = await Orion.postJSON(`/api/plans/${encodeURIComponent(pid)}/experiments/${encodeURIComponent(eid)}/run`, {});
      act.innerHTML = `<span class="ok">[+] ${Orion.esc(res.status)}</span> — <a href="/measure/${encodeURIComponent(res.trace_id)}">[ VIEW RESULTS → MEASURE ]</a>`;
    } catch (e) {
      act.innerHTML = `<span class="err" style="color:var(--red-bright)">[x] ${Orion.esc(e.message)}</span>`;
    }
  }

  (async function () {
    try {
      const data = await Orion.getJSON("/api/plans/" + encodeURIComponent(planId));
      render(data);
    } catch (e) {
      Orion.setState(root, "error", "[x] could not load plan: " + e.message);
    }
  })();
})();
