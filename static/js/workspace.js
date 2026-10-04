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
  const DEMO = q.get("demo") === "1";   // conference mode: make comparison prominent

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

  // Fetch a run's evidence and render the reusable image-attack comparison into
  // `host`. Type-aware: non-image runs leave the host empty and return false.
  async function injectComparison(host, runId, opts) {
    if (!host || !runId) return false;
    try {
      const rec = await Orion.getJSON("/api/runs/" + encodeURIComponent(runId));
      const html = Orion.imageComparison(rec, runId, opts || {});
      host.innerHTML = html || "";
      if (html) Orion.wireImageAmplify(host);
      return !!html;
    } catch (e) { host.innerHTML = ""; return false; }
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
      const done = ws.stages.attack === "COMPLETE";
      if (done) {
        wrap("ATTACK", `<div class="kv">
          <div class="k">Experiment</div><div class="v">${Orion.esc(prop.name || "—")}</div>
          <div class="k">Status</div><div class="v">${Orion.statusBadge(ws.stages.attack)}</div>
          <div class="k">Run</div><div class="v">${Orion.esc(ws.attack_run_id || "—")}</div></div>
          <div class="btn-row"><button class="btn" id="to-measure">[ VIEW MEASUREMENTS ]</button>
            <a class="btn btn-ghost" href="/runs/${encodeURIComponent(ws.attack_run_id)}">[ VIEW EVIDENCE ]</a></div>
          <div id="attack-compare" style="margin-top:0.6rem;"></div>`);
        injectComparison(document.getElementById("attack-compare"), ws.attack_run_id, { demo: DEMO });
        const tm = document.getElementById("to-measure");
        if (tm) tm.addEventListener("click", () => renderStage(ws, plan, "measure"));
        return;
      }
      // Not run yet: derive access level, then offer the matching attack catalog.
      await renderAttackChooser(el, ws, plan, prop);
      return;
    }

    if (stage === "measure") {
      Orion.setState(el, "running", "loading measurements…");
      try {
        const m = await Orion.postJSON(`/api/experiment/${wsid}/measure`, {});
        const met = m.metrics || {};
        // Family-aware: render every numeric metric the experiment produced
        // (image-adversarial, GenAI or agentic), not a fixed adversarial-ML list.
        const rows = Object.keys(met)
          .filter(k => met[k] && typeof met[k] === "object" && typeof met[k].value === "number")
          .map(k => `<tr><td>${k.replace(/_/g, " ")}</td><td>${Orion.esc(met[k].value)}${met[k].unit ? " " + Orion.esc(met[k].unit) : ""}</td></tr>`).join("");
        const fnd = m.finding;
        const fndLine = fnd ? `<div class="sub" style="margin-top:0.5rem;">Finding: <a href="/findings/${encodeURIComponent(fnd.id)}">${Orion.esc(fnd.id)}</a> · ${Orion.statusBadge(fnd.status)} · severity ${Orion.esc(fnd.severity)}</div>` : "";
        wrap("MEASURE", `<div class="sub">status: ${Orion.statusBadge(m.status)}${m.family ? " · family: " + Orion.esc(m.family.replace(/_/g, " ")) : ""}</div>
          <table class="orion"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>${rows || '<tr><td colspan=2 class="sub">no numeric metrics</td></tr>'}</tbody></table>
          ${fndLine}
          <div class="btn-row" style="margin-top:0.5rem;">
            <button class="btn btn-blue" id="to-defend">[ OPEN DEFEND ]</button>
            <a class="btn btn-ghost" href="/runs/${encodeURIComponent(ws.attack_run_id)}">[ VIEW EVIDENCE ]</a>
            <a class="btn btn-ghost" href="/agent?mission=${encodeURIComponent('Explain these measurement results and the trade-off.')}&context=${encodeURIComponent('trace=' + ws.attack_run_id)}">[ EXPLAIN RESULTS ]</a>
          </div>
          <div id="measure-compare" style="margin-top:0.6rem;"></div>`);
        injectComparison(document.getElementById("measure-compare"), ws.attack_run_id, { demo: DEMO });
        document.getElementById("to-defend").addEventListener("click", () => { loadWorkspace(wsid); });
      } catch (e) { Orion.setState(el, "error", "[x] " + e.message); }
      return;
    }

    if (stage === "defend") {
      wrap("DEFEND", `<div class="sub">A defense is not validated until the attack is replayed (Retest).
        Pick the control whose <em>mechanism</em> matches the attack's vector — the Retest will tell you if it worked.</div>
        <pre class="term-pre">FAILURE ......... ${Orion.esc(prop.name || ws.attack_catalog_id || "attack")} (see Measure)
STATUS .......... ${Orion.esc(ws.stages.defend)}</pre>
        <div class="term-title">CANDIDATE CONTROL</div>
        <div id="control-list" class="atk-catalog"><div class="state">loading controls…</div></div>
        <div class="btn-row" style="margin-top:0.4rem;"><button class="btn btn-blue" id="apply-defense">[ APPLY CONTROL ]</button>
          <a class="btn btn-ghost" href="/agent?mission=${encodeURIComponent('Suggest candidate controls for this result.')}">[ SUGGEST CONTROLS ]</a></div>
        <div id="defend-out" class="sub" style="margin-top:0.4rem;"></div>`);
      try {
        const cc = await Orion.getJSON(`/api/experiment/${wsid}/controls`);
        const list = (cc.controls || []);
        document.getElementById("control-list").innerHTML = list.length ? list.map((c, i) =>
          `<label class="atk-opt"><div><input type="radio" name="ctrl-sel" value="${Orion.esc(c.id)}" ${i === 0 ? "checked" : ""}>
            <strong>${Orion.esc(c.name)}</strong></div><div class="sub">${Orion.esc(c.description)}</div></label>`).join("")
          : '<div class="state">No catalogued controls for this attack.</div>';
      } catch (e) { document.getElementById("control-list").innerHTML = '<div class="state error">[x] ' + e.message + '</div>'; }
      document.getElementById("apply-defense").addEventListener("click", async () => {
        const sel = document.querySelector('input[name="ctrl-sel"]:checked');
        Orion.setState(document.getElementById("defend-out"), "running", "applying control…");
        try { await Orion.postJSON(`/api/experiment/${wsid}/defend`, { control_id: sel ? sel.value : null }); loadWorkspace(wsid); }
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
        <div id="retest-out" style="margin-top:0.4rem;"></div>
        <div id="retest-images" style="margin-top:0.6rem;"></div>`);
      // Before/after visual comparison: the original adversarial example, and the
      // hardened replay's example if it produced a new image artifact.
      async function renderRetestImages(afterId) {
        const host = document.getElementById("retest-images");
        if (!host) return;
        host.innerHTML = `<div class="term-title">BEFORE DEFENSE (original adversarial)</div><div id="rt-before"></div>
          <div class="term-title" style="margin-top:0.5rem;">AFTER DEFENSE (hardened retest)</div><div id="rt-after"></div>`;
        const hadBefore = await injectComparison(document.getElementById("rt-before"), ws.attack_run_id, { demo: DEMO });
        const hadAfter = afterId ? await injectComparison(document.getElementById("rt-after"), afterId, { demo: DEMO }) : false;
        if (!hadBefore && !hadAfter) { host.innerHTML = ""; return; }   // non-image experiment
        if (!hadAfter) document.getElementById("rt-after").innerHTML = '<div class="state">NO NEW IMAGE ARTIFACT — hardened retest reports metrics only.</div>';
      }
      const rb = document.getElementById("run-retest");
      if (rb) rb.addEventListener("click", async () => {
        Orion.setState(document.getElementById("retest-out"), "running", "replaying same attack with new posture…");
        try {
          const r = await Orion.postJSON(`/api/experiment/${wsid}/retest`, {});
          let html = comparison(r.comparison, ws.attack_run_id, r.retest_run_id);
          if (r.finding) html += `<div class="sub" style="margin-top:0.4rem;">Finding <a href="/findings/${encodeURIComponent(r.finding.id)}">${Orion.esc(r.finding.id)}</a>: ${Orion.statusBadge(r.finding.status)} · control ${Orion.statusBadge(r.finding.retest_status || "—")}</div>`;
          document.getElementById("retest-out").innerHTML = html;
          renderRetestImages(r.retest_run_id);
        } catch (e) { Orion.setState(document.getElementById("retest-out"), "error", "[x] " + e.message); }
      });
      if (done) { try {
        const cmp = await Orion.getJSON(`/orion/compare?before=${encodeURIComponent(ws.attack_run_id)}&after=${encodeURIComponent(ws.retest_run_id)}`);
        document.getElementById("retest-out").innerHTML = comparison(cmp, ws.attack_run_id, ws.retest_run_id);
        renderRetestImages(ws.retest_run_id);
      } catch (e) {} }
      return;
    }
  }

  // ATTACK stage: derive access level, then offer the matching attack catalog
  // with base tuning. White-box runs the real torch+ART attack on the weights;
  // black-box runs the decision-based query evasion against the live endpoint.
  const FAMILY_TABS = [["traditional_ml", "Traditional ML"], ["generative_ai", "Generative AI"], ["agentic_ai", "Agentic AI"]];

  async function renderAttackChooser(el, ws, plan, prop) {
    const wsid = ws.experiment_workspace_id;
    el.innerHTML = `<div class="term-title">EXPERIMENT / ATTACK</div><div class="state">loading attack catalog…</div>`;
    let opt;
    try { opt = await Orion.getJSON(`/api/experiment/${wsid}/attack-options`); }
    catch (e) { Orion.setState(el, "error", "[x] " + e.message); return; }
    const families = opt.families || {};
    const family = el._family || "traditional_ml";

    const tabs = `<div class="btn-row">` + FAMILY_TABS.map(([k, label]) =>
      `<button class="btn ${family === k ? "btn-red" : "btn-ghost"}" data-fam="${k}">${label}</button>`).join("") + `</div>`;

    let body;
    if (family === "traditional_ml") body = traditionalBody(opt, el);
    else body = agenticBody(families[family] || [], family);

    el.innerHTML = `<div class="term-title">EXPERIMENT / ATTACK</div>
      <div class="term-title" style="margin-top:0.3rem;">AI SECURITY FAMILY</div>
      ${tabs}<hr class="term-rule">${body}
      <div id="attack-out" class="sub" style="margin-top:0.4rem;"></div>`;

    el.querySelectorAll("[data-fam]").forEach(b => b.addEventListener("click", () => {
      el._family = b.dataset.fam; el._accessOverride = null; renderAttackChooser(el, ws, plan, prop);
    }));

    if (family === "traditional_ml") wireTraditional(el, ws, plan, prop, opt);
    else wireAgentic(el, ws, wsid);
  }

  function tuningInputs(a) {
    const bp = a.base_params || {};
    return Object.keys(bp).map(k =>
      `<label class="atk-tune">${k} <input type="number" step="any" data-atk="${Orion.esc(a.id)}" data-param="${Orion.esc(k)}" value="${Orion.esc(bp[k])}"></label>`).join("");
  }

  function traditionalBody(opt, el) {
    const derived = opt.derived || {};
    const access = el._accessOverride || derived.access_level || "white_box";
    const cat = opt.catalog || {};
    const liveUrl = opt.live_url || "";
    const wbOn = access === "white_box";
    const list = cat[access] || [];
    const attackList = list.length ? list.map((a, i) => `<label class="atk-opt">
        <div><input type="radio" name="atk-sel" value="${Orion.esc(a.id)}" data-backend="${Orion.esc(a.backend)}" ${i === 0 ? "checked" : ""}>
          <strong>${Orion.esc(a.name)}</strong></div>
        <div class="sub">${Orion.esc(a.about)}</div>
        <div class="atk-tuning">${tuningInputs(a)}</div></label>`).join("")
      : `<div class="state">No ${access.replace("_", "-")} attacks catalogued for this modality.</div>`;
    const wbInputs = `<div class="kv">
        <div class="k">Weights path</div><div class="v"><input type="text" id="wb-weights" value="weights/vit_teacher.pth" style="width:100%"></div>
        <div class="k">Input image</div><div class="v"><input type="text" id="wb-image" value="static/fake/0001_00_00_01_0.jpg" style="width:100%"></div>
        <div class="k">num_outputs</div><div class="v"><input type="number" id="wb-num" placeholder="(inferred from weights)" style="width:180px"></div></div>`;
    const bbInputs = `<div class="kv">
        <div class="k">Target URL</div><div class="v"><input type="text" id="bb-url" value="${Orion.esc(liveUrl)}" style="width:100%"></div>
        <div class="k">Endpoint path</div><div class="v"><input type="text" id="bb-path" value="/" style="width:100%"></div>
        <div class="k">File field</div><div class="v"><input type="text" id="bb-field" value="image" style="width:120px"></div>
        <div class="k">Input image</div><div class="v"><input type="text" id="bb-image" value="static/fake/0001_00_00_01_0.jpg" style="width:100%"></div></div>`;
    return `<div class="term-title">ATTACKER ACCESS LEVEL</div>
      <div class="btn-row">
        <button class="btn ${wbOn ? "btn-red" : "btn-ghost"}" id="acc-wb">WHITE-BOX (weights)</button>
        <button class="btn ${!wbOn ? "btn-red" : "btn-ghost"}" id="acc-bb">BLACK-BOX (query-only)</button></div>
      <p class="sub">Derived: <strong>${Orion.esc((derived.access_level || "—").replace("_", "-"))}</strong> — ${Orion.esc(derived.reason || "")}</p>
      <div class="term-title">${wbOn ? "WHITE-BOX" : "BLACK-BOX"} ATTACKS — base tuning applied, override as needed</div>
      <div class="atk-catalog">${attackList}</div>
      ${wbOn ? wbInputs : bbInputs}
      <div class="btn-row" style="margin-top:0.4rem;"><button class="btn btn-red" id="run-atk">[ RUN ATTACK ]</button>
        <span class="sub" style="align-self:center">${wbOn ? "real torch+ART attack on the weights (local)" : "real queries to the live endpoint — authorised targets only"}</span></div>`;
  }

  function wireTraditional(el, ws, plan, prop, opt) {
    const wsid = ws.experiment_workspace_id;
    const wb = document.getElementById("acc-wb"), bb = document.getElementById("acc-bb");
    if (wb) wb.addEventListener("click", () => { el._accessOverride = "white_box"; renderAttackChooser(el, ws, plan, prop); });
    if (bb) bb.addEventListener("click", () => { el._accessOverride = "black_box"; renderAttackChooser(el, ws, plan, prop); });
    const run = document.getElementById("run-atk");
    if (!run) return;
    run.addEventListener("click", async () => {
      const sel = el.querySelector('input[name="atk-sel"]:checked');
      if (!sel) return;
      const id = sel.value, backend = sel.dataset.backend;
      const params = {};
      el.querySelectorAll(`.atk-tuning input[data-atk="${id}"]`).forEach(i => { params[i.dataset.param] = i.value; });
      const out = document.getElementById("attack-out");
      if (backend === "whitebox_image") {
        Orion.setState(out, "running", `running white-box ${id} …`);
        try {
          await Orion.postJSON(`/api/experiment/${wsid}/attack-whitebox`, {
            weights_path: document.getElementById("wb-weights").value,
            image: document.getElementById("wb-image").value,
            num_outputs: document.getElementById("wb-num").value || null, attack: id, params });
          loadWorkspace(wsid);
        } catch (e) { Orion.setState(out, "error", "[x] " + e.message); }
      } else {
        if (!run.dataset.ok) { run.dataset.ok = "1"; run.textContent = "[ CONFIRM: real queries to " + (document.getElementById("bb-url").value || "endpoint") + " ]"; return; }
        Orion.setState(out, "running", `running black-box ${id} …`);
        try {
          await Orion.postJSON(`/api/experiment/${wsid}/attack-blackbox`, {
            url: document.getElementById("bb-url").value, path: document.getElementById("bb-path").value,
            field: document.getElementById("bb-field").value, image: document.getElementById("bb-image").value,
            epsilon: params.epsilon, max_queries: params.max_queries });
          loadWorkspace(wsid);
        } catch (e) { Orion.setState(out, "error", "[x] " + e.message); }
      }
    });
  }

  function agenticBody(list, family) {
    if (!list.length) return `<div class="state">No ${family.replace("_", " ")} attacks catalogued.</div>`;
    const items = list.map((a, i) => `<label class="atk-opt">
      <div><input type="radio" name="atk-sel" value="${Orion.esc(a.id)}" ${i === 0 ? "checked" : ""}>
        <strong>${Orion.esc(a.name)}</strong> <span class="sub">${Orion.esc(a.id)}</span></div>
      <div class="sub">${Orion.esc(a.description)}</div>
      <div class="sub">success: ${Orion.esc((a.success_criteria || []).join(", "))}</div></label>`).join("");
    return `<div class="term-title">${family === "generative_ai" ? "GENERATIVE AI" : "AGENTIC"} EXPERIMENTS — controlled lab</div>
      <p class="sub">Deterministic lab: an agent with tools over (possibly untrusted) content. Findings are derived from the recorded trace, not asserted.</p>
      <div class="atk-catalog">${items}</div>
      <div class="kv"><div class="k">Trials</div><div class="v"><input type="number" id="ag-trials" value="3" min="1" max="20" style="width:120px"></div></div>
      <div class="btn-row" style="margin-top:0.4rem;"><button class="btn btn-red" id="run-atk">[ RUN EXPERIMENT ]</button>
        <span class="sub" style="align-self:center">controlled lab (no live LLM / MCP)</span></div>`;
  }

  function wireAgentic(el, ws, wsid) {
    const run = document.getElementById("run-atk");
    if (!run) return;
    run.addEventListener("click", async () => {
      const sel = el.querySelector('input[name="atk-sel"]:checked');
      if (!sel) return;
      const out = document.getElementById("attack-out");
      Orion.setState(out, "running", `running ${sel.value} in the controlled lab …`);
      try {
        await Orion.postJSON(`/api/experiment/${wsid}/attack-agentic`, {
          attack_id: sel.value, trials: document.getElementById("ag-trials").value || 3 });
        loadWorkspace(wsid);
      } catch (e) { Orion.setState(out, "error", "[x] " + e.message); }
    });
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
          ${Orion.imageComparison(rec, trace, { demo: DEMO })}
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
      Orion.wireImageAmplify(root);
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
