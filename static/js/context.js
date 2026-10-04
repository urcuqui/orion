/* Analysis Context: select Self/Target/Environment profiles, correlate, hand off.
   Self + Target + Environment → Analysis Context → Threat Model → Experiment Plan. */
(function () {
  "use strict";
  const pickers = document.getElementById("ctx-pickers");
  const resultEl = document.getElementById("ctx-result");
  const handoffEl = document.getElementById("ctx-handoff");
  const q = new URLSearchParams(location.search);

  function sel(id, label, options, preset) {
    const opts = ['<option value="">○ NOT PROVIDED — optional input</option>']
      .concat(options.map(o => `<option value="${Orion.esc(o.id)}" ${o.id === preset ? "selected" : ""}>${Orion.esc(o.id)}${o.label ? " · " + Orion.esc(o.label) : ""}</option>`));
    return `<label class="field"><span class="field-label">${label}</span>
      <select id="${id}">${opts.join("")}</select></label>`;
  }

  async function loadPickers() {
    Orion.setState(pickers, "running", "listing profiles…");
    try {
      const p = await Orion.getJSON("/api/profiles");
      pickers.innerHTML =
        sel("ctx-self", "SYSTEM · Know Yourself", p.self || [], q.get("self_profile_id")) +
        sel("ctx-target", "TARGET · Know Your Target", p.target || [], q.get("target_profile_id")) +
        sel("ctx-env", "ENVIRONMENT · Know the Terrain", p.environment || [], q.get("environment_profile_id"));
      const summary = document.createElement("p");
      summary.className = "sub"; summary.setAttribute("role", "status");
      pickers.appendChild(summary);
      function updateSummary() {
        const count = pickers.querySelectorAll("select");
        summary.textContent = `Using ${Array.from(count).filter(s => s.value).length} of 3 profile inputs. Missing inputs are explicit; no profile is silently added.`;
      }
      pickers.querySelectorAll("select").forEach(s => s.addEventListener("change", updateSummary));
      updateSummary();
      if ((p.self||[]).length + (p.target||[]).length + (p.environment||[]).length === 0) {
        pickers.innerHTML += '<div class="state">No profiles yet. Create them in Know Yourself / Know Your Target / Know The Environment.</div>';
      }
    } catch (e) { Orion.setState(pickers, "error", "[x] " + e.message); }
  }

  function coverageLine(cov) {
    const b = (k) => `<div>${k.toUpperCase()} ${".".repeat(Math.max(2, 16 - k.length))} ${cov[k] === "COMPLETE" ? '<span class="ok">COMPLETE</span>' : '<span class="off">' + cov[k] + "</span>"}</div>`;
    return `<pre class="term-pre">${b("self")}${b("target")}${b("environment")}</pre>`;
  }

  function renderAssessment(a) {
    let html = `<div class="assessment-box"><div class="term-title">ORION // ANALYSIS CONTEXT — ${Orion.esc(a.analysis_context_id)}</div>
      ${coverageLine(a.coverage || {})}
      <div class="statusline"><div>STATUS: <span class="ok">${Orion.esc(a.status || "")}</span></div>
        ${a.self_profile_id ? `<div><span class="ok">[+]</span> SELF: ${Orion.esc(a.self_profile_id)}</div>` : '<div><span class="off">[-]</span> SELF: NOT_AVAILABLE</div>'}
        ${a.target_profile_id ? `<div><span class="ok">[+]</span> TARGET: ${Orion.esc(a.target_profile_id)}</div>` : '<div><span class="off">[-]</span> TARGET: NOT_AVAILABLE</div>'}
        ${a.environment_profile_id ? `<div><span class="ok">[+]</span> ENVIRONMENT: ${Orion.esc(a.environment_profile_id)}</div>` : '<div><span class="off">[-]</span> ENVIRONMENT: NOT_AVAILABLE</div>'}
      </div></div>`;

    // Source-aware threat model
    const tm = a.threat_model || {};
    html += `<div class="term"><div class="term-title">THREAT MODEL (source-aware)</div>`;
    html += `<div class="sub">ASSETS</div>` + (tm.assets||[]).map(x => `<div class="analysis-item"><strong>${Orion.esc(x.asset)}</strong> <span class="ttype">${Orion.esc(x.kind||"")}</span><div class="meta">source: ${Orion.esc(x.source)} · evidence: ${(x.evidence||[]).length}</div></div>`).join("") || "";
    html += `<div class="sub space-above-sm">TRUST BOUNDARIES</div>` + ((tm.trust_boundaries||[]).map(x => `<div class="analysis-item">${Orion.esc(x.boundary)}<div class="meta">source: ${Orion.esc(x.source)}</div></div>`).join("") || '<div class="sub">none</div>');
    html += `<div class="sub space-above-sm">OBJECTIVES</div>` + ((tm.objectives||[]).map(x => `<div class="analysis-item">${Orion.esc(x.objective)}<div class="meta">source: ${Orion.esc(x.source)}</div></div>`).join("") || '<div class="sub">none (define in Target Profile)</div>');
    const adv = tm.adversary || {};
    html += `<div class="sub space-above-sm">ADVERSARY</div><pre class="term-pre">goal ....... ${Orion.esc(adv.goal)}
knowledge .. ${Orion.esc(adv.knowledge)}
budget ..... ${Orion.esc(adv.budget)}
network .... ${Orion.esc(adv.network_access)}
creds ...... ${Orion.esc(adv.credential_access)}</pre>`;
    html += `</div>`;

    // Applicability (source-attributed)
    const exps = a.suggested_experiments || [];
    const ap = exps.filter(e => e.applicability === "APPLICABLE");
    const na = exps.filter(e => e.applicability !== "APPLICABLE");
    html += `<div class="term"><div class="term-title">APPLICABLE EXPERIMENTS</div>`;
    html += ap.length ? ap.map(e => `<div class="exp"><span class="ename">${Orion.esc(e.name)}</span>
        <span class="appl APPLICABLE">APPLICABLE</span>
        <div class="sub">${Orion.esc(e.rationale)} · source: ${Orion.esc((e.source_profiles||[]).join(", ") || "—")}</div></div>`).join("")
      : '<div class="state">No applicable experiments for this context.</div>';
    html += `<div style="margin-top:0.5rem;"><button class="btn btn-ghost" id="ctx-na">Show Non-Applicable (${na.length})</button>
      <div id="ctx-na-list" class="hidden space-above-sm">`
      + na.map(e => `<div class="exp na"><span class="ename">${Orion.esc(e.name)}</span> <span class="appl ${Orion.esc(e.applicability)}">${Orion.esc(e.applicability.replace("_"," "))}</span><div class="sub">${Orion.esc(e.rationale)}${e.missing&&e.missing.length?" · missing: "+Orion.esc(e.missing.join(", ")):""}</div></div>`).join("")
      + `</div></div>`;
    html += `<div class="btn-row space-above"><button class="btn btn-ghost" id="ctx-approve">Review &amp; Approve Plan</button></div></div>`;

    resultEl.innerHTML = html;
    resultEl.classList.remove("hidden");
    const naBtn = document.getElementById("ctx-na");
    if (naBtn) naBtn.addEventListener("click", () => document.getElementById("ctx-na-list").classList.toggle("hidden"));
    document.getElementById("ctx-approve").addEventListener("click", () => {
      handoffEl.classList.remove("hidden");
      Orion.planReview(handoffEl, "analysis_context", a);
    });
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function run() {
    const body = {
      self_profile_id: document.getElementById("ctx-self").value || null,
      target_profile_id: document.getElementById("ctx-target").value || null,
      environment_profile_id: document.getElementById("ctx-env").value || null,
    };
    if (!body.self_profile_id && !body.target_profile_id && !body.environment_profile_id) {
      Orion.setState(resultEl, "error", "Select at least one profile."); resultEl.classList.remove("hidden"); return;
    }
    Orion.setState(resultEl, "running", "correlating profiles into analysis context…");
    resultEl.classList.remove("hidden");
    try { renderAssessment(await Orion.postJSON("/api/context/analyze", body)); }
    catch (e) { Orion.setState(resultEl, "error", "[x] " + e.message); }
  }

  document.getElementById("ctx-run").addEventListener("click", run);
  document.getElementById("ctx-refresh").addEventListener("click", loadPickers);
  loadPickers().then(() => {
    // Auto-run when any profile id was deep-linked.
    if (q.get("self_profile_id") || q.get("target_profile_id") || q.get("environment_profile_id")) run();
  });
})();
