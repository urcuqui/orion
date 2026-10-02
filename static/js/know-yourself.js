/* Know Yourself: profile an AI system (Traditional ML / Generative AI / Hybrid).
   Profiling only — recommended experiments are proposals; execution is the Attack phase. */
(function () {
  "use strict";
  const resultEl = document.getElementById("ky-result");
  const statusEl = document.getElementById("ky-status");
  const forcedEl = document.getElementById("ky-forced");
  let forceType = "";
  const attackUrl = document.body.dataset.attackUrl || "/adversarial";

  document.querySelectorAll("#ky-menu li").forEach(li =>
    li.addEventListener("click", () => {
      forceType = li.dataset.type || "";
      document.querySelectorAll("#ky-menu li").forEach(x => x.classList.toggle("active", x === li));
      forcedEl.textContent = forceType ? "branch: " + forceType : "branch: auto-detect";
    }));

  const statusTone = (s) => ({ APPLICABLE: "good", CONDITIONAL: "warn", NOT_APPLICABLE: "", INSUFFICIENT_EVIDENCE: "warn" }[s] || "");

  function ctrlBadge(s) {
    const cls = { ENABLED: "good", PARTIAL: "warn", NOT_FOUND: "bad", UNKNOWN: "", NOT_APPLICABLE: "" }[s] || "";
    return `<span class="metric ${cls}" style="display:inline-block;padding:0.05rem 0.4rem;font-size:0.68rem;">${Orion.esc(s)}</span>`;
  }

  function item(it) {
    const c = it.classification;
    return `<div class="analysis-item cls-${c}"><span class="cls cls-${c}">${c}</span>${Orion.esc(it.statement)}
      <div class="meta">confidence: ${Orion.esc(it.confidence)} · evidence: ${(it.evidence||[]).length}</div>
      ${it.rationale ? `<div class="why-box open" style="border:none;padding-top:0.2rem;">${Orion.esc(it.rationale)}</div>` : ""}</div>`;
  }

  function render(r) {
    const st = r.system_type;
    statusEl.outerHTML = `<span class="${st==='unknown'?'off':'ok'}" id="ky-status">[+] SYSTEM TYPE: ${Orion.esc(st.toUpperCase())} · confidence ${Orion.esc(r.detection.confidence)}</span>`;
    const card = r.summary_card || {};
    let html = "";

    // Summary card
    html += `<div class="assessment-box"><div class="term-title">${st==='generative_ai'?'SYSTEM':'MODEL'} SECURITY SUMMARY</div><div class="kv">`;
    html += `<div class="k">Type</div><div class="v">${Orion.esc(card.subtype || card.model_type || st)}</div>`;
    if (card.framework) html += `<div class="k">Framework</div><div class="v">${Orion.esc(card.framework)}</div>`;
    if (card.access) html += `<div class="k">Access</div><div class="v">${Orion.esc(card.access)}</div>`;
    if (card.context_sources != null) html += `<div class="k">Context sources</div><div class="v">${card.context_sources}</div>`;
    if (card.trust_boundaries != null) html += `<div class="k">Trust boundaries</div><div class="v">${card.trust_boundaries}</div>`;
    if (card.tools != null) html += `<div class="k">Tools</div><div class="v">${card.tools} (sensitive: ${card.sensitive_tools||0})</div>`;
    if (card.mcp != null) html += `<div class="k">MCP</div><div class="v">${card.mcp?'ENABLED':'—'}</div>`;
    if (card.human_approval) html += `<div class="k">Human approval</div><div class="v">${Orion.esc(card.human_approval)}</div>`;
    html += `<div class="k">Applicable attacks</div><div class="v">${card.applicable_attacks} (conditional: ${card.conditional_attacks})</div>`;
    if (card.unknown_assumptions != null) html += `<div class="k">Unknown assumptions</div><div class="v">${card.unknown_assumptions}</div>`;
    if (card.unknown_controls != null) html += `<div class="k">Unknown controls</div><div class="v">${card.unknown_controls}</div>`;
    html += `</div>${r.detection.rationale ? `<div class="sub" style="margin-top:0.4rem;">${Orion.esc(r.detection.rationale)}</div>`:''}</div>`;

    // Traditional ML
    if (r.traditional_ml) {
      const fp = r.traditional_ml.fingerprint;
      html += `<div class="term"><div class="term-title">MODEL PROFILE (fingerprint)</div><pre class="term-pre">`
        + Orion.esc(JSON.stringify({system_type:fp.system_type,model_type:fp.model_type,framework:fp.framework,
            task:fp.task,input_type:fp.input_type,input_shape:fp.input_shape,num_classes:fp.num_classes,
            access:fp.access,supports_gradients:fp.supports_gradients,inference_method:fp.inference_method,
            model_hash:fp.model_hash?fp.model_hash.slice(0,16)+'…':null,model_size_bytes:fp.model_size_bytes}, null, 2))
        + `</pre></div>`;
      html += `<div class="term"><div class="term-title">MODEL ASSUMPTIONS</div>${r.traditional_ml.assumptions.map(item).join("")}</div>`;
      html += controlsBlock("CONTROL INVENTORY", r.traditional_ml.controls);
    }

    // Generative AI
    if (r.generative_ai) {
      const g = r.generative_ai;
      html += `<div class="term"><div class="term-title">SYSTEM PROFILE (fingerprint)</div><pre class="term-pre">`
        + Orion.esc(JSON.stringify(g.fingerprint, null, 2)) + `</pre></div>`;
      html += `<div class="term"><div class="term-title">PROMPT &amp; CONTEXT SURFACE</div>`
        + g.context_surface.map(s=>`<div class="analysis-item"><strong>${Orion.esc(s.source)}</strong> — trust: ${Orion.esc(s.trust)}</div>`).join("") + `</div>`;
      html += `<div class="term"><div class="term-title">TRUST BOUNDARIES</div>`
        + g.trust_boundaries.map(b=>`<div class="analysis-item"><strong>${Orion.esc(b.boundary)}</strong> — ${Orion.esc(b.data_crossing)} · trust ${Orion.esc(b.trust)}<div class="meta">impact: ${Orion.esc(b.impact)} · controls: ${Orion.esc(b.controls)}</div></div>`).join("") + `</div>`;
      if (g.tools.length) html += `<div class="term"><div class="term-title">TOOLS / MCP</div>`
        + g.tools.map(t=>`<div class="analysis-item"><strong>${Orion.esc(t.name)}</strong> ${t.sensitive?'<span class="proposed">SENSITIVE</span>':''}<div class="meta">permissions: ${Orion.esc(t.permissions)} · external side effects: ${t.external_side_effects} · approval: ${Orion.esc(t.approval_required)}</div></div>`).join("") + `</div>`;
      html += controlsBlock("CONTROL INVENTORY", g.controls);
    }

    // Security posture
    html += `<div class="term"><div class="term-title">SECURITY POSTURE</div><table class="orion"><thead><tr><th>Attack</th><th>Status</th><th>Rationale</th></tr></thead><tbody>`;
    r.security_posture.forEach(p => {
      html += `<tr><td>${Orion.esc(p.attack)}</td><td><span class="metric ${statusTone(p.status)}" style="display:inline-block;padding:0.05rem 0.4rem;font-size:0.68rem;">${Orion.esc(p.status.replace("_"," "))}</span></td><td class="sub">${Orion.esc(p.rationale)}${p.prerequisites_missing.length?` <span style="color:var(--muted)">(missing: ${Orion.esc(p.prerequisites_missing.join(", "))})</span>`:''}</td></tr>`;
    });
    html += `</tbody></table></div>`;

    // Recommended experiments (proposals; SEND TO ATTACK)
    html += `<div class="term"><div class="term-title">RECOMMENDED NEXT EXPERIMENTS <span class="proposed">PROPOSED</span></div>`;
    if (r.recommended_experiments.length) {
      r.recommended_experiments.forEach(x => {
        const q = "?ky=" + encodeURIComponent(r.trace_id || "") + "&attack=" + encodeURIComponent(x.attack);
        // Only evasion-family experiments map to the adversarial attack runner.
        const canRun = /evasion|pgd|fgsm|c&w|deepfool/i.test(x.attack);
        html += `<div class="exp"><span class="ename">${Orion.esc(x.attack)}</span>
          <span class="risk ${x.applicability==='HIGH'?'LOW':'MEDIUM'}">${Orion.esc(x.applicability)}</span>
          <div class="sub">${Orion.esc(x.reason)} · evidence: ${x.evidence_count} · risk: ${Orion.esc(x.risk)}</div>
          <div class="btn-row" style="margin-top:0.3rem;">
            <button class="btn btn-ghost btn-review">[ REVIEW ]</button>
            <a class="btn btn-red" href="${attackUrl}${canRun ? q : ''}">[ SEND TO ATTACK ]</a>
          </div></div>`;
      });
    } else {
      html += `<div class="state">No applicable experiments for this system with current evidence.</div>`;
    }
    html += `<p class="sub" style="color:var(--warning);margin-top:0.4rem;">⚠ Proposals only — Orion never auto-executes. Review, then send to the Attack phase.</p></div>`;

    if (r.trace_id) html += `<div class="sub">Evidence: <a href="/api/artifacts/${encodeURIComponent(r.trace_id)}/know_yourself.json" target="_blank">know_yourself.json</a></div>`;

    resultEl.innerHTML = html;
    resultEl.classList.remove("hidden");
    resultEl.querySelectorAll(".btn-review").forEach(b => b.addEventListener("click", () => {
      b.parentElement.previousElementSibling; // no-op placeholder
      b.textContent = "[ REVIEWED ]"; b.disabled = true;
    }));
    resultEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function controlsBlock(title, controls) {
    return `<div class="term"><div class="term-title">${title}</div>`
      + controls.map(c=>`<div class="analysis-item" style="border-left-color:var(--border);">${Orion.esc(c.control.replace(/_/g," "))} — ${ctrlBadge(c.status)}</div>`).join("")
      + `<p class="sub">UNKNOWN is not NOT_FOUND — it means Orion has no evidence either way.</p></div>`;
  }

  document.getElementById("ky-analyze").addEventListener("click", async function () {
    const url = (document.getElementById("ky-url").value || "").trim();
    const context = (document.getElementById("ky-context").value || "").trim();
    const active = document.getElementById("ky-active").checked;
    if (!url && !context) { Orion.setState(resultEl, "error", "Provide a URL or a description."); resultEl.classList.remove("hidden"); return; }
    Orion.setState(resultEl, "running", "profiling system…");
    resultEl.classList.remove("hidden");
    try {
      const body = { active };
      if (url) body.url = url;
      if (context) body.context = context;
      if (forceType) body.force_type = forceType;
      const r = await Orion.postJSON("/api/know-yourself/analyze", body);
      render(r);
    } catch (e) {
      Orion.setState(resultEl, "error", "[x] profile failed: " + e.message);
    }
  });
})();
