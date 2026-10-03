/* Know The Environment: landing (recon / build-from-URL) + Environment Profile view. */
(function () {
  "use strict";
  const root = document.getElementById("env-root");
  const envId = root.dataset.envId || new URLSearchParams(location.search).get("env_id");
  const targetUrl = document.body.dataset.targetUrl || "/target-analysis";
  const reconUrl = document.body.dataset.reconUrl || "/know-environment.html";

  function list(items, fn) { return (items && items.length) ? items.map(fn).join("") : '<div class="sub">none observed</div>'; }

  function topologyText(topo) {
    if (!topo || !topo.edges || !topo.edges.length) return "(no evidence-based relationships)";
    return topo.edges.map(e =>
      `${e.source}  ──${e.relationship || "→"}${e.inferred ? " (inferred)" : ""}──▶  ${e.destination}`).join("\n");
  }

  function renderProfile(p) {
    const c = p.counts || {};
    let html = `<div class="assessment-box"><div class="term-title">ORION // ENVIRONMENT PROFILE — ${Orion.esc(p.environment_profile_id)}</div>
      <div class="statusline"><span class="ok">[+] STATUS: COMPLETE</span></div>
      <pre class="term-pre">TARGET ............... ${Orion.esc(p.target_reference || "unknown")}
Assets ............... ${c.assets}
Endpoints ............ ${c.endpoints}
Technologies ......... ${c.technologies}
External Services .... ${c.external_services}
AI Dependencies ...... ${c.ai_dependencies}
MCP Servers .......... ${c.mcp_servers}
Trust Relationships .. ${c.trust_relationships}
Findings ............. ${c.findings}</pre>
      <div class="btn-row" style="margin-top:0.5rem;">
        <a class="btn btn-red" href="${targetUrl}?environment_profile_id=${encodeURIComponent(p.environment_profile_id)}">[ SEND TO TARGET ANALYSIS ]</a>
        <a class="btn btn-blue" href="/context?environment_profile_id=${encodeURIComponent(p.environment_profile_id)}">[ BUILD ANALYSIS CONTEXT ]</a>
        <a class="btn btn-ghost" href="/api/environment/${encodeURIComponent(p.environment_profile_id)}" target="_blank">[ VIEW EVIDENCE (JSON) ]</a>
      </div></div>`;

    html += `<div class="term"><div class="term-title">AI DEPENDENCIES</div>`
      + list(p.ai_dependencies, d => `<div class="analysis-item"><strong>${Orion.esc(d.name)}</strong> <span class="ttype">${Orion.esc(d.type)}</span>
          <div class="meta">trust: ${Orion.esc(d.trust)} · auth: ${Orion.esc(d.authentication)} · confidence: ${Orion.esc(d.confidence)} · evidence: ${(d.evidence||[]).length}</div></div>`)
      + `</div>`;

    html += `<div class="term"><div class="term-title">MCP / TOOL ECOSYSTEM</div>`
      + list(p.mcp_servers, d => `<div class="analysis-item"><strong>${Orion.esc(d.name)}</strong> <span class="ttype">${Orion.esc(d.type)}</span></div>`)
      + `</div>`;

    html += `<div class="term"><div class="term-title">TRUST RELATIONSHIPS</div>`
      + list(p.trust_relationships, r => `<div class="analysis-item">${Orion.esc(r.source)} ──${Orion.esc(r.relationship)}──▶ ${Orion.esc(r.destination)}
          <div class="meta">auth: ${Orion.esc(r.authentication)} · trust: ${Orion.esc(r.trust)}${r.inferred ? " · INFERRED" : ""}</div></div>`)
      + `</div>`;

    html += `<div class="term"><div class="term-title">ENVIRONMENT TOPOLOGY</div><pre class="term-pre">${Orion.esc(topologyText(p.topology))}</pre>
      <p class="sub">Only OBSERVED/INFERRED relationships are drawn; inferred ones are marked.</p></div>`;

    html += `<div class="term"><div class="term-title">TECHNOLOGIES / IDENTITY</div>
      <div class="ttypes">${(p.technologies||[]).map(t => `<span class="ttype">${Orion.esc(t)}</span>`).join("") || '<span class="sub">none</span>'}</div>
      <div class="sub" style="margin-top:0.4rem;">identity: ${(p.identity_context||[]).map(i => Orion.esc(i.type)).join(", ") || "UNKNOWN"}</div></div>`;

    root.innerHTML = html;
  }

  async function renderLanding() {
    let recon = [];
    try { recon = (await Orion.getJSON("/api/recon/runs")).runs || []; } catch (e) {}
    let html = `<div class="term"><div class="term-title">Build an Environment Profile</div>
      <p class="sub">Recon collects → observations → environment evidence → Environment Profile.</p>
      <div class="btn-row"><a class="btn btn-red" href="${reconUrl}">[ LAUNCH RECONNAISSANCE ]</a></div>
      <hr class="term-rule">
      <div class="panel-title">Build from a running URL (direct probe)</div>
      <div class="btn-row">
        <input type="text" id="env-url" placeholder="http://127.0.0.1:5001" style="flex:1;min-width:220px;">
        <label class="toggle" style="margin:0 0.5rem;"><input type="checkbox" id="env-active"> active</label>
        <button class="btn" id="env-probe">[ BUILD PROFILE ]</button>
      </div></div>`;
    html += `<div class="term"><div class="term-title">Recent recon runs</div>`;
    if (recon.length) {
      html += recon.map(r => `<div class="recon-run-row"><div class="meta"><span class="rid">${Orion.esc(r.display_id)}</span> · ${Orion.esc(r.target)} · ${Orion.statusBadge(r.status)}</div>
        <button class="btn btn-ghost btn-env" data-run="${Orion.esc(r.run_id)}">[ BUILD ENVIRONMENT PROFILE ]</button></div>`).join("");
    } else { html += '<div class="state">No recon runs yet. Launch reconnaissance above.</div>'; }
    html += `</div>`;
    root.innerHTML = html;

    document.getElementById("env-probe").addEventListener("click", async () => {
      const url = document.getElementById("env-url").value.trim();
      if (!url) { return; }
      Orion.setState(root, "running", "probing & building environment profile…");
      try {
        const p = await Orion.postJSON("/api/environment/from-url", { url, active: document.getElementById("env-active") && document.getElementById("env-active").checked });
        location.href = "/environment/" + encodeURIComponent(p.environment_profile_id);
      } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
    });
    root.querySelectorAll(".btn-env").forEach(b => b.addEventListener("click", async () => {
      Orion.setState(root, "running", "normalizing recon into environment profile…");
      try {
        const p = await Orion.postJSON("/api/environment/from-recon/" + encodeURIComponent(b.dataset.run), {});
        location.href = "/environment/" + encodeURIComponent(p.environment_profile_id);
      } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
    }));
  }

  if (envId) {
    Orion.getJSON("/api/environment/" + encodeURIComponent(envId))
      .then(renderProfile)
      .catch(e => Orion.setState(root, "error", "[x] " + e.message));
  } else {
    renderLanding();
  }
})();
