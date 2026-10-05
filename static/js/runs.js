/* Evidence contains Executions and Artifacts: two projections of recorded evidence.
   Both are read-only projections of the same existing persisted records. */
(function () {
  "use strict";
  const el = document.getElementById("runs-table");
  const evidenceView = el.dataset.view === "evidence";
  const search = document.getElementById("runs-search");
  const filters = { type: "all", status: "all" };
  const records = new Map();
  const findingsByRun = new Map();
  let allRuns = [], loading = true, recordErrors = 0, findingError = false;
  const query = new URLSearchParams(location.search);
  if (query.get("type")) filters.type = query.get("type");

  const runLink = id => `<a class="mono" href="/runs/${encodeURIComponent(id)}">${Orion.esc(Orion.shortId(id))}</a>`;
  const experimentId = r => ((records.get(r.trace_id) || {}).provenance || {}).experiment_workspace_id;
  const experimentLink = id => id ? `<a class="mono" href="/experiment/${encodeURIComponent(id)}">${Orion.esc(id)}</a>` : '<span class="sub">Not linked</span>';
  const findingLink = id => id ? `<a class="mono" href="/findings/${encodeURIComponent(id)}">${Orion.esc(id)}</a>` : '<span class="sub">Not linked</span>';
  const artifactURL = (id, file) => "/api/artifacts/" + encodeURIComponent(id) + "/" + encodeURIComponent(file);
  function typeOf(file) {
    const extension = String(file).split(".").pop().toLowerCase();
    return ({ json: "JSON record", png: "PNG image", md: "Markdown report" })[extension] || "Artifact";
  }
  function lineage(provenance) {
    const p = provenance || {};
    const nodes = [["System", p.self_profile_id], ["Target", p.target_profile_id],
      ["Environment", p.environment_profile_id], ["Analysis context", p.analysis_context_id],
      ["Threat model", p.threat_model_id], ["Plan", p.plan_id], ["Original run", p.retest_of], ["Control", p.defense_id]];
    const present = nodes.filter(([, value]) => value);
    if (!present.length) return '<span class="sub">Not recorded</span>';
    return `<details class="evidence-lineage"><summary>View provenance</summary><dl class="kv">${present.map(([label, value]) => `<dt class="k">${label}</dt><dd class="v">${Orion.esc(value)}</dd>`).join("")}</dl></details>`;
  }
  function visibleRuns() {
    const needle = search.value.trim().toLowerCase();
    return allRuns.filter(r => (filters.type === "all" || r.type === filters.type) &&
      (filters.status === "all" || String(r.status || "").toUpperCase() === filters.status) &&
      (!needle || [r.trace_id, r.scenario, r.target, r.technique, experimentId(r), findingsByRun.get(r.trace_id),
        ...Object.values((records.get(r.trace_id) || {}).artifacts || {})].join(" ").toLowerCase().includes(needle)));
  }
  function executionTable(runs) {
    const hasExperiments = runs.some(experimentId);
    const hasFindings = runs.some(r => findingsByRun.has(r.trace_id));
    return `<table class="orion runs-history"><caption>Executions and their recorded security results</caption><thead><tr>
      <th scope="col">Run ID</th>${hasExperiments ? '<th scope="col">Experiment</th>' : ''}<th scope="col">Scenario</th><th scope="col">Target</th><th scope="col">Result</th><th scope="col">Timestamp</th>${hasFindings ? '<th scope="col">Related finding</th>' : ''}
      </tr></thead><tbody>${runs.map(r => `<tr><td>${runLink(r.trace_id)}</td>${hasExperiments ? `<td>${experimentLink(experimentId(r))}</td>` : ''}<td>${Orion.esc(r.scenario || r.type || "UNKNOWN")}</td><td class="mono">${Orion.esc(r.target || "UNKNOWN")}</td><td>${Orion.statusBadge(r.status)}</td><td>${Orion.esc(Orion.fmtTime(r.timestamp))}</td>${hasFindings ? `<td>${findingLink(findingsByRun.get(r.trace_id))}</td>` : ''}</tr>`).join("")}</tbody></table>`;
  }
  function evidenceTable(runs) {
    const hasExperiments = runs.some(experimentId);
    const hasFindings = runs.some(r => findingsByRun.has(r.trace_id));
    const hasProvenance = runs.some(r => Object.keys((records.get(r.trace_id) || {}).provenance || {}).length);
    const rows = runs.flatMap(r => {
      const record = records.get(r.trace_id);
      if (!record) return [];
      const provenance = record.provenance || {};
      // Every loaded record has an existing experiment.json. Other artifacts
      // come only from the record's artifact map; no speculative report links.
      const artifacts = [["Execution record", "experiment.json"], ...Object.entries(record.artifacts || {})];
      return artifacts.map(([name, file]) => `<tr>
        <td><a class="evidence-artifact" href="${artifactURL(r.trace_id, file)}">${Orion.esc(name.replace(/_/g, " "))}</a><div class="sub mono">${Orion.esc(file)}</div></td>
        <td>${typeOf(file)}</td><td class="evidence-source">${Orion.esc(provenance.source_type || record.scenario_name || r.scenario || "UNKNOWN")}</td>
        <td>${runLink(r.trace_id)}</td>${hasExperiments ? `<td>${experimentLink(experimentId(r))}</td>` : ''}${hasFindings ? `<td>${findingLink(findingsByRun.get(r.trace_id))}</td>` : ''}<td>${Orion.statusBadge(r.status)}</td>${hasProvenance ? `<td>${lineage(provenance)}</td>` : ''}</tr>`);
    });
    if (!rows.length) return `<div class="state">${loading ? 'Loading artifact records…' : 'No artifact records could be loaded.'} Execution summaries remain available in <a href="/runs">Executions</a>.</div>`;
    return `<table class="orion evidence-catalog"><caption>Recorded artifacts, their sources, and originating executions</caption><thead><tr><th scope="col">Evidence / artifact</th><th scope="col">Type</th><th scope="col">Source</th><th scope="col">Run</th>${hasExperiments ? '<th scope="col">Experiment</th>' : ''}${hasFindings ? '<th scope="col">Finding</th>' : ''}<th scope="col">Result</th>${hasProvenance ? '<th scope="col">Provenance</th>' : ''}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
  }
  function render() {
    const runs = visibleRuns();
    const partial = (recordErrors || findingError) ? `<div class="state">! PARTIAL — ${recordErrors ? `${recordErrors} execution record(s) unavailable. ` : ''}${findingError ? 'Finding links unavailable. ' : ''}Available records are preserved.</div>` : '';
    if (!runs.length) {
      el.innerHTML = partial + `<div class="state">${allRuns.length ? `No matching ${evidenceView ? 'evidence' : 'runs'}. Adjust search or filters.` : 'No executions recorded. Run an approved experiment to produce evidence.'}</div>`;
      return;
    }
    el.innerHTML = partial + (evidenceView ? evidenceTable(runs) : executionTable(runs));
  }
  function bindFilters() {
    const update = () => document.querySelectorAll(".flt").forEach(b => {
      b.setAttribute("aria-pressed", String(filters[b.dataset.k] === b.dataset.v));
    });
    document.querySelectorAll(".flt").forEach(b => b.addEventListener("click", () => {
      filters[b.dataset.k] = b.dataset.v; update(); render();
    }));
    search.addEventListener("input", render); update();
  }
  (async function () {
    try {
      const data = await Orion.getJSON("/api/runs");
      allRuns = data.runs || []; bindFilters(); render();
      try {
        const d = await Orion.getJSON("/api/findings");
        (d.findings || []).forEach(f => [...(f.evidence_refs || []), ...(f.retest_refs || [])].forEach(id => findingsByRun.set(id, f.id)));
      } catch (_) { findingError = true; }
      // Both views need existing record lineage for honest related-object links.
      for (let i = 0; i < allRuns.length; i += 8) {
        const batch = allRuns.slice(i, i + 8);
        const results = await Promise.allSettled(batch.map(r => Orion.getJSON("/api/runs/" + encodeURIComponent(r.trace_id))));
        results.forEach((result, j) => { if (result.status === "fulfilled") records.set(batch[j].trace_id, result.value); else recordErrors++; });
        render();
      }
      loading = false; render();
    } catch (e) {
      Orion.setState(el, "error", `${evidenceView ? 'Evidence' : 'Runs'} unavailable: ` + e.message);
    }
  })();
})();
