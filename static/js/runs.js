/* Evidence history: list artifact-backed runs with client-side filters. */
(function () {
  "use strict";
  const el = document.getElementById("runs-table");
  let allRuns = [];
  const filters = { type: "all", status: "all" };
  const evidenceView = el.dataset.view === "evidence";
  const search = document.getElementById("runs-search");
  const records = new Map();
  let partial = 0;
  const findingsByRun = new Map();

  // Honour ?type=... deep links from the dashboard.
  const p = new URLSearchParams(window.location.search);
  if (p.get("type")) filters.type = p.get("type");

  function render() {
    let runs = allRuns.filter(r =>
      (filters.type === "all" || r.type === filters.type) &&
      (filters.status === "all" || (r.status || "").toUpperCase() === filters.status) &&
      (!search.value || [r.trace_id,r.scenario,r.target,r.technique].join(" ").toLowerCase().includes(search.value.toLowerCase())));
    if (!runs.length) {
      el.innerHTML = '<div class="state">No matching runs. Launch an experiment to generate evidence.</div>';
      return;
    }
    const rows = runs.map(r => `<tr>
      <td><a href="/runs/${encodeURIComponent(r.trace_id)}">${Orion.shortId(r.trace_id)}</a></td>
      <td>${Orion.esc(r.type)}</td>
      <td>${Orion.esc(r.scenario || "—")}</td>
      <td>${Orion.esc(r.target || "—")}</td>
      <td>${Orion.statusBadge(r.status)}</td>
      <td>${Orion.esc(Orion.fmtTime(r.timestamp))}</td>
      ${evidenceView ? `<td>${artifactLinks(r.trace_id)}</td>` : ""}</tr>`).join("");
    el.innerHTML = (partial ? `<div class="state">! PARTIAL — ${partial} evidence records could not be loaded. Available execution summaries remain visible.</div>` : "") + `<table class="orion"><thead><tr>
      <th>Originating run</th><th>Type</th><th>Scenario</th><th>Target</th><th>Result</th><th>Timestamp</th>${evidenceView ? "<th>Evidence artifacts</th>" : ""}
      </tr></thead><tbody>${rows}</tbody></table>`;
  }

  function artifactLinks(id) {
    const record = records.get(id);
    if (!record) return '<span class="sub">? Record not loaded</span>';
    const base = "/api/artifacts/" + encodeURIComponent(id) + "/";
    const finding = findingsByRun.get(id);
    const prov = record.provenance || {};
    const exp = prov.experiment_workspace_id;
    const related = (finding ? ` · <a href="/findings/${encodeURIComponent(finding)}">Finding</a>` : "") + (exp ? ` · <a href="/experiment/${encodeURIComponent(exp)}">Experiment</a>` : "");
    return related + ` · <a href="${base}experiment.json">Execution record</a>` + Object.entries(record.artifacts || {}).map(([label, file]) =>
      ` · <a href="${base + encodeURIComponent(file)}">${Orion.esc(label)}</a>`).join("");
  }
  function bindFilters() {
    const update = () => document.querySelectorAll(".flt").forEach(b => {
      b.setAttribute("aria-pressed", String(filters[b.dataset.k] === b.dataset.v));
      b.classList.toggle("active-flt", filters[b.dataset.k] === b.dataset.v);
    });
    document.querySelectorAll(".flt").forEach(b => b.addEventListener("click", () => {
      filters[b.dataset.k] = b.dataset.v; update(); render();
    }));
    update(); search.addEventListener("input", render);
  }

  (async function () {
    try {
      const data = await Orion.getJSON("/api/runs");
      allRuns = data.runs || [];
      bindFilters();
      render();
      if (evidenceView) {
        try {
          const d = await Orion.getJSON("/api/findings");
          (d.findings || []).forEach(f => [...(f.evidence_refs || []), ...(f.retest_refs || [])].forEach(id => findingsByRun.set(id, f.id)));
        } catch (_) { partial++; }
        // Bounded concurrent reads of existing records; no evidence copies.
        for (let i = 0; i < allRuns.length; i += 8) {
          const batch = allRuns.slice(i, i + 8);
          const results = await Promise.allSettled(batch.map(r => Orion.getJSON("/api/runs/" + encodeURIComponent(r.trace_id))));
          results.forEach((result, j) => { if (result.status === "fulfilled") records.set(batch[j].trace_id, result.value); else partial++; });
        }
        render();
      }
    } catch (e) {
      Orion.setState(el, "error", "Evidence unavailable: " + e.message);
    }
  })();
})();
