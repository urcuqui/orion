/* Measure: quantify an experiment's result; bridge to Defend/Evidence. */
(function () {
  "use strict";
  const root = document.getElementById("measure-root");
  const trace = root.dataset.trace || new URLSearchParams(location.search).get("trace_id");

  function metricRows(metrics) {
    const keys = ["clean_accuracy", "robust_accuracy", "attack_success_rate",
                  "perturbation_linf", "confidence_shift"];
    let rows = "";
    keys.forEach(k => {
      const m = metrics[k];
      if (m && typeof m === "object" && "value" in m)
        rows += `<tr><td>${k.replace(/_/g," ")}</td><td>${Orion.esc(m.value)}</td></tr>`;
    });
    return rows || '<tr><td colspan="2" class="sub">No numeric metrics.</td></tr>';
  }

  async function renderOne(id) {
    try {
      const r = await Orion.getJSON("/api/runs/" + encodeURIComponent(id));
      root.innerHTML = `<div class="assessment-box"><div class="term-title">RUN ${Orion.esc(id.slice(0,8))}</div>
        <div class="kv">
          <div class="k">Status</div><div class="v">${Orion.statusBadge(r.status)}</div>
          <div class="k">Attack</div><div class="v">${Orion.esc(r.attack_technique || r.scenario_name || "—")}</div>
          ${r.provenance && r.provenance.plan_id ? `<div class="k">Plan</div><div class="v">${Orion.esc(r.provenance.plan_id)}</div>` : ""}
        </div></div>
        <div class="term"><div class="term-title">METRICS</div>
          <table class="orion"><thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>${metricRows(r.metrics || {})}</tbody></table></div>
        <div class="term"><div class="btn-row">
          <a class="btn btn-blue" href="/defend/${encodeURIComponent(id)}">[ OPEN DEFEND ]</a>
          <a class="btn btn-ghost" href="/runs/${encodeURIComponent(id)}">[ VIEW EVIDENCE ]</a>
        </div></div>`;
    } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
  }

  async function renderList() {
    try {
      const data = await Orion.getJSON("/api/runs");
      const runs = (data.runs || []).slice(0, 15);
      if (!runs.length) { root.innerHTML = '<div class="state">No runs yet. Run an experiment in the Attack workspace.</div>'; return; }
      root.innerHTML = `<div class="term"><div class="term-title">Select a run to measure</div>`
        + `<table class="orion"><thead><tr><th>Run</th><th>Type</th><th>Status</th><th>Time</th></tr></thead><tbody>`
        + runs.map(r => `<tr><td><a href="/measure/${encodeURIComponent(r.trace_id)}">${Orion.shortId(r.trace_id)}</a></td>
            <td>${Orion.esc(r.type)}</td><td>${Orion.statusBadge(r.status)}</td><td>${Orion.fmtTime(r.timestamp)}</td></tr>`).join("")
        + `</tbody></table></div>`;
    } catch (e) { Orion.setState(root, "error", "[x] " + e.message); }
  }

  if (trace) renderOne(trace); else renderList();
})();
