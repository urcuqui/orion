/* Defend: apply controls and retest the same attack (replay), then compare. */
(function () {
  "use strict";
  const root = document.getElementById("defend-root");
  const trace = root.dataset.trace || new URLSearchParams(location.search).get("trace_id");

  if (!trace) {
    root.innerHTML = '<div class="state">Open Defend from a run (Measure → [ OPEN DEFEND ]).</div>';
    return;
  }

  function fmt(v) { return (v === null || v === undefined) ? "—" : v; }

  function cmpTable(cmp) {
    const cols = cmp.columns || [];
    if (!cmp.matrix || !cmp.matrix.length) return '<div class="state">No comparable metrics.</div>';
    const head = `<tr><th>Metric</th>${cols.map(c => `<th>${Orion.esc(c)}</th>`).join("")}</tr>`;
    const statusRow = `<tr><td class="sub">status</td>${cols.map(c =>
      `<td>${Orion.statusBadge((cmp.statuses||{})[c] || "—")}</td>`).join("")}</tr>`;
    const rows = cmp.matrix.map(m => {
      const tone = { BASELINE: "col-baseline", ATTACK: "col-attack", HARDENED: "col-hardened", RETEST: "col-hardened" };
      return `<tr><td>${m.metric.replace(/_/g," ")}</td>`
        + cols.map(c => `<td class="${tone[c]||''}">${fmt(m[c])}</td>`).join("") + `</tr>`;
    }).join("");
    return `<table class="orion compare"><thead>${head}</thead><tbody>${statusRow}${rows}</tbody></table>`;
  }

  async function render() {
    let rec;
    try { rec = await Orion.getJSON("/api/runs/" + encodeURIComponent(trace)); }
    catch (e) { Orion.setState(root, "error", "[x] " + e.message); return; }
    const defenses = (rec.hardening || []).map(d => d.name || d).filter(Boolean);
    const canRetest = defenses.length > 0;
    root.innerHTML = `<div class="assessment-box"><div class="term-title">DEFEND · RUN ${Orion.esc(trace.slice(0,8))}</div>
      <div class="kv">
        <div class="k">Current status</div><div class="v">${Orion.statusBadge(rec.status)}</div>
        <div class="k">Attack</div><div class="v">${Orion.esc(rec.attack_technique || rec.scenario_name || "—")}</div>
        <div class="k">Candidate controls</div><div class="v">${defenses.length ? Orion.esc(defenses.join(", ")) : "none declared in this run"}</div>
      </div>
      <p class="sub">A defense is not validated until the attack is replayed.</p>
      <div class="btn-row">
        ${canRetest ? `<button class="btn btn-blue" id="btn-retest">[ RETEST SAME EXPERIMENT ]</button>` : `<span class="sub">This run declares no hardening defenses to retest. Use a scenario with a hardening block.</span>`}
        <a class="btn btn-ghost" href="/runs/${encodeURIComponent(trace)}">[ VIEW EVIDENCE ]</a>
      </div>
      <div id="retest-out" class="hidden" style="margin-top:0.6rem;"></div></div>`;

    const btn = document.getElementById("btn-retest");
    if (btn) btn.addEventListener("click", async () => {
      const out = document.getElementById("retest-out");
      btn.disabled = true;
      Orion.setState(out, "running", "building baseline, hardened & retest runs (same attack, different posture)…");
      out.classList.remove("hidden");
      try {
        // Same attack across postures: baseline (clean), hardened, and a retest
        // (second hardened run — deterministic, verifies reproducibility).
        const baseline = await Orion.postJSON("/api/replay/" + encodeURIComponent(trace), { mode: "baseline" });
        const hardened = await Orion.postJSON("/api/replay/" + encodeURIComponent(trace), { mode: "hardened" });
        const retest = await Orion.postJSON("/api/replay/" + encodeURIComponent(trace), { mode: "hardened" });
        const cmp = await Orion.postJSON("/api/compare", { traces: {
          BASELINE: baseline.trace_id, ATTACK: trace, HARDENED: hardened.trace_id, RETEST: retest.trace_id } });
        out.innerHTML = `<div class="term-title" style="color:var(--green)">SAME ATTACK · DIFFERENT SECURITY POSTURE</div>
          ${cmpTable(cmp)}
          <div class="btn-row" style="margin-top:0.5rem;">
            <a class="btn btn-ghost" href="/runs/${encodeURIComponent(hardened.trace_id)}">[ VIEW HARDENED EVIDENCE ]</a>
            <a class="btn btn-ghost" href="/measure/${encodeURIComponent(hardened.trace_id)}">[ MEASURE HARDENED ]</a>
          </div>`;
      } catch (e) {
        Orion.setState(out, "error", "[x] retest failed: " + e.message);
        btn.disabled = false;
      }
    });
  }
  render();
})();
