/* Evidence history: list artifact-backed runs with client-side filters. */
(function () {
  "use strict";
  const el = document.getElementById("runs-table");
  let allRuns = [];
  const filters = { type: "all", status: "all" };

  // Honour ?type=... deep links from the dashboard.
  const p = new URLSearchParams(window.location.search);
  if (p.get("type")) filters.type = p.get("type");

  function render() {
    let runs = allRuns.filter(r =>
      (filters.type === "all" || r.type === filters.type) &&
      (filters.status === "all" || (r.status || "").toUpperCase() === filters.status));
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
      <td>${Orion.fmtTime(r.timestamp)}</td></tr>`).join("");
    el.innerHTML = `<table class="orion"><thead><tr>
      <th>ID</th><th>Type</th><th>Scenario</th><th>Target</th><th>Result</th><th>Timestamp</th>
      </tr></thead><tbody>${rows}</tbody></table>`;
  }

  function bindFilters() {
    document.querySelectorAll(".flt").forEach(b => {
      if (filters[b.dataset.k] === b.dataset.v) b.classList.add("active-flt");
      b.addEventListener("click", () => {
        filters[b.dataset.k] = b.dataset.v;
        document.querySelectorAll(`.flt[data-k="${b.dataset.k}"]`).forEach(x => x.style.color = "");
        b.style.color = "var(--green)";
        render();
      });
    });
  }

  (async function () {
    try {
      const data = await Orion.getJSON("/api/runs");
      allRuns = data.runs || [];
      bindFilters();
      render();
    } catch (e) {
      Orion.setState(el, "error", "Evidence unavailable: " + e.message);
    }
  })();
})();
