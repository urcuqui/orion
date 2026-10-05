/* Lightweight filtering over server-rendered findings; no lifecycle decisions. */
(function () {
  "use strict";
  const filters = document.getElementById("finding-filters"); if (!filters) return;
  const rows = Array.from(document.querySelectorAll("[data-finding-row]"));
  const search = document.getElementById("finding-search");
  const selects = Array.from(filters.querySelectorAll("select"));
  selects.forEach(select => {
    Array.from(new Set(rows.map(row => row.dataset[select.dataset.filter]))).filter(Boolean).sort().forEach(value => {
      const option = document.createElement("option"); option.value = value; option.textContent = value; select.appendChild(option);
    });
  });
  function update() {
    const needle = search.value.trim().toLowerCase();
    rows.forEach(row => {
      row.hidden = !row.textContent.toLowerCase().includes(needle) || selects.some(select => select.value && row.dataset[select.dataset.filter] !== select.value);
    });
    document.getElementById("finding-count").textContent = `${rows.filter(row => !row.hidden).length} of ${rows.length} findings shown`;
  }
  search.addEventListener("input", update); selects.forEach(select => select.addEventListener("change", update)); update();
})();
