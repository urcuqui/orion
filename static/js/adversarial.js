/* Adversarial ML page: configure -> run (real C&W) -> redirect to evidence view. */
(function () {
  "use strict";
  const form = document.getElementById("adv-form");
  const statusEl = document.getElementById("adv-status");
  const presetBtn = document.getElementById("btn-preset");

  function running(msg) {
    Orion.setState(statusEl, "running", msg || "");
    form.querySelectorAll("button").forEach(b => (b.disabled = true));
  }
  function idle() { form.querySelectorAll("button").forEach(b => (b.disabled = false)); }

  async function go(result) {
    // A run always produces a trace; the evidence page is the results view.
    if (result && result.trace_id) {
      statusEl.innerHTML =
        `<div class="state"><span style="color:var(--green)">Attack complete.</span> ` +
        `Status: ${Orion.statusBadge(result.status)} &mdash; opening evidence…</div>`;
      window.location.href = "/runs/" + encodeURIComponent(result.trace_id);
    }
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    const weights = document.getElementById("f-weights").files[0];
    const image = document.getElementById("f-image").files[0];
    if (!weights || !image) {
      Orion.setState(statusEl, "error", "Provide both model weights and an image, or use the demo preset.");
      return;
    }
    running("crafting adversarial example (Carlini & Wagner L2)…");
    try {
      const fd = new FormData();
      fd.append("weights", weights);
      fd.append("file", image);
      fd.append("numberoutputs", document.getElementById("f-outputs").value || "2");
      const res = await Orion.postForm("/api/adversarial/run", fd);
      await go(res);
    } catch (err) {
      Orion.setState(statusEl, "error", "Experiment failed: " + err.message);
      idle();
    }
  });

  presetBtn.addEventListener("click", async function () {
    running("running demo preset against weights/vit_teacher.pth…");
    try {
      const res = await Orion.postJSON("/api/adversarial/run", { preset: true });
      await go(res);
    } catch (err) {
      Orion.setState(statusEl, "error", "Demo preset failed: " + err.message);
      idle();
    }
  });
})();
