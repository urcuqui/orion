/* Adversarial ML page: configure -> run (real C&W) -> redirect to evidence view. */
(function () {
  "use strict";
  const form = document.getElementById("adv-form");
  const statusEl = document.getElementById("adv-status");
  const presetBtn = document.getElementById("btn-preset");

  // ---- Know Yourself hand-off: prefill from a profiled model fingerprint ----
  (async function kyHandoff() {
    const q = new URLSearchParams(window.location.search);
    const ky = q.get("ky");
    const attack = q.get("attack") || "evasion";
    const banner = document.getElementById("ky-handoff");
    if (!ky || !banner) return;
    try {
      const prof = await Orion.getJSON("/api/artifacts/" + encodeURIComponent(ky) + "/know_yourself.json");
      const tml = prof.traditional_ml;
      const fp = tml ? tml.fingerprint : null;
      if (!fp) { return; }
      if (fp.num_classes) document.getElementById("f-outputs").value = fp.num_classes;
      const localArtifact = fp.artifact && /\.(pt|pth)$/i.test(fp.artifact) && fp.access === "white_box";
      let html = `<div class="term-title" style="color:var(--green);">From Know Yourself · ${Orion.esc(attack)}</div>
        <div class="statusline">
          <div><span class="ok">[+]</span> model: ${Orion.esc(fp.model_type || "model")} · framework: ${Orion.esc(fp.framework)} · access: ${Orion.esc(fp.access)}</div>
          ${fp.num_classes ? `<div><span class="ok">[+]</span> classes prefilled: ${fp.num_classes}</div>` : ""}
          ${fp.artifact ? `<div><span class="ok">[+]</span> artifact: ${Orion.esc(fp.artifact)}</div>` : ""}
        </div>`;
      if (localArtifact) {
        html += `<div class="btn-row" style="margin-top:0.6rem;">
          <button class="btn btn-red" id="btn-run-profiled">[ RUN WITH PROFILED MODEL (white-box) ]</button></div>
          <p class="sub">Runs the real attack against the profiled local artifact — no re-upload needed.</p>`;
      } else {
        html += `<p class="sub">Black-box / no local artifact: upload weights + image below, or use the demo preset.</p>`;
      }
      banner.innerHTML = html;
      banner.classList.remove("hidden");
      if (localArtifact) {
        document.getElementById("btn-run-profiled").addEventListener("click", async () => {
          running("running " + attack + " against " + fp.artifact + "…");
          try {
            const res = await Orion.postJSON("/api/adversarial/run", {
              weights_path: fp.artifact, num_outputs: fp.num_classes || 2 });
            await go(res);
          } catch (err) { Orion.setState(statusEl, "error", "Run failed: " + err.message); idle(); }
        });
      }
    } catch (e) { /* profile not available; ignore */ }
  })();

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
