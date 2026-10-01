/* Orion Agent page: run supervisor (/chat or /chat_stream), visualize state.
   Timeline steps are DERIVED from actual state (todo/messages) — no fake stages. */
(function () {
  "use strict";
  const missionEl = document.getElementById("mission");
  const timelineEl = document.getElementById("agent-timeline");
  const reportPanel = document.getElementById("report-panel");
  const reportEl = document.getElementById("agent-report");
  const extras = document.getElementById("extras");
  const toolsEl = document.getElementById("agent-tools");
  const findingsEl = document.getElementById("agent-findings");
  const runBtn = document.getElementById("btn-run");

  // Prefill from query params (contextual entry points, e.g. [EXPLAIN RESULT]).
  (function prefill() {
    const q = new URLSearchParams(window.location.search);
    if (q.get("mission")) missionEl.value = q.get("mission");
    const ctxEl = document.getElementById("context");
    if (ctxEl && q.get("context")) ctxEl.value = q.get("context");
  })();

  function marker(status) {
    if (status === "done") return "[+]";
    if (status === "in_progress") return "[*]";
    if (status === "error") return "[!]";
    return "[-]";
  }

  function renderState(state) {
    if (!state) return;
    let html = "";
    html += `<div class="sub">objective: ${Orion.esc(state.objective || "")}</div>`;
    html += `<div class="sub">iteration: ${state.iteration ?? 0}${state.done ? " · done" : ""}</div>`;
    const todo = state.todo || [];
    if (todo.length) {
      html += '<div style="margin-top:0.6rem;">';
      todo.forEach(t => {
        const cls = t.status === "done" ? "type-done" : "";
        html += `<div class="log-line ${cls}" style="font-size:0.82rem; margin:0.15rem 0; color:${t.status === "done" ? "var(--green)" : "var(--muted)"}">` +
          `${marker(t.status)} ${Orion.esc(t.task)}` +
          (t.tools && t.tools.length ? ` <span style="color:var(--dim)">(${Orion.esc(t.tools.join(", "))})</span>` : "") +
          `</div>`;
      });
      html += "</div>";
    }
    const msgs = state.messages || [];
    if (msgs.length) {
      html += '<div style="margin-top:0.6rem; border-top:1px solid var(--border); padding-top:0.5rem;">';
      msgs.slice(-8).forEach(m => { html += `<div class="sub" style="margin:0.1rem 0;">› ${Orion.esc(m)}</div>`; });
      html += "</div>";
    }
    timelineEl.innerHTML = html || '<div class="state">No state.</div>';

    // Tools used + results (only when present).
    if (state.tools && state.tools.length) {
      extras.style.display = "";
      toolsEl.innerHTML = state.tools.map(t => `<div class="tool-item"><div class="tname">${Orion.esc(t)}</div></div>`).join("");
    }
    if (state.results && state.results.length) {
      extras.style.display = "";
      findingsEl.innerHTML = state.results.map(r =>
        `<div class="finding sev-info"><strong>${Orion.esc(r.task)}</strong><div class="sub">${Orion.esc((r.response || "").slice(0, 400))}</div></div>`).join("");
    }
  }

  function showReport(md) {
    if (!md) return;
    reportPanel.style.display = "";
    reportEl.innerHTML = Orion.markdown(md);
  }

  async function runStreaming(mission) {
    const resp = await fetch("/chat_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: mission }),
    });
    if (!resp.ok || !resp.body) throw new Error("stream unavailable (HTTP " + resp.status + ")");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split("\n\n");
      buf = chunks.pop();
      for (const chunk of chunks) {
        const line = chunk.split("\n").find(l => l.startsWith("data:"));
        if (!line) continue;
        try {
          const payload = JSON.parse(line.slice(5).trim());
          if (payload.state) renderState(payload.state);
          if (payload.response) showReport(payload.response);
        } catch (e) { /* ignore partial */ }
      }
    }
  }

  async function runOnce(mission) {
    const data = await Orion.postJSON("/chat", { message: mission });
    if (data.state) renderState(data.state);
    if (data.response) showReport(data.response);
  }

  runBtn.addEventListener("click", async function () {
    let mission = (missionEl.value || "").trim();
    const ctxEl = document.getElementById("context");
    const ctx = ctxEl ? (ctxEl.value || "").trim() : "";
    if (ctx) mission = "CONTEXT: " + ctx + "\n\nMISSION: " + mission;
    if (!mission) { Orion.setState(timelineEl, "error", "Enter a mission first."); return; }
    runBtn.disabled = true;
    reportPanel.style.display = "none";
    extras.style.display = "none";
    Orion.setState(timelineEl, "running", "agent planning & executing…");
    try {
      if (document.getElementById("opt-stream").checked) await runStreaming(mission);
      else await runOnce(mission);
    } catch (e) {
      Orion.setState(timelineEl, "error", "Agent run failed: " + e.message);
    } finally {
      runBtn.disabled = false;
    }
  });

  // MCP tools inspector
  document.getElementById("btn-tools").addEventListener("click", async function () {
    const panel = document.getElementById("mcp-panel");
    const list = document.getElementById("mcp-list");
    panel.classList.remove("hidden");
    Orion.setState(list, "running", "querying MCP registry…");
    try {
      const data = await Orion.getJSON("/api/mcp_tools");
      if (data.state === "ERROR") { Orion.setState(list, "error", "MCP tool registry unavailable: " + (data.error || "")); return; }
      const tools = data.tools || [];
      if (!tools.length) { Orion.setState(list, "idle", "No MCP tools available."); return; }
      list.innerHTML = tools.map(t => Orion.toolItem(t)).join("");
    } catch (e) {
      Orion.setState(list, "error", "MCP tool registry unavailable: " + e.message);
    }
  });
})();
