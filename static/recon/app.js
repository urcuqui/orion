// Know The Environment: launch form + live run console.
// Ported from HexAgent's web UI (github.com/urcuqui/HexAgent).
(function () {
  "use strict";

  function initLaunchForm() {
    const form = document.getElementById("launch-form");
    if (!form) return;

    const nucleiToggle = form.querySelector('input[name="enable_nuclei"]');
    const nucleiWarning = document.getElementById("nuclei-warning");
    if (nucleiToggle && nucleiWarning) {
      const syncNucleiWarning = () => {
        nucleiWarning.classList.toggle("hidden", !nucleiToggle.checked);
      };
      nucleiToggle.addEventListener("change", syncNucleiWarning);
      syncNucleiWarning();
    }

    const playwrightCheckbox = document.getElementById("playwright-checkbox");
    const playwrightSection = document.getElementById("playwright-section");
    if (playwrightCheckbox && playwrightSection) {
      const syncPlaywright = () => {
        playwrightSection.classList.toggle("hidden", !playwrightCheckbox.checked);
      };
      playwrightCheckbox.addEventListener("change", syncPlaywright);
      syncPlaywright();
    }

    const errorEl = document.getElementById("launch-error");
    const launchBtn = form.querySelector(".btn-launch");

    form.addEventListener("submit", async (evt) => {
      evt.preventDefault();
      errorEl.classList.add("hidden");
      launchBtn.disabled = true;
      launchBtn.textContent = "Launching…";
      try {
        const response = await fetch("/know-environment/run", {
          method: "POST",
          body: new FormData(form),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to start run");
        }
        window.location.href = "/know-environment/run/" + data.run_id;
      } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove("hidden");
        launchBtn.disabled = false;
        launchBtn.textContent = "Launch run";
      }
    });
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function describeEvent(event) {
    switch (event.type) {
      case "start":
        return `session started — objective="${event.objective}" target="${event.target}"`;
      case "plan":
        return `plan created (${event.step_count} step(s)) — ${event.rationale || ""}`;
      case "replan":
        return `replanned (now ${event.step_count} step(s)) — ${event.rationale || ""}`;
      case "execute":
        if (event.tool_name) {
          if (event.browser_tool) {
            return describeBrowserExecute(event);
          }
          const base = `${event.tool_name} -> ${event.status} — ${event.summary}`;
          return event.error ? `${base} (${event.error})` : base;
        }
        return event.message || "execute step";
      case "evaluate":
        if (event.message) return event.message;
        return `evaluated result — ${event.findings.length} finding(s)` +
          (event.needs_replan ? ` — requesting replan (${event.replan_reason})` : "");
      case "human_checkpoint":
        return event.message;
      case "approval_requested":
        return `approval requested for ${event.tool_name}`;
      case "approval_resolved":
        return `approval ${event.approved ? "GRANTED" : "DENIED"}`;
      case "report":
        return `report ready — stopped: ${event.stopped_reason}`;
      case "done":
        return "run finished";
      case "error":
        return `error: ${event.message}`;
      case "heartbeat":
        return "… still running";
      case "timeout":
        return "stream idle timeout — reconnect if the run is still active";
      default:
        return JSON.stringify(event);
    }
  }

  function describeBrowserExecute(event) {
    const ok = event.browser_success !== false;
    const icon = ok ? "🌐" : "✖";
    let desc = `${icon} ${event.tool_name} -> ${event.status}`;
    if (event.current_url) desc += ` | ${event.current_url}`;
    if (event.page_title) desc += ` (${event.page_title})`;
    const parts = [];
    if (event.links_count) parts.push(`${event.links_count} link(s)`);
    if (event.forms_count) parts.push(`${event.forms_count} form(s)`);
    if (event.network_count) parts.push(`${event.network_count} req(s) captured`);
    if ((event.api_endpoints || []).length) parts.push(`${event.api_endpoints.length} API hint(s)`);
    if (parts.length) desc += " | " + parts.join(", ");
    if (event.screenshot_filename) desc += ` | 📷 ${event.screenshot_filename}`;
    return desc;
  }

  function initRunView() {
    const view = document.getElementById("run-view");
    if (!view) return;

    const runId = view.dataset.runId;
    const log = document.getElementById("log");
    const statusBadge = document.getElementById("status-badge");
    const runMeta = document.getElementById("run-meta");
    const findingsList = document.getElementById("findings");
    const approvalPanel = document.getElementById("approval-panel");
    const approvalDetail = document.getElementById("approval-detail");
    const reportPanel = document.getElementById("report-panel");
    const reportHtml = document.getElementById("report-html");
    const reportRaw = document.getElementById("report-raw");
    const toggleRawBtn = document.getElementById("btn-toggle-raw");

    // Browser panel elements (may be absent in older templates).
    const browserPanel = document.getElementById("browser-panel");
    const browserStatusDot = document.getElementById("browser-status-dot");
    const browserStatusLabel = document.getElementById("browser-status-label");
    const browserSummaryEl = document.getElementById("browser-summary");
    const browserErrorsEl = document.getElementById("browser-errors");
    const browserUrlEl = document.getElementById("browser-url");
    const browserApiList = document.getElementById("browser-api-list");
    const screenshotGallery = document.getElementById("screenshot-gallery");
    const seenScreenshots = new Set();

    toggleRawBtn.addEventListener("click", () => {
      const showingRaw = !reportRaw.classList.contains("hidden");
      reportRaw.classList.toggle("hidden", showingRaw);
      reportHtml.classList.toggle("hidden", !showingRaw);
      toggleRawBtn.textContent = showingRaw ? "view raw markdown" : "view rendered report";
    });

    function setStatus(status) {
      statusBadge.className = "status-badge status-" + status;
      statusBadge.textContent = status.replace("_", " ");
    }

    function appendLog(event) {
      const line = document.createElement("div");
      let cls = "log-line type-" + event.type;
      if (event.type === "execute" && event.browser_tool) cls += " browser-event";
      line.className = cls;
      line.innerHTML = '<span class="tag">[' + event.type + "]</span> " + escapeHtml(describeEvent(event));
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
    }

    function renderFindings(findings) {
      findingsList.innerHTML = "";
      if (!findings.length) {
        findingsList.innerHTML = '<li class="finding sev-info">No findings yet.</li>';
        return;
      }
      findings.forEach((f) => {
        const li = document.createElement("li");
        li.className = "finding sev-" + f.severity;
        li.innerHTML =
          '<span class="sev sev-' + f.severity + '">' + f.severity + "</span>" + escapeHtml(f.title);
        findingsList.appendChild(li);
      });
    }

    function showApproval(event) {
      approvalDetail.textContent = event.tool_name + " " + JSON.stringify(event.arguments);
      approvalPanel.classList.remove("hidden");
      setStatus("awaiting_approval");
    }

    function hideApproval() {
      approvalPanel.classList.add("hidden");
    }

    function updateBrowserPanel(event) {
      if (!browserPanel) return;
      browserPanel.classList.remove("hidden");

      const active = event.tool_name !== "browser_close" && event.browser_success !== false;
      if (browserStatusDot) browserStatusDot.classList.toggle("active", active);
      if (browserStatusLabel) browserStatusLabel.textContent = active ? "active" : "closed";

      if (event.tool_name !== "browser_close" && event.summary && browserSummaryEl) {
        browserSummaryEl.textContent = event.summary;
      }

      const errors = event.browser_errors || [];
      if (browserErrorsEl) {
        if (errors.length) {
          browserErrorsEl.innerHTML = errors
            .map((err) => '<div class="browser-error-item">' + escapeHtml(err) + "</div>")
            .join("");
          browserErrorsEl.classList.remove("hidden");
        } else if (event.tool_name !== "browser_close") {
          browserErrorsEl.innerHTML = "";
          browserErrorsEl.classList.add("hidden");
        }
      }

      if (event.current_url && browserUrlEl) {
        browserUrlEl.textContent = event.current_url;
      }

      const endpoints = event.api_endpoints || [];
      if (endpoints.length && browserApiList) {
        browserApiList.innerHTML = endpoints
          .map((ep) => '<div class="browser-api-item">' + escapeHtml(ep) + "</div>")
          .join("");
      }

      if (event.screenshot_filename) addScreenshot(event.screenshot_filename);
    }

    function addScreenshot(filename) {
      if (!screenshotGallery || seenScreenshots.has(filename)) return;
      seenScreenshots.add(filename);
      const anchor = document.createElement("a");
      anchor.href = "/know-environment/screenshots/" + encodeURIComponent(filename);
      anchor.target = "_blank";
      anchor.title = filename;
      const img = document.createElement("img");
      img.src = "/know-environment/screenshots/" + encodeURIComponent(filename);
      img.alt = filename;
      img.className = "screenshot-thumb";
      anchor.appendChild(img);
      screenshotGallery.appendChild(anchor);
    }

    async function sendDecision(approved) {
      approvalPanel.querySelectorAll("button").forEach((b) => (b.disabled = true));
      try {
        await fetch("/know-environment/approve/" + runId, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ approved: approved }),
        });
      } finally {
        approvalPanel.querySelectorAll("button").forEach((b) => (b.disabled = false));
      }
    }

    document.getElementById("btn-approve").addEventListener("click", () => sendDecision(true));
    document.getElementById("btn-deny").addEventListener("click", () => sendDecision(false));

    const source = new EventSource("/know-environment/events/" + runId);
    source.onmessage = (msg) => {
      const event = JSON.parse(msg.data);
      appendLog(event);

      if (event.type === "start") {
        runMeta.textContent = 'objective="' + event.objective + '"  target="' + event.target + '"';
      } else if (event.type === "evaluate" && event.findings) {
        renderFindings(event.findings);
      } else if (event.type === "approval_requested") {
        showApproval(event);
      } else if (event.type === "approval_resolved") {
        hideApproval();
        setStatus("running");
      } else if (event.type === "execute" && event.browser_tool) {
        updateBrowserPanel(event);
      } else if (event.type === "report") {
        // report_html is sanitised server-side (markdown -> nh3.clean); report_markdown
        // (raw text) is only ever shown via .textContent, never innerHTML.
        reportHtml.innerHTML = event.report_html || "";
        reportRaw.textContent = event.report_markdown || "";
        reportPanel.classList.remove("hidden");
      } else if (event.type === "done") {
        setStatus("completed");
        source.close();
      } else if (event.type === "error") {
        setStatus("error");
        source.close();
      }
    };
    source.onerror = () => {
      // EventSource retries automatically; if the run already finished this is a no-op.
    };
  }

  document.addEventListener("DOMContentLoaded", () => {
    initLaunchForm();
    initRunView();
  });
})();
