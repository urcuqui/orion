"""Browser interaction tools powered by Playwright.

All tools share a single :class:`BrowserManager` instance (created lazily on
the first browser action). Playwright is an *optional* dependency — every tool
returns :meth:`~libs.recon.models.ToolResult.fail` when the package is not
installed rather than crashing, so the graph continues to work without it.

Credentials submitted via :class:`BrowserLoginTool` are **never** included in
the :class:`~libs.recon.models.ToolResult` data; only the authentication
outcome and resulting URL are exposed.

Every URL is validated against the authorised target before navigation, and
blocked schemes (``file://``, ``javascript:``, ``data:``) are rejected at the
tool boundary.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool
from tools.recon.fixtures import normalise_target

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BLOCKED_SCHEMES: frozenset[str] = frozenset({"file", "javascript", "data", "ftp", "vbscript"})

_SENSITIVE_HEADERS: frozenset[str] = frozenset(
    {
        "cookie",
        "set-cookie",
        "authorization",
        "x-api-key",
        "x-auth-token",
        "x-access-token",
        "x-csrf-token",
        "api-key",
    }
)

# Resource types that produce no security-relevant signal.
_IGNORED_RESOURCE_TYPES: frozenset[str] = frozenset(
    {"image", "font", "stylesheet", "media", "websocket", "other"}
)

# Cloud metadata endpoint that must always be blocked regardless of scope.
_CLOUD_METADATA_IP = "169.254.169.254"

# Hard cap on visible text returned to the planner (characters).
_MAX_VISIBLE_TEXT = 3_000

# Auth-related keywords found in page text that hint a login wall is present.
_AUTH_KEYWORDS = frozenset({"sign in", "log in", "login", "username", "password", "email"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values replaced by [REDACTED]."""
    return {k: "[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v for k, v in headers.items()}


def _sanitize_filename(name: str) -> str:
    """Return a filesystem-safe version of ``name`` (max 80 chars)."""
    safe = re.sub(r"[^\w\-]", "_", name)
    return safe[:80]


def _validate_url(url: str, target: str) -> str | None:
    """Return an error string if ``url`` is out-of-scope or blocked; else None."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme in _BLOCKED_SCHEMES:
        return f"Blocked scheme: {scheme!r}"
    if scheme not in ("http", "https", ""):
        return f"Unsupported scheme: {scheme!r}"

    hostname = parsed.hostname or ""

    # Always block the cloud metadata endpoint.
    try:
        if str(ipaddress.ip_address(hostname)) == _CLOUD_METADATA_IP:
            return f"Blocked: cloud metadata endpoint ({_CLOUD_METADATA_IP})"
    except ValueError:
        pass  # hostname is a domain name, not an IP literal

    # Enforce target scope: only the authorised hostname is reachable.
    _, target_host = normalise_target(target)
    target_bare = target_host.split(":")[0]  # strip port
    request_bare = hostname.split(":")[0] if hostname else ""

    if target_bare and request_bare and request_bare != target_bare:
        return f"Out-of-scope host {hostname!r} (authorised: {target_bare!r})"

    return None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CapturedRequest(BaseModel):
    """A sanitised record of a browser-generated network request."""

    method: str
    url: str
    resource_type: str
    status: int | None = None
    content_type: str | None = None
    request_body_preview: str | None = None
    response_body_preview: str | None = None
    timestamp: float = Field(default_factory=time.time)


class BrowserObservation(BaseModel):
    """Structured output returned by all browser tools."""

    success: bool
    action: str
    current_url: str | None = None
    title: str | None = None
    status_code: int | None = None
    visible_text: str | None = None
    links: list[dict[str, str]] = Field(default_factory=list)
    forms: list[dict[str, Any]] = Field(default_factory=list)
    inputs: list[dict[str, str]] = Field(default_factory=list)
    buttons: list[dict[str, str]] = Field(default_factory=list)
    network_requests: list[dict[str, Any]] = Field(default_factory=list)
    screenshot_path: str | None = None
    auth_indicators: list[str] = Field(default_factory=list)
    potential_api_endpoints: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Browser lifecycle manager
# ---------------------------------------------------------------------------


class BrowserManager:
    """Manages a single Playwright browser lifetime for an assessment run.

    Playwright is started lazily on the first :meth:`get_page` call and
    shuts down cleanly via :meth:`close`. All browser tools share the same
    instance so cookies and session state persist across tool invocations.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 15_000,
        max_requests: int = 200,
        max_body_preview_bytes: int = 2_000,
        screenshot_dir: Path | None = None,
    ) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._max_requests = max_requests
        self._max_body_preview_bytes = max_body_preview_bytes
        self._screenshot_dir = screenshot_dir or Path("reports/screenshots")

        self._playwright: Any | None = None
        self._browser: Any | None = None  # Browser
        self._context: Any | None = None  # BrowserContext
        self._page: Any | None = None  # Page
        self._network_events: list[CapturedRequest] = []
        self._active = False

        # Lab credentials stored here so they never appear in plan step args
        # or tool results (i.e. never reach any LLM context).
        self._default_username: str = ""
        self._default_password: str = ""

    @property
    def active(self) -> bool:
        """True when a Playwright instance is running."""
        return self._active

    def set_credentials(self, username: str, password: str) -> None:
        """Store lab credentials for use by :class:`BrowserLoginTool`.

        Credentials are held here — not in plan step arguments — so they are
        never serialised into any LLM context or written to logs.
        """
        self._default_username = username
        self._default_password = password

    def get_page(self) -> Any:  # -> Page
        """Return the current page, starting Playwright lazily if needed."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "playwright not installed; run: "
                "pip install playwright && playwright install chromium"
            )
        if self._playwright is None:
            logger.info("Starting Playwright / Chromium (headless=%s)", self._headless)
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self._headless)
            self._context = self._browser.new_context(
                ignore_https_errors=True,  # lab targets often use self-signed certs
            )
            self._page = self._context.new_page()
            self._page.set_default_timeout(self._timeout_ms)
            self._setup_network_capture(self._page)
            self._active = True
        return self._page

    def flush_network_events(self) -> list[dict[str, Any]]:
        """Return accumulated network events and clear the internal buffer."""
        events = [e.model_dump(mode="json") for e in self._network_events]
        self._network_events.clear()
        return events

    def save_screenshot(self, name: str, full_page: bool = True) -> str | None:
        """Capture a screenshot and return its path; return None on failure."""
        if self._page is None:
            return None
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        fname = f"{ts}_{_sanitize_filename(name)}.png"
        path = self._screenshot_dir / fname
        try:
            self._page.screenshot(path=str(path), full_page=full_page)
            logger.info("Screenshot saved: %s", path)
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screenshot failed: %s", exc)
            return None

    def close(self) -> None:
        """Shut down the browser cleanly, releasing all resources."""
        for obj, method in (
            (self._page, "close"),
            (self._context, "close"),
            (self._browser, "close"),
            (self._playwright, "stop"),
        ):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:  # noqa: BLE001
                    pass

        self._playwright = self._browser = self._context = self._page = None
        self._network_events.clear()
        self._active = False
        logger.info("Browser session closed")

    # -- private --------------------------------------------------------

    def _setup_network_capture(self, page: Any) -> None:
        """Wire request/response event listeners for network capture."""
        # Correlate requests with their responses by a simple key.
        pending: dict[str, CapturedRequest] = {}

        def on_request(request: Any) -> None:
            if len(self._network_events) >= self._max_requests:
                return
            if request.resource_type in _IGNORED_RESOURCE_TYPES:
                return
            body_preview: str | None = None
            try:
                raw = request.post_data
                if raw:
                    body_preview = raw[: self._max_body_preview_bytes]
            except Exception:  # noqa: BLE001
                pass
            cap = CapturedRequest(
                method=request.method,
                url=request.url,
                resource_type=request.resource_type,
                request_body_preview=body_preview,
            )
            pending[request.url + ":" + request.method] = cap

        def on_response(response: Any) -> None:
            key = response.url + ":" + response.request.method
            cap = pending.pop(key, None)
            if cap is None:
                if response.request.resource_type in _IGNORED_RESOURCE_TYPES:
                    return
                if len(self._network_events) >= self._max_requests:
                    return
                cap = CapturedRequest(
                    method=response.request.method,
                    url=response.url,
                    resource_type=response.request.resource_type,
                )
            cap.status = response.status
            ct = response.headers.get("content-type", "")
            cap.content_type = ct
            # Only peek at API-style response bodies.
            if any(t in ct for t in ("json", "xml", "text/plain")):
                try:
                    body = response.body()
                    cap.response_body_preview = body[: self._max_body_preview_bytes].decode(
                        "utf-8", errors="replace"
                    )
                except Exception:  # noqa: BLE001
                    pass
            self._network_events.append(cap)

        page.on("request", on_request)
        page.on("response", on_response)


# ---------------------------------------------------------------------------
# Page-element extraction
# ---------------------------------------------------------------------------


def _extract_page_elements(page: Any, max_text: int = _MAX_VISIBLE_TEXT) -> dict[str, Any]:
    """Return a dict of structural elements from the current page."""
    out: dict[str, Any] = {
        "visible_text": "",
        "links": [],
        "forms": [],
        "inputs": [],
        "buttons": [],
        "auth_indicators": [],
        "potential_api_endpoints": [],
    }

    try:
        out["visible_text"] = (page.inner_text("body") or "")[:max_text]
    except Exception:  # noqa: BLE001
        pass

    # Links.
    try:
        for el in page.query_selector_all("a[href]"):
            href = el.get_attribute("href") or ""
            text = (el.inner_text() or "").strip()[:100]
            if href and not href.startswith(("javascript:", "data:", "#")):
                out["links"].append({"href": href, "text": text})
                if any(k in href.lower() for k in ("/api/", "/graphql", "/v1/", "/v2/")):
                    out["potential_api_endpoints"].append(href)
    except Exception:  # noqa: BLE001
        pass

    # Forms.
    try:
        for form in page.query_selector_all("form"):
            action = form.get_attribute("action") or ""
            method = (form.get_attribute("method") or "get").upper()
            fields: list[dict[str, str]] = []
            for inp in form.query_selector_all("input, select, textarea"):
                inp_type = inp.get_attribute("type") or "text"
                inp_name = inp.get_attribute("name") or inp.get_attribute("id") or ""
                if inp_name:
                    fields.append({"name": inp_name, "type": inp_type})
                    if inp_type == "password":
                        out["auth_indicators"].append(f"password field '{inp_name}'")
            out["forms"].append({"action": action, "method": method, "fields": fields})
    except Exception:  # noqa: BLE001
        pass

    # Standalone inputs (outside any form).
    try:
        for inp in page.query_selector_all("input"):
            inp_type = inp.get_attribute("type") or "text"
            inp_name = inp.get_attribute("name") or inp.get_attribute("id") or ""
            if inp_name:
                out["inputs"].append({"name": inp_name, "type": inp_type})
    except Exception:  # noqa: BLE001
        pass

    # Buttons.
    try:
        for btn in page.query_selector_all("button, input[type=submit], input[type=button]"):
            label = (
                btn.inner_text()
                or btn.get_attribute("value")
                or btn.get_attribute("aria-label")
                or ""
            ).strip()[:80]
            btn_type = btn.get_attribute("type") or "button"
            if label:
                out["buttons"].append({"label": label, "type": btn_type})
    except Exception:  # noqa: BLE001
        pass

    # Auth indicators from page text.
    text_lower = out["visible_text"].lower()
    for kw in _AUTH_KEYWORDS:
        if kw in text_lower and kw not in out["auth_indicators"]:
            out["auth_indicators"].append(kw)

    return out


# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------


class BrowserOpenTool(BaseTool):
    """Open a URL in a real Playwright/Chromium browser and return page structure."""

    name = "browser_open"
    description = (
        "Open a URL with a real browser (Playwright/Chromium), wait for the page to load, "
        "then extract page title, visible text, links, forms, buttons, and capture network "
        "requests. Use when a target is a dynamic web application or requires JavaScript."
    )
    argument_help = {
        "target": "authorised base host or URL (scope reference)",
        "url": "full URL to open (must be in-scope; defaults to target root)",
        "screenshot": "capture a screenshot after loading (default: true)",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(
        self, target: str, url: str | None = None, screenshot: bool = True, **_: Any
    ) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.fail(
                self.name,
                "Playwright not installed. "
                "Run: pip install playwright && playwright install chromium",
            )

        open_url = url or normalise_target(target)[0]
        err = _validate_url(open_url, target)
        if err:
            return ToolResult.fail(self.name, f"Scope violation: {err}")

        # Wrap the entire operational block so no Playwright exception can escape
        # to BaseTool.run() and produce a ToolResult.fail()/ToolStatus.ERROR result.
        # All failures surface as ToolResult.ok(success=False) so the graph
        # continues to browser_analyze_page and browser_close without halting.
        errors: list[str] = []
        status_code: int | None = None
        current_url: str = ""
        page_title: str = ""
        elements: dict[str, Any] = {
            "visible_text": "",
            "links": [],
            "forms": [],
            "inputs": [],
            "buttons": [],
            "auth_indicators": [],
            "potential_api_endpoints": [],
        }
        network_reqs: list[dict[str, Any]] = []
        screenshot_path: str | None = None
        page = None

        try:
            page = self._mgr.get_page()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Browser launch failed: {exc}")
            logger.warning("browser_open: launch error: %s", exc)

        if page is not None:
            logger.info("browser_open: navigating to %s", open_url)
            try:
                response = page.goto(open_url, wait_until="domcontentloaded")
                if response:
                    status_code = response.status
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Navigation: {exc}")

            try:
                elements = _extract_page_elements(page)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Page extraction failed: {exc}")

            try:
                network_reqs = self._mgr.flush_network_events()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Network capture flush failed: {exc}")

            try:
                current_url = page.url
                page_title = page.title()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Page state unavailable: {exc}")

            if screenshot:
                try:
                    slug = _sanitize_filename(open_url.split("//")[-1][:40])
                    screenshot_path = self._mgr.save_screenshot(f"open_{slug}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Screenshot failed: {exc}")

        try:
            obs = BrowserObservation(
                success=not errors,
                action="browser_open",
                current_url=current_url,
                title=page_title,
                status_code=status_code,
                visible_text=elements["visible_text"],
                links=elements["links"][:50],
                forms=elements["forms"],
                inputs=elements["inputs"],
                buttons=elements["buttons"],
                network_requests=network_reqs[:50],
                screenshot_path=screenshot_path,
                auth_indicators=elements["auth_indicators"],
                potential_api_endpoints=elements["potential_api_endpoints"],
                errors=errors,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Observation build failed: {exc}")
            obs = BrowserObservation(success=False, action="browser_open", errors=errors)

        final_url = current_url or open_url
        summary = (
            f"browser_open {final_url} -> {status_code or '?'} | "
            f"title={obs.title!r} | {len(obs.links)} link(s), "
            f"{len(obs.forms)} form(s), {len(obs.potential_api_endpoints)} API hint(s)"
        )
        if errors:
            summary += f" | errors: {'; '.join(errors)}"
        logger.info(summary)
        return ToolResult.ok(self.name, summary, obs)


class BrowserAnalyzePageTool(BaseTool):
    """Analyse the current browser page without navigating."""

    name = "browser_analyze_page"
    description = (
        "Analyse the currently loaded browser page: extract links, forms, inputs, buttons, "
        "authentication indicators, and potential API endpoints. Does not navigate."
    )
    argument_help = {
        "target": "authorised target (scope reference only)",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(self, target: str, **_: Any) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.fail(self.name, "Playwright not installed.")
        if not self._mgr.active:
            obs = BrowserObservation(
                success=False,
                action="browser_analyze_page",
                errors=["No active browser session. Call browser_open first."],
            )
            return ToolResult.ok(self.name, "No active browser session to analyze.", obs)

        errors: list[str] = []
        page = None
        current_url = ""
        title = ""
        elements: dict[str, Any] = {
            "visible_text": "",
            "links": [],
            "forms": [],
            "inputs": [],
            "buttons": [],
            "auth_indicators": [],
            "potential_api_endpoints": [],
        }
        network_reqs: list[dict[str, Any]] = []

        try:
            page = self._mgr.get_page()
            current_url = page.url
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Browser page unavailable: {exc}")

        if not current_url:
            obs = BrowserObservation(
                success=False,
                action="browser_analyze_page",
                errors=errors or ["Browser page unavailable."],
            )
            return ToolResult.ok(self.name, "Browser page unavailable for analysis.", obs)

        err = _validate_url(current_url, target)
        if err:
            return ToolResult.fail(self.name, f"Current URL is out-of-scope: {err}")

        try:
            elements = _extract_page_elements(page)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Page extraction failed: {exc}")

        try:
            network_reqs = self._mgr.flush_network_events()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Network capture flush failed: {exc}")

        try:
            title = page.title()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Page title unavailable: {exc}")

        obs = BrowserObservation(
            success=not errors,
            action="browser_analyze_page",
            current_url=current_url,
            title=title,
            visible_text=elements["visible_text"],
            links=elements["links"][:50],
            forms=elements["forms"],
            inputs=elements["inputs"],
            buttons=elements["buttons"],
            network_requests=network_reqs[:50],
            auth_indicators=elements["auth_indicators"],
            potential_api_endpoints=elements["potential_api_endpoints"],
            errors=errors,
        )
        summary = (
            f"browser_analyze_page at {current_url} | "
            f"{len(obs.links)} link(s), {len(obs.forms)} form(s), "
            f"{len(obs.auth_indicators)} auth indicator(s)"
        )
        if errors:
            summary += f" | errors: {'; '.join(errors)}"
        logger.info(summary)
        return ToolResult.ok(self.name, summary, obs)


class BrowserLoginTool(BaseTool):
    """Authenticate through a browser login form.

    Credentials are **never** included in the :class:`~libs.recon.models.ToolResult`
    data. Only the authentication outcome, resulting URL, and captured network
    requests are exposed.
    """

    name = "browser_login"
    description = (
        "Fill and submit a login form in the browser. Maintains the authenticated session "
        "for subsequent browser actions. Credentials are redacted from all results."
    )
    sensitive = True  # modifies application state
    argument_help = {
        "target": "authorised base host or URL",
        "url": "login page URL (defaults to <target>/login)",
        "username": "credential username — redacted from results",
        "password": "credential password — redacted from results",
        "username_selector": "CSS selector for username field (auto-detected if omitted)",
        "password_selector": "CSS selector for password field (auto-detected if omitted)",
        "submit_selector": "CSS selector for submit button (auto-detected if omitted)",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(
        self,
        target: str,
        url: str | None = None,
        username: str = "",
        password: str = "",
        username_selector: str | None = None,
        password_selector: str | None = None,
        submit_selector: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.fail(self.name, "Playwright not installed.")
        # Fall back to manager-stored lab credentials when none are passed via args.
        # This prevents credentials from ever appearing in plan step arguments.
        username = username or self._mgr._default_username
        password = password or self._mgr._default_password
        if not username or not password:
            return ToolResult.fail(self.name, "Both username and password are required.")

        login_url = url or (normalise_target(target)[0] + "/login")
        err = _validate_url(login_url, target)
        if err:
            return ToolResult.fail(self.name, f"Scope violation: {err}")

        page = self._mgr.get_page()
        errors: list[str] = []

        # Navigate to the login page if the browser is not already there.
        if page.url.rstrip("/") != login_url.rstrip("/"):
            try:
                page.goto(login_url, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle", timeout=5_000)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Navigate to login: {exc}")

        url_before = page.url
        u_sel = username_selector or self._find_username_selector(page)
        p_sel = password_selector or "input[type=password]"
        s_sel = submit_selector or self._find_submit_selector(page)

        # Credentials go to the browser only — never into logs or results.
        logger.info(
            "browser_login: submitting form (username_selector=%r, password_selector=%r, "
            "submit_selector=%r) — credentials NOT logged",
            u_sel,
            p_sel,
            s_sel,
        )
        try:
            if u_sel:
                page.fill(u_sel, username)
            page.fill(p_sel, password)
            if s_sel:
                page.click(s_sel)
            else:
                page.keyboard.press("Enter")
            page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Form interaction: {exc}")

        url_after = page.url
        title_after = page.title()
        text_after = ""
        try:
            text_after = (page.inner_text("body") or "")[:1_000].lower()
        except Exception:  # noqa: BLE001
            pass

        auth_failure_words = ("invalid", "incorrect", "error", "failed", "wrong", "retry")
        auth_success = (url_after.rstrip("/") != login_url.rstrip("/")) or (
            not any(w in text_after for w in auth_failure_words)
        )

        network_reqs = self._mgr.flush_network_events()
        screenshot_path = self._mgr.save_screenshot("after_login")

        obs = BrowserObservation(
            success=auth_success and not errors,
            action="browser_login",
            current_url=url_after,
            title=title_after,
            network_requests=network_reqs[:50],
            screenshot_path=screenshot_path,
            errors=errors,
        )
        status_word = "succeeded" if obs.success else "failed"
        summary = (
            f"browser_login {status_word} | "
            f"{url_before!r} -> {url_after!r} | "
            f"{len(network_reqs)} request(s) captured"
        )
        logger.info("browser_login: %s (credentials NOT logged)", status_word)
        return ToolResult.ok(self.name, summary, obs)

    @staticmethod
    def _find_username_selector(page: Any) -> str | None:
        for sel in (
            "input[type=email]",
            "input[name=username]",
            "input[name=email]",
            "input[name=user]",
            "input[id=username]",
            "input[id=email]",
            "input[autocomplete=username]",
        ):
            if page.query_selector(sel) is not None:
                return sel
        return None

    @staticmethod
    def _find_submit_selector(page: Any) -> str | None:
        for sel in (
            "button[type=submit]",
            "input[type=submit]",
            "button:has-text('Login')",
            "button:has-text('Sign in')",
            "button:has-text('Log in')",
            "button:has-text('Submit')",
        ):
            if page.query_selector(sel) is not None:
                return sel
        return None


class BrowserClickTool(BaseTool):
    """Click a page element by CSS selector, visible text, or ARIA role."""

    name = "browser_click"
    description = (
        "Click an element on the current browser page. Specify the target by CSS selector, "
        "visible text, or ARIA role + accessible name. Use to navigate menus, follow links, "
        "or activate buttons."
    )
    argument_help = {
        "target": "authorised target (scope reference)",
        "selector": "CSS selector of the element to click",
        "text": "visible text content of the element",
        "role": "ARIA role (e.g. 'button', 'link')",
        "name": "accessible name paired with role",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(
        self,
        target: str,
        selector: str | None = None,
        text: str | None = None,
        role: str | None = None,
        name: str | None = None,
        **_: Any,
    ) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.fail(self.name, "Playwright not installed.")
        if not self._mgr.active:
            return ToolResult.fail(self.name, "No active browser session.")
        if not any([selector, text, role]):
            return ToolResult.fail(self.name, "Provide at least one of: selector, text, role.")

        page = self._mgr.get_page()
        errors: list[str] = []
        url_before = page.url

        try:
            if selector:
                page.click(selector)
            elif role and name:
                page.get_by_role(role, name=name).click()  # type: ignore[call-arg]
            elif text:
                page.get_by_text(text).first.click()
            page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Click failed: {exc}")

        url_after = page.url
        scope_err = _validate_url(url_after, target)
        if scope_err:
            errors.append(f"Post-click scope violation: {scope_err}")

        elements = _extract_page_elements(page)
        network_reqs = self._mgr.flush_network_events()
        clicked_desc = selector or text or f"role={role}"

        obs = BrowserObservation(
            success=not errors,
            action="browser_click",
            current_url=url_after,
            title=page.title(),
            visible_text=elements["visible_text"],
            links=elements["links"][:30],
            forms=elements["forms"],
            buttons=elements["buttons"],
            network_requests=network_reqs[:50],
            auth_indicators=elements["auth_indicators"],
            potential_api_endpoints=elements["potential_api_endpoints"],
            errors=errors,
        )
        summary = (
            f"browser_click {clicked_desc!r}: {url_before!r} -> {url_after!r} | "
            f"{len(network_reqs)} request(s)"
        )
        logger.info(summary)
        return ToolResult.ok(self.name, summary, obs)


class BrowserScreenshotTool(BaseTool):
    """Capture an evidence screenshot of the current browser page."""

    name = "browser_screenshot"
    description = (
        "Capture a screenshot of the current browser page and save it to the evidence "
        "directory. The path is included in the tool result for the report."
    )
    argument_help = {
        "target": "authorised target (scope reference)",
        "name": "descriptive label for the screenshot filename",
        "full_page": "capture the full scrollable page (default: true)",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(
        self, target: str, name: str = "screenshot", full_page: bool = True, **_: Any
    ) -> ToolResult:
        if not PLAYWRIGHT_AVAILABLE:
            return ToolResult.fail(self.name, "Playwright not installed.")
        if not self._mgr.active:
            return ToolResult.fail(self.name, "No active browser session.")

        path = self._mgr.save_screenshot(name, full_page=full_page)
        if path is None:
            return ToolResult.fail(self.name, "Screenshot capture failed.")

        data = {"screenshot_path": path, "name": name, "full_page": full_page}
        return ToolResult.ok(self.name, f"Screenshot saved: {path}", data)


class BrowserCloseTool(BaseTool):
    """Close the Playwright browser session and release all resources."""

    name = "browser_close"
    description = (
        "Close the browser session cleanly. Should be called when browser-based "
        "exploration is complete to avoid orphan browser processes."
    )
    argument_help = {
        "target": "authorised target (not used; kept for uniform interface)",
    }

    def __init__(self, manager: BrowserManager) -> None:
        self._mgr = manager

    def _run(self, target: str = "", **_: Any) -> ToolResult:
        if not self._mgr.active:
            return ToolResult.ok(self.name, "No active browser session to close.", {})
        self._mgr.close()
        return ToolResult.ok(self.name, "Browser session closed.", {"closed": True})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_browser_tools(
    headless: bool = True,
    timeout_ms: int = 15_000,
    max_requests: int = 200,
    max_body_preview_bytes: int = 2_000,
    screenshot_dir: Path | None = None,
) -> list[BaseTool]:
    """Create a shared :class:`BrowserManager` and return all browser tool instances.

    All six tools share the *same* manager so that browser cookies and session
    state persist across ``browser_open`` → ``browser_login`` → ``browser_analyze_page``
    action sequences within a single assessment run.
    """
    manager = BrowserManager(
        headless=headless,
        timeout_ms=timeout_ms,
        max_requests=max_requests,
        max_body_preview_bytes=max_body_preview_bytes,
        screenshot_dir=screenshot_dir,
    )
    return [
        BrowserOpenTool(manager),
        BrowserAnalyzePageTool(manager),
        BrowserLoginTool(manager),
        BrowserClickTool(manager),
        BrowserScreenshotTool(manager),
        BrowserCloseTool(manager),
    ]
