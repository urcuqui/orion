"""Real Nuclei wrapper for controlled, template-based vulnerability scanning.

Mirrors the mock/live HTTP split elsewhere in ``tools/recon``: shells out to a
locally installed ``nuclei`` binary and is registered only when explicitly
enabled (``enable_nuclei=True``) so the default registry stays fully
mock/offline. Every match becomes a :class:`~libs.recon.models.NucleiFinding`
*candidate* observation — never an auto-confirmed vulnerability — which the
evaluator turns into a ``Finding`` with ``validation_status="candidate"`` and
the planner routes to the existing HTTP tool for confirmation. Use only
against hosts you are explicitly authorised to test.

Two scanning tools are provided, plus a diagnostic:

- ``NucleiScanUrlTool`` (``nuclei_scan_url``): scan a single target/path.
- ``NucleiScanUrlsTool`` (``nuclei_scan_urls``): scan a deduplicated, capped
  batch of URLs via a temporary ``-list`` file.
- ``NucleiCheckInstallationTool`` (``nuclei_check_installation``): verify the
  binary is present and runnable without scanning anything.

Safety is layered the same way as everywhere else in ``tools/recon``: static
allow/block lists gate template *tags*, and ``BaseTool.is_call_sensitive``
(see ``tools/recon/base.py``) makes the existing approval gate in
``libs/recon/specialists.py`` call-aware — the safe default profile runs
automatically, anything that escalates beyond it (custom templates, high/
critical severity, a raised rate limit, an oversized URL batch) is gated
behind human approval, with no second approval mechanism required.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from libs.recon.models import NucleiFinding, NucleiScanResult, ToolResult
from tools.recon.base import BaseTool
from tools.recon.browser_tools import _validate_url as validate_url_scope
from tools.recon.fixtures import normalise_target

logger = logging.getLogger(__name__)

# Conservative charset for a single tag/severity/template token. Rejecting
# anything else (notably a leading '-') stops a crafted value from being
# parsed by nuclei as an extra CLI flag once it lands in a comma-joined
# -tags/-severity value or a -t path.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]*$")

# Safe-default profile: passive discovery and misconfiguration checks only.
# Anything outside this needs explicit human approval -- enforced call-by-call
# in is_call_sensitive() below, not by refusing to register the tool at all.
ALLOWED_DEFAULT_TAGS = {
    "exposure",
    "misconfig",
    "headers",
    "tech",
    "panel",
    "files",
    "tokens",
}
BLOCKED_TAGS = {
    "bruteforce",
    "brute-force",
    "fuzz",
    "fuzzing",
    "dos",
    "rce",
    "intrusive",
    "destructive",
    "sqli",
    "injection",
    "exploit",
    "malware",
    "takeover",
}
ALLOWED_SEVERITIES = {"info", "low", "medium"}
ESCALATED_SEVERITIES = {"high", "critical"}
_ALL_SEVERITIES = ALLOWED_SEVERITIES | ESCALATED_SEVERITIES

_MAX_OUTPUT_BYTES = 5_000_000  # cap how much JSONL output we ever read into memory
_SECRET_RE = re.compile(
    r"(authorization|cookie|api[_-]?key|x-api-key|token)\s*[:=]\s*[^'\"\n]+", re.IGNORECASE
)
_SECRET_ENV_RE = re.compile(r"(API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.IGNORECASE)

_TAGS_LINE_RE = re.compile(r"^\s*tags:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_SEVERITY_LINE_RE = re.compile(r"^\s*severity:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def _redact(text: str | None) -> str | None:
    """Strip secret-looking header/token values out of e.g. curl commands."""
    if not text:
        return text
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}: [redacted]", text)


def _safe_env() -> dict[str, str]:
    """Environment for the subprocess with obviously-sensitive vars stripped.

    Keeps PATH/HOME etc intact (nuclei needs them to resolve its templates
    directory) while never handing the app's own API keys/tokens to a
    shelled-out process.
    """
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV_RE.search(k)}


def _validate_tokens(values: list[str] | None, kind: str) -> str | None:
    """Return an error string if any token in ``values`` is unsafe, else None."""
    for value in values or []:
        if not _TOKEN_RE.match(value):
            return f"Invalid {kind} token {value!r}"
    return None


def validate_profile(
    tags: list[str],
    severity: list[str],
    templates: list[str] | None,
    allow_high: bool,
    allow_critical: bool,
) -> str | None:
    """Validate a requested tag/severity/template combination against the
    allow/block lists. Shared by both the single-URL and batch tools."""
    for kind, values in (("tag", tags), ("severity", severity), ("template", templates or [])):
        error = _validate_tokens(values, kind)
        if error:
            return error
    blocked = set(tags) & BLOCKED_TAGS
    if blocked:
        return f"Refusing to run blocked tag(s): {sorted(blocked)}"
    unknown = set(severity) - _ALL_SEVERITIES
    if unknown:
        return f"Unknown severity value(s): {sorted(unknown)}"
    if "critical" in severity and not allow_critical:
        return "Severity 'critical' is disabled by configuration (allow_critical=false)"
    if "high" in severity and not allow_high:
        return "Severity 'high' is disabled by configuration (allow_high=false)"
    return None


def _check_template_metadata(path: Path) -> str | None:
    """Best-effort guard against a blocked tag/severity in a template file.

    Nuclei templates are YAML, but we only need the ``info:`` block's tags and
    severity for this safety check, so a targeted regex avoids adding a YAML
    dependency just for this. This is a defence in depth, not a full parser.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:20_000]
    except OSError as exc:
        return f"Could not read template {path}: {exc}"
    tags_match = _TAGS_LINE_RE.search(text)
    if tags_match:
        declared = {t.strip().strip("[]\"'").lower() for t in tags_match.group(1).split(",")}
        blocked = declared & BLOCKED_TAGS
        if blocked:
            return f"Template {path.name} declares blocked tag(s): {sorted(blocked)}"
    severity_match = _SEVERITY_LINE_RE.search(text)
    if severity_match and severity_match.group(1).strip().lower() == "critical":
        return f"Template {path.name} declares severity 'critical'; not permitted as a template"
    return None


def resolve_templates(
    templates: list[str] | None, templates_dir: str | None
) -> tuple[list[str] | None, str | None]:
    """Resolve explicit template IDs to validated, absolute file paths.

    Every template must exist inside ``templates_dir`` (no ``..`` escape) and
    pass the metadata guard above. Returns ``(None, None)`` when no explicit
    templates were requested (the safe default profile doesn't use ``-t``).
    """
    if not templates:
        return None, None
    if not templates_dir:
        return None, "Explicit templates require a configured templates directory"
    base = Path(templates_dir).resolve()
    resolved: list[str] = []
    for template_id in templates:
        candidate = (base / template_id).resolve()
        if candidate != base and base not in candidate.parents:
            return None, f"Template {template_id!r} escapes the templates directory"
        if not candidate.is_file():
            return None, f"Template {template_id!r} not found under the templates directory"
        metadata_error = _check_template_metadata(candidate)
        if metadata_error:
            return None, metadata_error
        resolved.append(str(candidate))
    return resolved, None


def _is_call_sensitive(
    kwargs: dict[str, Any],
    allowed_severity: set[str],
    allowed_tags: set[str],
    default_rate_limit: int,
) -> bool:
    """Shared escalation check: True when a call goes beyond the safe default
    profile and therefore needs human approval (see module docstring)."""
    if kwargs.get("templates"):
        return True  # explicit templates are always gated
    severity = kwargs.get("severity")
    if severity and (not set(severity) <= allowed_severity or set(severity) & ESCALATED_SEVERITIES):
        return True
    tags = kwargs.get("tags")
    if tags and not set(tags) <= allowed_tags:
        return True
    rate_limit = kwargs.get("rate_limit")
    return bool(rate_limit and int(rate_limit) > default_rate_limit)


def _binary_path(binary: str) -> str | None:
    return shutil.which(binary)


def check_installation(binary: str = "nuclei", timeout: float = 15.0) -> ToolResult:
    """Diagnostic: verify the nuclei binary is installed and runnable.

    Never updates templates -- that stays manual/opt-in per the safety brief,
    and is not performed by this function at all.
    """
    path = _binary_path(binary)
    if path is None:
        return ToolResult.fail(
            "nuclei_check_installation",
            f"'{binary}' not found on PATH; install via "
            "'go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest'",
        )
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_safe_env(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ToolResult.fail(
            "nuclei_check_installation", f"Failed to run '{binary} -version': {exc}"
        )

    version_output = (proc.stdout or proc.stderr or "").strip()
    data = {"binary": path, "version_output": version_output}
    return ToolResult.ok(
        "nuclei_check_installation",
        f"nuclei available at {path}: {version_output or 'unknown version'}",
        data,
    )


def _run_nuclei_command(
    binary_path: str,
    target_args: list[str],
    tags: list[str],
    severity: list[str],
    templates: list[str] | None,
    rate_limit: int,
    process_timeout: float,
    max_results: int,
) -> tuple[bool, list[NucleiFinding], str, list[str]]:
    """Execute nuclei with ``target_args`` (``['-u', url]`` or ``['-list', path]``)
    and return ``(success, findings, command_summary, errors)``.

    Output is captured via ``-jsonl -o <tmpfile>`` (not parsed from stdout,
    which may also carry banners/progress) and the temp file is always removed.
    """
    fd, out_path = tempfile.mkstemp(prefix="orion-nuclei-", suffix=".jsonl")
    os.close(fd)
    try:
        command = [
            binary_path,
            *target_args,
            "-jsonl",
            "-o",
            out_path,
            "-silent",
            "-no-color",
            "-disable-update-check",
            "-rate-limit",
            str(rate_limit),
        ]
        if tags:
            command += ["-tags", ",".join(tags)]
        if severity:
            command += ["-severity", ",".join(severity)]
        if templates:
            for template_path in templates:
                command += ["-t", template_path]

        command_summary = " ".join(command)
        logger.info("Executing: %s", command_summary)
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=process_timeout,
                check=False,
                env=_safe_env(),
            )
        except subprocess.TimeoutExpired:
            return False, [], command_summary, [f"nuclei scan timed out after {process_timeout}s"]
        except OSError as exc:
            return False, [], command_summary, [f"Failed to execute nuclei: {exc}"]

        if proc.returncode != 0:
            stderr = (_redact((proc.stderr or "").strip()) or "")[:2000]
            return False, [], command_summary, [f"nuclei exited {proc.returncode}: {stderr}"]

        findings, parse_notes = _parse_jsonl(out_path, max_results)
        return True, findings, command_summary, parse_notes
    finally:
        with contextlib.suppress(OSError):
            os.remove(out_path)


def _finding_from_record(record: dict[str, Any]) -> NucleiFinding:
    info = record.get("info") or {}
    references = info.get("reference") or []
    if isinstance(references, str):
        references = [references]
    extracted = record.get("extracted-results") or []
    return NucleiFinding(
        template_id=record.get("template-id") or record.get("templateID"),
        template_name=info.get("name"),
        severity=info.get("severity"),
        matched_at=record.get("matched-at") or record.get("host") or record.get("url"),
        matcher_name=record.get("matcher-name"),
        extracted_results=list(extracted) if isinstance(extracted, list) else [str(extracted)],
        description=info.get("description"),
        tags=list(info.get("tags") or []),
        references=list(references),
        curl_command=_redact(record.get("curl-command")),
    )


def _parse_jsonl(path: str, max_results: int) -> tuple[list[NucleiFinding], list[str]]:
    """Parse nuclei's JSONL output file into findings, tolerating malformed lines."""
    notes: list[str] = []
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return [], [f"Failed to read nuclei output: {exc}"]

    if len(raw) > _MAX_OUTPUT_BYTES:
        raw = raw[:_MAX_OUTPUT_BYTES]
        notes.append(f"nuclei output exceeded {_MAX_OUTPUT_BYTES} bytes; output was truncated")

    findings: list[NucleiFinding] = []
    text = raw.decode("utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            notes.append(f"Skipped malformed JSONL line {line_no}")
            continue
        findings.append(_finding_from_record(record))
        if len(findings) >= max_results:
            notes.append(f"capped at max_results={max_results}; later matches dropped")
            break
    return findings, notes


class NucleiScanUrlTool(BaseTool):
    """Run a controlled, template-based Nuclei scan against a single URL."""

    name = "nuclei_scan_url"
    description = (
        "Template-based vulnerability/misconfiguration scan of a single URL via "
        "the local nuclei binary. The safe-default profile (exposure/misconfig/"
        "headers/tech/panel/files/tokens tags, info/low/medium severity) runs "
        "automatically; custom templates, high/critical severity or a raised "
        "rate limit require human approval. Every result is a candidate, not a "
        "confirmed finding -- validate matched_at with http_get."
    )
    sensitive = True
    argument_help = {
        "target": "host or URL to scan (lab/authorised targets only)",
        "path": "URL path to scan, default '/'",
        "tags": f"template tags, default: {sorted(ALLOWED_DEFAULT_TAGS)}",
        "severity": (
            f"severities, default: {sorted(ALLOWED_SEVERITIES)} (high/critical need approval)"
        ),
        "templates": "explicit template file paths inside the templates directory (needs approval)",
        "rate_limit": "requests/sec, default from configuration",
        "timeout": "overall scan timeout in seconds, default from configuration",
    }

    def __init__(
        self,
        binary: str = "nuclei",
        templates_dir: str | None = None,
        default_tags: set[str] | None = None,
        default_severity: set[str] | None = None,
        allow_high: bool = False,
        allow_critical: bool = False,
        rate_limit: int = 5,
        timeout: float = 600.0,
        max_results: int = 100,
    ) -> None:
        self.binary = binary
        self.templates_dir = templates_dir
        self.default_tags = default_tags or set(ALLOWED_DEFAULT_TAGS)
        self.default_severity = default_severity or set(ALLOWED_SEVERITIES)
        self.allow_high = allow_high
        self.allow_critical = allow_critical
        self.default_rate_limit = rate_limit
        self.default_timeout = timeout
        self.max_results = max_results

    def is_call_sensitive(self, **kwargs: Any) -> bool:
        return _is_call_sensitive(
            kwargs, self.default_severity, ALLOWED_DEFAULT_TAGS, self.default_rate_limit
        )

    def _run(
        self,
        target: str,
        path: str = "/",
        tags: list[str] | None = None,
        severity: list[str] | None = None,
        templates: list[str] | None = None,
        rate_limit: int | None = None,
        timeout: int | None = None,
        **_: Any,
    ) -> ToolResult:
        base_url, _host = normalise_target(target)
        url = base_url + (path if path.startswith("/") else f"/{path}")

        scope_error = validate_url_scope(url, target)
        if scope_error:
            return ToolResult.ok(
                self.name,
                f"Skipped out-of-scope URL: {scope_error}",
                NucleiScanResult(
                    success=True,
                    action=self.name,
                    targets_skipped=[{"url": url, "reason": scope_error}],
                ),
            )

        tags = list(tags) if tags else sorted(self.default_tags)
        severity = list(severity) if severity else sorted(self.default_severity)

        error = validate_profile(tags, severity, templates, self.allow_high, self.allow_critical)
        if error:
            return ToolResult.fail(self.name, error)

        resolved_templates, template_error = resolve_templates(templates, self.templates_dir)
        if template_error:
            return ToolResult.fail(self.name, template_error)

        binary_path = shutil.which(self.binary)
        if binary_path is None:
            return ToolResult.fail(
                self.name, f"'{self.binary}' not found on PATH; install nuclei"
            )

        effective_rate_limit = int(rate_limit) if rate_limit else self.default_rate_limit
        effective_timeout = float(timeout) if timeout else self.default_timeout

        success, findings, command_summary, notes = _run_nuclei_command(
            binary_path,
            ["-u", url],
            tags,
            severity,
            resolved_templates,
            effective_rate_limit,
            effective_timeout,
            self.max_results,
        )
        if not success:
            return ToolResult.fail(self.name, "; ".join(notes) or "nuclei scan failed")

        result = NucleiScanResult(
            success=True,
            action=self.name,
            targets_scanned=[url],
            command_summary=command_summary,
            findings=findings,
            result_count=len(findings),
            errors=notes,
        )
        summary = f"{url}: {len(findings)} candidate finding(s)"
        return ToolResult.ok(self.name, summary, result)


class NucleiScanUrlsTool(BaseTool):
    """Run a controlled Nuclei scan across a batch of URLs via ``-list``."""

    name = "nuclei_scan_urls"
    description = (
        "Template-based scan across multiple URLs (e.g. discovered by "
        "url_crawler or browser_analyze_page) via a temporary -list file. Same "
        "safe-default profile and approval rules as nuclei_scan_url; "
        "duplicates and out-of-scope URLs are dropped, oversized batches are "
        "capped and gated behind approval."
    )
    sensitive = True
    argument_help = {
        "urls": (
            "URLs to scan (deduplicated, capped at max_targets, "
            "out-of-scope entries skipped)"
        ),
        "target": "authorised host/URL used to enforce scope on every entry in 'urls'",
        "tags": f"template tags, default: {sorted(ALLOWED_DEFAULT_TAGS)}",
        "severity": (
            f"severities, default: {sorted(ALLOWED_SEVERITIES)} (high/critical need approval)"
        ),
        "templates": "explicit template file paths inside the templates directory (needs approval)",
    }

    def __init__(
        self,
        binary: str = "nuclei",
        templates_dir: str | None = None,
        default_tags: set[str] | None = None,
        default_severity: set[str] | None = None,
        allow_high: bool = False,
        allow_critical: bool = False,
        rate_limit: int = 5,
        timeout: float = 600.0,
        max_results: int = 100,
        max_targets: int = 20,
    ) -> None:
        self.binary = binary
        self.templates_dir = templates_dir
        self.default_tags = default_tags or set(ALLOWED_DEFAULT_TAGS)
        self.default_severity = default_severity or set(ALLOWED_SEVERITIES)
        self.allow_high = allow_high
        self.allow_critical = allow_critical
        self.default_rate_limit = rate_limit
        self.default_timeout = timeout
        self.max_results = max_results
        self.max_targets = max_targets

    def is_call_sensitive(self, **kwargs: Any) -> bool:
        urls = kwargs.get("urls") or []
        if len(urls) > self.max_targets:
            return True
        return _is_call_sensitive(
            kwargs, self.default_severity, ALLOWED_DEFAULT_TAGS, self.default_rate_limit
        )

    def _run(
        self,
        urls: list[str],
        target: str = "",
        tags: list[str] | None = None,
        severity: list[str] | None = None,
        templates: list[str] | None = None,
        rate_limit: int | None = None,
        timeout: int | None = None,
        **_: Any,
    ) -> ToolResult:
        if not urls:
            return ToolResult.fail(self.name, "No URLs provided")

        deduped = list(dict.fromkeys(urls))
        scope_target = target or deduped[0]

        in_scope: list[str] = []
        skipped: list[dict[str, str]] = []
        for url in deduped:
            error = validate_url_scope(url, scope_target)
            if error:
                skipped.append({"url": url, "reason": error})
            else:
                in_scope.append(url)

        if len(in_scope) > self.max_targets:
            overflow, in_scope = in_scope[self.max_targets :], in_scope[: self.max_targets]
            reason = f"exceeds max_targets={self.max_targets}"
            skipped.extend({"url": u, "reason": reason} for u in overflow)

        if not in_scope:
            return ToolResult.ok(
                self.name,
                "No in-scope URLs to scan",
                NucleiScanResult(success=True, action=self.name, targets_skipped=skipped),
            )

        tags = list(tags) if tags else sorted(self.default_tags)
        severity = list(severity) if severity else sorted(self.default_severity)
        error = validate_profile(tags, severity, templates, self.allow_high, self.allow_critical)
        if error:
            return ToolResult.fail(self.name, error)

        resolved_templates, template_error = resolve_templates(templates, self.templates_dir)
        if template_error:
            return ToolResult.fail(self.name, template_error)

        binary_path = shutil.which(self.binary)
        if binary_path is None:
            return ToolResult.fail(
                self.name, f"'{self.binary}' not found on PATH; install nuclei"
            )

        effective_rate_limit = int(rate_limit) if rate_limit else self.default_rate_limit
        effective_timeout = float(timeout) if timeout else self.default_timeout

        list_fd, list_path = tempfile.mkstemp(prefix="orion-nuclei-targets-", suffix=".txt")
        try:
            with os.fdopen(list_fd, "w") as fh:
                fh.write("\n".join(in_scope) + "\n")
            success, findings, command_summary, notes = _run_nuclei_command(
                binary_path,
                ["-list", list_path],
                tags,
                severity,
                resolved_templates,
                effective_rate_limit,
                effective_timeout,
                self.max_results,
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(list_path)

        if not success:
            return ToolResult.fail(self.name, "; ".join(notes) or "nuclei scan failed")

        result = NucleiScanResult(
            success=True,
            action=self.name,
            targets_scanned=in_scope,
            targets_skipped=skipped,
            command_summary=command_summary,
            findings=findings,
            result_count=len(findings),
            errors=notes,
        )
        summary = (
            f"{len(in_scope)} URL(s) scanned: {len(findings)} candidate finding(s), "
            f"{len(skipped)} skipped"
        )
        return ToolResult.ok(self.name, summary, result)


class NucleiCheckInstallationTool(BaseTool):
    """Diagnostic tool: verify nuclei is installed without scanning anything."""

    name = "nuclei_check_installation"
    description = (
        "Verify the nuclei binary is installed and can run (no scanning, no template update)."
    )
    sensitive = False

    def __init__(self, binary: str = "nuclei") -> None:
        self.binary = binary

    def _run(self, **_: Any) -> ToolResult:
        return check_installation(self.binary)
