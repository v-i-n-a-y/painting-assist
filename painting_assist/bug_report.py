# Copyright 2026 Vinay Williams

"""Compose a pre-filled GitHub "new issue" URL for user-initiated bug reports.

Qt-free so the URL-building and length-budgeting logic is unit-testable without
a GUI. Nothing here sends anything: the caller opens the returned URL in the
user's browser, and the report is only submitted when the user reviews it and
clicks *Submit* on GitHub. That keeps the whole flow consent-based and avoids
shipping any GitHub credential inside the application.

The body carries a short environment block plus the tail of the current session
log. GitHub accepts fairly long query strings but not unbounded ones, so
:func:`issue_url` shrinks the log excerpt until the whole URL fits under
:data:`MAX_URL_LEN`.
"""

from __future__ import annotations

import platform
from urllib.parse import urlencode

# The public GitHub repository that receives the reports.
GITHUB_REPO = "https://github.com/v-i-n-a-y/painting-assist"
_NEW_ISSUE = f"{GITHUB_REPO}/issues/new"

# Conservative ceiling for the whole URL. GitHub's practical limit is around
# 8 KB; staying well under it leaves room for the browser and any redirects.
MAX_URL_LEN = 7000

# Longest issue title we will send (GitHub truncates very long titles anyway).
_TITLE_MAX = 120

# How much log tail to include before any URL-length shrinking kicks in.
DEFAULT_LOG_CHARS = 6000

_TRUNCATED_NOTE = (
    "…(log truncated to fit a shareable link — full log via Help ▸ View Logs)"
)


def _tail(text: str, max_chars: int) -> str:
    """Return the last ``max_chars`` of ``text``, cut at a clean line boundary.

    When the text is trimmed, the excerpt starts at the next newline after the
    cut point (so it never begins mid-line) and is prefixed with an ellipsis.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[-max_chars:]
    newline = cut.find("\n")
    if newline != -1:
        cut = cut[newline + 1 :]
    return "…\n" + cut


def environment_lines(version: str) -> str:
    """Return the markdown environment block (app version, OS, Python)."""
    return (
        f"- Painting Assist: {version}\n"
        f"- OS: {platform.platform()}\n"
        f"- Python: {platform.python_version()}"
    )


def build_body(
    version: str,
    log_text: str,
    *,
    intro: str = "",
    log_chars: int = DEFAULT_LOG_CHARS,
) -> str:
    """Compose the markdown issue body: description, environment, log excerpt."""
    description = intro.strip() or (
        "_Describe what you were doing when the problem occurred._"
    )
    log = _tail(log_text, log_chars)
    return (
        "**What happened**\n"
        f"{description}\n\n"
        "**Environment**\n"
        f"{environment_lines(version)}\n\n"
        "**Recent log** _(please review before submitting — may contain file "
        "paths)_\n"
        "```text\n"
        f"{log}\n"
        "```\n"
    )


def _compose(title: str, body: str, labels: tuple[str, ...]) -> str:
    """Build the full new-issue URL from a title, body and optional labels."""
    params = {"title": title[:_TITLE_MAX], "body": body}
    if labels:
        params["labels"] = ",".join(labels)
    return f"{_NEW_ISSUE}?{urlencode(params)}"


def issue_url(
    version: str,
    log_text: str,
    *,
    title: str,
    intro: str = "",
    labels: tuple[str, ...] = ("bug",),
    max_url_len: int = MAX_URL_LEN,
    max_log_chars: int = DEFAULT_LOG_CHARS,
) -> str:
    """Return a pre-filled GitHub new-issue URL, bounded to ``max_url_len``.

    The description and environment block are always kept; only the log excerpt
    is shrunk (repeatedly, then flagged as truncated) until the encoded URL fits.
    """
    log_chars = max_log_chars
    while True:
        body = build_body(version, log_text, intro=intro, log_chars=log_chars)
        url = _compose(title, body, labels)
        if len(url) <= max_url_len or log_chars <= 200:
            if len(url) > max_url_len:
                # Even a minimal excerpt overflows; drop the log entirely and
                # tell the reader where the full log lives.
                body = build_body(
                    version,
                    _TRUNCATED_NOTE,
                    intro=intro,
                    log_chars=len(_TRUNCATED_NOTE),
                )
                url = _compose(title, body, labels)
            return url
        log_chars = int(log_chars * 0.8)
