# Copyright 2026 Vinay Williams

"""Tests for the pre-filled GitHub issue-URL builder (bug_report)."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from painting_assist import bug_report


def test_issue_url_points_at_the_repo_new_issue_form():
    url = bug_report.issue_url("0.12.0", "some log", title="Hello")
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/v-i-n-a-y/painting-assist/issues/new"


def test_issue_url_carries_title_body_and_labels():
    url = bug_report.issue_url("0.12.0", "LOGLINE-XYZ", title="Crash: RuntimeError")
    q = parse_qs(urlparse(url).query)
    assert q["title"] == ["Crash: RuntimeError"]
    assert "0.12.0" in q["body"][0]  # environment block includes the version
    assert "LOGLINE-XYZ" in q["body"][0]  # the log excerpt is embedded
    assert q["labels"] == ["bug"]


def test_body_includes_environment_and_intro():
    body = bug_report.build_body("9.9.9", "tail", intro="Clicked a swatch")
    assert "Clicked a swatch" in body
    assert "Painting Assist: 9.9.9" in body
    assert "```text" in body and "tail" in body


def test_tail_trims_to_a_line_boundary():
    text = "line-a\nline-b\nline-c\nline-d\n"
    tail = bug_report._tail(text, 20)
    assert tail.startswith("…\n")
    assert "line-a" not in tail  # the head was dropped
    assert tail.endswith("line-d\n")
    # No mid-line start: every kept line after the ellipsis is whole.
    for kept in ("line-c", "line-d"):
        assert f"{kept}\n" in tail
    assert not tail.startswith("…\nne-")  # never begins mid-word


def test_tail_returns_whole_text_when_short():
    assert bug_report._tail("short", 100) == "short"


def test_issue_url_stays_within_the_length_budget():
    huge = "x" * 5_000_000  # a pathologically large log
    url = bug_report.issue_url(
        "0.12.0", huge, title="Crash", max_url_len=bug_report.MAX_URL_LEN
    )
    assert len(url) <= bug_report.MAX_URL_LEN
    # The body must still contain the environment block after shrinking.
    body = parse_qs(urlparse(url).query)["body"][0]
    assert "Painting Assist: 0.12.0" in body


def test_title_is_capped():
    url = bug_report.issue_url("0.12.0", "log", title="T" * 500)
    title = parse_qs(urlparse(url).query)["title"][0]
    assert len(title) <= bug_report._TITLE_MAX
