#!/usr/bin/env python3
"""Reject Chinese in authored repository text while preserving research inputs.

Run from any directory: python scripts/check_english.py [--json].
The inventory includes tracked files and non-ignored new files. Raw benchmark
inputs, grading assets, and archived trajectories are intentionally immutable.
Paper/image binaries are listed separately and require visual/text inspection.
"""
from __future__ import annotations

import argparse
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
RAW_INPUTS = frozenset({
    "data/assets.tar.gz",
    "data/bench1231.jsonl",
    "data/deepswe113.jsonl",
    "data/dev192_multilingual.jsonl",
})
REVIEWED_BINARIES = frozenset({
    "docs/static/images/method.png", "data/method_editable.pdf",
    "data/assets/logo.png", "docs/static/images/logo-transparent.png",
    "docs/static/images/organization-avatar.png",
    "reports/SI2CA-Technical-Report.pdf",
})
RAW_ARCHIVE = re.compile(
    r"paper_data/table[12]_[^/]+/(?:[^/]+/)*(?:selection[^/]*|trajectories)[.]jsonl[.]gz$"
)
# Include Han, Bopomofo, radicals, and East Asian prose punctuation. Model-format
# delimiters such as the fullwidth vertical bar remain valid protocol literals.
RANGES = (
    (0x2E80, 0x2FFF), (0x3001, 0x3003), (0x3008, 0x301F),
    (0x3100, 0x312F), (0x31A0, 0x31BF), (0x31C0, 0x31EF),
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0xFE10, 0xFE19), (0xFE30, 0xFE4F), (0x20000, 0x323AF),
)
PUNCTUATION = (0xFF01, 0xFF08, 0xFF09, 0xFF0C, 0xFF0E, 0xFF1A, 0xFF1B, 0xFF1F)
CHINESE = re.compile("[" + "".join(chr(lo) + "-" + chr(hi) for lo, hi in RANGES)
                     + "".join(map(chr, PUNCTUATION)) + "]")
ESCAPES = re.compile(re.escape(chr(92)) + (
    r"(?:u\{([0-9a-fA-F]{1,6})\}|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|N\{([^}]+)\})"
))
SURROGATES = re.compile("([" + chr(0xD800) + "-" + chr(0xDBFF) + "])"
                        "([" + chr(0xDC00) + "-" + chr(0xDFFF) + "])")


def decoded_text(text: str) -> str:
    """Expose Unicode escapes and HTML entities; escaping is not translation."""
    def replace(match):
        try:
            if match[4] is not None:
                return unicodedata.lookup(match[4])
            return chr(int(next(g for g in match.groups()[:3] if g is not None), 16))
        except (ValueError, KeyError):
            return match[0]

    for _ in range(3):
        decoded = unescape(ESCAPES.sub(replace, text))
        if decoded == text:
            break
        text = decoded
    return SURROGATES.sub(
        lambda m: chr(0x10000 + ((ord(m[1]) - 0xD800) << 10) + ord(m[2]) - 0xDC00), text
    )


def audit(root: Path = ROOT) -> dict:
    """Inventory every repository file, reporting authored-text violations."""
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    ).decode("utf-8").split(chr(0))
    report = {"files": [], "violations": []}
    for name in sorted(set(names) - {""}):
        path = root / name
        if not path.exists():  # A tracked deletion is absent from the next commit.
            continue
        if CHINESE.search(decoded_text(name)):
            report["violations"].append({"file": name, "line": 0, "reason": "Chinese filename"})
        if name in RAW_INPUTS or RAW_ARCHIVE.fullmatch(name):
            report["files"].append({"file": name, "scope": "preserved research material"})
            continue
        if name in REVIEWED_BINARIES:
            report["files"].append({"file": name, "scope": "separately reviewed binary"})
            continue
        report["files"].append({"file": name, "scope": "authored text"})
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            report["violations"].append({"file": name, "line": 0, "reason": "Unreviewed non-UTF-8 file"})
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if CHINESE.search(decoded_text(line)):
                report["violations"].append({"file": name, "line": number, "reason": "Chinese text"})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the per-file inventory as JSON")
    args = parser.parse_args()
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        scopes = {scope: sum(row["scope"] == scope for row in report["files"])
                  for scope in sorted({row["scope"] for row in report["files"]})}
        print(f"English-only audit: {len(report['files'])} files; {scopes}")
        for issue in report["violations"]:
            print(f"{issue['file']}:{issue['line']}: {issue['reason']}")
        print(f"Authored-text violations: {len(report['violations'])}")
    return int(bool(report["violations"]))


if __name__ == "__main__":
    raise SystemExit(main())
