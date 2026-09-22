"""Build the privileged-context message that carries the golden patch.

The template lives in ``prompts/gold_guided_scoring.md`` (its leading HTML comment
documents the design and is stripped here). Substitution is plain text, not
jinja: a diff is arbitrary source code and routinely contains ``{{`` / ``{%``.

Placement is configurable (``select.gold_position``); the two arms differ in
exactly this and nothing else:

* ``tail`` -- block last, immediately before the candidate. Shares the whole
  rollout prefix with the generation call's cache, so scoring costs one prefill
  of the block plus the candidate. Measured 2026-08-16 over 264,865 candidates:
  the block's mere presence raises the candidate's NLL by +0.221 while the
  answer's content adds only +0.066 on top, and argmin(gold) agrees with
  argmin(an unrelated patch) 81.8% of the time. The interruption at the
  continuation point, not the answer, was doing most of the work.
* ``head`` -- block right after the problem statement, before the first agent
  turn, so the trajectory reads as if it had been written with the reference in
  view. Costs a second cached prefix per trajectory for its whole life; the
  measured pool (1.64-1.85M tokens/replica) covers it at concurrency 64, which
  the 958k pool of 2026-08-07 would not have.

Substitution is plain text, not jinja: a diff is arbitrary source code and
routinely contains ``{{`` / ``{%``.
"""

from __future__ import annotations

from si2ca.prompts import PROMPT_DIR, MESSAGES

import re

_TEMPLATE = {"tail": PROMPT_DIR / "gold_guided_scoring.md",
             "head": PROMPT_DIR / "gold_guided_scoring_head.md",
             
             "hint": PROMPT_DIR / "gold_guided_scoring_hint.md"}
_LEADING_COMMENT = re.compile(r"\A\s*<!--.*?-->\s*", re.S)

_GIT_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)$", re.M)
_PLUS_HEADER = re.compile(r"^\+\+\+ (?:b/)?(?P<p>[^\t\n]+)$", re.M)

_TRUNCATION_MARK = MESSAGES['judge']['patch_omitted']

_template_cache: dict[str, str] = {}

def template(position: str = "tail") -> str:
    """The privileged-context block for this placement, design comment stripped."""
    if position not in _TEMPLATE:
        raise ValueError(f"gold position {position!r}: expected one of {sorted(_TEMPLATE)}")
    if position not in _template_cache:
        _template_cache[position] = _LEADING_COMMENT.sub("", _TEMPLATE[position].read_text()).strip()
    return _template_cache[position]

def patch_files(patch: str) -> list[str]:
    """Paths the patch touches, in order, de-duplicated."""
    out: list[str] = []
    for m in _GIT_HEADER.finditer(patch or ""):
        p = m.group("b") if m.group("b") != "/dev/null" else m.group("a")
        if p not in out:
            out.append(p)
    if not out:
        for m in _PLUS_HEADER.finditer(patch or ""):
            p = m.group("p")
            if p != "/dev/null" and p not in out:
                out.append(p)
    return out

def truncate_patch(patch: str, max_chars: int, files: list[str]) -> tuple[str, bool]:
    """Head+tail truncation, naming what was dropped and where the fix lives.

    Never silently drops the tail: the last hunks are as diagnostic as the first,
    and a candidate that edits the last file must not look surprising merely
    because the scorer never saw it. The file list rides in the marker rather
    than in the prompt body, so an untruncated patch (92% of this pool) adds
    nothing to the block at all.
    """
    patch = patch or ""
    if max_chars <= 0 or len(patch) <= max_chars:
        return patch, False
    head = int(max_chars * 0.6)
    tail = max_chars - head
    mark = _TRUNCATION_MARK.format(n=len(patch) - max_chars,
                                   files=", ".join(files) if files else MESSAGES['judge']['no_file_header'])
    return patch[:head] + mark + patch[-tail:], True

def gold_message(patch: str, max_chars: int = 12000, position: str = "tail") -> tuple[dict, dict]:
    """The privileged-context user message, plus what went into it.

    Returns ``(message, meta)`` where meta records the patch size, whether it was
    truncated and the files it touches -- all of which belong in the turn log, so
    a later analysis can ask whether the effect depends on patch size.
    """
    files = patch_files(patch)
    body, truncated = truncate_patch(patch, max_chars, files)
    text = template(position).replace("{{reference_patch}}", body)
    meta = {
        "gold_chars": len(patch or ""),
        "gold_chars_used": len(body),
        "gold_truncated": truncated,
        "gold_files": files,
        "gold_position": position,
    }
    return {"role": "user", "content": text}, meta
