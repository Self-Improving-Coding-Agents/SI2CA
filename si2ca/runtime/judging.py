"""Pure judge rendering for the unified runner; no harness state or oracle monkeypatches."""
import json
import re
from si2ca.prompts import PROMPT_DIR, MESSAGES
from si2ca.runtime.protocol import bash_actions

RUBRIC_KEYS = ("R1", "R2", "R3", "A1", "A2", "A3", "A4")
_PR_DESCRIPTION = re.compile(r"<pr_description>(.*?)</pr_description>", re.S)


def system_prompt(weights, gold=False):
    template = (PROMPT_DIR / ("judge_oracle.md" if gold else "judge_system.md")).read_text().removesuffix("\n")
    formula = " + ".join(f"{v:.12g}*{k}" for k, v in weights.items())
    return template.format(formula=formula)


def extract_commands(message):
    actions = bash_actions(message)
    return "\n".join(command for _, command in actions) if actions else None


def _elide(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 40:
        return text
    suffix = text[-tail:] if tail else ""
    return f"{text[:head]}\n[... {len(text) - head - tail} chars elided ...]\n{suffix}"

def render_prefix(messages: list[dict], cfg) -> str:
    """Serialize the live conversation for the judge.

    Issue description (from the first user message's <pr_description>), then one
    block per event: agent tool calls, environment results (head/tail elided),
    and harness notices (format-error nudges). Previous tool-calling assistant
    reasoning is omitted here, but retained in policy history. When the total exceeds the budget
    the OLDEST blocks are elided first: the issue and recent turns matter most.
    """
    issue = ""
    blocks: list[str] = []
    turn = -1
    head, tail = cfg.prefix_tool_result_head_chars, cfg.prefix_tool_result_tail_chars
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            continue
        if not issue and role == "user":
            match = _PR_DESCRIPTION.search(content)
            issue = (match.group(1) if match else content).strip()
            continue
        if role == "assistant":
            turn += 1
            command = extract_commands(m)
            if command is not None:
                blocks.append(MESSAGES['judge']['turn_tool'].format(turn=turn, content=command))
            else:
                blocks.append(MESSAGES['judge']['turn_reply'].format(turn=turn, content=_elide(content, head, tail)))
        elif role == "tool":
            blocks.append(MESSAGES['judge']['turn_result'].format(turn=turn, content=_elide(content, head, tail)))
        elif role == "user":
            blocks.append(MESSAGES['judge']['turn_notice'].format(turn=turn, content=_elide(content, head, tail)))

    budget = max(cfg.prefix_max_chars - len(issue), 0)
    kept: list[str] = []
    used = 0
    for i, block in enumerate(reversed(blocks)):
        if used + len(block) > budget and kept:
            kept.append(MESSAGES['judge']['earlier_blocks'].format(count=len(blocks) - i))
            break
        kept.append(block)
        used += len(block)
    body = "\n\n".join(reversed(kept)) if kept else MESSAGES['judge']['no_prior_turns']
    return MESSAGES['judge']['trajectory'].format(issue=issue, body=body)

def render_candidate(message: dict) -> str:
    command = extract_commands(message)
    if command is not None:
        tool_call = f"bash: {command}"
    else:
        tool_call = (
            "(malformed or missing tool call: "
            f"{json.dumps(message.get('tool_calls'), ensure_ascii=False)[:500]})"
        )
    analysis = "\n\n".join(
        part
        for part in (
            (message.get("reasoning_content") or "").strip(),
            (message.get("content") or "").strip(),
        )
        if part
    )
    return MESSAGES['judge']['candidate'].format(analysis=analysis or '(empty)', tool_call=tool_call)
