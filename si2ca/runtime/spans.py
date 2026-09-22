from __future__ import annotations
import json as _json
import logging
from typing import Any
from si2ca.prompts import PROMPT_DIR
logger = logging.getLogger(__name__)

BASH_TOOL = _json.loads((PROMPT_DIR / "bash_tool.json").read_text(encoding="utf-8"))

def _render_text(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tools=[BASH_TOOL], tokenize=False, add_generation_prompt=False)

def _encode(tok, text: str):
    return tok(text, add_special_tokens=False, return_offsets_mapping=True)

def candidate_message(msg: dict) -> dict:
    """The candidate turn in the shape the chat template expects."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        out["reasoning_content"] = msg["reasoning_content"]
    calls = msg.get("tool_calls") or []
    if calls:
        fn = (calls[0] or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = _json.loads(args or "{}")
            except Exception:
                args = {"command": args}
        out["tool_calls"] = [{"type": "function", "function": {"name": fn.get("name") or "bash", "arguments": args}}]
    return out

def normalize_prefix(msgs: list[dict]) -> list[dict]:
    """Apply the dict-arguments fix to the LIVE history before rendering."""
    out: list[dict] = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(candidate_message(m))
        else:
            out.append(m)
    return out

def build_spans(tok, prefix_msgs: list[dict], cand_msg: dict) -> dict | None:
    """Token ids for prefix+candidate, plus the candidate's own / think / content spans.

    ``prefix_msgs`` is whatever context this variant scores under -- the live
    rollout prefix, or that prefix plus the privileged-reference message. The
    candidate is byte-identical across variants, so any difference in the
    returned NLL is a difference in what the model was conditioned on.
    """
    try:
        pre = normalize_prefix(prefix_msgs)
        pre_ids = list(_encode(tok, _render_text(tok, pre))["input_ids"])
        post_text = _render_text(tok, pre + [cand_msg])
        enc = _encode(tok, post_text)
        ids = list(enc["input_ids"])
        offs = enc["offset_mapping"]
    except Exception as e:
        logger.warning("[gpgd] render failed: %s: %s", type(e).__name__, str(e)[:200])
        return None

    n = min(len(pre_ids), len(ids))
    start = 0
    while start < n and pre_ids[start] == ids[start]:
        start += 1
    if start >= len(ids):
        return None

    think: list[int] = []
    if cand_msg.get("reasoning_content"):
        c0 = offs[start][0]
        i0 = post_text.find("<think>", c0)
        i1 = post_text.find("</think>", i0) if i0 >= 0 else -1
        if i0 >= 0 and i1 > i0:
            a, b = i0 + len("<think>"), i1
            think = [k for k, (s0, s1) in enumerate(offs) if k >= start and s0 >= a and s1 <= b]
    think_set = set(think)
    content = [k for k in range(start, len(ids)) if k not in think_set]
    return {"ids": ids, "start": start, "think": think, "content": content}
