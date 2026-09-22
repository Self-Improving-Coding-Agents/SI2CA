#!/usr/bin/env python3
"""Shared execution loop for all new SI2CA experiments.

Standard, self-judgement, self-likelihood, discovered strategies and student
runs share workspace preparation, policy history, tool execution and result
accounting. Selectors change candidate allocation/scoring; benchmark adapters
retain their grading contracts, including DeepSWE in-place grading.
See docs/HARNESS.md for the versioned protocol and historical differences.
"""
from __future__ import annotations

from si2ca.prompts import PROMPT_DIR, MESSAGES, AGENT_TEMPLATES, MODEL_TEMPLATES

import argparse
import asyncio
import importlib.util
import json
import os
import random
import re
import shlex
import sys
import uuid
import time
from pathlib import Path
from types import SimpleNamespace
from si2ca.results import infrastructure_failure
from si2ca.runtime.protocol import (HARNESS_PROTOCOL, SUBMIT_MARKER, MUTATION_MARKERS,
                                    bash_actions, is_submission)

os.environ.setdefault("SWE_AGENT_UID0", "1")            
os.environ.setdefault("SLIME_AGENT_SANDBOX_BACKEND", "local_docker")

import httpx  
from si2ca.backends import chat_payload
from jinja2 import Template  

ENDPOINT_WAIT_SEC = int(os.environ.get("ENDPOINT_WAIT_SEC", "600"))

NULL_CONTROL_RATE = 0.15
SCAFFOLDS = ("none", "SJ", "SL", "SJ+SL")

def split_endpoints(base_url: str) -> list[str]:
    """Split comma-separated base URLs into endpoint roots without /v1.

    Per-request data-parallel routing can move consecutive turns to different
    replicas, losing prefix-cache locality. A historical H100 comparison showed
    a 5.4x difference. Independent replicas with per-task endpoint affinity keep
    a whole trajectory on one server; run_one_bounded acquires/releases the slot.

    A single endpoint does not create a health-checked pool, preserving support
    for hosted relays that lack /health.
    """
    out = []
    for u in base_url.split(","):
        u = u.strip().rstrip("/")
        if not u:
            continue
        out.append(u[: -len("/v1")].rstrip("/") if u.endswith("/v1") else u)
    return out

BASH_TOOL = json.loads((PROMPT_DIR / "bash_tool.json").read_text(encoding="utf-8"))

class TurnContext:
    """Expose strategy context without explicit task-identity fields."""

    def __init__(self, step: int, history: list[dict], last_obs: str, last_rc: int | None,
                 has_edited: bool, prior_commands: list[str], rng=None,
                 gold_in_judge: bool = False):

        self.gold_in_judge = gold_in_judge

        self.rng = rng if rng is not None else random.Random(step)
        self.step = step
        self.n_prior_turns = len(prior_commands)
        self.last_observation = last_obs
        self.last_returncode = last_rc
        self.has_edited = has_edited
        self.prior_commands = prior_commands
        self._history = history

class Strategy:
    """Baseline: sample one continuation per turn and execute it unchanged."""

    name = "baseline"

    def plan(self, ctx: TurnContext) -> int:
        return 1

    async def select(self, ctx: TurnContext, cands: list[dict], judge) -> tuple[int, str, dict]:
        """Return (winner_index, decision_label, judge_usage)."""
        return 0, "single_candidate", {}

class SelfGuideK2(Strategy):
    """Sample k=2 candidates and score them with the same model endpoint.

    Identical candidate commands have nothing to rank, so judging is skipped.
    This avoided judge calls on about 19.9% of historical turns.
    """

    name = "selfguide_k2"
    K = 2
    RUBRIC = ("progress", "correctness", "efficiency")

    @staticmethod
    def _score_prompt(ctx, c: dict) -> str:
        """Use the same prompt for scoring and null controls to keep them comparable."""
        return (
            MESSAGES['judge']['simple_action'].format(recent_commands=ctx.prior_commands[-5:], last_result=ctx.last_observation[:1500], command=c['command'])
        )

    def plan(self, ctx: TurnContext) -> int:
        return self.K

    async def select(self, ctx, cands, judge):
        valid = [c for c in cands if c.get("command")]
        if len(valid) < 2:
            return (cands.index(valid[0]) if valid else 0,
                    "only_valid_toolcall" if valid else "both_invalid", {})
        if len({c["command"] for c in valid}) == 1:

            if ctx.rng.random() < NULL_CONTROL_RATE:
                usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
                for c in valid:
                    _, usage = await judge(self._score_prompt(ctx, c))
                    for k in usage_total:
                        usage_total[k] += int((usage or {}).get(k) or 0)
                return cands.index(valid[0]), "identical_command_null_control", usage_total
            return cands.index(valid[0]), "identical_command", {}

        scores, usage_total = [], {"prompt_tokens": 0, "completion_tokens": 0}
        for c in valid:
            text, usage = await judge(self._score_prompt(ctx, c))
            for k in usage_total:
                usage_total[k] += int((usage or {}).get(k) or 0)
            total = 0.0
            try:
                obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
                total = sum(float(obj.get(k, 0)) for k in self.RUBRIC)
            except Exception:  
                total = -1.0
            scores.append(total)

        if max(scores) < 0:
            return cands.index(valid[0]), "parse_fallback_random", usage_total
        best = max(range(len(valid)), key=lambda i: scores[i])
        label = "score_comparison" if scores.count(scores[best]) == 1 else "random_tie"
        return cands.index(valid[best]), label, usage_total

GOLD_SLOT = "<<REFERENCE_SOLUTION>>"

def _parse_judge_scores(text: str) -> dict | None:
    """Parse numeric rubric fields and justification from a judge response.

    Preserve keys such as R1, A4, G1, and total; return None on parse failure.
    """
    try:
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:  
        return None
    if not isinstance(obj, dict):
        return None
    out = {k: v for k, v in obj.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if isinstance(obj.get("justification"), str):
        out["justification"] = obj["justification"]
    return out or None

def _infer_cand_idx(prompt: str, cands: list[dict]) -> int | None:
    """Identify the candidate command occurring last in the judge prompt.

    This attributes samples even when strategies loop internally; return None
    if no command matches rather than guessing.
    """
    best, best_pos = None, -1
    for i, c in enumerate(cands or []):
        cmd = c.get("command")
        if cmd:
            pos = prompt.rfind(cmd)
            if pos > best_pos:
                best, best_pos = i, pos
    return best

_RESCORE_TOK = None

_RESCORE_SEM: "asyncio.Semaphore | None" = None

def _rescore_sem() -> "asyncio.Semaphore":
    global _RESCORE_SEM
    if _RESCORE_SEM is None:
        _RESCORE_SEM = asyncio.Semaphore(int(os.environ.get("RESCORE_CONCURRENCY", "4")))
    return _RESCORE_SEM

def _rescore_tokenizer(path: str):
    global _RESCORE_TOK
    if _RESCORE_TOK is None:
        from transformers import AutoTokenizer
        _RESCORE_TOK = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _RESCORE_TOK

async def rescore_privileged(native_client, tok, prefix_msgs: list[dict],
                             block_msg: dict, cands: list[dict],
                             position: str = "head") -> None:
    """Attach mean_nll_privileged to each candidate, or None on scoring failure.

    head inserts the block after the first user message, creating a separate
    scoring prefix that can be cached. tail inserts it just before the candidate,
    sharing the generation prefix but prefilling the block on every turn.
    Historical tail-placement measurements found a +0.221 NLL shift from the
    block's presence alone; which position selects better is an empirical question.

    Rendering, HTTP, or missing-logprob failures leave None. Strategies must
    label unavailable signals explicitly rather than silently choosing at random.
    """
    from si2ca.runtime import spans as S
    if position == "tail":
        priv_prefix = list(prefix_msgs) + [block_msg]
    else:
        i = next((k for k, m in enumerate(prefix_msgs) if m.get("role") == "user"), 0)
        priv_prefix = list(prefix_msgs[: i + 1]) + [block_msg] + list(prefix_msgs[i + 1:])
    for c in cands:
        c["mean_nll_privileged"] = None
        try:
            spans = S.build_spans(tok, priv_prefix, S.candidate_message(c["msg"]))
        except Exception:  
            spans = None
        if not spans:
            continue

        if len(spans["ids"]) - spans["start"] > 4096:
            continue
        payload = {"input_ids": spans["ids"],
                   "sampling_params": {"temperature": 0, "max_new_tokens": 0,
                                       "skip_special_tokens": False},
                   "return_logprob": True,
                   "logprob_start_len": max(0, spans["start"] - 1)}
        try:
            async with _rescore_sem():
                r = await native_client.post("/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            lps = [x[0] for x in (data.get("meta_info") or {}).get("input_token_logprobs") or []]
            if not lps:
                continue
            
            off = len(spans["ids"]) - len(lps)
            v = [lps[k - off] for k in range(spans["start"], len(spans["ids"]))
                 if 0 <= k - off < len(lps) and isinstance(lps[k - off], (int, float))]
            if v:
                c["mean_nll_privileged"] = round(-sum(v) / len(v), 6)
        except Exception:  
            continue

class SelfGuideRubrics(Strategy):
    """Score candidates with self-judgement rubrics in the unified harness.

    R1-R3 assess analysis: grounding, diagnostic insight, and plan coherence.
    A1-A4 assess tool calls: alignment, correctness, progress, and efficiency.
    Average SCORE_SAMPLES independent scores per candidate. With gold_in_judge,
    add G1 goal alignment with default weight 0.30 and scale the other weights
    to 0.70. Prompt rendering uses the shared judging module and prompt assets.
    """

    name = "selfguide_rubrics"
    K = 2
    SCORE_SAMPLES = 3
    GOLD_WEIGHT = .3
    KEYS7 = ("R1", "R2", "R3", "A1", "A2", "A3", "A4")

    def _weights(self, gold: bool) -> dict:
        base = {"R1": 0.15, "R2": 0.15, "R3": 0.10,
                "A1": 0.15, "A2": 0.20, "A3": 0.15, "A4": 0.10}
        if not gold:
            return base
        w = {k: v * (1-self.GOLD_WEIGHT) for k, v in base.items()}
        w["G1"] = self.GOLD_WEIGHT
        return w

    @staticmethod
    def _system(gold: bool, weights: dict) -> str:
        from si2ca.runtime.judging import system_prompt
        return system_prompt(weights, gold)

    @staticmethod
    def _parse(text: str, keys) -> dict | None:
        """Parse scores for an explicit key set, including optional G1."""
        if not text:
            return None
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
        s, e = t.find("{"), t.rfind("}")
        if s < 0 or e <= s:
            return None
        try:
            obj = json.loads(t[s: e + 1])
        except Exception:  
            return None
        if not isinstance(obj, dict):
            return None
        out: dict = {}
        for k in keys:
            v = obj.get(k)
            if isinstance(v, bool):
                return None
            if isinstance(v, str):
                try:
                    v = float(v.strip())
                except ValueError:
                    return None
            if not isinstance(v, (int, float)) or not 1 <= v <= 10:
                return None
            out[k] = float(v)
        return out

    def plan(self, ctx: TurnContext) -> int:
        return self.K

    async def select(self, ctx, cands, judge):
        from si2ca.runtime.judging import render_candidate, render_prefix
        indices = [i for i, c in enumerate(cands) if c.get("command")]
        valid = [cands[i] for i in indices]
        if len(valid) < 2:
            return (indices[0] if valid else 0,
                    "only_valid_toolcall" if valid else "both_invalid", {})
        if len({tuple(cmd for _, cmd in bash_actions(c['msg'])) for c in valid}) == 1:
            return ctx.rng.choice(indices), "identical_command", {}

        gold = bool(getattr(ctx, "gold_in_judge", False))
        keys = self.KEYS7 + (("G1",) if gold else ())
        weights = self._weights(gold)
        system = self._system(gold, weights)
        prefix = render_prefix(ctx._history, getattr(self, "prefix_config", SimpleNamespace(
            prefix_tool_result_head_chars=1500, prefix_tool_result_tail_chars=1500, prefix_max_chars=80000)))
        slot = f"\n\n{GOLD_SLOT}" if gold else ""

        usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
        totals: list[float | None] = []
        for c in valid:
            user = f"{prefix}{slot}\n\n{render_candidate(c['msg'])}"
            samples = []
            for _ in range(self.SCORE_SAMPLES):
                text, usage = await judge(user, system=system)
                for k in usage_total:
                    usage_total[k] += int((usage or {}).get(k) or 0)
                sc = self._parse(text, keys)
                if sc is not None:
                    
                    samples.append(sum(weights[k] * sc[k] for k in keys))
            totals.append(sum(samples) / len(samples) if samples else None)

        scored = [(t, i) for i, t in enumerate(totals) if t is not None]
        if not scored:
            return indices[0], "parse_fallback_first", usage_total
        best_t, best_i = max(scored)
        tie = sum(1 for t, _ in scored if t == best_t) > 1
        if tie:

            best_i = ctx.rng.choice([i for t, i in scored if t == best_t])
        label = "rubrics_tie_random" if tie else "rubrics_score"
        return indices[best_i], label, usage_total

class SLArgminNLL(Strategy):
    """Base SL selector: generate without privileged information, then rescore.

    This base class uses k=2; public SL configurations override the count.
    Choose the lowest mean_nll_privileged after the runner scores candidates.
    Both plain and privileged NLL remain available for the gain variant.
    """

    name = "sl_argmin_nll"
    K = 2

    def plan(self, ctx: TurnContext) -> int:
        return self.K

    async def select(self, ctx, cands, judge):
        indices = [i for i, c in enumerate(cands) if c.get("command")]
        valid = [cands[i] for i in indices]
        if len(valid) < 2:
            return (indices[0] if valid else 0,
                    "only_valid_toolcall" if valid else "both_invalid", {})
        if len({tuple(cmd for _, cmd in bash_actions(c['msg'])) for c in valid}) == 1:
            return ctx.rng.choice(indices), "identical_command", {}
        scored = [(c.get("mean_nll_privileged"), i) for i, c in enumerate(valid)
                  if isinstance(c.get("mean_nll_privileged"), (int, float))]
        if not scored:
            
            return indices[0], "priv_nll_unavailable_first", {}
        best = min(t for t, _ in scored)
        ties = [i for t, i in scored if t == best]
        best_i = ctx.rng.choice(ties) if len(ties) > 1 else ties[0]
        return indices[best_i], "priv_nll_tie_random" if len(ties) > 1 else "priv_nll_argmin", {}

BUILTIN = {"baseline": Strategy, "selfguide_k2": SelfGuideK2,
           "selfguide_rubrics": SelfGuideRubrics, "sl_argmin_nll": SLArgminNLL}

def load_strategy(spec: str) -> Strategy:
    """Load a built-in strategy or a candidate file defining Strategy."""
    if spec in BUILTIN:
        return BUILTIN[spec]()
    p = Path(spec)
    if not p.exists():
        raise SystemExit(f"strategy {spec!r} is neither a builtin {sorted(BUILTIN)} nor a file")

    mod_name = f"candidate_strategy_{p.parent.name or p.stem}"
    mod_spec = importlib.util.spec_from_file_location(mod_name, p)
    if mod_spec is None or mod_spec.loader is None:
        raise SystemExit(f"{p}: not importable as a Python module")
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[mod_name] = mod
    try:
        mod_spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    for attr in ("Strategy", "STRATEGY"):
        obj = getattr(mod, attr, None)
        if obj is None:
            continue
        inst = obj() if isinstance(obj, type) else obj

        if not getattr(inst, "name", None):
            inst.name = p.parent.name if p.parent.name != "." else p.stem
        for required in ("plan", "select"):
            if not callable(getattr(inst, required, None)):
                raise SystemExit(f"{p}: Strategy is missing a callable {required}()")
        return inst
    raise SystemExit(f"{p} defines neither Strategy nor STRATEGY")

def load_official_config(python_bin: str) -> dict:
    import subprocess
    import yaml
    r = subprocess.run(
        [python_bin, "-c",
         "import importlib.util,pathlib;s=importlib.util.find_spec('minisweagent');"
         "print(pathlib.Path(s.origin).parent/'config'/'benchmarks'/'swebench.yaml')"],
        capture_output=True, text=True, timeout=60,
    )
    path = (r.stdout or "").strip()
    if not path or not Path(path).exists():
        raise SystemExit(f"cannot locate mini-swe-agent swebench.yaml via {python_bin} ({r.stderr[:200]})")
    config = yaml.safe_load(Path(path).read_text())
    config.setdefault("agent", {}).update(AGENT_TEMPLATES)
    config.setdefault("model", {}).update(MODEL_TEMPLATES)
    return config

_HARMONY_CALL = re.compile(
    r"to=functions\.(?P<name>[A-Za-z_]\w*).*?<\|message\|>(?P<args>.*?)(?:<\|call\|>|<\|end\|>|$)",
    re.S)
_HARMONY_TOKEN = re.compile(r"<\|(?:start|end|call|return|message|channel|constrain)\|>")

def normalize_harmony_message(msg: dict) -> bool:
    """Repair malformed Harmony messages in place; report tool-call recovery."""
    content = msg.get("content") or ""
    salvaged = False
    if not (msg.get("tool_calls") or []) and "<|message|>" in content:

        calls = []
        first = None
        for m in _HARMONY_CALL.finditer(content):
            args = m.group("args").strip()
            try:
                json.loads(args)
            except json.JSONDecodeError:
                continue
            if first is None:
                first = m
            calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                          "function": {"name": m.group("name"), "arguments": args}})
        if calls:
            msg["tool_calls"] = calls

            head = content[: first.start()]
            marks = [i for i in (head.rfind("<|start|>"), head.rfind("<|channel|>")) if i >= 0]
            content = head[: min(marks)] if marks else head
            salvaged = True
    cleaned = _HARMONY_TOKEN.sub("", content)
    if cleaned != content or salvaged:
        msg["content"] = cleaned.strip()
    return salvaged

def parse_actions(msg: dict) -> list[tuple[str, str]]:
    actions = bash_actions(msg)
    if actions:
        for call in msg['tool_calls']:
            if not call.get('id'):
                call['id'] = f"call_{uuid.uuid4().hex[:12]}"
        actions = bash_actions(msg)
    return actions

_EDIT_HINT = MUTATION_MARKERS

async def run_one(row: dict, cfg: dict, args, strategy: Strategy, sem: asyncio.Semaphore,
                  sample: int = 0, aux: dict | None = None, endpoint: str | None = None) -> dict:
    if args.scaffold not in SCAFFOLDS:
        raise ValueError("Scaffold must be none, SJ, SL or SJ+SL")
    from si2ca.runtime import swe
    from si2ca.runtime.sandbox import create_sandbox

    md = row["metadata"]
    iid, workdir = md["instance_id"], (md.get("workdir") or "/testbed")
    ag, mo, en = cfg.get("agent", {}), cfg.get("model", {}), cfg.get("environment", {})
    step_limit = int(getattr(args, "max_turns", None) or ag.get("step_limit") or 250)
    cmd_timeout = int(en.get("timeout") or 60)
    max_fmt_err = int(ag.get("max_consecutive_format_errors") or 3)

    clients_to_close: list = []
    _stage = md.get("_stage")
    def _mark(where: str) -> None:
        if isinstance(_stage, dict):
            _stage["at"], _stage["t"] = where, time.time()
    rec = {"instance_id": iid, "sample": sample, "repo": md.get("repo"), "reward": 0.0, "turns": 0, "abort": None,
           "diff_len": 0, "applied": None, "strategy": strategy.name,
           "harness_protocol": HARNESS_PROTOCOL,
           "gen_prompt_tokens": 0, "gen_completion_tokens": 0,
           "judge_prompt_tokens": 0, "judge_completion_tokens": 0,
           "n_candidates_drawn": 0, "harmony_salvaged": 0, "empty_draws": 0,
           "dropped_calls": 0, "decisions": {}}
    traj: list[dict] = []
    messages: list[dict] = []
    deepswe = bool(md.get("deepswe")) or md.get("benchmark") == "deepswe" or getattr(args, "benchmark", None) == "deepswe"
    if "gold_validated" in md:
        rec["gold_validated"] = md["gold_validated"]

    async with sem:
        t0 = time.time()
        try:
            _mark("create_sandbox")
            async with create_sandbox(md["image"]) as sb:

                await swe.prepare_workspace(sb, workdir, md)

                _mark("baseline_dirty_scan")
                _, _dirty0, _ = await sb.exec(
                    f"cd {shlex.quote(workdir)} && git config --global --add safe.directory {shlex.quote(workdir)} 2>/dev/null && "
                    f"git status --porcelain 2>/dev/null | sed 's/^...//'", timeout=120)
                baseline_dirty = sorted({p.strip().rstrip("/") for p in (_dirty0 or "").splitlines()
                                         if p.strip()})
                rec["baseline_dirty"] = baseline_dirty

                sys_prompt = Template(ag["system_template"].replace("/testbed", workdir)).render()
                user_prompt = Template(ag["instance_template"].replace("/testbed", workdir)).render(
                    task=md["problem_statement"])
                messages = [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}]

                hint_text = (aux or {}).get("hints", {}).get(iid) or ""
                gold_text = (aux or {}).get("gold", {}).get(iid) or (md.get("gold_patch") or "")
                scaffold = args.scaffold
                rec["scaffold"] = scaffold
                sj_gold_block = ""
                priv_block_msg = None
                if scaffold in ("SL", "SJ+SL"):

                    body = hint_text if args.inject_source == "hint" else gold_text
                    rec["priv_source"] = args.inject_source
                    rec["priv_position"] = args.inject_position
                    if body:
                        from si2ca.runtime import privileged_prompt as gold_prompt
                        if args.inject_template:
                            tpl = Path(args.inject_template).read_text()
                            priv_block_msg = {"role": "user",
                                              "content": tpl.replace("{{reference_patch}}", body)}
                        else:

                            tpl_name = ("hint" if args.inject_source == "hint"
                                        else args.inject_position)
                            priv_block_msg, _hm = gold_prompt.gold_message(
                                body, len(body) + 1, tpl_name)
                        rec["priv_used"], rec["priv_chars"] = True, len(body)
                    else:
                        rec["priv_used"] = False  
                if scaffold in ("SJ", "SJ+SL"):
                    rec["gold_in_judge"] = bool(gold_text)
                    if gold_text:

                        sj_gold_block = (
                            MESSAGES['judge']['reference_solution'].format(patch=gold_text[:getattr(args, "gold_max_chars", 100000000)]))

                chat_base = f"{endpoint}/v1" if endpoint else args.base_url
                native_base = endpoint if endpoint else args.base_url.rsplit("/v1", 1)[0]

                request_timeout = float(getattr(args, "request_timeout", 900))
                client = httpx.AsyncClient(
                    base_url=chat_base,
                    timeout=httpx.Timeout(request_timeout, connect=60.0),
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=2))
                clients_to_close.append(client)
                judge_client = httpx.AsyncClient(timeout=httpx.Timeout(request_timeout, connect=60.0))
                clients_to_close.append(judge_client)
                judge_url = getattr(args, "judge_url", None) or chat_base.rstrip("/") + "/chat/completions"
                retries = int(getattr(args, "request_retries", 2))
                native_client = None
                if priv_block_msg is not None:
                    
                    native_client = httpx.AsyncClient(
                        base_url=native_base,
                        headers={"Authorization": f"Bearer {os.environ.get('SI2CA_API_KEY', 'dummy')}"},
                        timeout=httpx.Timeout(300.0, connect=60.0),
                        limits=httpx.Limits(max_connections=8, max_keepalive_connections=2))
                    clients_to_close.append(native_client)

                async def draw_once(overrides: dict | None = None) -> dict:
                    payload = {
                        "model": args.model, "messages": messages, "tools": [BASH_TOOL],
                        "parallel_tool_calls": False,
                    }

                    if args.reasoning_effort:
                        payload["reasoning_effort"] = args.reasoning_effort
                    if args.temperature is not None:
                        payload["temperature"] = args.temperature
                    if args.top_p is not None:
                        payload["top_p"] = args.top_p
                    if args.top_k is not None:
                        payload["top_k"] = args.top_k

                    if args.logprobs:
                        payload["logprobs"] = True

                    if hasattr(strategy, "payload_overrides"):
                        try:
                            base_ov = strategy.payload_overrides()
                            if base_ov:
                                payload.update(base_ov)
                        except Exception as e:
                            print(f"[payload_overrides] {iid} error: {e}", flush=True)

                    if overrides:
                        payload.update(overrides)

                    if args.max_gen_tokens:
                        payload["max_tokens"] = min(int(payload.get("max_tokens") or args.max_gen_tokens),
                                                    args.max_gen_tokens)

                    body = None
                    _dt = 0
                    while body is None:
                        try:
                            r = await client.post("/chat/completions", json=chat_payload(payload),
                                                  headers={"Authorization": f"Bearer {os.environ.get('SI2CA_API_KEY', 'dummy')}"})
                            r.raise_for_status()
                            body = r.json()
                        except httpx.HTTPStatusError as e:
                            if (e.response.status_code < 500 and e.response.status_code != 429) or _dt >= retries:
                                raise
                            _dt += 1
                            await asyncio.sleep(min(15 * _dt, 60))
                        except httpx.TransportError:
                            if _dt >= retries:
                                raise
                            _dt += 1
                            await asyncio.sleep(min(15 * _dt, 60))
                    u = body.get("usage") or {}
                    choice = body["choices"][0]
                    msg = choice["message"]

                    raw_content = msg.get("content") or ""
                    raw_reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")
                    salvaged = normalize_harmony_message(msg)
                    acts = parse_actions(msg)

                    lp_items = ((choice.get("logprobs") or {}).get("content")) or []
                    logps = [t["logprob"] for t in lp_items
                             if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))]
                    mean_nll = (-sum(logps) / len(logps)) if logps else None

                    return {"msg": msg, "usage": u, "salvaged": salvaged,
                            "finish_reason": choice.get("finish_reason"),
                            "raw_content": raw_content, "raw_reasoning": raw_reasoning,
                            "command": "\n".join(cmd for _, cmd in acts) if acts else None,
                            "tool_call_id": acts[0][0] if acts else "", "actions": acts,
                            "logprobs": logps or None, "mean_nll": mean_nll,
                            "n_gen_tokens": len(logps)}

                async def draw() -> dict:
                    """Resample missing actions without adding format-correction history.

                    Historical gpt-oss runs sometimes ended after analysis without an action.
                    Adding a correction message could trap the model in discussing missing
                    tool calls; three smoke-test tasks aborted early that way. Resampling
                    raised the estimated action yield from about 85% to 99.7% over three draws.

                    Count retry tokens where they are spent to avoid understating model cost.
                    """
                    out = await draw_once()
                    for attempt in range(1, max(1, args.draw_retries)):
                        if out["actions"]:
                            return out

                        ov = None
                        if hasattr(strategy, "retry_overrides"):
                            try:
                                ov = strategy.retry_overrides(attempt)
                            except Exception as e:
                                print(f"[retry_overrides] {iid} error: {e}", flush=True)
                        retry = await draw_once(ov)
                        if ov and retry["actions"]:
                            
                            tag = ",".join(f"{k}={v}" for k, v in sorted(ov.items()))
                            rec.setdefault("retry_rescued", {})
                            rec["retry_rescued"][tag] = rec["retry_rescued"].get(tag, 0) + 1
                        for k in ("prompt_tokens", "completion_tokens"):
                            retry["usage"][k] = int(retry["usage"].get(k) or 0) + \
                                int(out["usage"].get(k) or 0)
                        retry["n_empty_draws"] = out.get("n_empty_draws", 0) + 1
                        retry["n_trunc_draws"] = out.get("n_trunc_draws", 0) + \
                            (1 if out.get("finish_reason") == "length" else 0)
                        out = retry
                    return out

                judge_log: list[dict] = []          
                cur_cands: dict = {"cands": []}    

                async def judge_call(prompt: str, system: str | None = None) -> tuple[str, dict]:

                    if sj_gold_block:
                        if GOLD_SLOT in prompt:
                            prompt = prompt.replace(GOLD_SLOT, sj_gold_block)
                        else:
                            prompt = sj_gold_block + "\n\n" + prompt
                    elif GOLD_SLOT in prompt:
                        prompt = prompt.replace(GOLD_SLOT, MESSAGES['judge']['no_reference'])
                    msgs = ([{"role": "system", "content": system}] if system else []) \
                        + [{"role": "user", "content": prompt}]
                    jpayload = {"model": getattr(args, "judge_model", None) or args.model, "messages": msgs,
                                "temperature": getattr(args, "judge_temperature", 1.0),
                                "top_p": getattr(args, "judge_top_p", .95),
                                "chat_template_kwargs": {"enable_thinking": getattr(args, "judge_thinking", True)}}
                    judge_effort = getattr(args, "judge_reasoning_effort", args.reasoning_effort)
                    if judge_effort:
                        jpayload["reasoning_effort"] = judge_effort
                    
                    judge_cap = getattr(args, "judge_max_tokens", args.max_gen_tokens)
                    if judge_cap:
                        jpayload["max_tokens"] = judge_cap

                    b = None
                    _jt = 0
                    while b is None:
                        try:
                            r = await judge_client.post(judge_url, json=chat_payload(jpayload, judge=True),
                                                  headers={"Authorization": f"Bearer {os.environ.get('SI2CA_JUDGE_API_KEY') or os.environ.get('SI2CA_API_KEY', 'dummy')}"})
                            r.raise_for_status()
                            b = r.json()
                        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                            permanent = isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429
                            if permanent or _jt >= retries:
                                raise
                            _jt += 1
                            await asyncio.sleep(min(15 * _jt, 60))
                    text = b["choices"][0]["message"].get("content") or ""
                    usage = b.get("usage") or {}

                    ci = getattr(judge_call, "ctx", {}).get("cand_idx") if getattr(judge_call, "ctx", None) else None
                    if ci is None:
                        ci = _infer_cand_idx(prompt, cur_cands["cands"])
                    si = sum(1 for e in judge_log if e.get("cand_idx") == ci) if ci is not None else None
                    judge_log.append({"cand_idx": ci, "sample_idx": si,
                                      "prompt_len": len(prompt), "prompt_tail": prompt[-2000:],
                                      "reply": text, "scores": _parse_judge_scores(text),
                                      "usage": {k: usage.get(k) for k in
                                                ("prompt_tokens", "completion_tokens")}})
                    return text, usage

                cot_msgs: list[dict] = []     
                fmt_err, submitted, has_edited = 0, False, False
                prior_cmds: list[str] = []
                last_obs, last_rc = "", None

                for step in range(step_limit):
                    ctx = TurnContext(step, messages, last_obs, last_rc, has_edited, prior_cmds,
                                      rng=random.Random(f"{args.seed}:{iid}:{sample}:{step}"),
                                      gold_in_judge=bool(sj_gold_block))
                    k = max(1, int(strategy.plan(ctx)))
                    _mark(f"draw(step={step})")
                    drawn = await asyncio.gather(*[draw() for _ in range(k)], return_exceptions=True)
                    cands = [c for c in drawn if not isinstance(c, BaseException)]
                    rec["generation_failures"] = rec.get("generation_failures", 0) + len(drawn) - len(cands)
                    if not cands:

                        err = next((e for e in drawn if isinstance(e, BaseException)), None)
                        if isinstance(err, httpx.HTTPStatusError):
                            rec["draw_error"] = (f"HTTP {err.response.status_code}: "
                                                 f"{err.response.text[:240]}")
                        elif err is not None:
                            rec["draw_error"] = f"{type(err).__name__}: {str(err)[:240]}"
                        rec["abort"] = "all_draws_failed"
                        print(f"[draw-fail] {iid} step={step} {rec.get('draw_error')}", flush=True)
                        break
                    rec["n_candidates_drawn"] += len(cands)
                    for c in cands:
                        rec["gen_prompt_tokens"] += int(c["usage"].get("prompt_tokens") or 0)
                        rec["gen_completion_tokens"] += int(c["usage"].get("completion_tokens") or 0)

                    if priv_block_msg is not None:

                        _mark(f"rescore(step={step})")
                        await rescore_privileged(
                            native_client, _rescore_tokenizer(args.tokenizer_path),
                            messages, priv_block_msg, cands,
                            position=args.inject_position)

                    if len(cands) == 1:
                        idx, label, jusage = 0, "single_candidate", {}
                    else:
                        cur_cands["cands"] = cands
                        idx, label, jusage = await strategy.select(ctx, cands, judge_call)
                    rec["judge_prompt_tokens"] += int((jusage or {}).get("prompt_tokens") or 0)
                    rec["judge_completion_tokens"] += int((jusage or {}).get("completion_tokens") or 0)
                    rec["decisions"][label] = rec["decisions"].get(label, 0) + 1

                    win = cands[idx]
                    msg = win["msg"]
                    rec["turns"] = step + 1
                    rec["harmony_salvaged"] += int(bool(win.get("salvaged")))
                    rec["empty_draws"] += sum(int(c.get("n_empty_draws") or 0) for c in cands)
                    rec["dropped_calls"] += int(msg.get("_dropped_calls") or 0)

                    keep = {k2: v for k2, v in msg.items()
                            if k2 in ("role", "content", "tool_calls", "reasoning_content", "reasoning")}

                    if not win["actions"]:
                        keep.pop("tool_calls", None)
                    elif not getattr(args, "keep_reasoning", True):
                        keep["content"] = ""
                        keep.pop("reasoning_content", None)
                        keep.pop("reasoning", None)
                    if args.cot_in_content and not (keep.get("content") or "").strip():

                        keep["content"] = (msg.get("reasoning_content")
                                           or msg.get("reasoning") or "")

                        cot_msgs.append(keep)
                        for stale in cot_msgs[:-args.cot_in_content]:
                            stale["content"] = ""
                    messages.append(keep)

                    turn_rec = {"action": win["command"] or (msg.get("content") or ""),
                                "finish_reason": win.get("finish_reason"),
                                "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or ""),
                                "observation": "", "decision": label, "k": len(cands),

                                "judge_calls": list(judge_log),

                                "candidates": [{"command": c.get("command"),
                                                "reasoning": ((c.get("msg") or {}).get("reasoning_content")
                                                              or (c.get("msg") or {}).get("reasoning") or ""),
                                                "content": (c.get("msg") or {}).get("content") or "",
                                                "mean_nll": c.get("mean_nll"),
                                                "mean_nll_privileged": c.get("mean_nll_privileged"),
                                                "n_gen_tokens": c.get("n_gen_tokens"),
                                                "logprobs": c.get("logprobs"),
                                                "salvaged": bool(c.get("salvaged")),
                                                "n_empty_draws": c.get("n_empty_draws")}
                                               for c in cands],

                                "cand_mean_nll": [c.get("mean_nll") for c in cands],
                                "cand_mean_nll_privileged":
                                    [c.get("mean_nll_privileged") for c in cands],
                                "chosen_idx": idx,
                                "obs_state": {"last_returncode": ctx.last_returncode,
                                              "has_edited": ctx.has_edited,
                                              "obs_had_error": bool(ctx.last_observation) and
                                                  any(t in ctx.last_observation for t in
                                                      ("Traceback", "Error", "error:", "FAILED")),
                                              "obs_truncated": "elided" in (ctx.last_observation or ""),
                                              "obs_len": len(ctx.last_observation or "")}}
                    judge_log.clear()

                    fr = win.get("finish_reason") or "none"
                    rec.setdefault("finish_reasons", {})
                    rec["finish_reasons"][fr] = rec["finish_reasons"].get(fr, 0) + 1
                    rec["trunc_draws"] = rec.get("trunc_draws", 0) + win.get("n_trunc_draws", 0)
                    if not win["actions"] and fr == "length":
                        rec["trunc_noaction_turns"] = rec.get("trunc_noaction_turns", 0) + 1

                    if not win["actions"]:

                        turn_rec["raw_content"] = (win.get("raw_content") or "")[:20000]
                        turn_rec["raw_reasoning"] = (win.get("raw_reasoning") or "")[:20000]
                        fmt_err += 1
                        if fmt_err >= max_fmt_err:
                            rec["abort"] = "format_errors"
                            traj.append(turn_rec)
                            break
                        nudge = Template(
                            mo.get("format_error_template")
                            or MESSAGES['harness']['strategy_format_error']).render(
                                finish_reason="stop", error="no tool call", actions=[])
                        messages.append({"role": "user", "content": nudge})
                        turn_rec["observation"] = nudge
                        traj.append(turn_rec)
                        continue
                    fmt_err = 0

                    obs_all = []
                    for action_index, (tc_id, cmd) in enumerate(win["actions"]):
                        prior_cmds.append(cmd)
                        if any(h in cmd for h in _EDIT_HINT):
                            has_edited = True
                        rc, so, se = await sb.exec(
                            f"cd {shlex.quote(workdir)} && {cmd}", user="agent",
                            env={str(k): str(v) for k, v in (en.get("env") or {}).items() if k != "BASH_ENV"},
                            timeout=cmd_timeout)
                        submitted = is_submission(rc, (so or "") + (se or ""))
                        obs = Template(
                            mo.get("observation_template")
                            or MESSAGES['harness']['strategy_observation']).render(
                            output={"returncode": rc, "output": (so or "") + (se or ""),
                                    "exception_info": ""})
                        messages.append({"role": "tool", "tool_call_id": tc_id, "content": obs})
                        obs_all.append(obs)
                        last_rc = rc
                        if submitted:
                            rec["skipped_after_submit"] = len(win["actions"]) - action_index - 1
                            keep["tool_calls"] = keep["tool_calls"][:action_index + 1]
                            break
                    last_obs = "\n".join(obs_all)
                    turn_rec["observation"] = last_obs[:20000]

                    if os.environ.get("SLIME_ALLOW_NUDGE") == "1" and hasattr(strategy, "nudge"):
                        try:
                            nctx = TurnContext(step + 1, messages, last_obs, last_rc,
                                               has_edited, prior_cmds,
                                               rng=random.Random(f"{args.seed}:{iid}:{sample}:n{step}"),
                                               gold_in_judge=bool(sj_gold_block))
                            nctx.instance_id, nctx.sample = iid, sample
                            hint = strategy.nudge(nctx)
                        except Exception as e:  
                            print(f"[nudge] {iid} step={step} error: {type(e).__name__}: {e}", flush=True)
                            hint = None
                        if hint:
                            messages.append({"role": "user", "content": hint})
                            rec["nudges"] = rec.get("nudges", 0) + 1
                            turn_rec["nudge"] = hint[:200]

                    traj.append(turn_rec)
                    if submitted:
                        break
                else:
                    rec["abort"] = "max_turns"

                rec["submitted"] = submitted

                excl = " ".join(shlex.quote(f":(exclude){p}") for p in baseline_dirty)
                _, diff, _ = await sb.exec(
                    f"cd {shlex.quote(workdir)} && git add -A -- . >/dev/null 2>&1 && "
                    f"git diff --cached -- . {excl}".rstrip(), timeout=120)
                diff = diff or ""
                rec["diff_len"] = len(diff)
                patches = Path(args.out).parent / "patches"
                patches.mkdir(parents=True, exist_ok=True)
                patch_path = patches / f"{iid}_s{sample}.patch"
                patch_path.write_text(diff, encoding="utf-8")
                rec["patch_path"] = str(patch_path)
                if deepswe:
                    from si2ca.runtime.deepswe_grading import grade
                    rec["reward"], rec["grade_status"], rec["grade_detail"] = await grade(sb, md)
                    rec["applied"] = rec["grade_status"] == "OK"
                    if not rec["applied"]:
                        rec["abort"] = rec["abort"] or "grading_infrastructure"
                rec["grading_protocol"] = "deepswe_in_place" if deepswe else ("multilingual" if md.get("benchmark") == "swebench_multilingual" else "fresh_patch")

            if diff.strip() and not deepswe:
                _mark("grading")
                if md.get("benchmark") == "swebench_multilingual":

                    from si2ca.runtime.multilingual import evaluate_multilingual
                    reward, applied, gd = await evaluate_multilingual(
                        image=md["image"], workdir=workdir, diff_text=diff,
                        metadata=md, timeout_sec=args.eval_timeout)
                    
                    rec["grade_detail"] = gd
                    if gd.get("error") or gd.get("markers_missing"):
                        rec["abort"] = "grading_infrastructure"
                        rec["error"] = str(gd)
                else:
                    reward, applied = await swe.evaluate(
                        image=md["image"], workdir=workdir, diff_text=diff,
                        swepro=md.get("swepro"), eval_cmd=md.get("eval_cmd"),
                        eval_files=md.get("eval_files"), pre_commands=md.get("pre_commands"),
                        timeout_sec=args.eval_timeout)
                rec["reward"], rec["applied"] = float(reward), bool(applied)
                if not applied:
                    rec["abort"] = rec["abort"] or "grading_infrastructure"
        except asyncio.CancelledError:
            rec["abort"] = "task_timeout"
        except Exception as e:  
            rec["abort"] = rec["abort"] or f"exception:{type(e).__name__}"
            rec["error"] = str(e)[:400]

        for _c in clients_to_close:
            try:
                await _c.aclose()
            except Exception as _e:  
                print(f"[warn] {iid}: could not close HTTP client: {type(_e).__name__}", flush=True)

        rec["elapsed"] = round(time.time() - t0, 1)
        rec["total_tokens"] = (rec["gen_prompt_tokens"] + rec["gen_completion_tokens"]
                               + rec["judge_prompt_tokens"] + rec["judge_completion_tokens"])
        rec["execution_status"] = "infrastructure_failure" if infrastructure_failure(rec) else "completed"
        if args.traj_dir:
            d = Path(args.traj_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / (f"{iid}.json" if args.n_samples == 1 else f"{iid}_s{sample}.json")).write_text(json.dumps(
                {"instance_id": iid, "sample": sample, "reward": rec["reward"],
                 "messages": messages, "result": rec,
                 "strategy": strategy.name,
                 "problem_statement": md.get("problem_statement", ""), "turns": traj},
                ensure_ascii=False), encoding="utf-8")
        print(f"[done] {iid[:46]:<46} reward={rec['reward']} turns={rec['turns']:>3} "
              f"abort={rec['abort']} diff={rec['diff_len']}B {rec['elapsed']}s "
              f"tok={rec['total_tokens']:,}", flush=True)
        return rec

async def run_one_bounded(row: dict, cfg: dict, args, strategy: Strategy,
                          sem: asyncio.Semaphore, sample: int = 0,
                          aux: dict | None = None, pool=None) -> dict:
    """Hold the concurrency slot outside a task-wide wall-clock timeout.

    Per-operation timeouts alone can leave all slots occupied by stalled
    sandbox calls. A historical six-slot run sent no requests for 82 minutes
    despite healthy endpoints, finishing only 93 of 384 rollouts before shutdown.

    The limit must exceed a complete legitimate task, not just one operation.
    The default 3,600 seconds allows room for the 1,800-second grading budget.
    """
    async with sem:
        t0 = time.time()

        endpoint = None
        if pool:
            try:
                endpoint = await asyncio.wait_for(pool.acquire(), timeout=ENDPOINT_WAIT_SEC)
            except asyncio.TimeoutError:
                md = row["metadata"]
                print(f"[fatal] {md['instance_id'][:40]} no healthy endpoint within "
                      f"{ENDPOINT_WAIT_SEC}s; check serving logs for "
                      f"kill_process_tree", flush=True)
                return {"instance_id": md["instance_id"], "sample": sample, "repo": md.get("repo"),
                        "reward": 0.0, "turns": 0, "abort": "no_healthy_endpoint",
                        "diff_len": 0, "applied": None, "strategy": strategy.name,
                        "harness_protocol": HARNESS_PROTOCOL,
                        "gen_prompt_tokens": 0, "gen_completion_tokens": 0,
                        "judge_prompt_tokens": 0, "judge_completion_tokens": 0,
                        "n_candidates_drawn": 0, "harmony_salvaged": 0, "empty_draws": 0,
                        "decisions": {}, "elapsed": round(time.time() - t0, 1), "total_tokens": 0}

        stage = {"at": "start", "t": time.time()}
        row.setdefault("metadata", {})["_stage"] = stage

        async def _watch():
            while True:
                await asyncio.sleep(600)
                held = time.time() - stage["t"]
                if held > 600:
                    print(f"[stall] {row['metadata']['instance_id'][:40]} has remained at stage "
                          f"{stage['at']!r} for {held / 60:.0f} minutes", flush=True)

        watcher = asyncio.ensure_future(_watch())
        try:
            
            return await asyncio.wait_for(
                run_one(row, cfg, args, strategy, asyncio.Semaphore(1 << 20), sample,
                        aux=aux, endpoint=endpoint),
                timeout=args.task_timeout)
        except asyncio.TimeoutError:
            md = row["metadata"]
            iid = md["instance_id"]
            print(f"[stall] {iid[:40]} reached task_timeout; last stage: {stage['at']!r}",
                  flush=True)

            print(f"[done] {iid[:46]:<46} reward=0.0 turns=  0 abort=task_timeout "
                  f"diff=0B {round(time.time() - t0, 1)}s tok=0", flush=True)
            return {"instance_id": iid, "sample": sample, "repo": md.get("repo"),
                    "reward": 0.0, "turns": 0, "abort": "task_timeout",
                    "diff_len": 0, "applied": None, "strategy": strategy.name,
                    "harness_protocol": HARNESS_PROTOCOL,
                    "gen_prompt_tokens": 0, "gen_completion_tokens": 0,
                    "judge_prompt_tokens": 0, "judge_completion_tokens": 0,
                    "n_candidates_drawn": 0, "decisions": {},
                    "elapsed": round(time.time() - t0, 1), "total_tokens": 0,
                    "stalled_at": stage["at"]}
        finally:
            watcher.cancel()
            if pool:
                await pool.release(endpoint)

async def amain(args) -> None:
    if args.scaffold not in SCAFFOLDS:
        raise ValueError("Scaffold must be none, SJ, SL or SJ+SL")
    for name in ("max_turns", "k", "score_samples", "branch_until_edit", "request_timeout"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if getattr(args, "request_retries", 2) < 0:
        raise ValueError("request_retries must be nonnegative")
    gold_weight = getattr(args, "gold_weight", None)
    if gold_weight is not None and not 0 <= gold_weight <= 1:
        raise ValueError("gold_weight must be in [0, 1]")
    cfg = load_official_config(args.config_python)
    strategy = load_strategy(args.strategy)
    if hasattr(args, "k") and args.k is not None:
        strategy.K = args.k
        if hasattr(strategy, "BRANCH_K"):
            strategy.BRANCH_K = args.k
    if hasattr(args, "score_samples") and args.score_samples is not None:
        strategy.SCORE_SAMPLES = args.score_samples
    if hasattr(args, "gold_weight") and args.gold_weight is not None:
        strategy.GOLD_WEIGHT = args.gold_weight
    if hasattr(args, "branch_until_edit") and args.branch_until_edit is not None:
        strategy.MAX_PRIOR_MUTATIONS = args.branch_until_edit - 1
    strategy.prefix_config = SimpleNamespace(
        prefix_tool_result_head_chars=getattr(args, "prefix_head_chars", 1500),
        prefix_tool_result_tail_chars=getattr(args, "prefix_tail_chars", 1500),
        prefix_max_chars=getattr(args, "prefix_max_chars", 80000))
    rows = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    aux = {"hints": {}, "gold": {}}
    if args.hints_file:
        for ln in Path(args.hints_file).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("instance_id") and (r.get("hint") or "").strip():
                    aux["hints"][r["instance_id"]] = r["hint"].strip()
    if args.gold_file:
        for ln in Path(args.gold_file).read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                md_ = r.get("metadata") or {}
                iid_ = md_.get("instance_id") or r.get("instance_id")
                g = (md_.get("gold_patch") or r.get("gold_patch") or "").strip()
                if iid_ and g:
                    aux["gold"][iid_] = g
    if args.scaffold in ("SL", "SJ+SL") and not args.logprobs:
        
        print("[scaffold] SL requires logprobs; enabling --logprobs", flush=True)
        args.logprobs = True
    if args.scaffold != "none":
        ids = [r["metadata"]["instance_id"] for r in rows]
        hcov = sum(1 for i in ids if i in aux["hints"])
        gcov = sum(1 for i in ids if i in aux["gold"]
                   or (next((r for r in rows if r["metadata"]["instance_id"] == i), {})
                       .get("metadata", {}).get("gold_patch")))
        if args.scaffold == "SJ+SL":
            
            need, cov = (("hints+gold", min(hcov, gcov)) if args.inject_source == "hint"
                         else ("gold", gcov))
        elif args.scaffold == "SL":
            need, cov = (("hints", hcov) if args.inject_source == "hint"
                         else ("gold", gcov))
        else:
            need, cov = ("gold", gcov)
        print(f"[scaffold] {args.scaffold}: hints {hcov}/{len(ids)} · gold {gcov}/{len(ids)}",
              flush=True)
        if cov < len(ids) and not args.allow_partial_scaffold:
            raise SystemExit(
                f"[scaffold] {args.scaffold} requires {need} coverage for {len(ids)}/{len(ids)} tasks, "
                f"but found {cov}. Complete the sidecar or explicitly use --allow-partial-scaffold")

    eps = split_endpoints(args.base_url)
    pool = None
    if len(eps) > 1:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   
        from si2ca.runtime.endpoint_pool import EndpointPool
        pool = EndpointPool(eps)

    print(f"[run] {len(rows)} tasks · strategy={strategy.name} · scaffold={args.scaffold} · "
          f"model={args.model} · conc={args.concurrency} · "
          f"endpoints={len(eps)}{'(sticky)' if pool else ''} · "
          f"reasoning_effort={args.reasoning_effort or 'endpoint-default'} · "
          f"step_limit={cfg['agent'].get('step_limit')}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    recs: list[dict] = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    done_keys: set = set()
    if args.resume and Path(args.out).exists():
        prev = json.loads(Path(args.out).read_text())
        if not isinstance(prev, dict) or prev.get("harness_protocol") != HARNESS_PROTOCOL:
            raise ValueError("Output uses another or unversioned harness; choose a new --out")
        if (prev.get("model") != args.model or prev.get("strategy") != strategy.name
                or prev.get("scaffold") != args.scaffold):
            raise ValueError("Output configuration differs; choose a new --out")
        for r in prev.get("results", []):
            if r.get("harness_protocol") != HARNESS_PROTOCOL:
                raise ValueError("Output contains mixed harness versions; choose a new --out")
            if r.get("scaffold", args.scaffold) != args.scaffold:
                raise ValueError("Output contains different scaffolds; choose a new --out")
            if infrastructure_failure(r):
                continue
            key = (r.get("instance_id"), r.get("sample", 0))
            if key[0] is None or key in done_keys:
                continue
            done_keys.add(key)
            recs.append(r)
        print(f"[resume] {len(recs)} completed trials retained", flush=True)

    jobs = [run_one_bounded(r, cfg, args, strategy, sem, sample=si, aux=aux, pool=pool)
            for si in range(max(1, args.n_samples)) for r in rows
            if (r["metadata"]["instance_id"], si) not in done_keys]
    if done_keys:
        print(f"[resume] {len(jobs)} pending; skipped {len(done_keys)} completed records", flush=True)
    health_task = asyncio.create_task(pool.health_loop()) if pool else None
    try:
        for fut in asyncio.as_completed(jobs):
            recs.append(await fut)
        
            Path(args.out).write_text(json.dumps(
                {"harness_protocol": HARNESS_PROTOCOL, "strategy": strategy.name, "model": args.model, "scaffold": args.scaffold,
                 "results": recs},
                ensure_ascii=False, indent=2), encoding="utf-8")

    finally:
        if health_task is not None:
            health_task.cancel()
            await asyncio.gather(health_task, return_exceptions=True)

    from si2ca.results import summary
    result_summary = summary({(r['instance_id'], r.get('sample', 0)): r for r in recs},
                             len(rows) * max(1, args.n_samples))
    solved = result_summary['solved']
    gen = sum(r["gen_prompt_tokens"] + r["gen_completion_tokens"] for r in recs)
    jud = sum(r["judge_prompt_tokens"] + r["judge_completion_tokens"] for r in recs)
    dec: dict[str, int] = {}
    for r in recs:
        for k, v in (r.get("decisions") or {}).items():
            dec[k] = dec.get(k, 0) + v
    print("\n" + json.dumps(result_summary, sort_keys=True))
    print(f"    tokens: generation {gen:,} + judge {jud:,} = {gen + jud:,}"
          f"   per solve: {int((gen + jud) / max(solved, 1)):,}")
    print(f"    decisions: {dec}")
    print(f"    -> {args.out}", flush=True)

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-dir", default=None)
    ap.add_argument("--strategy", default="baseline",
                    help=f"{sorted(BUILTIN)} or a path to a candidate strategy.py")
    
    ap.add_argument("--model", default="gpt-chat-latest_2026-05-28")
    ap.add_argument("--base-url", default="http://127.0.0.1:8799/v1",
                    help="OpenAI-compatible endpoint(s). Comma-separated endpoints assign each task "
                         "to the least-busy healthy server and retain it for prefix-cache locality. "
                         "A single endpoint does not use pool health checks.")
    ap.add_argument("--cot-in-content", type=int, default=0, metavar="N",
                    help="Replay reasoning from the last N turns in content (0 disables this). "
                         "Intended for Harmony/gpt-oss templates that read reasoning from content. "
                         "Do not enable for templates such as Qwen that treat content as user-facing text.")
    ap.add_argument("--draw-retries", type=int, default=1,
                    help="Total draw attempts when a turn has no action (default 1: no retry). "
                         "Resampling avoids injecting format-correction messages into history.")
    ap.add_argument("--reasoning-effort", default=None,
                    help="Set reasoning effort per request (gpt-oss: low/medium/high). "
                         "If omitted, use the model endpoint's default.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--score-samples", type=int, default=None)
    ap.add_argument("--gold-weight", type=float, default=None)
    ap.add_argument("--branch-until-edit", type=int, default=None)
    ap.add_argument("--gold-max-chars", type=int, default=100000000)
    ap.add_argument("--judge-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-temperature", type=float, default=1.0)
    ap.add_argument("--judge-top-p", type=float, default=.95)
    ap.add_argument("--judge-reasoning-effort", default=None)
    ap.add_argument("--prefix-head-chars", type=int, default=1500)
    ap.add_argument("--prefix-tail-chars", type=int, default=1500)
    ap.add_argument("--prefix-max-chars", type=int, default=80000)
    ap.add_argument("--request-retries", type=int, default=2)
    ap.add_argument("--request-timeout", type=float, default=900)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                    help="Do not resume existing results in --out (resuming is the default)")
    ap.add_argument("--task-timeout", type=int, default=3600,
                    help="Per-task wall-clock limit in seconds; allow time for a complete task.")
    ap.add_argument("--eval-timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="Independent runs per task; use more than one to estimate sampling variance.")
    ap.add_argument("--config-python", default=sys.executable)
    
    ap.add_argument("--scaffold", choices=SCAFFOLDS, default="none",
                    help="SJ=gold in judge requests; SL=privileged likelihood scoring; SJ+SL=both; none=no privileged information")
    ap.add_argument("--hints-file", default=None,
                    help="JSONL with {instance_id, hint} per row; used by SL with --inject-source hint.")
    ap.add_argument("--gold-file", default=None,
                    help="JSONL with instance_id and gold_patch at top level or under metadata; "
                         "optional for SJ when the dataset already supplies metadata.gold_patch.")
    ap.add_argument("--inject-source", choices=("hint", "gold"), default="hint",
                    help="SL scoring content: hint uses a gold-derived hint (default, to reduce length "
                         "confounding); gold uses the full reference patch. Generation stays unprivileged.")
    ap.add_argument("--inject-position", choices=("head", "tail"), default="head",
                    help="SL scoring-block position: head follows the problem statement (default); "
                         "tail follows the history, before the candidate. Position is a search dimension.")
    ap.add_argument("--inject-template", default=None,
                    help="SL scoring-block template with a literal {{reference_patch}} placeholder. "
                         "Defaults to the published hint/head/tail template for the content and position.")
    ap.add_argument("--tokenizer-path", default=None,
                    help="Tokenizer for SL scoring; its chat template must match the served model.")
    ap.add_argument("--allow-partial-scaffold", action="store_true",
                    help="Allow incomplete sidecars instead of failing; each task records hint_used "
                         "and gold_in_judge so partial coverage remains visible.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature; if omitted, use the endpoint default. Historical "
                         "minimal runs used temperature=1.0, top_p=0.95, top_k=10; set all three for comparison.")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-gen-tokens", type=int, default=4096,
                   help="Token limit per generation, shared by sampling and judging; 0 disables it. "
                        "Default 4096 truncated 0.06%% (62/99,030) of historical 122B generations "
                        "(median 98 tokens, p99 1,673) while bounding runaway turns.")
    ap.add_argument("--logprobs", action="store_true", default=False,
                    help="Request token logprobs for each draw. Disabled by default because some "
                         "API relays reject this parameter; SL enables it automatically for NLL scoring.")
    return ap


if __name__ == "__main__":
    asyncio.run(amain(build_parser().parse_args()))
