#!/usr/bin/env python3
"""Local chat/completions adapter for Copilot REST endpoints.

Why use an adapter?
1. Some deployed models accept /responses rather than the harness's chat API.
2. The adapter explicitly supplies reasoning effort. Historical probes found
   zero reasoning tokens when effort was omitted on some deployments.
3. EndpointPool requires a local GET /health route.
4. gh auth token supplies credentials, cached for 300 seconds.

Unlike the historical CLI wrapper, direct REST calls avoid per-request process
startup and CLI concurrency limits.

Usage:
    python -m si2ca.responses_shim --model gpt-5.6-sol --effort high --port 8901
    python -m si2ca.responses_shim --model gpt-5.6-sol --probe
    curl http://127.0.0.1:8901/health
    curl http://127.0.0.1:8901/stats

Point the harness at the local endpoint. The incoming model field is ignored:
--model fixes one model per shim instance; use separate ports for comparisons.

Translation from chat to Responses:
* system/user messages become role/content input items.
* assistant content becomes an assistant message with output_text content.
* tool calls become function_call items with call_id, name, and JSON arguments.
* tool replies become function_call_output items keyed by tool_call_id.
* Nested function tools become flat function definitions.

On return, function_call items become message.tool_calls. Arguments remain
JSON strings and call_id is preserved for matching the next tool reply.
Incomplete output due to max_output_tokens maps to finish_reason="length";
otherwise tool calls map to "tool_calls", and ordinary output maps to "stop".

Historical reasoning items are not replayed. Requests use store=false rather
than retaining a server-side conversation.
"""
from __future__ import annotations

from si2ca.prompts import MESSAGES

import argparse
import asyncio
import json
import os
import subprocess
import time

from aiohttp import ClientSession, ClientTimeout, web

COPILOT_BASE = os.environ.get("COPILOT_BASE_URL", "https://api.githubcopilot.com")
HEADERS_STATIC = {
    "Content-Type": "application/json",
    "Copilot-Integration-Id": "copilot-cli",
    "X-GitHub-Api-Version": "2026-01-09",
}

# Unknown models start with chat/completions. On unsupported_api_for_model,
# switch to /responses and remember the route.
_RESPONSES_PREFIXES = ("gpt-5", "grok", "o3", "o4")


def _default_route(model: str) -> str:
    return "responses" if model.lower().startswith(_RESPONSES_PREFIXES) else "chat"


class TokenCache:
    """Fetch gh auth token on demand and cache it for 300 seconds."""

    def __init__(self) -> None:
        self._token = ""
        self._ts = 0.0
        self._lock = asyncio.Lock()

    async def get(self) -> str:
        # Tests may use COPILOT_TOKEN=dummy with COPILOT_BASE_URL pointing to a mock.
        # Prefer refreshed credentials in production; never hard-code tokens.
        fixed = os.environ.get("COPILOT_TOKEN")
        if fixed:
            return fixed
        async with self._lock:
            if self._token and time.monotonic() - self._ts < 300:
                return self._token
            # Older gh installations may lack auth token. Search GH_BIN first,
            # then ~/.local/bin/gh, then PATH.
            candidates = [c for c in (os.environ.get("GH_BIN"),
                                      os.path.expanduser("~/.local/bin/gh"),
                                      "gh") if c]
            last_err = "no gh binary tried"
            token = ""
            for gh in candidates:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        gh, "auth", "token",
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = await proc.communicate()
                except FileNotFoundError:
                    last_err = f"{gh}: not found"
                    continue
                if proc.returncode == 0 and out.decode().strip():
                    token = out.decode().strip()
                    break
                last_err = f"{gh}: rc={proc.returncode} {err.decode()[:120] or out.decode()[:120]}"
            if not token:
                raise RuntimeError(f"gh auth token failed: {last_err}")
            self._token = token
            self._ts = time.monotonic()
            return self._token


# ------------------------------------------------------------- chat -> responses
def chat_to_responses_input(messages: list[dict]) -> list[dict]:
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):  # Join content parts as a fallback for non-string input.
            content = "".join(
                c.get("text", "") for c in content if isinstance(c, dict))
        if role in ("system", "user"):
            items.append({"role": role, "content": content})
        elif role == "assistant":
            if content:
                items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                })
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": call.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "{}",
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id") or "",
                "output": content,
            })
    return items


def chat_tools_to_responses(tools: list[dict] | None) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function") or {}
        out.append({
            "type": "function",
            "name": fn.get("name"),
            "description": fn.get("description"),
            "parameters": fn.get("parameters"),
        })
    return out


def responses_to_chat(r: dict) -> dict:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in r.get("output") or []:
        t = item.get("type")
        if t == "message":
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    text_parts.append(c.get("text") or "")
        elif t == "function_call":
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or "",
                "type": "function",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "{}",
                },
            })
    msg: dict = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        finish = "tool_calls"
    elif (r.get("status") == "incomplete"
          and (r.get("incomplete_details") or {}).get("reason") == "max_output_tokens"):
        finish = "length"
    else:
        finish = "stop"
    usage_in = r.get("usage") or {}
    usage = {
        "prompt_tokens": usage_in.get("input_tokens", 0),
        "completion_tokens": usage_in.get("output_tokens", 0),
        "total_tokens": usage_in.get("total_tokens", 0),
        "prompt_tokens_details": usage_in.get("input_tokens_details") or {},
        "completion_tokens_details": usage_in.get("output_tokens_details") or {},
    }
    return {
        "id": r.get("id") or "shim",
        "object": "chat.completion",
        "model": r.get("model") or "",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": usage,
    }


# ------------------------------------------------------------------------ shim
class TrapiChannel:
    """Optional Azure OpenAI-compatible channel with a rate-limit pool separate from Copilot.

    Configured entirely through the environment: TRAPI_BASE_URL (chat endpoint
    base), TRAPI_DEPLOYMENTS (JSON object mapping model name to deployment
    name) and TRAPI_TOKEN_RESOURCE (the resource passed to az account
    get-access-token). Tokens are cached for 30 minutes. Unknown deployments
    return None so the caller can fall back to Copilot without losing the request.
    """

    BASE = os.environ.get("TRAPI_BASE_URL", "")
    DEPLOYMENTS = json.loads(os.environ.get("TRAPI_DEPLOYMENTS", "{}"))
    TOKEN_RESOURCE = os.environ.get("TRAPI_TOKEN_RESOURCE", "")

    def __init__(self) -> None:
        self._tok = ""
        self._ts = 0.0
        self._lock = asyncio.Lock()
        self._session: ClientSession | None = None

    async def _token(self) -> str:
        async with self._lock:
            if self._tok and time.monotonic() - self._ts < 1800:
                return self._tok
            proc = await asyncio.create_subprocess_exec(
                "az", "account", "get-access-token", "--resource", self.TOKEN_RESOURCE,
                "--query", "accessToken", "-o", "tsv",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = await proc.communicate()
            self._tok = out.decode().strip() if proc.returncode == 0 else ""
            self._ts = time.monotonic()
            return self._tok

    async def post(self, payload: dict, model: str):
        """Return (status, body), or (None, None) to allow caller fallback."""
        dep = self.DEPLOYMENTS.get(model)
        if not dep or not self.BASE or not self.TOKEN_RESOURCE:
            return None, None
        tok = await self._token()
        if not tok:
            return None, None
        if self._session is None or self._session.closed:
            self._session = ClientSession(timeout=ClientTimeout(
                total=None, connect=30, sock_read=3300))
        body = {**payload, "model": dep}
        # This Azure endpoint uses max_completion_tokens instead of max_tokens.
        if "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        try:
            async with self._session.post(f"{self.BASE}/chat/completions",
                                          json=body,
                                          headers={"Authorization": f"Bearer {tok}",
                                                   "Content-Type": "application/json"}) as r:
                raw = await r.text()
                try:
                    return r.status, json.loads(raw)
                except Exception:
                    return r.status, raw
        except Exception:
            return None, None


class Shim:
    def __init__(self, model: str, effort: str, max_output_tokens: int,
                 max_inflight: int) -> None:
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self.route = _default_route(model)
        self.tokens = TokenCache()
        self.sem = asyncio.Semaphore(max_inflight)
        self.stats = {"requests": 0, "ok": 0, "fail": 0, "retries_429": 0,
                      "retries_integrator": 0, "trapi_calls": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "reasoning_tokens": 0, "cached_tokens": 0}
        self._session: ClientSession | None = None
        self._rr = 0
        self._last_was_trapi = False
        self.trapi = TrapiChannel() if os.environ.get("SHIM_USE_TRAPI") else None

    async def session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            # Allow long reasoning turns while detecting connection failures quickly.
            self._session = ClientSession(timeout=ClientTimeout(
                total=None, connect=30, sock_read=3300))
        return self._session

    async def _post(self, path: str, payload: dict) -> tuple[int, dict | str]:
        # Route alternating chat requests to the optional TRAPI channel, which
        # has an independent rate-limit pool. TRAPI uses the Azure chat API;
        # Responses-format requests remain on Copilot in this helper.
        if self.trapi and path == "/chat/completions":
            self._rr += 1
            if self._rr % 2 == 0:
                st, body = await self.trapi.post(payload, self.model)
                if st is not None:
                    self.stats["trapi_calls"] = self.stats.get("trapi_calls", 0) + 1
                    return st, body
                # Missing credentials or an unknown deployment: fall back to Copilot.
                # Optional routing must not introduce a new failure source.
        sess = await self.session()
        headers = {**HEADERS_STATIC,
                   "Authorization": f"Bearer {await self.tokens.get()}"}
        async with sess.post(f"{COPILOT_BASE}{path}", json=payload,
                             headers=headers) as resp:
            raw = await resp.text()
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw

    async def upstream(self, chat_req: dict) -> tuple[int, dict | str]:
        self._last_was_trapi = False
        """Call upstream with transient-error backoff and automatic API routing."""
        messages = chat_req.get("messages") or []
        tools = chat_req.get("tools")
        last: tuple[int, dict | str] = (599, "no attempt")
        for attempt in range(6):
            # For supported models, send alternate requests to TRAPI using chat
            # payloads even when the Copilot route uses Responses. Mark the channel
            # so downstream parsing uses the chat schema, not the Responses schema;
            # parsing a chat response as Responses would yield empty content.
            used_trapi = False
            if (self.trapi and self.route == "responses"
                    and self.trapi.DEPLOYMENTS.get(self.model)):
                self._rr += 1
                if self._rr % 2 == 0:
                    tp = {**chat_req, "model": self.model}
                    if self.effort:
                        tp.setdefault("reasoning_effort", self.effort)
                    tp.setdefault("max_tokens", self.max_output_tokens)
                    st, bd = await self.trapi.post(tp, self.model)
                    if st == 200:
                        self.stats["trapi_calls"] = self.stats.get("trapi_calls", 0) + 1
                        self._last_was_trapi = True
                        return st, bd
                    # Fall back to Copilot on a non-200 response or unavailable TRAPI route.

            if self.route == "responses":
                payload: dict = {
                    "model": self.model,
                    "input": chat_to_responses_input(messages),
                    "store": False,
                    # Caller max_tokens takes precedence over the startup default.
                    # Unlike the chat branch, this payload is rebuilt from scratch.
                    # Omitting the override once caused 361/2466 historical outputs
                    # to exceed 8192 tokens, reaching 19123 tokens.
                    "max_output_tokens": int(chat_req.get("max_tokens")
                                             or self.max_output_tokens),
                }
                if tools:
                    payload["tools"] = chat_tools_to_responses(tools)
                if self.effort:
                    payload["reasoning"] = {"effort": self.effort}
                status, body = await self._post("/responses", payload)
            else:
                payload = {**chat_req, "model": self.model}
                # Some chat deployments also accept reasoning_effort. Historical
                # Claude probes accepted low through max but rejected none.
                # Missing reasoning_tokens in usage does not imply no reasoning.
                if self.effort:
                    payload.setdefault("reasoning_effort", self.effort)
                # Supply the configured output limit when the caller omits it;
                # do not depend on the upstream provider's default.
                payload.setdefault("max_tokens", self.max_output_tokens)
                status, body = await self._post("/chat/completions", payload)

            if status == 200:
                return status, body
            btxt = body if isinstance(body, str) else json.dumps(body)
            if "unsupported_api_for_model" in btxt:
                self.route = "chat" if self.route == "responses" else "responses"
                print(f"[shim] {self.model}: switched API route -> {self.route}", flush=True)
                continue
            # "not available for integrator" has been observed as a transient 400:
            # identical credentials can succeed on immediate retry. Retry only
            # this specific 400; deterministic validation/policy errors pass through.
            # Otherwise transient routing failures could exhaust harness retries
            # and invalidate a whole task (exit_code=1).
            transient_400 = status == 400 and "not available for integrator" in btxt
            if transient_400:
                self.stats["retries_integrator"] = self.stats.get("retries_integrator", 0) + 1
            if status in (429, 500, 502, 503, 504, 408) or transient_400:
                self.stats["retries_429"] += status == 429
                # Retry-After may be absent. Use exponential backoff, longer for 429;
                # transient integrator-routing failures get a shorter initial delay.
                delay = (min(2 * (2 ** attempt), 30) if transient_400
                         else min((15 if status == 429 else 5) * (2 ** attempt), 300))
                print(f"[shim] upstream {status}, retry in {delay}s "
                      f"(attempt {attempt + 1}/6): {btxt[:160]}", flush=True)
                await asyncio.sleep(delay)
                last = (status, body)
                continue
            return status, body  # Pass through non-retryable errors unchanged.
        return last

    async def handle_chat(self, request: web.Request) -> web.Response:
        t0 = time.monotonic()
        chat_req = await request.json()
        sid = (request.headers.get("Authorization") or "")[-12:]
        self.stats["requests"] += 1
        async with self.sem:
            try:
                status, body = await self.upstream(chat_req)
            except Exception as e:
                self.stats["fail"] += 1
                print(f"[shim] EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
                return web.json_response(
                    {"error": {"message": f"shim: {type(e).__name__}: {e}"}},
                    status=502)
        if status != 200:
            self.stats["fail"] += 1
            print(f"[shim] FAIL {status} sid={sid} "
                  f"{(body if isinstance(body, str) else json.dumps(body))[:200]}",
                  flush=True)
            return web.json_response(
                body if isinstance(body, dict) else {"error": {"message": body}},
                status=status if 400 <= status < 600 else 502)

        if self.route == "responses" and not self._last_was_trapi:
            out = responses_to_chat(body)  # type: ignore[arg-type]
        else:
            out = body  # type: ignore[assignment]

        # Moderation can return HTTP 200 with an empty choices list. Passing
        # that directly to LiteLLM produces a misleading missing-choices error.
        # Synthesize an empty assistant turn so the harness handles it through
        # its normal missing-tool-call retry/correction path.
        if isinstance(out, dict) and not (out.get("choices") or []):
            self.stats["moderation_empty"] = self.stats.get("moderation_empty", 0) + 1
            print(f"[shim] moderation/empty-choices sid={sid} -> synthesized empty turn", flush=True)
            out = {**out, "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant",
                            "content": MESSAGES['shim']['content_filter_retry']},
            }]}
        u = out.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        self.stats["ok"] += 1
        self.stats["prompt_tokens"] += u.get("prompt_tokens") or 0
        self.stats["completion_tokens"] += u.get("completion_tokens") or 0
        self.stats["reasoning_tokens"] += rt or 0
        self.stats["cached_tokens"] += ct or 0
        n_tc = len((out.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or [])
        print(f"[shim] ok sid={sid} {time.monotonic() - t0:.1f}s "
              f"in={u.get('prompt_tokens')} (cached={ct}) out={u.get('completion_tokens')} "
              f"(reasoning={rt}) tool_calls={n_tc} "
              f"finish={(out.get('choices') or [{}])[0].get('finish_reason')}", flush=True)
        return web.json_response(out)

    async def handle_health(self, _r: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def handle_stats(self, _r: web.Request) -> web.Response:
        return web.json_response({"model": self.model, "effort": self.effort,
                                  "route": self.route, **self.stats})


# ------------------------------------------------------------------------ probe
async def probe(model: str) -> None:
    """Probe supported Responses reasoning-effort levels and token accounting.

    Availability varies by deployment, and omitted effort may disable reasoning.
    Try levels from highest to lowest and report every supported setting.
    """
    shim = Shim(model, "", 4000, 4)
    for effort in ("max", "xhigh", "high", "medium", "low", "minimal", "none", ""):
        payload = {
            "model": model, "store": False, "max_output_tokens": 4000,
            "input": [{"role": "user",
                       "content": MESSAGES['shim']['smoke_test']}],
        }
        if effort:
            payload["reasoning"] = {"effort": effort}
        status, body = await shim._post("/responses", payload)
        if status == 200 and isinstance(body, dict):
            u = body.get("usage") or {}
            rt = (u.get("output_tokens_details") or {}).get("reasoning_tokens")
            text = "".join(
                c.get("text", "")
                for item in body.get("output") or [] if item.get("type") == "message"
                for c in item.get("content") or [])
            print(f"effort={effort or '(unset)':8s} OK  reasoning_tokens={rt} "
                  f"output_tokens={u.get('output_tokens')} answer={text.strip()[:40]!r}")
        else:
            btxt = body if isinstance(body, str) else json.dumps(body)
            print(f"effort={effort or '(unset)':8s} {status} {btxt[:140]}")
    if shim._session:
        await shim._session.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--effort", default="high",
                   help="Fixed upstream reasoning effort; empty string omits the field")
    p.add_argument("--port", type=int, default=8901)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-output-tokens", type=int, default=65536)
    p.add_argument("--max-inflight", type=int, default=16,
                   help="Maximum concurrent upstream requests")
    p.add_argument("--probe", action="store_true", help="Make real API requests to probe supported effort levels, then exit")
    args = p.parse_args()

    if args.probe:
        asyncio.run(probe(args.model))
        return

    shim = Shim(args.model, args.effort, args.max_output_tokens, args.max_inflight)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/health", shim.handle_health)
    app.router.add_get("/stats", shim.handle_stats)
    app.router.add_post("/v1/chat/completions", shim.handle_chat)
    print(f"[shim] {args.model} effort={args.effort or '(none)'} route={shim.route} "
          f"max_output_tokens={args.max_output_tokens} on http://{args.host}:{args.port}",
          flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
