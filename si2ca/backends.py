"""Request compatibility shared by all policy and judge transports."""
import os
import math
import httpx


def chat_payload(payload, *, judge=False):
    """Adapt a chat request to an API's declared capabilities, without changing history."""
    key = "SI2CA_JUDGE_BACKEND" if judge else "SI2CA_BACKEND"
    backend = os.environ.get(key, os.environ.get("SI2CA_BACKEND", "self-hosted"))
    if backend != "api":
        return payload
    body = dict(payload)
    for name in ("top_k", "chat_template_kwargs", "logprobs", "top_logprobs"):
        body.pop(name, None)
    if os.environ.get("SI2CA_API_SAMPLING", "0") != "1":
        for name in ("temperature", "top_p"):
            body.pop(name, None)
    if body.get("n") == 1:
        body.pop("n")
    token_field = os.environ.get("SI2CA_API_TOKEN_FIELD", "max_tokens")
    if "max_tokens" in body and token_field != "max_tokens":
        body[token_field] = body.pop("max_tokens")
    return body


def check_loglikelihood(base_url):
    """Verify the native forward capability before launching an SL task batch.

    This is a short real prefill (zero generated tokens), called only during
    execution, never by --help.
    """
    for endpoint in base_url.split(','):
        response = httpx.post(endpoint + "/generate", timeout=120,
            headers={"Authorization": f"Bearer {os.environ.get('SI2CA_API_KEY', 'dummy')}"},
            json={"text":"SI2CA checks conditional token likelihood.",
                  "sampling_params":{"max_new_tokens":0,"temperature":0},
                  "return_logprob":True,"logprob_start_len":0})
        response.raise_for_status()
        values = (response.json().get("meta_info") or {}).get("input_token_logprobs") or []
        if not any(isinstance(x, (list,tuple)) and x and isinstance(x[0], (int,float))
                   and math.isfinite(x[0]) for x in values):
            raise ValueError(f"{endpoint} did not return native input-token log-likelihoods; SL cannot run on this service")
