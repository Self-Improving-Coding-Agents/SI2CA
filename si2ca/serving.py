"""Reuse an explicitly matching service or own the lifecycle of a new one."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx


def matching_server(url, model, checkpoint, *, revision=None, cache_dir=None):
    try:
        response = httpx.get(url + "/v1/models", timeout=3)
    except httpx.ConnectError:
        return False
    response.raise_for_status()
    names = {row.get("id") for row in response.json().get("data", [])}
    if model not in names:
        raise ValueError(f"Port already serves {sorted(names)}, not {model}; choose --port or --base-url explicitly")
    info = httpx.get(url + "/get_model_info", timeout=3)
    info.raise_for_status()
    actual = info.json().get("model_path")
    from si2ca.model_files import is_hf_id
    if is_hf_id(checkpoint) and not Path(checkpoint).is_dir() and (revision or actual != checkpoint):
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            expected = snapshot_download(repo_id=checkpoint, revision=revision, cache_dir=cache_dir, local_files_only=True)
        except LocalEntryNotFoundError:
            expected = None
        same = bool(expected and actual and Path(expected).resolve() == Path(actual).expanduser().resolve())
    else:
        same = actual == checkpoint or (actual and Path(actual).expanduser().resolve() == Path(checkpoint).expanduser().resolve())
    if not same:
        raise ValueError(f"Existing service model_path does not match {checkpoint}; refusing to reuse or replace it")
    health = httpx.get(url + "/health", timeout=3)
    health.raise_for_status()
    return True


@contextmanager
def managed_service(c):
    checkpoint = c.get("managed_model_path")
    if not checkpoint:
        yield
        return
    from si2ca.serve import launch, stop
    if matching_server(c["base_url"], c["model"], checkpoint, revision=c.get("revision"), cache_dir=c.get("model_cache_dir")):
        print(f"Reusing {c['model']} at {c['base_url']}; this command will not stop it", flush=True)
        yield
        return
    gpus = c.get("gpu_ids")
    if not gpus:
        raise ValueError("No matching server is running. Supply --gpu-ids (for example 0,1) to start one, or --base-url for an existing endpoint")
    import importlib.util
    if importlib.util.find_spec("sglang") is None:
        raise ValueError("Automatic serving requires SGLang: install 'si2ca[serve]' for a compatible NVIDIA environment, or use your compatible ROCm SGLang image")
    log = Path(c["out"]) / "serve.log"
    if log.exists():
        log = log.with_name(f"serve-{uuid.uuid4().hex[:8]}.log")
    process = launch(SimpleNamespace(model_path=checkpoint, model_id=c["model"], name=c["model"], gpus=gpus,
        tp=c.get("tp") or len(gpus.split(',')), port=c["port"], context_length=c.get("context_length"),
        tool_call_parser=c.get("tool_call_parser", "auto"), reasoning_parser=c.get("reasoning_parser", "auto"),
        revision=c.get("revision"), model_cache_dir=c.get("model_cache_dir"), local_files_only=c.get("local_files_only", False),
        mem_fraction=c["mem_fraction"], log=str(log), rocm=c["rocm"]))
    try:
        yield
    finally:
        if c.get("keep_server"):
            print(f"Server kept at {c['base_url']}, PID {process.pid}", flush=True)
        else:
            stop(process)
