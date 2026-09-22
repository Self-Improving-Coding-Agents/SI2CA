"""Resolve an experiment configuration and run evaluation."""
import argparse
import asyncio
import json
import os
import re
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

from si2ca.data import RESOURCE_ROOT, asset_root, load_dataset
from si2ca.runtime.protocol import HARNESS_PROTOCOL


def parser(*, add_help=True, public=False):
    p = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    if public:
        methods = p.add_mutually_exclusive_group(required=True)
        methods.add_argument("--method",
            type=lambda value: "DISCOVERED" if value.upper() == "STRATEGY6" else value.upper(),
            choices=["STANDARD", "SJ", "SL", "DISCOVERED", "SFT"],
            help="Paper method (case insensitive); Discovered selects the Discovered Early-Commit Window Strategy")
        methods.add_argument("--setting", help="Advanced: exact named setting, including historical DeepSWE recipes")
        p.add_argument("--pi", choices=["none", "gold", "hint"], help="Privileged information: SJ defaults to gold, SL to hint")
        p.add_argument("--placement", choices=["head", "tail"], help="SL scoring-context placement; default head")
        p.add_argument("--score", choices=["likelihood", "gain"], help="SL selection score; default likelihood")
        p.add_argument("--gpu-ids", "--gpus", "--gpu", help="Physical GPU IDs for automatic serving, e.g. 0,1,2,3")
        p.add_argument("--tp", type=int, help="Automatic serving tensor parallelism; defaults to the GPU count")
        p.add_argument("--port", type=int, default=8151, help="Local port to reuse or start a self-hosted model")
        p.add_argument("--rocm", action="store_true", help="Set ROCm device visibility for automatic serving")
        p.add_argument("--context-length", type=int, help="Server context length; default: SGLang reads the model configuration")
        p.add_argument("--model-path", help="Local weights directory for --model org/model; missing files download here")
        p.add_argument("--tool-call-parser", default="auto", help="SGLang parser: auto, none, or a parser name supported by your SGLang version")
        p.add_argument("--reasoning-parser", default="auto", help="SGLang reasoning parser: auto, none, or an explicit parser name")
        p.add_argument("--mem-fraction", type=float, default=.8, help="Automatically started server static memory fraction")
        p.add_argument("--keep-server", action="store_true", help="Leave a newly started server running after this experiment")
        p.add_argument("--gold-patch", action=argparse.BooleanOptionalAction, default=None, help="SJ: show/hide the reference patch in judge context")
        p.add_argument("--gold-weight", type=float, help="SJ G1 weight in [0,1]; default .3; other rubric weights share the remainder")
        p.add_argument("--branch-until-edit", type=int, help="Discovered early-commit window: branch through this many mutation-like commands; default 3")
    else:
        p.add_argument("--setting", required=True, help="Name in configs/experiments.yaml")
    p.add_argument("--gold-field", help="Dotted row field containing the reference patch; otherwise detect common field names")
    p.add_argument("--config", default=str(RESOURCE_ROOT / "configs/experiments.yaml"))
    p.add_argument("--backend", choices=["self-hosted", "api"], help="self-hosted: SGLang generation and log-likelihood; api: chat generation only")
    p.add_argument("--judge-backend", choices=["self-hosted", "api"], help="Defaults to policy backend; external judges default to api")
    p.add_argument("--api-sampling", action="store_true", help="API supports temperature/top_p; otherwise omit these fields")
    p.add_argument("--api-token-field", choices=["max_tokens", "max_completion_tokens"], default="max_tokens", help="Output-limit field accepted by the API")
    p.add_argument("--benchmark", choices=["verified", "pro", "both", "deepswe", "validation"])
    p.add_argument("--dataset", help="Optional custom JSONL in the documented task schema")
    p.add_argument("--validated-only", action="store_true", help="Use only tasks marked gold_validated in the selected dataset")
    p.add_argument("--allow-unvalidated", action="store_true", help="Explicitly include search tasks whose gold grading chain has not yet been validated")
    p.add_argument("--model", required=True, help="Served/API model ID, or local checkpoint/Hugging Face ID for automatic serving")
    p.add_argument("--base-url", required=not public, help="Existing endpoint root or /v1; omit to automatically serve --model")
    p.add_argument("--judge-url", help="Full /v1/chat/completions endpoint; defaults to the policy's endpoint")
    p.add_argument("--judge-model", help="Judge model name; defaults to policy model")
    p.add_argument("--judge-mode", choices=["local_openai", "trapi", "terra_responses"])
    p.add_argument("--judge-reasoning-effort")
    p.add_argument("--reasoning-effort")
    p.add_argument("--tokenizer-path", help="SL tokenizer directory or HF ID; defaults to --model-path or --model")
    p.add_argument("--revision", help="Hugging Face branch, tag or commit for downloaded model/tokenizer files")
    p.add_argument("--model-cache-dir", help="Hugging Face cache directory (otherwise use standard HF cache settings)")
    p.add_argument("--local-files-only", action="store_true", help="Never download model/tokenizer files; fail if local files are incomplete")
    p.add_argument("--out", "--output-dir", required=not public, help="Output directory; the public CLI defaults to a timestamped runs/ directory")
    p.add_argument("--limit", type=int, default=0, help="0 = full dataset; positive values are debugging subsets")
    for key in ("concurrency", "seed", "n_samples", "k", "score_samples", "max_turns", "max_gen_tokens", "task_timeout", "eval_timeout", "top_k", "gold_max_chars", "judge_max_tokens"):
        aliases = {"k": ["--num-candidates", "--branch"], "score_samples": ["--judge-samples", "--average"], "max_gen_tokens": ["--max-tokens"]}
        p.add_argument("--" + key.replace("_", "-"), *aliases.get(key, []), type=int)
    for key in ("temperature", "top_p", "judge_temperature", "judge_top_p"):
        p.add_argument("--" + key.replace("_", "-"), type=float)
    return p


def resolve(args):
    if getattr(args, "method", None):
        method = args.method
        if getattr(args, "gold_patch", None) is not None:
            if method != "SJ":
                raise ValueError("--gold-patch/--no-gold-patch applies to --method SJ")
            choice = "gold" if args.gold_patch else "none"
            if args.pi is not None and args.pi != choice:
                raise ValueError("--pi conflicts with --gold-patch/--no-gold-patch")
            args.pi = choice
        pi = args.pi or ("hint" if method == "SL" else "gold")
        if method != "SL" and (args.placement or args.score):
            raise ValueError("--placement and --score apply only to SL")
        if method == "SL":
            if pi == "none":
                raise ValueError("SL requires --pi hint or gold")
            args.setting = f"sl_{pi}_{args.placement or 'head'}" + ("_gain" if args.score == "gain" else "")
        elif method == "SJ":
            if pi == "hint":
                raise ValueError("SJ uses --pi gold or none; hints are an SL ablation")
            args.setting = "sj_no_pi" if pi == "none" else "sj"
            if args.benchmark == "deepswe":
                if pi == "none":
                    raise ValueError("The packaged DeepSWE SJ recipe requires gold")
                args.setting = "ds_sj"
        else:
            if args.pi is not None:
                raise ValueError("--pi is an SJ/SL option")
            args.setting = {"STANDARD": "ds_standard" if args.benchmark == "deepswe" else "standard", "DISCOVERED": "discovered", "SFT": "sft_eval"}[method]
    elif getattr(args, "pi", None) or getattr(args, "placement", None) or getattr(args, "score", None):
        raise ValueError("Use --method with PI/placement/score options, or select an exact --setting")
    registry = yaml.safe_load(Path(args.config).read_text())
    if getattr(args, "gold_weight", None) is not None and args.setting not in ("sj", "sj_no_pi", "sj_strong", "ds_sj", "ds_sj_medium", "discovered"):
        raise ValueError("--gold-weight applies only to SJ settings")
    if args.setting not in registry["settings"]:
        raise ValueError(f"Unknown setting {args.setting!r}; choose {', '.join(registry['settings'])}")
    cfg = registry["defaults"] | registry["settings"][args.setting]
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k not in ("pi", "placement", "score", "method", "command")})
    cfg["backend"] = cfg.get("backend") or ("api" if cfg["benchmark"] == "deepswe" else "self-hosted")
    cfg["gold_weight"] = cfg.get("gold_weight", .3)
    cfg["branch_until_edit"] = cfg.get("branch_until_edit", 3)
    if getattr(args, "branch_until_edit", None) is not None and args.setting != "discovered":
        raise ValueError("--branch-until-edit applies only to the Discovered Early-Commit Window Strategy")
    if cfg["branch_until_edit"] < 1:
        raise ValueError("--branch-until-edit must be at least 1")
    if not math.isfinite(cfg["gold_weight"]) or not 0 <= cfg["gold_weight"] <= 1:
        raise ValueError("--gold-weight must be a finite number in [0,1]")
    if cfg["pi"] == "none":
        if getattr(args, "gold_weight", None) not in (None, 0):
            raise ValueError("A positive --gold-weight requires gold-patch guidance")
        cfg["gold_weight"] = 0.
    if not cfg.get("out"):
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg["model"]).strip("_")[-64:]
        cfg["out"] = f"runs/{cfg['setting']}-{tag}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
    if not cfg.get("base_url"):
        if not hasattr(args, "method"):
            raise ValueError("base-url must contain an HTTP(S) endpoint")
        if cfg["backend"] == "api":
            raise ValueError("API models require --base-url; GPU options are only for self-hosted models")
        cfg["managed_model_path"] = cfg.get("model_path") or cfg["model"]
        cfg["base_url"] = f"http://127.0.0.1:{cfg.get('port', 8151)}"
    elif cfg.get("gpu_ids"):
        raise ValueError("Use --base-url for an existing server or --gpu-ids for automatic serving, not both")
    cfg["judge_backend"] = cfg.get("judge_backend") or ("api" if args.judge_url or args.judge_model else cfg["backend"])
    if cfg["backend"] == "api" and cfg.get("strategy", "").startswith("sl"):
        raise ValueError("SL requires --backend self-hosted and a SGLang /generate log-likelihood endpoint; API models support Standard and SJ only")
    if cfg["judge_backend"] == "api":
        if not args.judge_reasoning_effort and cfg["benchmark"] != "deepswe":
            cfg["judge_reasoning_effort"] = args.reasoning_effort or ""
    if cfg.get("gpu_ids"):
        devices = cfg["gpu_ids"].split(',')
        if not all(i.isdigit() for i in devices) or len(devices) != len(set(devices)):
            raise ValueError("--gpu-ids must contain distinct nonnegative physical indices")
        if cfg.get("tp") is not None and cfg["tp"] != len(devices):
            raise ValueError("--tp must match the number of --gpu-ids for one replica")
    if cfg.get("managed_model_path"):
        path = cfg["managed_model_path"]
        from si2ca.model_files import is_hf_id
        if not Path(path).expanduser().is_dir() and not is_hf_id(cfg["model"]):
            raise ValueError("Automatic serving requires a local checkpoint or Hugging Face org/model ID; use --base-url for a served model name")
        if not 1 <= cfg["port"] < 55536 or not 0 < cfg["mem_fraction"] < 1:
            raise ValueError("Invalid automatic-serving port or memory fraction")
        if cfg.get("context_length") is not None and cfg["context_length"] < 1:
            raise ValueError("--context-length must be positive")
    if cfg["engine"] != "strategy":
        raise ValueError("New experiments require engine: strategy; use the current experiment configuration")
    cfg["harness_protocol"] = HARNESS_PROTOCOL
    cfg["out"] = str(Path(cfg["out"]).resolve())
    roots = [u.strip().rstrip("/").removesuffix("/v1") for u in cfg["base_url"].split(",") if u.strip()]
    if not roots or any(not u.startswith(("http://", "https://")) for u in roots):
        raise ValueError("base-url must contain HTTP(S) endpoint URLs")
    cfg["base_url"] = ",".join(roots)
    cfg["judge_url"] = cfg.get("judge_url") or roots[0] + "/v1/chat/completions"
    cfg["judge_model"] = cfg.get("judge_model") or cfg["model"]
    for key in ("concurrency", "n_samples", "k", "score_samples", "max_turns", "task_timeout", "eval_timeout"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["limit"] < 0 or cfg["max_gen_tokens"] < 0:
        raise ValueError("limit and max_gen_tokens must be nonnegative")
    if not 0 < cfg["top_p"] <= 1 or not math.isfinite(cfg["temperature"]) or cfg["temperature"] < 0:
        raise ValueError("Require top_p in (0, 1] and temperature >= 0")
    if cfg["engine"] == "strategy" and cfg.get("strategy", "").startswith("sl"):
        cfg["tokenizer_path"] = cfg.get("tokenizer_path") or cfg.get("model_path") or cfg["model"]
        tok = cfg.get("tokenizer_path") or ""
        downloadable_override = tok == cfg.get("model_path") and re.fullmatch(r"[\w.-]+/[\w.-]+", cfg["model"])
        if not tok or (not Path(tok).expanduser().is_dir() and not re.fullmatch(r"[\w.-]+/[\w.-]+", tok) and not downloadable_override):
            raise ValueError("SL requires a local --tokenizer-path or Hugging Face tokenizer ID")
    if cfg["setting"] == "sj_strong" and not args.judge_model:
        raise ValueError("sj_strong requires an explicit --judge-model and its --judge-url")
    if cfg["setting"].startswith("ds_") and cfg["benchmark"] != "deepswe":
        raise ValueError("DeepSWE settings require the DeepSWE grading protocol")
    if cfg["benchmark"] == "deepswe" and not cfg["setting"].startswith("ds_"):
        raise ValueError("Use ds_standard or ds_sj for the DeepSWE grading protocol")
    if cfg["judge_mode"] != "local_openai":
        raise ValueError("Use an OpenAI-compatible judge endpoint; Responses models require the included shim")
    if cfg["judge_batch_samples"]:
        raise ValueError("The unified judge uses independent requests; set judge_batch_samples: false")
    if cfg["rubric_family"]:
        raise ValueError("The unified judge uses the general rubric; set rubric_family: false")
    if not math.isfinite(cfg["judge_temperature"]) or cfg["judge_temperature"] < 0 or not 0 < cfg["judge_top_p"] <= 1:
        raise ValueError("Require judge_temperature >= 0 and judge_top_p in (0, 1]")
    for key in ("judge_max_tokens", "gold_max_chars", "prefix_head_chars", "prefix_tail_chars", "prefix_max_chars"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if cfg["strategy"] in ("selfguide_rubrics", "discovered") and cfg["k"] < 2:
        raise ValueError("SJ needs at least two candidates")
    return cfg


def environment(c):
    # Prevent unrelated inherited ablation flags from silently changing the setting.
    for key in ("SLIME_JUDGE_ACTION_ONLY", "SLIME_JUDGE_PROGRESS_DIFF", "SLIME_JUDGE_INSTANCE_RUBRIC", "SLIME_JUDGE_MULTI_RUBRIC", "SLIME_JUDGE_SELECT_WORST", "SLIME_JUDGE_IDENTICAL_PICK", "SLIME_SELECT_RULE", "SLIME_SELECT_BY_CONFIDENCE", "SLIME_BRANCH_GATE", "MINIMAL_READBACK", "MINIMAL_REPO_SNAPSHOT", "MINIMAL_PERSISTENT_SHELL", "SLIME_ALLOW_NUDGE"):
        os.environ.pop(key, None)
    settings = {
        "SWE_AGENT_UID0": "1", "SLIME_AGENT_SANDBOX_BACKEND": "local_docker",
        "SLIME_AGENT_DOCKER_NETWORK": "host", "SLIME_KEEP_REASONING": "1" if c["keep_reasoning"] else "",
        "MINIMAL_MAX_TURNS": str(c["max_turns"]), "MINIMAL_MAX_TOKENS": str(c["max_gen_tokens"]),
        "MINIMAL_TEMPERATURE": str(c["temperature"]), "MINIMAL_TOP_P": str(c["top_p"]),
        "MINIMAL_TOP_K": str(c["top_k"]), "MINIMAL_REASONING_EFFORT": c.get("reasoning_effort", ""),
        "SLIME_JUDGE_ORACLE_PATCH": "1" if c["pi"] == "gold" else "0",
        "SLIME_JUDGE_SKIP_IDENTICAL": "1", "SLIME_JUDGE_FAMILY_RUBRIC": str(int(c["rubric_family"])),
        "SLIME_GOLD_PATCH_MAX_CHARS": str(c["gold_max_chars"]),
        "SLIME_JUDGE_BATCH_SAMPLES": str(int(c["judge_batch_samples"])),
        "SLIME_TEACHER_ENDPOINT": c["judge_url"],
        "SLIME_TEACHER_CONCURRENCY": str(c["concurrency"]*2),
        "SLIME_BRANCH_LOG_DIR": str(Path(c["out"]) / "branch_logs"),
        "SI2CA_BACKEND": c["backend"], "SI2CA_JUDGE_BACKEND": c["judge_backend"],
        "SI2CA_API_SAMPLING": str(int(c["api_sampling"])), "SI2CA_API_TOKEN_FIELD": c["api_token_field"],
    }
    os.environ.update(settings)


async def run_strategy(c, rows):
    from si2ca.runtime import strategy as runner
    from si2ca.strategies import discovered, sl, sl_gain
    out = Path(c["out"])
    runner.BUILTIN.update(sl=sl.Strategy, sl_gain=sl_gain.Strategy, discovered=discovered.Strategy)
    dataset = out / "tasks.jsonl"
    dataset.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    hints = out / "hints.jsonl"
    hints.write_text(''.join(json.dumps({"instance_id": r["metadata"]["instance_id"], "hint": r["metadata"].get("hint", "")}, ensure_ascii=False)+'\n' for r in rows))
    args = SimpleNamespace(**c)
    args.dataset, args.out, args.traj_dir = str(dataset), str(out / "results.json"), str(out / "traj")
    args.config_python, args.cot_in_content, args.draw_retries = sys.executable, 0, 1
    args.reasoning_effort = c.get("reasoning_effort")
    args.resume, args.limit = True, 0
    args.base_url = ','.join(u+'/v1' for u in c['base_url'].split(','))
    args.scaffold = "SL" if c["strategy"].startswith("sl") else ("SJ" if c["pi"] == "gold" else "none")
    args.hints_file, args.gold_file = str(hints), str(dataset)
    args.inject_source, args.inject_position, args.inject_template = c["pi"], c.get("position", "head"), None
    args.tokenizer_path = c.get("tokenizer_path")
    args.allow_partial_scaffold, args.logprobs = False, c["strategy"].startswith("sl")
    await runner.amain(args)


def execute_config(c):
    rows = load_dataset(c["benchmark"], c.get("dataset"), c["limit"], gold_field=c.get("gold_field"), extract_gold=c["pi"] == "gold")
    if c["benchmark"] == "deepswe" and c["pi"] == "gold" and not c.get("dataset"):
        task_assets = asset_root() / "data/deepswe/tasks"
        for row in rows:
            md = row["metadata"]
            if not md.get("gold_patch"):
                md["gold_patch"] = (task_assets / md["instance_id"] / "solution/solution.patch").read_text()
    if c["validated_only"]:
        rows = [r for r in rows if r["metadata"].get("gold_validated") is True]
        if not rows:
            raise ValueError("No tasks are marked gold_validated in this dataset")
    pending = sum(r["metadata"].get("gold_validated") is False for r in rows)
    if pending:
        print(f"Search dataset: {len(rows)-pending} gold-validated; {pending} pending grading validation.")
        if not c["allow_unvalidated"]:
            raise ValueError("Validate the remaining grading chains first, or explicitly use --allow-unvalidated; --validated-only is a diagnostic subset")
    if c["pi"] != "none":
        field = "hint" if c["pi"] == "hint" else "gold_patch"
        missing = [r["metadata"]["instance_id"] for r in rows if not r["metadata"].get(field)]
        if missing:
            raise ValueError(f"{len(missing)}/{len(rows)} tasks missing {field}: {', '.join(missing[:5])}. Supply the field or select --gold-field; no generation was started")
    print(json.dumps(c | {"tasks": len(rows)}, ensure_ascii=False, indent=2))
    if any(r["metadata"].get("benchmark") == "swebench_multilingual" for r in rows):
        import importlib.util
        if importlib.util.find_spec("swebench") is None:
            raise ValueError("Multilingual grading requires: pip install 'si2ca[search]'")
    out = Path(c["out"])
    out.mkdir(parents=True, exist_ok=True)
    effective = out / "config.json"
    if effective.exists() and json.loads(effective.read_text()) != c:
        raise ValueError("Output directory belongs to another configuration; choose a new --out")
    effective.write_text(json.dumps(c, ensure_ascii=False, indent=2)+'\n')
    environment(c)
    from si2ca.serving import managed_service
    with managed_service(c):
        if c.get("strategy", "").startswith("sl"):
            from si2ca.model_files import resolve_files
            local_override = c["tokenizer_path"] if c["tokenizer_path"] == c.get("model_path") else None
            c = c | {"tokenizer_path": resolve_files(c["model"] if local_override else c["tokenizer_path"],
                local_path=local_override, tokenizer_only=True,
                revision=c.get("revision"), cache_dir=c.get("model_cache_dir"),
                local_files_only=c.get("local_files_only", False))}
        if c.get("public_cli") and c.get("strategy", "").startswith("sl"):
            from si2ca.backends import check_loglikelihood
            check_loglikelihood(c["base_url"])
        asyncio.run(run_strategy(c, rows))


def main():
    execute_config(resolve(parser().parse_args()))


if __name__ == "__main__":
    main()
