"""Installed ``si2ca`` command-line interface."""
import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys

import yaml
import httpx

from si2ca.data import RESOURCE_ROOT, load_dataset, read_rows, unpack_assets
from si2ca.run import execute_config, parser as run_parser, resolve


def build_parser():
    p = argparse.ArgumentParser(prog="si2ca", description="Run Self-Judgement (SJ), Self-Likelihood (SL), and coding-agent baselines.")
    try:
        package_version = version("si2ca")
    except PackageNotFoundError:
        package_version = "0.2.4 (source tree)"
    p.add_argument("--version", action="version", version=f"si2ca {package_version}")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[run_parser(add_help=False, public=True)],
        description="Run an evaluation using a paper method.",
        epilog="Examples: si2ca run --method SJ --backend api --model MODEL --base-url URL --out runs/sj\n"
               "          si2ca run --method SL --backend self-hosted --model MODEL --base-url URL --tokenizer-path PATH --out runs/sl",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    listing = sub.add_parser("ls", help="List methods, named settings, benchmarks or search skills")
    listing.add_argument("what", choices=["methods", "settings", "benchmarks", "skills"], nargs="?", default="methods")
    data = sub.add_parser("data", help="Prepare and validate the bundled inputs")
    data.add_argument("action", choices=["prepare"])
    sub.add_parser("serve", help="Launch an explicit self-hosted SGLang replica", add_help=False)
    results = sub.add_parser("results", help="Summarize a JSON/JSONL result file")
    results.add_argument("paths", nargs="+")
    results.add_argument("--expected", type=int, required=True)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # The serving entry point owns its lifecycle/options, including --help.
    if argv and argv[0] == "serve":
        import runpy
        previous = sys.argv
        try:
            sys.argv = ["si2ca serve", *argv[1:]]
            runpy.run_module("si2ca.serve", run_name="__main__")
        finally:
            sys.argv = previous
        return
    p = build_parser()
    args = p.parse_args(argv)
    try:
        if args.command == "run":
            cfg = resolve(args)
            cfg["public_cli"] = True
            execute_config(cfg)
        elif args.command == "ls":
            if args.what == "methods":
                print("Standard    one candidate\nSJ          self-judgement; self-hosted or API\nSL          self-likelihood; self-hosted SGLang only\nDiscovered  Discovered Early-Commit Window Strategy (--branch-until-edit)\nSFT         standard decoding of a distilled student")
            elif args.what == "settings":
                print("\n".join(yaml.safe_load((RESOURCE_ROOT / "configs/experiments.yaml").read_text())["settings"]))
            elif args.what == "skills":
                for path in sorted((RESOURCE_ROOT / "skills").glob("*.md")):
                    print(f"{path.stem}: {path}")
            else:
                rows = read_rows(RESOURCE_ROOT / "data/dev192_multilingual.jsonl")
                validated = sum(r["metadata"].get("gold_validated") is True for r in rows)
                print("verified 500\npro 731\nboth 1231\ndeepswe 113\n"
                      f"validation {len(rows)} (SWE-bench Multilingual; {validated} gold-validated)")
        elif args.command == "data":
            print(f"Grading assets: {unpack_assets()}")
            for name in ("verified", "pro", "deepswe", "validation"):
                print(f"{name}: {len(load_dataset(name))} tasks, referenced assets present")
        elif args.command == "results":
            from si2ca.results import load_results, summary
            if args.expected < 1:
                raise ValueError("--expected must be positive")
            for path in args.paths:
                print(json.dumps({"path": path, **summary(load_results(path), args.expected)}, indent=2))
    except (ValueError, FileNotFoundError, httpx.HTTPError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
