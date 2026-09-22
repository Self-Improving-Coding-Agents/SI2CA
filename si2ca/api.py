"""Python API with per-experiment isolation for the retained stateful harnesses."""
import subprocess
import sys


def run_experiment(*, method, model, backend="self-hosted", base_url=None, out=None, **options):
    """Run the installed CLI in a separate interpreter; return CompletedProcess.

    Keyword options match CLI flags with underscores. Credentials are inherited
    from environment variables. Exceptions from a failed run propagate as
    subprocess.CalledProcessError; output streams remain attached to the caller.
    """
    argv = [sys.executable, "-m", "si2ca", "run", "--method", method,
            "--backend", backend, "--model", str(model)]
    if base_url:
        argv += ["--base-url", base_url]
    if out:
        argv += ["--out", str(out)]
    from si2ca.run import parser
    supported_flags = parser(public=True)._option_string_actions
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if flag not in supported_flags:
            raise TypeError(f"Unknown experiment option: {key}")
        if key == "gold_patch" and value is False:
            argv.append("--no-gold-patch")
            continue
        if value is None or value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return subprocess.run(argv, check=True)
