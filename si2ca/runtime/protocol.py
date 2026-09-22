"""Execution contract shared by every newly configured experiment."""
import json

HARNESS_PROTOCOL = "si2ca-unified-v1"
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MUTATION_MARKERS = ("sed -i", "tee ", "cat >", "cat <<", " > ", ">>",
                    "patch ", "python -c", "apply_patch")


def bash_actions(message):
    """Validate the complete tool-call batch; never silently discard a call."""
    actions = []
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return []
    for call in calls:
        if not isinstance(call, dict):
            return []
        fn = call.get("function") or {}
        if not isinstance(fn, dict) or fn.get("name") != "bash":
            return []
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                return []
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str) or not command.strip():
            return []
        actions.append((call.get("id") or "", command))
    return actions


def is_submission(returncode, output):
    lines = (output or "").lstrip().splitlines()
    return returncode == 0 and bool(lines) and lines[0].strip() == SUBMIT_MARKER


def early_commit_window(commands, *, k=2, max_prior_mutations=2):
    if not isinstance(commands, list) or any(not isinstance(c, str) for c in commands):
        return 1, "mutation_count_signal_unavailable"
    commands = [c for c in commands if c.strip()]
    if not commands:
        return 1, "skip_before_first_observation"
    mutations = sum(any(marker in c for marker in MUTATION_MARKERS) for c in commands)
    if mutations > max_prior_mutations:
        return 1, f"skip_after_{max_prior_mutations + 1}_mutations"
    if not mutations:
        return k, "branch_before_first_mutation"
    return k, f"branch_after_{mutations}_mutations"
