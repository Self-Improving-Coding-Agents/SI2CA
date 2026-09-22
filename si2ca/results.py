"""Summarize deduplicated trials using a fixed, explicit task denominator."""
import argparse
import json
import math
from pathlib import Path
import statistics


def infrastructure_failure(row):
    """Execution/transport/grade failures are not completed benchmark trials."""
    return (bool(row.get("error")) or row.get("exit_code") in (-1, 1)
            or bool(row.get("generation_failures"))
            or row.get("execution_status") == "infrastructure_failure"
            or str(row.get("grade_status", "")).startswith(("GRADEFAIL", "TIMEOUT"))
            or str(row.get("abort", "")).startswith(("exception", "task_timeout", "all_draws_failed",
                                                       "no_healthy_endpoint", "grading_infrastructure")))


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def load_results(path):
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("results", data.get("records", data.get("per_instance", [])))
    if not isinstance(data, list):
        raise ValueError("Unsupported result schema")
    records = {}
    for row in data:
        iid = row.get("instance_id") or row.get("task") or row.get("label")
        if not iid:
            raise ValueError("Result is missing instance ID")
        records[(iid, row.get("sample", 0))] = row
    return records


def summary(records, expected):
    rows = list(records.values())
    if expected < 1:
        raise ValueError("Expected trials must be positive")
    if len(rows) > expected:
        raise ValueError("Observed trials exceed the declared denominator")
    turns = [r["turns"] for r in rows if not infrastructure_failure(r)
             and isinstance(r.get("turns"), (int,float))]
    infrastructure = sum(infrastructure_failure(r) for r in rows)
    solved = sum(float(r.get("reward",0)) >= .999 and not infrastructure_failure(r) for r in rows)
    protocols = {r.get("harness_protocol", "unversioned") for r in rows}
    unvalidated = sum(r.get("gold_validated") is False for r in rows)
    return {"expected_trials": expected, "observed_trials": len(rows), "missing_trials": expected-len(rows),
            "solved": solved, "resolve_pct": solved/expected*100, "infrastructure_failures": infrastructure,
            "valid_trials": len(rows)-infrastructure, "harness_protocols": sorted(protocols),
            "turns_trials": len(turns),
            "turns_mean": statistics.mean(turns) if turns else None,
            "turns_median": quantile(turns,.5), "turns_p90": quantile(turns,.9),
            "unvalidated_grading_chains": unvalidated,
            "final": len(rows)==expected and infrastructure==0 and unvalidated==0 and len(protocols)<=1}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--expected", type=int, required=True, help="Tasks × independent samples (500/731/1231 or 113 per DeepSWE run)")
    a = p.parse_args()
    for path in a.paths:
        print(json.dumps({"path":path, **summary(load_results(path),a.expected)},ensure_ascii=False,indent=2))
