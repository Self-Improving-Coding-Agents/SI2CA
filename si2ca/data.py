"""Portable benchmark manifests and explicitly referenced grading assets."""
import json
import fcntl
import hashlib
import os
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT if (ROOT / "configs/experiments.yaml").is_file() else Path(__file__).parent / "resources"


def asset_root():
    """Writable, versioned cache; never extract into an installed package directory."""
    base = os.environ.get("SI2CA_CACHE_DIR")
    if not base:
        base = str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "si2ca")
    digest = hashlib.sha256((RESOURCE_ROOT / "data/assets.tar.gz").read_bytes()).hexdigest()[:16]
    return Path(base).expanduser().resolve() / digest


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unpack_assets():
    target_root = asset_root()
    target_root.mkdir(parents=True, exist_ok=True)
    with (target_root / ".extract.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _unpack_assets(target_root)
    return target_root


def _unpack_assets(target_root):
    with tarfile.open(RESOURCE_ROOT / "data/assets.tar.gz") as archive:
        for member in archive.getmembers():
            target = (target_root / member.name).resolve()
            if not target.is_relative_to(target_root / "data") or not member.isfile():
                raise ValueError(f"Invalid packaged asset: {member.name}")
            if target.exists():
                if target.read_bytes() != archive.extractfile(member).read():
                    raise ValueError(f"Existing asset differs: {target}")
            else:
                archive.extract(member, target_root, filter="data")


def load_dataset(benchmark, dataset=None, limit=0, gold_field=None, extract_gold=True):
    name = {"deepswe": "deepswe113", "validation": "dev192_multilingual"}.get(benchmark, "bench1231")
    path = Path(dataset).resolve() if dataset else RESOURCE_ROOT / "data" / f"{name}.jsonl"
    bundled_assets = unpack_assets() if not dataset else None
    rows = read_rows(path)
    if not dataset and benchmark in ("verified", "pro"):
        rows = [r for r in rows if bool(r["metadata"].get("swepro")) == (benchmark == "pro")]
    if limit:
        rows = rows[:limit]
    ids = [r["metadata"]["instance_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Dataset must contain nonempty, unique instance IDs")
    for row in rows:
        md = row["metadata"]
        from si2ca.gold import extract_patch
        if extract_gold:
            gold = extract_patch(row, gold_field)
            if gold:
                md["gold_patch"] = gold
            elif gold_field:
                md.pop("gold_patch", None)
        for key in ("image", "problem_statement"):
            if not md.get(key):
                raise ValueError(f"{md['instance_id']}: missing {key}")
        mappings = [md.get("eval_files", {}), md.get("eval_assets", {})]
        pro = md.get("swepro", {})
        mappings.append({k: pro[k] for k in ("run_script_path", "parser_script_path") if k in pro})
        for mapping in mappings:
            for key, value in mapping.items():
                p = Path(value)
                if not p.is_absolute():
                    if value.startswith(("data/assets/", "data/deepswe/")):
                        if bundled_assets is None:
                            bundled_assets = unpack_assets()
                        p = bundled_assets / p
                    else:
                        p = path.parent / p
                if not p.is_file():
                    raise FileNotFoundError(f"Missing grading asset: {p}; run python -m si2ca.data --unpack")
                if key in pro:
                    pro[key] = str(p.resolve())
                elif key in md.get("eval_assets", {}):
                    md["eval_assets"][key] = str(p.resolve())
                else:
                    md["eval_files"][key] = str(p.resolve())
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unpack", action="store_true", help="Extract and verify the bundled grading assets")
    args = parser.parse_args()
    if args.unpack:
        unpack_assets()
    for benchmark in ("verified", "pro", "deepswe", "validation"):
        print(f"{benchmark}: {len(load_dataset(benchmark))} tasks, all referenced assets present")
