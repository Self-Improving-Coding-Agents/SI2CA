"""Resolve local or Hugging Face files without mistaking a partial cache for weights."""
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


TOKENIZER_PATTERNS = ["*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.jinja2", "*.py"]


def is_hf_id(value):
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value)) and not value.startswith(("./", "../"))


def _present(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as stream:
        return not stream.read(128).startswith(b"version https://git-lfs.github.com/spec/")


def complete_files(directory, *, tokenizer_only=False):
    """Check tokenizer payloads or a complete weight set, including indexed shards.

    This is a presence check, not a checksum or SGLang compatibility certification.
    Hugging Face itself verifies downloads; explicit local files remain user inputs.
    """
    root = Path(directory)
    if tokenizer_only:
        return any(_present(p) for pattern in ("tokenizer.json", "*.model", "vocab.json", "vocab.txt", "*.tiktoken")
                   for p in root.glob(pattern))
    if not any(_present(root / name) for name in ("config.json", "params.json")):
        return False
    indexed_weights = set()
    for index in root.glob("*.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if mapping and all(_present(root / filename) for filename in set(mapping.values())):
            return True
        indexed_weights.update(mapping.values())
    weights = [p for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf") for p in root.glob(pattern)]
    groups = {}
    for weight in weights:
        if weight.name in indexed_weights or not _present(weight):
            continue
        # Training metadata/optimizer binaries are not an alternative weight set.
        if weight.suffix == ".bin" and not re.fullmatch(r"(?:pytorch_model|model)(?:-\d+-of-\d+)?\.bin", weight.name):
            continue
        shard = re.fullmatch(r"(.+)-(\d+)-of-(\d+)(\.[^.]+)", weight.name)
        if shard:
            prefix, number, count, suffix = shard.groups()
            groups.setdefault((prefix, int(count), suffix), set()).add(int(number))
        elif not (root / (weight.name + ".index.json")).exists():
            return True
    return any(numbers == set(range(1, count + 1)) for (_, count, _), numbers in groups.items())


def resolve_files(model, *, local_path=None, revision=None, cache_dir=None,
                  local_files_only=False, tokenizer_only=False):
    """Reuse a complete directory/cache first; download missing files only on execution."""
    explicit = Path(local_path or model).expanduser()
    if local_path or explicit.is_dir() or not is_hf_id(model):
        if explicit.is_dir() and complete_files(explicit, tokenizer_only=tokenizer_only):
            return str(explicit.resolve())
        if not is_hf_id(model):
            raise ValueError(f"Missing or incomplete {'tokenizer' if tokenizer_only else 'model'} directory: {explicit}. "
                             "Use --model org/model (and optionally --model-path DIR) to download missing files.")
        if local_files_only:
            raise ValueError(f"Missing files in {explicit}; --local-files-only forbids downloading")
        # Explicit local override: keep the official model ID for downloading and serving.
        destination = str(explicit.resolve())
    else:
        destination = None
        try:
            cached = snapshot_download(repo_id=model, revision=revision, cache_dir=cache_dir, local_files_only=True)
        except LocalEntryNotFoundError:
            cached = None
        if cached and complete_files(cached, tokenizer_only=tokenizer_only):
            return str(Path(cached).resolve())
        if local_files_only:
            raise ValueError(f"No complete cached {'tokenizer' if tokenizer_only else 'weights'} for {model}; "
                             "--local-files-only forbids downloading")
    print(f"Preparing {'tokenizer files' if tokenizer_only else 'model weights'} for {model}; reusing downloaded files", flush=True)
    path = snapshot_download(repo_id=model, revision=revision, cache_dir=cache_dir,
                             local_dir=destination, allow_patterns=TOKENIZER_PATTERNS if tokenizer_only else None)
    if not complete_files(path, tokenizer_only=tokenizer_only):
        raise ValueError(f"Downloaded {model}, but no complete {'tokenizer' if tokenizer_only else 'weight set'} was found in {path}")
    return str(Path(path).resolve())


def parser_defaults(model, directory=None):
    """Known SGLang parser conventions, never a model allowlist; explicit options win."""
    config = {}
    if directory and (Path(directory) / "config.json").is_file():
        config = json.loads((Path(directory) / "config.json").read_text())
    family = config.get("model_type", "").lower()
    name = model.lower()
    if family.startswith("qwen3_5") or "qwen3.5" in name or "qwen3-coder" in name:
        return "qwen3_coder", "qwen3"
    if family.startswith("qwen3") or "qwen3" in name:
        return "qwen", "qwen3"
    if family.startswith("qwen2") or "qwen2" in name:
        return "qwen", None
    if family == "llama" or "llama-3" in name:
        return "llama3", None
    if family in ("mistral", "mixtral"):
        return "mistral", None
    return None, None
