"""Verify a Qwen3.5 export and install its required stop-token configuration."""
import argparse
import json
from pathlib import Path


def finalize(path):
    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    if "qwen3_5" not in str(config.get("model_type", "")):
        raise ValueError("Expected a Qwen3.5 checkpoint")
    vocab_path = root / "tokenizer.json"
    vocab = json.loads(vocab_path.read_text())
    added = {r["content"]:r["id"] for r in vocab.get("added_tokens", [])}
    tokens = [added.get("<|im_end|>"), added.get("<|endoftext|>")]
    if any(t is None for t in tokens):
        raise ValueError("Tokenizer is missing required Qwen stop tokens")
    target = root / "generation_config.json"
    generation = json.loads(target.read_text()) if target.exists() else {}
    previous = generation.get("eos_token_id")
    previous_tokens = set(previous if isinstance(previous, list) else [previous]) if previous is not None else set()
    if not previous_tokens.issubset(set(tokens)):
        raise ValueError(f"Existing EOS configuration differs: {previous}; review it before replacing")
    generation["eos_token_id"] = tokens
    target.write_text(json.dumps(generation, indent=2)+'\n')
    print(f"Verified {target}: eos_token_id={tokens}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    finalize(p.parse_args().checkpoint)
