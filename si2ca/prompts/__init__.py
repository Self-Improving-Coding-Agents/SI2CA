"""Packaged prompt assets; text is kept in Markdown and JSON, not Python."""

import json
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent
MESSAGES = json.loads((PROMPT_DIR / "messages.json").read_text(encoding="utf-8"))

# Snapshot of the pinned mini-swe-agent 2.4.2 templates, now owned here.
# Keep file newlines: the existing Jinja renderers handle them as before.
AGENT_TEMPLATES = {
    name + "_template": (PROMPT_DIR / f"agent_{name}.md").read_text(encoding="utf-8")
    for name in ("system", "instance")
}
MODEL_TEMPLATES = {
    name + "_template": (PROMPT_DIR / f"agent_{name}.md").read_text(encoding="utf-8")
    for name in ("observation", "format_error")
}
