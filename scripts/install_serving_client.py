"""Install the controller without replacing a base image's core serving libraries."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="si2ca-serving-") as folder:
    constraints = []
    for name in ("sglang", "sglang-router", "torch", "transformers", "openai"):
        try:
            constraints.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    path = Path(folder) / "constraints.txt"
    path.write_text('\n'.join(constraints)+'\n')
    subprocess.run([sys.executable,"-m","pip","install","--no-cache-dir",
                    "--constraint",str(path),".[search]"],check=True)
