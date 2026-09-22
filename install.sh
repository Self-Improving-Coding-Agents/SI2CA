#!/usr/bin/env bash
# Create an isolated environment, install SI2CA, and prepare packaged input assets.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PROFILE=full
PYTHON_BIN=python3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash install.sh [--venv PATH] [--python PYTHON] [--profile api|full|serve]"
      echo "api: SJ/SL clients; full: add Multilingual grading; serve: also install NVIDIA SGLang."
      echo "Use an existing compatible ROCm/SGLang image for AMD serving. No model is launched."
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  api) EXTRAS="" ;;
  full) EXTRAS="[search]" ;;
  serve) EXTRAS="[search,serve]" ;;
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ is required"'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install "$PROJECT_DIR$EXTRAS"
"$VENV_DIR/bin/si2ca" data prepare
echo "Installed. Activate with: source $VENV_DIR/bin/activate"
echo "Try: si2ca run --help"
