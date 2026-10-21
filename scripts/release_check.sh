#!/usr/bin/env bash
set -euo pipefail

ruff check src tests scripts
pytest -q
python -m compileall -q src
python -m recsys_lab m9

echo "Release checks passed. Run 'python -m recsys_lab m10' only when ready to finalize the sealed test evaluation."
