#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

is_tk_compatible() {
  local py="$1"
  "$py" - <<'PY' >/dev/null 2>&1
import sys
try:
    import _tkinter
except Exception:
    sys.exit(1)
tcl = getattr(_tkinter, "TCL_VERSION", "") or ""
sys.exit(0 if not tcl.startswith("8.5") else 1)
PY
}

run_with() {
  local py="$1"
  exec "$py" frame_extractor.py
}

if [[ -x ".venv-tk/bin/python" ]] && is_tk_compatible ".venv-tk/bin/python"; then
  run_with ".venv-tk/bin/python"
fi

if [[ -x ".venv/bin/python" ]] && is_tk_compatible ".venv/bin/python"; then
  run_with ".venv/bin/python"
fi

cat <<'EOF'
No compatible Python environment was found for the Tk desktop app.

Why this happens:
- The existing `.venv` uses macOS Tcl/Tk 8.5, which crashes Tkinter apps.
- This app needs a Python build with Tcl/Tk 8.6+.

Recommended one-time fix:
1. Create a new venv with a compatible Python:
   python3 -m venv .venv-tk
2. Install dependencies into it:
   .venv-tk/bin/pip install -r requirements.txt
3. Start the app again:
   ./run.sh

Notes:
- On this machine, `python3` appears to use Tk 8.6, so it is a good candidate.
- If step 1 fails, install a Tk-enabled Python from python.org or Homebrew first.
EOF

exit 1
