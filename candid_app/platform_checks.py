import platform
import sys


def ensure_supported_tk() -> None:
    if platform.system() != "Darwin":
        return
    try:
        import _tkinter
    except Exception:
        return

    tcl = getattr(_tkinter, "TCL_VERSION", "") or ""
    if tcl.startswith("8.5") or tcl == "8.5":
        print(
            "Tcl/Tk 8.5 detected on macOS. This app needs Tcl/Tk 8.6+.\n"
            "Use a compatible Python, recreate the venv, then reinstall dependencies.\n"
            "Example:\n"
            "  python3 -m venv .venv-tk\n"
            "  .venv-tk/bin/pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

