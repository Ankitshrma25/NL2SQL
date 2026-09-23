"""Launcher used for packaging (PyInstaller) and for `python run_app.py`.

Starts the Streamlit server on app.py programmatically so a frozen
executable can run the UI without a separate `streamlit` command.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _base_dir() -> Path:
    # PyInstaller unpacks bundled files to sys._MEIPASS
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def main() -> None:
    base = _base_dir()
    os.chdir(base)
    os.environ.setdefault("NL2SQL_DB_URL", f"sqlite:///{base / 'data' / 'sample.db'}")
    os.environ.setdefault("NL2SQL_METADATA", str(base / "data" / "schema_metadata.json"))
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(base / "app.py"), "--global.developmentMode=false",
                "--server.headless=true", "--browser.gatherUsageStats=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
