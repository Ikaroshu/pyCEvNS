"""Command line that serves the Streamlit page on the loopback interface only."""

import sys
from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")


def streamlit_command(port: int) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.address", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--server.port", str(port),
    ]
