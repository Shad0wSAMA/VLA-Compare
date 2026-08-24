"""Small launcher shared by the local LeRobot convenience scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
LEROBOT_DIR = WORKSPACE / "lerobot"


def run_lerobot(entrypoint: str, args: list[str]) -> int:
    """Run a LeRobot console entrypoint from the repository checkout."""
    uv = shutil.which("uv")
    if uv:
        command = [uv, "run", "--directory", str(LEROBOT_DIR), entrypoint, *args]
        return subprocess.run(command, cwd=WORKSPACE).returncode

    # Fallback for machines where uv is not on PATH.  The source checkout is
    # added explicitly, so this still works with the workspace venv.
    workspace_python = WORKSPACE / "venv" / "Scripts" / "python.exe"
    python = str(workspace_python) if workspace_python.exists() else sys.executable
    env = os.environ.copy()
    source_dir = str(LEROBOT_DIR / "src")
    env["PYTHONPATH"] = source_dir + os.pathsep + env.get("PYTHONPATH", "")
    module = {
        "lerobot-teleoperate": "lerobot.scripts.lerobot_teleoperate",
        "lerobot-record": "lerobot.scripts.lerobot_record",
        "lerobot-calibrate": "lerobot.scripts.lerobot_calibrate",
        "lerobot-setup-motors": "lerobot.scripts.lerobot_setup_motors",
    }[entrypoint]
    return subprocess.run([python, "-m", module, *args], cwd=WORKSPACE, env=env).returncode
