"""Shared bootstrap for directly executable component tests."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_CONFIG = WORKSPACE_DIR / "configs" / "aruco_yolo_pick_place.example.json"
