"""Run Week 1 from the repository root: python scripts/run_week1.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cinematch.week1 import run_week1  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run_week1(ROOT), indent=2))

