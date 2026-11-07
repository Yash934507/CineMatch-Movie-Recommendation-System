"""Start CineMatch API: python scripts/run_api.py"""

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    uvicorn.run(
        "cinematch.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=str(ROOT / "src"),
    )
