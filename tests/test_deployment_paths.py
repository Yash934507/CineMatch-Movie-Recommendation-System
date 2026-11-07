from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_container_pythonpath_resolves_runtime_assets_outside_project_cwd(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["CINEMATCH_TEST_PROJECT_ROOT"] = str(project_root)
    code = """
import os
from pathlib import Path
from cinematch.api import PROJECT_ROOT
from cinematch.config import settings

root = Path(os.environ["CINEMATCH_TEST_PROJECT_ROOT"]).resolve()
assert PROJECT_ROOT == root
assert settings.model_file == root / "models" / "svd_model.pkl"
assert (root / "data" / "raw" / "movies.csv").is_file()
assert settings.model_file.is_file()
assert (root / "frontend" / "dist" / "index.html").is_file()
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
