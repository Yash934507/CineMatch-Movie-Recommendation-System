from pathlib import Path

from fastapi.testclient import TestClient

from cinematch.api import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unknown_api_paths_return_json_404_and_spa_paths_still_fall_back(tmp_path):
    frontend_root = tmp_path / "frontend"
    frontend_root.mkdir()
    (frontend_root / "index.html").write_text("<html>CineMatch</html>", encoding="utf-8")
    client = TestClient(create_app(PROJECT_ROOT, frontend_dir=frontend_root))

    api_response = client.get("/api/v1/not-a-route")
    frontend_response = client.get("/watchlist")

    assert api_response.status_code == 404
    assert api_response.json() == {"detail": "Not Found"}
    assert api_response.headers["content-type"].startswith("application/json")
    assert frontend_response.status_code == 200
    assert frontend_response.text == "<html>CineMatch</html>"

    client.close()
