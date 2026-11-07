# CineMatch Week 7: Deployment and Portfolio Guide

## What changed

CineMatch is now packaged as one production service. Vite compiles the React website and FastAPI serves that build alongside the recommendation endpoints. The frontend uses the relative `/api/v1` URL in production and Vite proxies `/api` to port 8000 during development.

The deployment files are:

- `Dockerfile`: reproducible Node build followed by a Python 3.12 runtime.
- `.dockerignore`: excludes development-only and regenerated files.
- `render.yaml`: declares a Docker web service and model-backed health check.
- `frontend/dist/`: the verified production frontend build.

## Local production check

Build the frontend, then start the combined service:

```powershell
.\scripts\run_frontend.ps1 -Build
& ".\.venv\Scripts\python.exe" -m uvicorn cinematch.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Verify:

1. The CineMatch interface loads.
2. User 1 returns personalized SVD recommendations.
3. An unknown ID such as 999999 shows the cold-start list.
4. Movie search and similar-title dialogs work.
5. `/api/v1/health` reports `status: ok`.

## Docker and Render

With Docker installed:

```powershell
docker build -t cinematch .
docker run --rm -p 8000:8000 cinematch
```

To deploy on Render, push the repository to a Git host, create a Blueprint from `render.yaml`, and wait for `/api/v1/health` to pass. Hosting limits and pricing can change, so confirm the selected plan before deploying. Do not commit secrets; CineMatch currently requires none.

## Verified evidence

The production frontend build passed. All 50 automated tests passed. A live combined-service smoke test returned HTTP 200 for the React root, model health, and user 1 recommendations. The observed first recommendation in that smoke test was `Shawshank Redemption, The (1994)`; this is recorded evidence from the persisted model, not a hand-written example.

Docker itself was not available in the verification environment, so the Dockerfile could not be executed here. The local combined service exercises the same Python application and compiled frontend artifacts.
