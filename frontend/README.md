# CineMatch React Frontend

## Run locally

Start the FastAPI backend from the repository root:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\run_api.py"
```

In a second terminal, return to the repository root and run:

```powershell
.\scripts\run_frontend.ps1
```

Open `http://127.0.0.1:5173`. The launcher uses Codex's bundled modern Node runtime when available because the machine's global Node 16 is too old for Vite 8. The frontend uses `http://127.0.0.1:8000/api/v1` by default. Set `VITE_API_URL` to override the API base URL.
