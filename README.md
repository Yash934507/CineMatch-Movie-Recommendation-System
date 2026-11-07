# CineMatch

CineMatch is a local, full-stack movie recommendation application. It combines a React interface with a FastAPI backend and MovieLens Small data to let viewers explore movies, compare recommendation approaches, and build a private account taste profile.

**Project status:** Local development and submission verification only. CineMatch is not deployed.

## Problem statement

Movie catalogs can be difficult to navigate, and general popularity alone does not reflect an individual viewer's preferences. CineMatch explores collaborative-filtering techniques and provides a working interface for discovering movies from historical ratings and a viewer's own feedback.

## Objectives

- Explore user-user and item-item collaborative filtering and matrix factorization.
- Serve MovieLens demo recommendations separately from private account recommendations.
- Support movie search, details, similar titles, and a temporary onboarding profile.
- Let registered users save ratings, watchlist entries, and feedback.
- Keep the application reproducible and usable locally with SQLite.

## Technologies

- **Frontend:** React, Vite, JavaScript, CSS.
- **Backend:** Python, FastAPI, Pydantic, Uvicorn.
- **Recommendations:** pandas, NumPy, scikit-surprise SVD, collaborative-filtering evaluation helpers.
- **Accounts and persistence:** SQLAlchemy, SQLite, Alembic, Argon2id password hashing, HttpOnly session cookies.
- **Tests/build:** pytest, Node's built-in test runner, pnpm.

The backend has been verified with Python 3.12.14. Use Node.js 22.12+ (or Node 20.19+) for the Vite 8 toolchain; the package lock pins frontend dependency versions and `package.json` specifies pnpm 9.15.0.

## Recommendation approaches

- **User-user collaborative filtering:** explores users with similar rating patterns.
- **Item-item collaborative filtering:** explores relationships between similarly rated movies.
- **SVD:** the saved 20-factor Surprise model powers MovieLens demo predictions and recommendations.
- **Cold start:** Bayesian/popularity-ranked recommendations are used before there are enough account ratings.
- **Account personalization:** once an account has at least three ratings, the service folds those ratings into the SVD item-factor space and ranks unseen movies. Before then, it returns the cold-start list.
- **Onboarding:** temporary ratings can be used to produce a taste-based list without saving those ratings to an account.

User-user and item-item methods are part of the project's experiments; the current web recommendation routes use the saved SVD model and cold-start fallback rather than running those experiments on each request.

## Main features

- Public catalog search, movie details, similar titles, and MovieLens demo recommendations/predictions.
- Registration, login, session restoration, and logout using server-managed sessions.
- Account-specific ratings, watchlist, feedback, and personalized recommendations.
- Public demo data is read-only and separate from account records.
- Responsive, cinematic React interface.

## System architecture

```text
React + Vite browser
        | /api/v1 (Vite proxies to FastAPI during development)
        v
FastAPI routes and recommendation services
   |                         |
   |                         +--> MovieLens CSVs + saved SVD model
   +--> SQLAlchemy --> local SQLite (accounts, sessions, ratings, watchlist, feedback)
```

In development, FastAPI serves the API and Vite serves the frontend on separate local ports. For a built single-service run, FastAPI can serve the Vite build from `frontend/dist`. This documentation describes local use only and does not claim that the service has been deployed.

## Data, model, and licensing

The application needs:

- `data/raw/movies.csv` and `data/raw/ratings.csv` to load and validate the MovieLens catalog and ratings.
- `models/svd_model.pkl` for the trained demo model.
- `data/processed/week4_cold_start_recommendations.csv` for cold-start recommendations.

The other MovieLens CSV files and the dataset's full attribution/license text are included for completeness. MovieLens Small may be redistributed only under the conditions in `data/raw/README.txt`; in particular, acknowledge GroupLens and do not use the dataset for commercial or revenue-bearing purposes without permission. The model artifact was trained from MovieLens data and is supplied with the same dataset attribution/license notice. The project code's license is in `LICENSE`.

If the dataset files are missing, download and verify MovieLens Small with:

```powershell
python .\scripts\download_movielens_small.py
```

The script checks a pinned SHA-256 for `ratings.csv`. The saved model must also be present; retraining is not part of normal application startup.

## Local installation (Windows PowerShell)

From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .[test]
```

Install the frontend dependencies from the `frontend` folder using the pinned package manager and lockfile:

```powershell
corepack enable
corepack prepare pnpm@9.15.0 --activate
Set-Location .\frontend
pnpm install --frozen-lockfile
Set-Location ..
```

The Vite build requires a supported modern Node version. If `corepack` is unavailable, install Node.js 22 and pnpm 9.15.0 using the official tools for your platform.

## Start the application

Use two PowerShell terminals, both from the project root.

**Terminal 1 — FastAPI:**

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\run_api.py
```

The development default is `sqlite:///./data/cinematch.db`. Development startup automatically runs Alembic migrations. The current schema revision is `20261107_auth_schema`. Do not point the application at a database containing records you need to preserve unless you have a backup and intend to use it.

For a disposable local demo/test account database, use this instead in Terminal 1:

```powershell
$env:CINEMATCH_ENV = "development"
$env:CINEMATCH_DATABASE_URL = "sqlite:///./data/cinematch_demo.db"
$env:CINEMATCH_ALLOWED_ORIGINS = "http://127.0.0.1:5173"
python .\scripts\run_api.py
```

This creates a separate `data/cinematch_demo.db`; it does not use or alter `data/cinematch.db`. Remove only this demo database after stopping the API if it is no longer needed.

**Terminal 2 — React/Vite:**

```powershell
.\scripts\run_frontend.ps1
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The frontend launcher uses a compatible bundled Node runtime on the original development machine when available.

Useful local endpoints:

- Homepage: `http://127.0.0.1:5173/`
- API health: `http://127.0.0.1:8000/api/v1/health`
- API readiness: `http://127.0.0.1:8000/api/v1/ready`
- API docs: `http://127.0.0.1:8000/docs`

## Read-only public demo mode

The container deployment template is configured for a public, read-only demonstration:

- Set `CINEMATCH_READ_ONLY_DEMO=true` and use an isolated SQLite file on the host's ephemeral filesystem.
- Account registration, login/logout, ratings, watchlist, feedback, and event writes are rejected by the API.
- The temporary onboarding recommendations endpoint remains available because it does not persist submitted ratings.
- The Docker frontend build sets `VITE_CINEMATCH_READ_ONLY_DEMO=true` so account prompts are not offered.
- The Render free-service template uses `/tmp/cinematch-demo.db`; its contents are disposable and must never contain real user records.

Render's free service may sleep after inactivity and its filesystem is ephemeral. This setup is for a read-only project demo, not production authentication or persistent accounts. The project is not deployed until a public URL has been created and verified.

### Deploying the read-only demo on Render Free

After the source repository is published and connected in Render:

1. Create a **Web Service** from the repository and select the existing `render.yaml` Blueprint configuration (Docker runtime, Free plan).
2. Keep the configured `CINEMATCH_READ_ONLY_DEMO=true` and `CINEMATCH_DATABASE_URL=sqlite:////tmp/cinematch-demo.db`; do not add real account data or credentials.
3. Wait for the `/api/v1/ready` health check to pass, then verify the homepage, `/api/v1/health`, movie search, and public demo recommendations at the generated Render URL.
4. Record the URL only after those public checks succeed.

The free service sleeps after inactivity and `/tmp` is ephemeral; it is intentionally unsuitable for persistent accounts. Render Free services are for demos and testing, not production workloads.

## Five-minute demonstration

Use the disposable database startup above before creating a test account.

1. Open the homepage and search for “Star Wars”; open a result to show movie details.
2. Show the public, read-only MovieLens demo recommendations (sample user ID `1`).
3. Register a disposable account or sign in with a test account.
4. Search for and rate at least three movies to meet the personalization threshold.
5. Return to recommendations and show the account-specific list.
6. Open a movie and use **Add to watchlist** and **Like**.
7. Sign out and confirm the public demo is shown again.

Do not use the MovieLens sample ID as an account ID. The demo ID selects public sample recommendations only; account actions use the signed-in session.

## Verification

Verified locally for this submission:

- Backend: `python -m pytest -q` — **137 passed**, 2 deprecation warnings.
- Frontend: `pnpm test` — **15 passed**.
- Production frontend build: `pnpm build` — **passed** with Node 24.22.1; output was built in a temporary directory to preserve the running app's served assets.
- Browser/API smoke test: homepage, movie search/details, MovieLens demo, and disposable-account registration/login, session-backed personalized recommendations, three ratings, watchlist, feedback, and logout — **passed**.
- SQLite migration revision read-only check — **`20261107_auth_schema`**.

The backend warnings are from Starlette's TestClient/httpx deprecation and `datetime.utcnow()` use in the rating repository; they did not fail tests.

To run tests yourself:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
Set-Location .\frontend
pnpm test
pnpm build
```

## Limitations and future scope

- MovieLens Small is a historical, relatively small research/development dataset, not a live movie catalog.
- New accounts initially receive popularity-based recommendations; personalized SVD fold-in requires at least three saved ratings.
- The local SQLite database is suitable for this project demo, not a multi-instance production database.
- Authentication rate limiting is process-local and is not shared across multiple backend workers.
- The frontend uses generated artwork rather than a live poster service.
- Future work could add richer account-history views, more evaluation/monitoring, and a shared database or rate limiter if the project is intentionally scaled. Those changes are not required for local submission.

## Repository hygiene

Do not include `.env` files with real values, local account databases, `.venv`, `node_modules`, generated build output, logs, or temporary test files in a submission archive. The example environment files contain templates only; every `VITE_` variable is public in the browser bundle and must never contain a secret.
