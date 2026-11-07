# CineMatch API Guide

## Start the backend

From the CineMatch repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_api.py
```

The service listens on `http://127.0.0.1:8000`. Open `/docs` for the generated interactive OpenAPI interface.

## Example requests

```text
GET /api/v1/health
GET /api/v1/movies?q=Star%20Wars&limit=10
GET /api/v1/movies/260
GET /api/v1/movies/260/similar?limit=5
GET /api/v1/recommendations/1?limit=10
GET /api/v1/recommendations/999999?limit=10
GET /api/v1/predictions/1/318
```

## Recommendation behavior

- Known MovieLens user: rank unseen catalog movies with the persisted 20-factor Surprise SVD model.
- Unknown user: return Bayesian-weighted popular movies and mark `cold_start` as `true`.
- Similar movies: compare learned SVD item-factor vectors with cosine similarity.
- Invalid limits: return HTTP 422 through FastAPI validation.
- Unknown movie: return HTTP 404.

## Frontend contract

Movie objects expose `movie_id`, `title`, `genres`, `rating_count`, and `mean_rating`. Recommendation objects add `predicted_rating`. The React development servers on ports 5173 and 3000 are explicitly allowed by CORS.

## Verification

The live server was started locally and returned HTTP 200 for:

- `/api/v1/health`
- `/api/v1/recommendations/1?limit=3`
- `/api/v1/recommendations/999999?limit=2`

All 28 automated project tests passed, including eight API endpoint tests.
