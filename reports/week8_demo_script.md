# CineMatch Demo Script

## Before the presentation

From the repository root, build the frontend and start the combined service:

```powershell
.\scripts\run_frontend.ps1 -Build
& ".\.venv\Scripts\python.exe" -m uvicorn cinematch.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and keep `http://127.0.0.1:8000/docs` available in a second tab.

## Five-minute walkthrough

### 1. Problem and data

“CineMatch reduces choice overload by learning from rating patterns. The included MovieLens snapshot has 100,836 ratings from 610 users. Its utility matrix is 98.3% sparse, so most user-movie pairs are unknown.”

### 2. Personalized recommendations

Enter user ID `1` and select **Find my matches**.

“This is a known MovieLens user. The API removes already-rated movies, predicts the remaining catalog with the saved 20-factor SVD model, and returns the highest-scoring unseen titles.”

Do not promise a specific ordering during the spoken demo unless it matches the screen. The persisted model is deterministic in this repository, but the interface should remain the source of truth.

### 3. Cold-start behavior

Enter user ID `999999`.

“This profile has no rating history. CineMatch switches to a Bayesian popularity score instead of pretending it can personalize. The prior reduces the influence of movies with very few ratings.”

### 4. Search and similarity

Search for `Star Wars`, open a movie, and show the similar-title section.

“These similarities come from cosine similarity between learned SVD item vectors. They reflect rating patterns, not plot descriptions.”

### 5. Evaluation and API

Open `/docs` and execute `/api/v1/health`.

“Item-Item achieved the lowest test RMSE at 0.8740. SVD achieved the best ranking results with Precision@10 of 0.375 and NDCG@10 of 0.497, so the product uses SVD for ranked discovery.”

## Questions to prepare for

**Why did SVD not have the lowest RMSE?**  
Different objectives can select different models. Item-Item predicted held-out ratings more accurately, while SVD ordered relevant unseen movies more effectively in the ranking evaluation.

**How did you prevent leakage?**  
Model selection used training and validation data only. The fixed outer test sample remained untouched until final evaluation. Recommendations also exclude movies already rated by the user.

**How do you support a new user?**  
The API returns Bayesian-weighted popular movies and marks the response as cold-start. A future onboarding flow could collect a small set of ratings before personalization.

**What would you improve with more time?**  
Add implicit feedback, temporal evaluation, richer content features, calibrated diversity, online experiments, and monitoring for drift.
