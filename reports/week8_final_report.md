# CineMatch Final Internship Report

## Project objective

CineMatch addresses movie-choice overload with collaborative filtering. The system learns from MovieLens rating behavior and ranks unseen movies for each user. The implementation progresses from exploratory analysis to memory-based filtering, matrix factorization, offline evaluation, an API, a React interface, and production packaging.

## Data preparation

The project uses the included MovieLens Small snapshot. It contains 100,836 ratings from 610 users and 9,742 rated movies. The utility matrix has 5,931,640 possible user-movie cells and 98.3000% sparsity. This high sparsity explains why neighbor coverage and fallback behavior matter.

## Models

Week 2 implemented mean-centered User-User and Item-Item collaborative filtering with cosine similarity. Week 3 tuned Surprise SVD with 20, 50, and 100 latent factors. The validation split selected 20 factors with RMSE 0.8779.

The final test comparison used the same 2,000 held-out ratings for every model:

| Model | RMSE | MAE |
|---|---:|---:|
| User-User | 0.9025 | 0.6873 |
| Item-Item | 0.8740 | 0.6617 |
| SVD | 0.8922 | 0.6835 |

Item-Item produced the lowest rating error. CineMatch selected SVD for the product because it performed best on the measured ranking objective:

| Model | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| User-User | 0.122 | 0.115 | 0.65 | 0.127 |
| Item-Item | 0.309 | 0.316 | 0.88 | 0.343 |
| SVD | 0.375 | 0.369 | 0.89 | 0.497 |

The ranking protocol evaluated 100 seeded users. Each model ranked the same candidate set per user: all held-out items plus 100 sampled unseen items. Relevant items had held-out ratings of at least 4.0.

## Product implementation

FastAPI loads the saved 20-factor model once at startup. It exposes health, search, movie detail, similar-movie, prediction, and recommendation endpoints. Known users receive unseen movies ranked by SVD. Unknown users receive a Bayesian popularity fallback with a prior weight of 50.

The React interface supports user profiles, personalized recommendations, cold-start messaging, title search, genre filters, movie details, and latent-space similarities. The production package serves the compiled React application and API from one FastAPI process.

## Verification

The final automated suite passed 50 tests. The React production build passed. A live combined-service check returned HTTP 200 for the website root, model health, and recommendations for user 1. The recorded first result in that check was `Shawshank Redemption, The (1994)`.

## Limitations

- MovieLens Small contains explicit ratings but no live viewing or click behavior.
- Offline ranking metrics do not measure real user satisfaction.
- New users require a popularity fallback until they provide ratings.
- SVD latent factors support similarity calculations but do not provide human-readable explanations by themselves.
- The dataset snapshot is small compared with a commercial streaming catalog.

## Conclusion

CineMatch demonstrates an end-to-end recommendation workflow with reproducible evidence. Item-Item best predicts individual ratings in this experiment, while SVD best ranks relevant unseen movies. The deployed product therefore uses SVD for personalized discovery and a Bayesian fallback for cold-start users.
