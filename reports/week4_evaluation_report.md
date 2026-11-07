# CineMatch Final Evaluation Report

## Evaluation design

- MovieLens Small: 100,836 explicit ratings.
- Shared per-user 80/20 holdout with seed 42.
- Rating evaluation: identical deterministic sample of 2,000 held-out ratings.
- Ranking evaluation: 100 users with at least one held-out rating >= 4.0.
- Each ranking candidate set contains all of that user's held-out movies plus 100 sampled unseen movies.
- K = 10.

Unrated sampled movies are evaluation candidates, not confirmed dislikes. Ranking results must therefore be interpreted under this sampled-candidate protocol.

## Rating prediction

| model | rmse | mae |
| --- | --- | --- |
| User-User | 0.9025 | 0.6873 |
| Item-Item | 0.8740 | 0.6617 |
| SVD | 0.8922 | 0.6835 |

Best rating model: **Item-Item**.

## Top-K ranking

| model | evaluated_users | k | precision_at_k | recall_at_k | hit_rate_at_k | ndcg_at_k | catalog_coverage_at_k | unique_recommended_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| User-User | 100 | 10 | 0.1220 | 0.1149 | 0.6500 | 0.1274 | 0.0781 | 761 |
| Item-Item | 100 | 10 | 0.3090 | 0.3158 | 0.8800 | 0.3429 | 0.0673 | 656 |
| SVD | 100 | 10 | 0.3750 | 0.3692 | 0.8900 | 0.4970 | 0.0520 | 507 |

Best ranking model by Precision@10, with NDCG@10 and RMSE as tie-breakers: **SVD**.

## Final selection

The recommended model for the CineMatch demo is **SVD** because the product's primary goal is ranked discovery. This selection is based on measured Precision@10 first, not on an assumption that matrix factorization must win.

## Cold start

For a new user with no ratings, collaborative filtering has no personal signal. CineMatch therefore uses a Bayesian-weighted popularity list with a prior weight of 50 ratings. This reduces the chance that a movie with one unusually high rating outranks broadly liked movies. A future hybrid can ask new users to select genres or rate several onboarding titles.

## Limitations

- Precision and recall are measured with sampled negatives because MovieLens does not identify true dislikes among unrated movies.
- The data is a historical development snapshot and contains no demographics or current catalog availability.
- Offline accuracy does not measure novelty, diversity, fairness, or real user satisfaction.
- Latent factors are not automatically interpretable genre labels.
