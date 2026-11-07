# Improvement 1: New-user taste onboarding

## Purpose

The original product returned Bayesian popularity recommendations for every unknown user. The improved flow lets a new viewer rate familiar movies and receive immediate session-based personalization.

## Method

The saved Surprise SVD model already contains an item bias and 20-factor vector for each trained movie. For the submitted ratings, the API solves a regularized least-squares problem for one temporary user bias and user-factor vector. Predictions use the same SVD equation as the trained model:

```text
global mean + user bias + item bias + dot(user factors, item factors)
```

The implementation requires at least three unique ratings, accepts scores from 0.5 to 5.0, excludes submitted movies, clips predictions to the MovieLens scale, and does not store the ratings.

## Actual verification

Two live requests used the same five movies with different ratings. Both returned HTTP 200 with strategy `svd-fold-in`, and their Top-5 lists differed.

Profile A rated Star Wars, Pulp Fiction, and The Shawshank Redemption highly. Its observed Top-5 was:

1. Godfather, The (1972), predicted 5.0000
2. Star Wars: Episode V - The Empire Strikes Back (1980), predicted 4.8382
3. Seven (a.k.a. Se7en) (1995), predicted 4.8112
4. Reservoir Dogs (1992), predicted 4.7966
5. Fight Club (1999), predicted 4.7952

Profile B used a different rating pattern and returned a different list beginning with `Singin' in the Rain (1952)`. These are recorded outputs from the saved model, not example recommendations.

## Limitations

With only a few ratings, a 20-factor user vector remains uncertain. Regularization reduces overfitting, but this session-based profile has not yet been evaluated with an offline onboarding protocol or real users. The next improvement should compare fold-in quality at 3, 5, and 10 onboarding ratings before selecting a recommended minimum.
