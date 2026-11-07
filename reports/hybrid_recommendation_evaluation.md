# Hybrid recommendation evaluation

## Validation selection rule

- Baseline is the selected hybrid model with weights 0.7 collaborative, 0.3 genre, 0.0 popularity.
- A reranking configuration qualifies only if validation NDCG@10 remains within 2% relative of the baseline.
- Among qualifying configurations, choose the highest intra-list diversity; tie-break by novelty and then by larger lambda.

- Selected lambda: 0.90
- Validation baseline NDCG@10: 0.5287
- Final test NDCG@10: 0.5289

## Validation results

- lambda=1.00: NDCG=0.5287, Precision=0.3960, Recall=0.4122, HitRate=0.9200, Diversity=0.7367, Novelty=11.1740.
- lambda=0.90: NDCG=0.5307, Precision=0.3990, Recall=0.4114, HitRate=0.9200, Diversity=0.7668, Novelty=11.1277.
- lambda=0.80: NDCG=0.5177, Precision=0.3840, Recall=0.3869, HitRate=0.9200, Diversity=0.7960, Novelty=11.1735.
- lambda=0.70: NDCG=0.5082, Precision=0.3730, Recall=0.3790, HitRate=0.9200, Diversity=0.8224, Novelty=11.2060.
- lambda=0.60: NDCG=0.4974, Precision=0.3660, Recall=0.3784, HitRate=0.9200, Diversity=0.8444, Novelty=11.2427.
- lambda=0.50: NDCG=0.4723, Precision=0.3420, Recall=0.3508, HitRate=0.9000, Diversity=0.8604, Novelty=11.3701.

## Final test summary

- strategy: final-test-mmr
- evaluated_users: 100
- precision_at_k: 0.3440000000000001
- recall_at_k: 0.4868922101514512
- hit_rate_at_k: 0.93
- ndcg_at_k: 0.5289041490713844
- intra_list_diversity: 0.7696046496713163
- mean_novelty: 11.405583387311397
- genre_count: 11.32
- catalog_coverage: 0.06179429275302813