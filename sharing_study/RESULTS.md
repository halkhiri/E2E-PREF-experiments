# Representation sharing experiment

Post hoc extension on previously evaluated datasets; all planned results reported. No new holdout. Intervals are exploratory conditional seed/user bootstrap intervals.

## movielens

| Model | Parameters | Catalog NDCG@10 | Sampled NDCG@10 |
|---|---:|---:|---:|
| full | 989,264 | 0.01776 ± 0.00131 | 0.55091 ± 0.00350 |
| separate | 1,864,496 | 0.01676 ± 0.00218 | 0.55052 ± 0.00518 |
| shared_dual | 1,864,496 | 0.01764 ± 0.00118 | 0.55224 ± 0.00173 |

| Contrast | Protocol | Difference | 95% conditional interval | Positive seeds |
|---|---|---:|---|---:|
| full minus separate | catalog | +0.00100 | [-0.00164, +0.00373] | 3/5 |
| shared_dual minus separate | catalog | +0.00089 | [-0.00208, +0.00372] | 4/5 |
| full minus separate | sampled | +0.00038 | [-0.00657, +0.00746] | 3/5 |
| shared_dual minus separate | sampled | +0.00172 | [-0.00333, +0.00755] | 3/5 |

### Selected settings

full: LR 0.003; selected epochs (seeds 101–105): 50, 44, 39, 49, 37.
separate: LR 0.01; selected epochs (seeds 101–105): 25, 28, 36, 18, 28.
shared_dual: LR 0.01; selected epochs (seeds 101–105): 49, 28, 33, 47, 28.

Both new variants select the upper grid boundary (0.01); the fixed grid was not expanded after seeing results.

## amazon

| Model | Parameters | Catalog NDCG@10 | Sampled NDCG@10 |
|---|---:|---:|---:|
| full | 124,614 | 0.04987 ± 0.00340 | 0.16825 ± 0.00336 |
| separate | 155,750 | 0.04801 ± 0.00451 | 0.16819 ± 0.00291 |
| shared_dual | 155,750 | 0.04846 ± 0.00367 | 0.16883 ± 0.00389 |

| Contrast | Protocol | Difference | 95% conditional interval | Positive seeds |
|---|---|---:|---|---:|
| full minus separate | catalog | +0.00186 | [-0.00057, +0.00558] | 3/5 |
| shared_dual minus separate | catalog | +0.00046 | [-0.00269, +0.00383] | 3/5 |
| full minus separate | sampled | +0.00006 | [-0.00420, +0.00407] | 2/5 |
| shared_dual minus separate | sampled | +0.00064 | [-0.00318, +0.00432] | 3/5 |

### Selected settings

full: LR 0.01; selected epochs (seeds 101–105): 13, 49, 35, 32, 43.
separate: LR 0.01; selected epochs (seeds 101–105): 30, 49, 40, 32, 43.
shared_dual: LR 0.01; selected epochs (seeds 101–105): 6, 19, 35, 32, 43.

Both new variants select the upper grid boundary (0.01); the fixed grid was not expanded after seeing results.

