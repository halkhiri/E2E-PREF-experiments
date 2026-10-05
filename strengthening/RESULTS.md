# Independent evaluation and component study

All planned results are reported. Model and checkpoint selection used validation only; primary outcomes use the complete eligible static catalog. Intervals describe conditional resampling of the observed five-seed/user grid.

## movielens

| Model | Full catalog NDCG@10 | Sampled NDCG@10 |
|---|---:|---:|
| E2E-PREF ranking only | 0.01776 ± 0.00131 | 0.55091 ± 0.00350 |
| Without graph encoder | 0.01898 ± 0.00173 | 0.54319 ± 0.00458 |
| Mean history instead of temporal encoder | 0.01886 ± 0.00179 | 0.60499 ± 0.00677 |
| Without frozen text features | 0.01715 ± 0.00198 | 0.55355 ± 0.00248 |
| Candidate-independent scoring | 0.01760 ± 0.00098 | 0.54782 ± 0.00361 |
| All auxiliary losses | 0.02097 ± 0.00290 | 0.60468 ± 0.01846 |
| Contrastive loss only | 0.02009 ± 0.00155 | 0.60299 ± 0.00401 |
| Temporal loss only | 0.01757 ± 0.00251 | 0.55218 ± 0.00338 |
| Semantic loss only | 0.01773 ± 0.00126 | 0.55136 ± 0.00374 |
| BPR-MF | 0.02246 ± 0.00261 | 0.55559 ± 0.00381 |
| SASRec port | 0.01645 ± 0.00285 | 0.57977 ± 0.00663 |
| Popularity | 0.01783 | 0.55631 |
| Item neighborhood | 0.03525 | 0.61381 |
| Matched joint continuation | 0.01721 ± 0.00176 | 0.54934 ± 0.00316 |
| Matched frozen continuation | 0.01742 ± 0.00200 | 0.54844 ± 0.00513 |

### Matched representation updating

catalog: joint minus frozen = -0.00021, 95% conditional interval [-0.00323, +0.00264], positive in 3/5 seeds. The 97.5% interval is [-0.00370, +0.00304].
sampled: joint minus frozen = +0.00090, 95% conditional interval [-0.00480, +0.00678], positive in 3/5 seeds. The 97.5% interval is [-0.00576, +0.00772].

### Exploratory full-catalog contrasts

| Left minus right | Difference | 95% conditional interval | Positive seeds |
|---|---:|---:|---:|
| E2E-PREF ranking only minus Without graph encoder | -0.00123 | [-0.00380, +0.00125] | 1/5 |
| E2E-PREF ranking only minus Mean history instead of temporal encoder | -0.00110 | [-0.00571, +0.00362] | 1/5 |
| E2E-PREF ranking only minus Without frozen text features | +0.00061 | [-0.00198, +0.00335] | 3/5 |
| E2E-PREF ranking only minus Candidate-independent scoring | +0.00015 | [-0.00171, +0.00205] | 4/5 |
| E2E-PREF ranking only minus BPR-MF | -0.00471 | [-0.00921, -0.00054] | 1/5 |
| E2E-PREF ranking only minus SASRec port | +0.00131 | [-0.00489, +0.00769] | 4/5 |
| All auxiliary losses minus E2E-PREF ranking only | +0.00322 | [-0.00112, +0.00783] | 5/5 |
| Contrastive loss only minus E2E-PREF ranking only | +0.00233 | [-0.00261, +0.00743] | 5/5 |
| Temporal loss only minus E2E-PREF ranking only | -0.00019 | [-0.00326, +0.00251] | 2/5 |
| Semantic loss only minus E2E-PREF ranking only | -0.00002 | [-0.00195, +0.00154] | 2/5 |

## amazon

| Model | Full catalog NDCG@10 | Sampled NDCG@10 |
|---|---:|---:|
| E2E-PREF ranking only | 0.04987 ± 0.00340 | 0.16825 ± 0.00336 |
| Without graph encoder | 0.04818 ± 0.00290 | 0.16780 ± 0.00167 |
| Mean history instead of temporal encoder | 0.04581 ± 0.00196 | 0.15781 ± 0.00617 |
| Without frozen text features | 0.04909 ± 0.00301 | 0.16810 ± 0.00279 |
| Candidate-independent scoring | 0.04938 ± 0.00258 | 0.16879 ± 0.00177 |
| All auxiliary losses | 0.04998 ± 0.00231 | 0.16466 ± 0.00684 |
| Contrastive loss only | 0.04997 ± 0.00142 | 0.16554 ± 0.00147 |
| Temporal loss only | 0.05060 ± 0.00304 | 0.16882 ± 0.00187 |
| Semantic loss only | 0.04753 ± 0.00368 | 0.16609 ± 0.00289 |
| BPR-MF | 0.02803 ± 0.00225 | 0.11564 ± 0.00793 |
| SASRec port | 0.04495 ± 0.00429 | 0.16004 ± 0.00594 |
| Popularity | 0.04404 | 0.15941 |
| Item neighborhood | 0.03627 | 0.15487 |
| Matched joint continuation | 0.04762 ± 0.00323 | 0.16683 ± 0.00147 |
| Matched frozen continuation | 0.04923 ± 0.00287 | 0.16778 ± 0.00091 |

### Matched representation updating

catalog: joint minus frozen = -0.00161, 95% conditional interval [-0.00419, +0.00118], positive in 2/5 seeds. The 97.5% interval is [-0.00456, +0.00170].
sampled: joint minus frozen = -0.00095, 95% conditional interval [-0.00410, +0.00204], positive in 1/5 seeds. The 97.5% interval is [-0.00452, +0.00258].

### Exploratory full-catalog contrasts

| Left minus right | Difference | 95% conditional interval | Positive seeds |
|---|---:|---:|---:|
| E2E-PREF ranking only minus Without graph encoder | +0.00169 | [-0.00036, +0.00429] | 5/5 |
| E2E-PREF ranking only minus Mean history instead of temporal encoder | +0.00406 | [-0.00237, +0.01085] | 5/5 |
| E2E-PREF ranking only minus Without frozen text features | +0.00078 | [-0.00072, +0.00257] | 4/5 |
| E2E-PREF ranking only minus Candidate-independent scoring | +0.00049 | [-0.00328, +0.00518] | 2/5 |
| E2E-PREF ranking only minus BPR-MF | +0.02185 | [+0.01271, +0.03108] | 5/5 |
| E2E-PREF ranking only minus SASRec port | +0.00493 | [-0.00234, +0.01180] | 5/5 |
| All auxiliary losses minus E2E-PREF ranking only | +0.00011 | [-0.00330, +0.00362] | 3/5 |
| Contrastive loss only minus E2E-PREF ranking only | +0.00009 | [-0.00302, +0.00364] | 1/5 |
| Temporal loss only minus E2E-PREF ranking only | +0.00073 | [-0.00020, +0.00181] | 5/5 |
| Semantic loss only minus E2E-PREF ranking only | -0.00235 | [-0.00826, +0.00401] | 1/5 |

## Scope

MovieLens uses 1,000 previously unused users, with zero overlap with the earlier pilot. Amazon Musical Instruments is an independent product domain with 1,092 eligible users and 900 products. This is a small 2014 5-core benchmark; the result does not imply evaluation on the entire Amazon catalog. Both datasets use per-user chronological splits and static side information rather than a globally time-causal deployment simulation.

SASRec is a source-informed PyTorch port using left padding, all valid next-item positions and one negative per position, with its original Adam/logistic training structure. Search equality covers learning-rate trials, maximum epochs and per-user batch opportunities; positive-pair counts and computational costs differ. BPR samples all training positives. Six preliminary MovieLens baseline validation trials were archived before test access following the implementation audit. Hyperparameters other than learning rate and item-neighborhood size are fixed. All component and auxiliary-loss comparisons are exploratory, with no multiplicity-controlled significance claim.
