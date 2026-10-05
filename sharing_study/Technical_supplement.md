# E2E-PREF technical supplement

Supporting material for the framework manuscript including the representation-sharing extension.

## S1 Representation-sharing experiment


Locked 2026-10-04 before this study's training or test evaluation. This is a post hoc extension motivated by the architectural claim, not independent preregistration or a new holdout. Prior results on these datasets are already known.

Question: Does routing a common graph representation into both the history encoder and item fusion improve ranking relative to independently learned graph pathways?

Reference: existing ranking-only full model and its saved five-seed predictions/checkpoints, unchanged. Each new model retains dimension 32, graph topology, history encoder, fusion, candidate attention, ranking objective, data, slates and optimization protocol.

Separate: two independently initialized graph banks, each consisting of a complete item embedding matrix plus two GAT layers. Bank A feeds the history encoder; bank B feeds item fusion. All remaining parameters are initialized identically to the original model for the same seed. Both banks receive ranking gradients through their respective paths. This model has more parameters than the original; report that difference explicitly.

Shared dual-bank capacity control: exactly the same two independently initialized banks and all other parameters as Separate. Compute their arithmetic mean and feed that same vector into both pathways. This has precisely the same parameter count and graph-bank compute as Separate. It tests shared routing/averaging at fixed parameter budget, not parameter tying alone. It is a diagnostic control, not a newly selected replacement framework. All three models are reported regardless of performance.

Datasets: existing MovieLens cohort and Amazon Musical Instruments, frozen data.pt and candidate sets. No data re-filtering or new features. New variants: separate and shared_dual. Each receives LR .001/.003/.01, seed501, 50 epochs, sampled validation NDCG@10 selection. Five confirmation seeds101–105,50 epochs with independently selected validation checkpoints. Same per-seed schedules and dropout random streams; second-bank initialization uses a separate RNG stream seed+40000. All tuning and confirmation on BOTH datasets must complete before evaluating any new test results. No test-driven retries or architecture changes.

Primary contrasts: original shared minus Separate; Shared dual-bank minus Separate. Report both datasets, full-catalog NDCG@10 as primary, sampled NDCG@10 secondary, Recall@20/MRR descriptive. Paired seed/user crossed bootstrap5000 draws,95% conditional intervals; exploratory, no multiplicity-corrected significance claim. Exact same user order required. Record parameters, times, chosen LR/epochs, saved ranks, independent rank-sort checks and source/data/checkpoint hashes. Full-catalog attention covers each complete eligible slate.

Code checks before training: both new variants have equal parameter counts and identical initial tensors; graph-only history/fusion gradients reach the intended banks; sparse graph computation matches dense computation; optimized full-catalog scorer matches direct complete-slate scoring. Existing original code/data stay unchanged.


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



### Interpretation

The original shared model has higher observed full-catalog means in both domains: 0.01776 versus 0.01676 on MovieLens and 0.04987 versus 0.04801 on Amazon. Its paired differences are +0.00100 [−0.00164, +0.00373] and +0.00186 [−0.00057, +0.00558], respectively. At equal parameter count, the shared-average control also has higher means than separate pathways, with differences of +0.00089 [−0.00208, +0.00372] on MovieLens and +0.00046 [−0.00269, +0.00383] on Amazon. Every interval includes zero, so these comparisons establish neither a ranking advantage nor equivalence. The direct architectural finding is the observed performance–size trade-off: the original sharing design uses fewer parameters while achieving slightly higher mean scores in these runs. All sampled-ranking intervals likewise include zero; the supplement reports every comparison.

The sharing study directly evaluates the connection between the history and item pathways. Retaining the original shared encoder gives the most compact of the tested designs and the highest observed full-catalog mean among the three sharing configurations in each domain. The equal-parameter control separates this observation from simply adding an encoder: its mean differences also favor shared representations, although the estimated effects remain uncertain. These results characterize representation sharing as a parsimonious architectural option under the tested protocol.

## S2 Original implementation and numerical records

Section and table numbers below retain their technical-record numbering. Reference numbers match the revised main manuscript.

3 The E2E-PREF Framework

E2E-PREF connects four stages in a shared ranking pipeline. A training-only item graph refines embeddings using collaborative relationships. A causal temporal encoder then summarizes the ordered graph-refined history into a user preference vector. In the item pathway, learned projections fuse graph embeddings with frozen text features and structured metadata. Finally, the user vector attends to the supplied candidate slate, and a shared product/sum scorer ranks its items. Ranking feedback reaches the graph, temporal, projection, fusion and scoring modules through this connection. Graph construction, text extraction and candidate generation supply fixed inputs. The framework therefore integrates complementary preference signals through a common differentiable objective; the following subsections specify each stage and its information boundaries.

See Figure 1 in the main manuscript for the architecture diagram.

3.1 Task and Differentiable Scope

For user u, a positive history Su precedes a target item i⁺. The model scores i within a supplied candidate set Cu using s(u,i | Cu). Ranking gradients reach the trainable user encoder, item encoder, and scoring head. Graph construction, candidate generation, frozen language-model inference, and raw metadata extraction are outside that gradient path. Training optimizes a differentiable ranking surrogate over supplied candidates.

3.2 Graph and Temporal Preference Encoder

Two items are connected if they appear in at least two users’ training-positive histories. Each item retains its eight most frequent neighbors (ties broken by ascending item index) and a self-loop. Two four-head graph-attention layers combine a learned source/neighbor compatibility term with a count prior, followed by ELU aggregation, residual connections and layer normalization. Isolated items keep only their self-loop, and the graph uses training interactions only.

With nᵢⱼ the number of training users shared by items i and j, the stored edge weight is wᵢⱼ = log(1 + nᵢⱼ), and the self-loop weight is one. The attention logit adds log(wᵢⱼ) to a LeakyReLU compatibility score with negative slope 0.2. Consequently, its unnormalized softmax weight is exp(compatibility) × log(1 + nᵢⱼ). This compresses count differences while retaining their ordering; all logarithms are natural.

Training computes the exact two-hop graph neighborhood needed by each batch and applies feature fusion to the required items. This sparse computation preserves the original graph operation; dense and sparse scores and gradients were numerically checked. Item representations are recomputed after parameter updates. Evaluation constructs the complete item representations and applies candidate attention over each user’s entire eligible slate.

The implementation uses dimension d = 32 and the last 30 positive events. Graph-refined embeddings receive learned absolute positions and time-gap embeddings. Gaps are bucketed as min(31, floor(log₂(1 + gap in hours))); the first retained gap is zero. Four causal transformer layers, each with four heads, feed-forward width 128, GELU and dropout 0.2, process the right-padded sequence. Learned tanh attention pooling produces the user vector:
Equation (1): see the linear equation transcriptions below.
Right-padding masks preserve valid history positions, and causal attention protects each supplied prefix from subsequent positions. The item graph is constructed from the static training partition described in Section 4.

3.3 Text and Structured Feature Fusion

The frozen all-MiniLM-L6-v2 encoder maps item titles and category words to normalized 384-dimensional vectors. Text is truncated at 256 wordpieces and pooled by the attention-mask-weighted mean. A learned linear layer projects these vectors to 32 dimensions. Category features are averages of learned 32-dimensional embeddings. A two-layer numerical-feature network uses movie release year, scaled as (year − 1950)/100, or log(1 + product price) for Amazon; missing values are zero. The checkpoint revision and feature rules are fixed before evaluation.

The item encoder concatenates the graph, text, category, and numerical vectors. A fusion network with two 64-dimensional GELU hidden layers and a 32-dimensional output yields vi. The pretrained language model itself remains frozen. The compact dimension and input length define the computational configuration used in this study.

3.4 Candidate-Aware Scoring
Equation (2): see the linear equation transcriptions below.Equation (3): see the linear equation transcriptions below.
The pooled user is a single query; candidate item vectors supply keys and values to four-head cross-attention with zero attention dropout. The linear product/sum scoring equation and item bias are used exactly as written. Within a slate, the user-side part of the sum term is identical for every candidate and does not affect ranking, so the sum branch acts as an item-only score. Candidate permutation preserves item-associated scores, whereas changing membership may change every score. Complete-catalog evaluation therefore attends over the whole eligible slate rather than scoring arbitrary independent chunks.

For one user, u, uC, vi, w1 and w2 are 32-dimensional vectors, VC is a |C| × 32 matrix, and bi is a scalar item bias that is zero-initialized for every catalog item.

4 Training Objectives and Experimental Design

4.1 Ranking and Auxiliary Losses

A training example is a positive prefix and its next positive target. The candidate set contains the target and 31 distinct uniform negatives. The sampler excludes the full permitted positive prefix and current target, even when the encoder truncates its input. Future training positives and held-out identities are not consulted. The ranking term is cross-entropy over the sampled candidate set:
Equation (4): see the linear equation transcriptions below.
Equation (4) is cross-entropy over the sampled candidate set, with logsumexp denoting the natural logarithm of the sum of exponentials; Equation (5) uses the same convention.

For the auxiliary joint variant, each valid history position is independently masked with probability 0.2 in two views; the first retained event is preserved to avoid empty attention. Position and gap values are not recomputed. Let zbj be the cosine similarity of the first view for user b and the second view for user j, divided by temperature 0.2. The asymmetric contrastive loss is
Equation (5): see the linear equation transcriptions below.
The temporal term is the batch-mean squared Euclidean distance between a sampled prefix representation and that of its immediately preceding prefix. Semantic alignment averages one minus cosine similarity between item vectors in the batch slates and a fixed, seeded Gaussian projection of the normalized frozen text vectors into 32 dimensions.

Both terms act on pooled user or item vectors before candidate cross-attention, and the frozen text target receives no gradient. Each user supplies one example per epoch, so contrastive negatives are the other users in the same batch; no cross-batch memory is used.
Equation (6): see the linear equation transcriptions below.
The auxiliary weights reach full strength at epoch five. Ranking-only variants omit all auxiliary terms. The weights and fixed projection are specified independently of test outcomes.

TABLE I  AUXILIARY OBJECTIVES AND FIXED WEIGHTS

| Term | Alignment target | Weight | Hypothesized purpose |
| --- | --- | --- | --- |
| Contrastive | Two masked histories of the same user | 0.10 | View consistency |
| Temporal | Immediately preceding prefix | 0.05 | Limit abrupt representation changes |
| Semantic | Fixed projected frozen text | 0.20 | Preserve content alignment |

All auxiliary coefficients are scaled by min(epoch/5, 1). The combined package and each individual loss are compared with ranking-only training; the architecture and individual coefficients remain fixed. Linear equation transcriptions and executable loss definitions accompany the supplementary material.

4.2 Datasets and Information Boundaries

MovieLens-20M [4] contributes a fresh sample of 1,000 users with at least ten ratings of four or above. Sampling uses seed 29092026 and explicitly excludes every user in the earlier exploratory pilot. The complete supplied catalog contains 27,278 movies. The independent Amazon 2014 Musical Instruments 5-core release [30] supplies 900 products; after retaining ratings of four or above and users with at least five distinct positive events, 1,092 users remain. Product metadata supplies titles, categories and price; review text, sales rank and co-purchase links are not features.

For each user, positive events are ordered by timestamp and item identifier. The last event is held out for test and the penultimate event for validation. Duplicate Amazon user–item positives retain their earliest event. Validation is appended to the permitted history at test time without parameter updates. Graph edges and popularity counts use training events only. Item catalogs and content are static side information; the split does not impose a single chronological cutoff across users.

TABLE II  INDEPENDENT EVALUATION DATA

| Quantity | MovieLens | Amazon |
| --- | --- | --- |
| Users | 1,000 | 1,092 |
| Catalog items | 27,278 | 900 |
| Positive events | 74,833 | 7,814 |
| Training events | 72,833 | 5,630 |
| Validation / test events | 1,000 / 1,000 | 1,092 / 1,092 |
| Test targets without training positives | 49 | 38 |

Training negatives exclude the permitted prefix and current target. Validation and test negatives exclude the available history and held-out target, which is restored as the sole relevant item. Evaluation uses either every eligible catalog item or a fixed slate of one target and 99 distinct uniform negatives. Sampling seeds 2909777 and 2909778 fix validation and test slates respectively. Internal item indices break score ties consistently for every model.

4.3 Search Budget and Baselines

Models are initialized and trained separately for each dataset. Each learned model and each ablation receives three learning-rate trials (0.001, 0.003 and 0.01), each with 50 epochs on tuning seed 501. The largest sampled-validation NDCG@10 selects the learning rate and checkpoint, with earliest epochs and the first listed rate resolving ties. The selected rate is evaluated on five new training seeds, 101–105; each seed again selects its checkpoint using validation only. Test evaluation begins after all planned model training and selection are complete on both datasets. All implementation and validation choices are finalized before test evaluation. The supplementary protocol records the pre-test baseline fidelity corrections and preserves the preliminary validation trials separately.

The experiments use PyTorch 2.2.2 on a local CPU; the supplement records exact dependency versions. Learned models use 32-dimensional representations and batches of 128 users. Sequential encoders process histories of at most 30 events. E2E-PREF and BPR use AdamW [31] with weight decay 10⁻⁵ and gradient-norm clipping at five. E2E-PREF variants share one precomputed prefix and 31-negative slate per user per epoch. BPR samples one positive from the entire training history and 31 negatives outside all training positives. Thus the E2E-PREF component and continuation comparisons share exact examples, while baseline families use their appropriate training objectives.

The SASRec port [3] uses two causal blocks, one attention head, dropout 0.2, left padding, Xavier initialization and Adam with second-moment coefficient 0.98. Every nonpadding next-item position in each user’s final 30 training transitions is supervised with one uniform negative outside that user’s training positives. This yields 24,764 positive positions per MovieLens epoch and 4,529 per Amazon epoch. Search trials, maximum epochs and per-user batch opportunities are matched; positive-pair counts and computational costs differ. The model is a source-informed PyTorch port, not an exact reproduction of the original TensorFlow benchmark.

BPR-MF uses dot-product user/item embeddings and pairwise negative log-sigmoid loss. Deterministic controls are training-positive popularity and cosine item-neighborhood scoring; neighborhood size is selected from 20, 50 and 100 using validation. Item-neighborhood scoring sums similarities over the complete permitted positive history. Every model is evaluated on the same eligible test items and held-out targets.

TABLE III  VALIDATION SELECTED LEARNING RATES

| Model | MovieLens | Amazon |
| --- | --- | --- |
| Ranking only | 0.003 | 0.01 |
| No graph | 0.01 | 0.01 |
| Mean history | 0.01 | 0.01 |
| No text | 0.01 | 0.01 |
| Independent scorer | 0.003 | 0.01 |
| All auxiliary losses | 0.003 | 0.003 |
| Contrastive only | 0.003 | 0.001 |
| Temporal only | 0.01 | 0.01 |
| Semantic only | 0.003 | 0.01 |
| BPR-MF | 0.01 | 0.003 |
| SASRec port | 0.01 | 0.01 |

4.4 Matched Updating and Component Comparisons

The mechanism comparison shares 25 epochs of scaled dot-product pretraining and then branches from the identical final state into 25 epochs of joint or frozen continuation. The pretraining score divides the user–item dot product by √32 and adds the item bias. Both branches use the ranking-only learning rate selected above, the same continuation examples, a reset optimizer and disabled encoder dropout. The frozen branch updates only cross-attention, score vectors and item biases; the joint branch additionally updates graph, temporal and feature-fusion parameters. Checkpoints are selected separately by validation. Sharing initialization and examples prevents their differences from being mistaken for an updating effect. Separate validation-selected checkpoints form part of the comparison; its scope is continued adaptation after shared pretraining.

Four architectural comparisons remove the entire graph encoder and use base item embeddings, replace temporal encoding by a masked history mean, remove projected text features, or replace candidate attention by the existing user vector while retaining the product/sum scoring head. All other pathways remain unchanged. Additional variants add each auxiliary loss individually or all three together. Every from-scratch variant receives its own identical three-rate search; these comparisons describe the effects of the specified substitutions under validation-based selection. Added auxiliary views consume random numbers and can change later dropout realizations, despite shared initial parameters and example plans.

4.5 Metrics Uncertainty and Verification

Primary evaluation uses complete-catalog NDCG@10. Recall@20, mean reciprocal rank and 100-candidate sampled ranking are complementary outcomes. With one relevant target at rank r, these metrics are respectively 1[r ≤ 10]/log₂(r + 1), 1[r ≤ 20] and 1/r, averaged over users. Absolute learned-model results report means and sample standard deviations across five seeds. Deterministic controls have one value.

Paired differences first subtract per-user outcomes within each seed. A crossed bootstrap resamples the five seeds and the users independently, reusing the same sampled users across selected seeds; 5,000 replicates yield 95% conditional intervals. These intervals characterize the observed seed–user grid. Component and auxiliary comparisons are exploratory, without multiplicity-controlled significance claims. The supplement also reports 97.5% intervals for assessing the two primary domain contrasts more conservatively.

Automated checks verify split exclusions, training-only graph reconstruction, distinct negative candidates, finite outputs, gradient flow, candidate permutation behavior and frozen-parameter immutability. Dense and sparse graph computations agree numerically. Full-catalog attention is checked against direct scoring of an explicit eligible slate, and every saved test rank is independently verified by sorting. The executable supplement contains source hashes, training logs, validation choices and per-user ranks. Trained checkpoints are retained locally; the redistribution package includes their checksums and instructions to regenerate them.

As a post hoc sensitivity analysis, we recompute every planned paired contrast after omitting each of the five training seeds in turn, using unchanged saved test ranks. The resulting minimum and maximum four-seed means describe dependence on a single realized seed. They are descriptive ranges, not confidence intervals, and do not add independent datasets or replace the crossed seed–user intervals.

5 Results

5.1 Joint Preference Adaptation

TABLE IV  MATCHED JOINT MINUS FROZEN NDCG DIFFERENCES

| Dataset | Full catalog difference [95% interval] | Sampled difference [95% interval] |
| --- | --- | --- |
| MovieLens | -0.00021 [-0.00323, +0.00264] | +0.00090 [-0.00480, +0.00678] |
| Amazon | -0.00161 [-0.00419, +0.00118] | -0.00095 [-0.00410, +0.00204] |

The matched experiment starts both branches from identical pretrained representations. Complete-catalog NDCG@10 is 0.01721 for joint continuation and 0.01742 for frozen continuation on MovieLens; the corresponding Amazon values are 0.04762 and 0.04923. Table IV reports the paired differences for both candidate protocols.

The 95% and 97.5% conditional intervals include zero in both domains and under both protocols. Thus the experiment does not establish an incremental ranking advantage from continued representation updating within the prescribed schedule. The intervals permit effects in either direction. This inference concerns additional adaptation after shared end-to-end pretraining; the from-scratch framework is assessed separately against external baselines.

5.2 Baseline Comparisons

TABLE V  COMPLETE CATALOG NDCG AT TEN

| Model | MovieLens | Amazon |
| --- | --- | --- |
| E2E-PREF ranking only | 0.01776 ± 0.00131 | 0.04987 ± 0.00340 |
| E2E-PREF with auxiliaries | 0.02097 ± 0.00290 | 0.04998 ± 0.00231 |
| BPR-MF | 0.02246 ± 0.00261 | 0.02803 ± 0.00225 |
| SASRec port | 0.01645 ± 0.00285 | 0.04495 ± 0.00429 |
| Popularity | 0.01783 | 0.04404 |
| Item neighborhood | 0.03525 | 0.03627 |

On Amazon, the ranking-only framework achieves a mean complete-catalog NDCG@10 of 0.04987, above the means of all four implemented external baselines. Its paired difference from BPR-MF is +0.02185 (95% conditional interval [+0.01271, +0.03108]). The difference from the SASRec port is +0.00493 (95% conditional interval [-0.00234, +0.01180]), indicating a smaller and less precisely resolved comparison. The framework also has higher mean Recall@20 and MRR than these two learned baselines; the supplement reports all secondary metrics.

MovieLens presents a different ordering. Item-neighborhood scoring reaches 0.03525, the highest baseline mean, while BPR-MF reaches 0.02246 and ranking-only E2E-PREF reaches 0.01776. The E2E-PREF minus BPR-MF contrast is -0.00471 (95% conditional interval [-0.00921, -0.00054]). Its difference from the SASRec port is +0.00131 (95% conditional interval [-0.00489, +0.00769]). Together, these results locate the framework’s strongest baseline performance in the product domain.

5.3 Components and Auxiliary Objectives

TABLE VI  COMPONENT EFFECTS ON COMPLETE CATALOG NDCG AT TEN

| Ranking-only model minus ablation | MovieLens difference [95% interval] | Amazon difference [95% interval] |
| --- | --- | --- |
| No graph | -0.00123 [-0.00380, +0.00125] | +0.00169 [-0.00036, +0.00429] |
| Mean history | -0.00110 [-0.00571, +0.00362] | +0.00406 [-0.00237, +0.01085] |
| No text | +0.00061 [-0.00198, +0.00335] | +0.00078 [-0.00072, +0.00257] |
| Independent scorer | +0.00015 [-0.00171, +0.00205] | +0.00049 [-0.00328, +0.00518] |

On Amazon, the full architecture has a higher mean complete-catalog score than each of the four architectural ablations. The graph-encoder and temporal-encoder contrasts are positive in all five seeds, while their crossed seed–user intervals still include zero. On MovieLens, removing the graph encoder or replacing temporal encoding with mean pooling produces higher complete-catalog point estimates than the full model. All eight full-catalog component intervals in Table VI include zero, so these measurements characterize the tested substitutions without establishing that every component is necessary.

Sampled MovieLens ranking gives a more differentiated picture: retaining the graph encoder produces +0.00772 (95% conditional interval [+0.00101, +0.01441]), whereas replacing temporal encoding with a history mean raises the mean score from 0.55091 to 0.60499. Candidate-independent scoring remains close to the full model in both domains, with intervals spanning zero. The effects of the substitutions therefore depend on the domain and candidate protocol.

TABLE VII  AUXILIARY EFFECTS ON COMPLETE CATALOG NDCG AT TEN

| Auxiliary variant minus ranking only | MovieLens difference [95% interval] | Amazon difference [95% interval] |
| --- | --- | --- |
| All auxiliary losses | +0.00322 [-0.00112, +0.00783] | +0.00011 [-0.00330, +0.00362] |
| Contrastive only | +0.00233 [-0.00261, +0.00743] | +0.00009 [-0.00302, +0.00364] |
| Temporal only | -0.00019 [-0.00326, +0.00251] | +0.00073 [-0.00020, +0.00181] |
| Semantic only | -0.00002 [-0.00195, +0.00154] | -0.00235 [-0.00826, +0.00401] |

On MovieLens, the combined auxiliary objective increases mean complete-catalog NDCG@10 from 0.01776 to 0.02097, and contrastive-only training reaches 0.02009. Both contrasts are positive in all five seeds, although their 95% conditional intervals include zero (Table VII). Under sampled ranking, the combined package produces +0.05377 (95% conditional interval [+0.03473, +0.07096]), and contrastive-only training produces +0.05208 (95% conditional interval [+0.04093, +0.06339]). Both sampled improvements occur in all five seeds.

The contrastive-only sampled mean (0.60299) is close to that of the combined package (0.60468), identifying history-view regularization as a useful tested component in this setting. Temporal-only and semantic-only objectives do not produce comparably resolved MovieLens gains. On Amazon, every auxiliary contrast has a 95% interval that includes zero; temporal-only training has the highest observed mean (0.05060), but its difference from ranking-only learning remains uncertain. These exploratory results support a domain-specific interpretation of auxiliary learning.

5.4 Sensitivity to Individual Training Seeds

Omitting any one seed preserves the positive Amazon complete-catalog mean difference against BPR-MF (+0.02107 to +0.02337) and SASRec (+0.00434 to +0.00591). The latter remains uncertain under the original crossed bootstrap. The sampled MovieLens contrastive-only advantage also remains positive (+0.05109 to +0.05387). In contrast, the MovieLens matched sampled difference ranges from -0.00024 to +0.00194, crossing zero. This sensitivity check reinforces the distinction between the stable directional findings and the unresolved continuation effect. All 44 planned paired contrasts are included in the supplementary sensitivity table.

6 Discussion

E2E-PREF connects preference modeling and recommendation through shared graph representations, a temporal user pathway, content fusion and slate-conditioned ranking. Its strongest baseline result is in Amazon Musical Instruments, where ranking-only learning has higher mean complete-catalog NDCG@10 than the four evaluated external baselines and a positive conditional interval against BPR-MF. The matched study addresses a separate question: additional encoder adaptation after shared end-to-end pretraining. Its unresolved contrasts support reporting the framework’s observed ranking quality separately from the incremental effect of further joint updating.

The component study supplies a practical finding about the tested design: adding representation pathways does not uniformly improve ranking, and auxiliary objectives depend on the evaluation setting. Contrastive history-view learning improves sampled MovieLens ranking, while mean-history aggregation is also effective there. Amazon graph and temporal contrasts are directionally favorable across seeds, with substantial user-level uncertainty. These observations support evaluating each preference pathway in its application context. They do not identify catalog size or graph density as causal explanations, because those factors were not independently manipulated. Every planned variant is reported; test outcomes do not select a new primary model.

Candidate protocol is part of the model’s operating context. In this framework, changing the candidate slate changes both the competing items and the cross-attention input. Sampled gains and complete-catalog gains are therefore separate empirical findings, consistent with prior work on sampled-metric interpretation [28]. Reporting both protocols, together with a candidate-independent scorer, makes these distinctions explicit. The full-catalog results remain primary, and the implementation, fixed selections and saved prediction ranks provide a reproducible basis for examining each comparison.

7 Limitations and Future Work

The study covers one newly sampled MovieLens cohort and one small Amazon product domain. Static catalogs, per-user splitting, positivity filtering and the Amazon release’s pre-existing 5-core selection define its scope. Timestamp ties use deterministic identifier ordering and affect 555 MovieLens users and 935 Amazon users; the Amazon sequence order therefore does not always recover within-day event order. Conditional intervals do not include uncertainty from dataset selection, filtering or architecture search. SASRec is a PyTorch port, and matching trial and epoch budgets does not equalize positive-pair counts or computation. Dimensions, history length, graph rules and auxiliary coefficients are fixed; a selected boundary checkpoint does not establish convergence. The external baseline set does not include a direct implementation of every related integrated or slate-aware model. Broader model-specific tuning and additional independent cohorts are needed to assess generality beyond the present configurations.

Future work will evaluate larger product categories, additional datasets and multiple held-out targets under historically available feature constraints. Broader model-specific tuning and official end-to-end baseline pipelines can extend the comparison. Online utility, candidate generation and serving latency are further directions. Tests across additional graph priors, representation dimensions and history lengths can investigate the conditions under which each preference pathway is useful.



## References

[1] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, “BPR: Bayesian personalized ranking from implicit feedback,” in Proc. UAI, 2009, pp. 452–461. [Online]. Available: https://arxiv.org/abs/1205.2618

[2] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural collaborative filtering,” in Proc. WWW, 2017, pp. 173–182. doi: 10.1145/3038912.3052569.

[3] W.-C. Kang and J. McAuley, “Self-attentive sequential recommendation,” in Proc. IEEE ICDM, 2018, pp. 197–206. doi: 10.1109/icdm.2018.00035.

[4] F. M. Harper and J. A. Konstan, “The MovieLens datasets: History and context,” ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, Article 19, 2015. doi: 10.1145/2827872.

[5] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, “Session-based recommendations with recurrent neural networks,” in Proc. ICLR, 2016. https://arxiv.org/abs/1511.06939

[6] A. Vaswani et al., “Attention is all you need,” in Advances in Neural Information Processing Systems, vol. 30, 2017. [Online]. Available: https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[7] F. Sun et al., “BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer,” in Proc. CIKM, 2019, pp. 1441–1450. doi: 10.1145/3357384.3357895.

[8] P. Veličković et al., “Graph attention networks,” in Proc. ICLR, 2018. https://arxiv.org/abs/1710.10903

[9] X. He, K. Deng, X. Wang, Y. Li, Y. Zhang, and M. Wang, “LightGCN: Simplifying and powering graph convolution network for recommendation,” in Proc. SIGIR, 2020, pp. 639–648. doi: 10.1145/3397271.3401063.

[10] X. Wang, X. He, Y. Cao, M. Liu, and T.-S. Chua, “KGAT: Knowledge graph attention network for recommendation,” in Proc. KDD, 2019, pp. 950–958. doi: 10.1145/3292500.3330989.

[11] G. Lee, K. Kim, and K. Shin, “Revisiting LightGCN: Unexpected inflexibility, inconsistency, and a remedy towards improved recommendation,” in Proc. RecSys, 2024, pp. 957–962. doi: 10.1145/3640457.3688176.

[12] Z. Fan, Z. Liu, J. Zhang, Y. Xiong, L. Zheng, and P. S. Yu, “Continuous-time sequential recommendation with temporal graph collaborative transformer,” in Proc. CIKM, 2021. doi: 10.1145/3459637.3482242.

[13] G. Zhou et al., “Deep interest network for click-through rate prediction,” in Proc. KDD, 2018, pp. 1059–1068. doi: 10.1145/3219819.3219823.

[14] C. Pei et al., “Personalized re-ranking for recommendation,” in Proc. RecSys, 2019, pp. 3–11. doi: 10.1145/3298689.3347000.

[15] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proc. EMNLP-IJCNLP, 2019, pp. 3982–3992. doi: 10.18653/v1/D19-1410.

[16] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 5776–5788. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

[17] Sentence Transformers, “all-MiniLM-L6-v2,” model card and model files, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. Accessed: Sep. 28, 2026. [Online]. Available: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41

[18] Y. Hou, S. Mu, W. X. Zhao, Y. Li, B. Ding, and J.-R. Wen, “Towards universal sequence representation learning for recommender systems,” in Proc. KDD, 2022, pp. 585–593. doi: 10.1145/3534678.3539381.

[19] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proc. ICML, PMLR, vol. 119, 2020, pp. 1597–1607. [Online]. Available: https://proceedings.mlr.press/v119/chen20j.html

[20] J. Wu et al., “Self-supervised graph learning for recommendation,” in Proc. SIGIR, 2021, pp. 726–735. doi: 10.1145/3404835.3462862.

[21] K. Zhou et al., “S³-Rec: Self-supervised learning for sequential recommendation with mutual information maximization,” in Proc. CIKM, 2020, pp. 1893–1902. doi: 10.1145/3340531.3411954.

[22] X. Xie et al., “Contrastive learning for sequential recommendation,” in Proc. IEEE ICDE, 2022, pp. 1259–1273. doi: 10.1109/ICDE53745.2022.00099.

[23] D. Zhang, J. Qin, J. Ma, Z. Yang, D. Cui, and P. Ji, “Item attributes fusion based on contrastive learning for sequential recommendation,” Multimedia Systems, vol. 30, Art. 291, 2024. doi: 10.1007/s00530-024-01486-7.

[24] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, “Item-based collaborative filtering recommendation algorithms,” in Proc. WWW, 2001, pp. 285–295. doi: 10.1145/371920.372071.

[25] M. Ferrari Dacrema, P. Cremonesi, and D. Jannach, “Are we really making much progress? A worrying analysis of recent neural recommendation approaches,” in Proc. RecSys, 2019, pp. 101–109. doi: 10.1145/3298689.3347058.

[26] S. Rendle, W. Krichene, L. Zhang, and J. Anderson, “Neural collaborative filtering vs. matrix factorization revisited,” in Proc. RecSys, 2020, pp. 240–248. doi: 10.1145/3383313.3412488.

[27] A. Klenitskiy and A. Vasilev, “Turning dross into gold loss: Is BERT4Rec really better than SASRec?” in Proc. RecSys, 2023, pp. 1120–1125. doi: 10.1145/3604915.3610644.

[28] W. Krichene and S. Rendle, “On sampled metrics for item recommendation,” in Proc. KDD, 2020, pp. 1748–1757. doi: 10.1145/3394486.3403226.

[29] B. L. Pereira, A. Said, and R. L. T. Santos, “On the reliability of sampling strategies in offline recommender evaluation,” in Proc. RecSys, 2025, pp. 360–369. doi: 10.1145/3705328.3748086.

[30] J. McAuley, C. Targett, Q. Shi, and A. van den Hengel, “Image-based recommendations on styles and substitutes,” in Proc. SIGIR, 2015, pp. 43–52. doi: 10.1145/2766462.2767755.

[31] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. ICLR, 2019. https://arxiv.org/abs/1711.05101

## Linear equation transcriptions
Accessible linear equivalents of manuscript Equations (1)–(6) and evaluation definitions

1. a_k = softmax over k of [w^T tanh(W_p h_k + b_p)]; u = sum_k a_k h_k.
2. u_C = MultiHeadAttention(u, V_C, V_C).
3. s(u,i | C) = w_1^T (u_C elementwise-product v_i) + w_2^T (u_C + v_i) + b_i.
4. L_rank = logsumexp_{j in C} s(u,j | C) - s(u,i_positive | C).
   logsumexp_j x_j means log(sum_j exp(x_j)); this is the same original sampled-softmax loss.
5. L_con = (1/B) sum_{b=1}^B [logsumexp_{j=1,...,B} z_bj - z_bb].
   This equals the original negative mean log-softmax on the paired view.
6. L_e = L_rank + min(e/5,1) * (0.1 L_con + 0.05 L_tem + 0.2 L_sem).
Evaluation: NDCG@10 for user u = indicator(r_u <= 10) / log2(r_u + 1).
Evaluation: Recall@20 for user u = indicator(r_u <= 20); MRR for user u = 1/r_u.

Paired contrast: Delta = [1/(S*U)] sum_{s=1}^S sum_{u=1}^U [m_joint(s,u) - m_frozen(s,u)].
   S=5 seeds; U=1000 fixed MovieLens user indices or1092 Amazon user indices; one held-out target per user.

B is batch size, e is epoch, r_u is target rank. Ranking loss is averaged over the batch in Equation 6. z_bj is cosine similarity of two augmented history representations divided by temperature 0.2. Temporal loss compares pre-attention pooled prefix vectors; semantic loss averages one minus cosine to the fixed projected-text prior over unique slate items. u, u_C, v_i, w_1 and w_2 have 32 entries; V_C has |C| rows and 32 columns. Native editable Word equations are used in the manuscript. Plain-text PDF extraction may flatten fractions and re-order summation limits, so this file is the unambiguous linear representation.
