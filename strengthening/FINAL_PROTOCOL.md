# Final experimental protocol

The final study evaluates E2E-PREF without changing its original integrated preference-learning idea. This file consolidates the initial local protocol and pre-test baseline fidelity corrections. PROTOCOL.md and PROTOCOL_AMENDMENT_BASELINES.md preserve that history. No external preregistration is claimed.

## Data

MovieLens-20M: 1,000 randomly sampled users with at least ten ratings ≥4, explicitly excluding every user in the previous pilot, with sampling seed 29092026. Full 27,278-item supplied metadata catalog. 74,833 positive events, 72,833 training events.

Amazon 2014 Musical Instruments 5-core: all 900 released products and 1,092 users with at least five distinct ratings ≥4. Earliest positive event retained for duplicate user/item pairs. 7,814 positive events, 5,630 training events. Product title, category and log1p(price) supply features; no review text, sales ranks or co-purchase links are features.

Both: chronological user sequences, ties by item ID, last event test, penultimate validation. Test histories include validation without parameter updates. Graph and popularity use training only. Static catalogs and side information do not enforce global historical availability. Last 30 input events; representations 32-dimensional. Frozen MiniLM checkpoint 1110a243fdf4706b3f48f1d95db1a4f5529b4d41.

## Models and selection

E2E-PREF ranking only; no graph encoder (base item embeddings replace both graph-attention layers); mean history instead of temporal encoder; no text; candidate-independent product/sum scoring; all auxiliaries; contrastive only; temporal only; semantic only; BPR-MF; SASRec PyTorch port. Popularity and item neighborhood are deterministic controls.

All11 learned configurations receive three rates (.001,.003,.01), 50 epochs and tuning seed 501. Each chooses the rate with maximum sampled-validation NDCG@10 over epochs. Earliest epoch and first listed rate resolve ties. The chosen rate is repeated on seeds 101–105, with per-seed validation checkpoint selection. Every planned configuration is retained regardless of test outcome.

E2E-PREF variants: one common precomputed positive prefix and 31 distinct negatives per user/epoch. Negatives exclude permitted prefix and current target. AdamW, weight decay 1e-5, clipping 5, batches 128. Auxiliary weights .1/.05/.2 with five-epoch warm-up. Added views can change subsequent dropout random-number consumption.

BPR: one sampled positive from all of each user's training positives, 31 negatives excluding that entire training-positive set. Pairwise log-sigmoid objective, same AdamW settings and per-user update count.

SASRec: original all-position logistic training structure, one negative per nonpadding position, excluding all training positives. Last 30 next-item training transitions, left padding, two blocks, one head, dropout 0.2, Xavier initialization and Adam beta2 = 0.98. Each epoch has 24,764 MovieLens or 4,529 Amazon positive positions. The original TensorFlow model is ported to PyTorch; official benchmark replication is not claimed. Equal rate-trial/epoch/per-user batch budgets do not mean equal positive-pair counts or FLOPs.

Item neighborhood: cosine training-only item similarities; k∈{20, 50, 100} selected by sampled validation. All available positive history is used at inference. Popularity counts training positives only.

Matched learning mechanism: 25 epochs shared dot-product pretraining, final state cloned into joint/frozen 25-epoch ranking-only continuations. Both use the selected ranking-only rate and identical examples, disabled encoder dropout and reset optimizer. Only joint continuation updates encoders; frozen branch updates scorer parameters. Five seeds, separate validation checkpoint selection. No continuation-specific learning-rate search.

Final fitting count: 103 runs per dataset (33 tuning,55 selected-model repeats,5 pretraining and 10 continuations), 206 total. Six earlier MovieLens validation-only baseline runs are archived separately after fidelity correction and never evaluated on test.

## Evaluation and audit

All training and selection on both datasets must finish before test evaluation. Primary complete eligible-catalog NDCG@10; secondary Recall@20, MRR and 100-candidate sampled ranking. Fixed candidate seeds 2909777 validation/2909778 test. History items excluded; target retained; numeric ID tie breaking. Full-catalog candidate attention attends over the whole eligible slate, with cached key/value projections verified against direct explicit-slate scoring.

Five-seed means and sample SDs. Paired within-seed/user differences, 5,000 crossed seed/user bootstrap replicates, 95% conditional intervals. Also report 97.5% intervals for the two primary domain comparisons. Component and auxiliary comparisons are exploratory; no multiplicity-controlled significance claim.

Save checkpoints, validation histories, source hashes, candidate and user IDs, sampled scores and full-catalog per-user ranks. Independent sorting verifies every test rank. CPU wall time is recorded under concurrent execution and is not a serving-latency benchmark. Dense/sparse training equivalence and final baseline causal/sampling behavior are checked. The first three Amazon ranking-only tuning trials retain their exact dense implementation; the rest use the equivalent sparse version.
