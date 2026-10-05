# Shared representation study

Locked 2026-10-04 before this study's training or test evaluation. This is a post hoc extension motivated by the architectural claim, not independent preregistration or a new holdout. Prior results on these datasets are already known.

Question: Does routing a common graph representation into both the history encoder and item fusion improve ranking relative to independently learned graph pathways?

Reference: existing ranking-only full model and its saved five-seed predictions/checkpoints, unchanged. Each new model retains dimension 32, graph topology, history encoder, fusion, candidate attention, ranking objective, data, slates and optimization protocol.

Separate: two independently initialized graph banks, each consisting of a complete item embedding matrix plus two GAT layers. Bank A feeds the history encoder; bank B feeds item fusion. All remaining parameters are initialized identically to the original model for the same seed. Both banks receive ranking gradients through their respective paths. This model has more parameters than the original; report that difference explicitly.

Shared dual-bank capacity control: exactly the same two independently initialized banks and all other parameters as Separate. Compute their arithmetic mean and feed that same vector into both pathways. This has precisely the same parameter count and graph-bank compute as Separate. It tests shared routing/averaging at fixed parameter budget, not parameter tying alone. It is a diagnostic control, not a newly selected replacement framework. All three models are reported regardless of performance.

Datasets: existing MovieLens cohort and Amazon Musical Instruments, frozen data.pt and candidate sets. No data re-filtering or new features. New variants: separate and shared_dual. Each receives LR .001/.003/.01, seed501, 50 epochs, sampled validation NDCG@10 selection. Five confirmation seeds101–105,50 epochs with independently selected validation checkpoints. Same per-seed schedules and dropout random streams; second-bank initialization uses a separate RNG stream seed+40000. All tuning and confirmation on BOTH datasets must complete before evaluating any new test results. No test-driven retries or architecture changes.

Primary contrasts: original shared minus Separate; Shared dual-bank minus Separate. Report both datasets, full-catalog NDCG@10 as primary, sampled NDCG@10 secondary, Recall@20/MRR descriptive. Paired seed/user crossed bootstrap5000 draws,95% conditional intervals; exploratory, no multiplicity-corrected significance claim. Exact same user order required. Record parameters, times, chosen LR/epochs, saved ranks, independent rank-sort checks and source/data/checkpoint hashes. Full-catalog attention covers each complete eligible slate.

Code checks before training: both new variants have equal parameter counts and identical initial tensors; graph-only history/fusion gradients reach the intended banks; sparse graph computation matches dense computation; optimized full-catalog scorer matches direct complete-slate scoring. Existing original code/data stay unchanged.
