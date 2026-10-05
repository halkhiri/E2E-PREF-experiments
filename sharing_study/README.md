# E2E-PREF representation sharing study

This is a post hoc extension of the existing evaluation. It includes all 32 new fitting records, five-seed test ranks for each new model/dataset, and the reused original full model records. No test-driven variant selection was performed. Read PROTOCOL.md and RESULTS.md together.

## Recompute reported results

With NumPy installed, from this extracted directory:

    python source/analyze_sharing.py --root . --original original_reference --output recomputed
    python source/plot_sharing.py --output recomputed

The first command verifies sampled ranks independently from stored scores, validates checkpoint and LR selections, checks paired user order, and recomputes all metrics and crossed bootstrap intervals. Full-catalog ranks were independently checked against sorted full score vectors during model evaluation; source/evaluation.py contains that check. The compact archive does not retain every full-catalog score.

## Repeat model fitting

Use the pinned data preparation recipe in the previous E2E_PREF_independent_evaluation_reproduction.zip archive to reconstruct the original data.pt files. Place the rebuilt data at a sibling directory named strengthening/data/{movielens,amazon}/data.pt, as referenced in source/sharing.py. Use a fresh copy of this directory without runs or completion markers, keeping source and PROTOCOL.md. Install the versions in environment.json in an isolated environment, run source/checks.py and then source/sharing.py. The runner uses four worker processes, completes both datasets before test evaluation, and resumes from done.json records. Existing output records must not be mistaken for a fresh retraining. Reuse or retrain the original full reference separately according to its original protocol.

Prepared data, pretrained language-model weights, raw datasets and trained checkpoints are not redistributed here. The original datasets/models remain subject to their providers' terms. Trained checkpoints remain in the local work/sharing_study/runs directory; hashes for every checkpoint are retained. Training used fixed cached text features, not downloaded or newly fine-tuned language models.

The two new variants have identical initial parameter tensors and parameter counts. The shared-average control isolates a routing/averaging choice at equal allocated capacity, not parameter tying alone. Original-versus-separate also changes capacity. Runtime is wall time on a shared local CPU and is not a serving-latency benchmark.
