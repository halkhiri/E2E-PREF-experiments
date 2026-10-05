# E2E PREF independent evaluation reproduction

This release contains the code, protocol, all training/validation logs, selected settings, sampled score archives and independently verified full-catalog per-user ranks. Raw datasets and pretrained model weights are downloaded separately. Final trained checkpoints and prepared feature arrays remain in the local work/strengthening directory; they are not embedded in this compact redistribution bundle.

## Recompute results without retraining

From the original workspace run `python source/analyze.py` after extracting so that source, data and runs are sibling directories. Edit the output path at the top of analyze.py if desired; by default it writes outputs/strengthened_study relative to the current directory. It recomputes tables and paired intervals from saved ranks. The model evaluation script independently checks each rank against a complete score sort during execution.

## Rebuild from raw data

Install requirements in an isolated Python3.12 environment. Download MovieLens20M and Amazon Musical Instruments2014 5-core/metadata from SOURCE_PROVENANCE.md. Download the exact MiniLM checkpoint revision recorded there, not an unpinned current model. Run:

```sh
python source/prepare_portable.py --ml20m /path/ml-20m.zip --amazon-reviews /path/reviews_Musical_Instruments_5.json.gz --amazon-metadata /path/meta_Musical_Instruments.json.gz --model-snapshot /path/pinned_snapshot --output .
python source/checks.py
python source/baseline_checks.py
python source/evaluation_checks.py
python source/run_parallel.py
```

Use run_parallel.py (or its run_all.py wrapper) as the experiment entry point. Historical training helpers inside study.py and study_dense_original.py are retained for source provenance; they are not the final baseline dispatcher. The runner trains both datasets, then evaluates only after both complete. Existing done.json files mark completed runs; for a fresh retraining, use a fresh directory containing source and the rebuilt data, without the archived runs directory. The training script checks every epoch on validation and never evaluates test. Source-informed SASRec adaptations, static-catalog assumptions and conditional uncertainty are documented in the protocol and provenance files. Start with FINAL_PROTOCOL.md. PROTOCOL.md and PROTOCOL_AMENDMENT_BASELINES.md preserve the initial plan and pre-test implementation corrections. The final BPR and SASRec implementations restore their native data exposure/objectives before any test access. The raw-data preparation recipe was checked by reconstructing both datasets and verifying exact equality of user/item IDs, histories, timestamps, candidate slates, graph arrays and structured features. Fixed text vectors were reused for that reconstruction check; the portable preparation script includes the pinned-model embedding recipe. Runtime measurements are wall time on a shared local CPU, not controlled serving-latency benchmarks. Parameter fields count allocated parameters and requires-grad flags; unused ablation modules remain allocated. Matched-run timings include separate pretraining and continuation fields; shared pretraining is executed once per seed, not independently for both branches.

Do not infer acceptance or general model superiority from artifact completeness. Consult RESULTS.md and all comparison tables for the actual measured outcomes.
