# E2E-PREF experiment code and results

Reproducibility materials for **End-to-End Learning for Joint Preference Modeling and Recommendation: A Unified Framework with Differentiable Preference-Aware Optimization**.

## Contents

- `strengthening/`: original independent evaluation, baselines, component substitutions, auxiliary objectives, and matched continuation study.
- `sharing_study/`: subsequent representation-sharing experiment, including separate graph pathways and an equal-parameter shared-average control.

Each study includes its protocol, source code, validation records, selected settings, saved test ranks, reported results, and an original SHA-256 manifest. These are preserved reproduction packages. Historical manuscript-generation helpers are retained for provenance and may depend on the original workspace; they are not experiment entry points.

## Findings and scope

The ranking-only framework has the highest observed full-catalog mean among the four external baselines in the studied Amazon Musical Instruments domain. The paired interval against BPR-MF is positive; the SASRec comparison is uncertain. On MovieLens, item-neighborhood scoring has the highest full-catalog mean. Contrastive learning improves sampled MovieLens ranking, without an established full-catalog benefit.

The original shared design uses fewer parameters than separate graph encoders and has slightly higher observed full-catalog means in both domains. All sharing intervals include zero: the study establishes neither ranking superiority nor equivalence. The sharing study is a post hoc extension on previously evaluated datasets. The equal-parameter control changes both routing and averaging, not parameter tying alone.

## Restore the complete experimental records

Clone or download this entire repository, including all six `reproduction.zip.part*` files. From its root run:

```sh
python restore_records.py
```

The script checks every archive part, reconstructs the archive, restores both studies, and verifies their original file manifests. It refuses to overwrite locally modified files. Source, protocols and summary results are also browsable directly in GitHub. Archive parts contain the original saved ranks and logs; they are not separate ZIP files.

## Recompute results from saved ranks

Install the dependencies documented by each study in an isolated environment. The sharing analysis requires NumPy; the original analysis also imports the training libraries. The recorded environment is in `strengthening/requirements.txt` and `sharing_study/environment.json`. Plotting requires Matplotlib (the original charts used 3.9.4).

From the repository root:

```sh
python strengthening/source/analyze.py
python sharing_study/source/analyze_sharing.py --root sharing_study --original sharing_study/original_reference --output recomputed/sharing
python sharing_study/source/plot_sharing.py --output recomputed/sharing
```

The original analysis writes to `outputs/strengthened_study`; sharing analysis writes to `recomputed/sharing`. These commands analyze saved predictions; they do not retrain models.

## Repeat model fitting

Follow `strengthening/README.md`, `strengthening/FINAL_PROTOCOL.md`, and `strengthening/SOURCE_PROVENANCE.md` for raw-data sources, the pinned text-model revision, preparation, and fitting commands. Follow `sharing_study/README.md` and `sharing_study/PROTOCOL.md` for the extension. The sibling directory layout in this repository matches the sharing runner's expected data path.

For fresh training, use a separate working copy containing the source and protocols but no archived runs or completion markers. Rebuild the data from the cited providers. Existing completion records make the runners resume or skip completed work; they must not be interpreted as new fitting.

Raw datasets, prepared feature caches, pretrained model weights, and trained checkpoints are not distributed here. Checkpoint hashes are retained. See the study protocols for static-catalog assumptions, timestamp handling, selection rules, and conditional uncertainty.

## Attribution and citation

The supplied third-party SASRec license and source attribution are retained in `strengthening/`. Dataset and pretrained-model terms remain those of their providers. No new blanket software license is assigned in this preparation step.

Repository: https://github.com/halkhiri/E2E-PREF-experiments

Suggested reference: halkhiri, “E2E-PREF experiments: Code and reproducibility materials,” GitHub repository, 2026. https://github.com/halkhiri/E2E-PREF-experiments . Use the exact commit identifier when citing a fixed snapshot.
