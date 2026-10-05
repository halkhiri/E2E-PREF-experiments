# E2E-PREF reference validation

Checked on 28 September 2026. Reference numbers refer to the findings-focused manuscript and the accompanying references-checked copy.

## Outcome

All 28 entries were located as real sources: 27 research publications and one official model resource. No fabricated reference or citation-topic mismatch was identified in this audit. Ten entries were enhanced with verified identifiers, publication page ranges or more precise source links. No reference was removed.

All 28 references are cited in the manuscript, their first appearances follow numerical order, and no duplicate entry was found. Section III and all other main-text sections, numerical results, tables, equations and figures are unchanged.

## What this check establishes

The check covers source identity, available bibliographic metadata and whether the limited statement attached to each citation is supported by the source abstract, relevant text or official documentation. Publisher records, original author manuscripts, official proceedings, author repositories and institutional publication records were used. Several ACM/IEEE pages blocked automated retrieval; author copies and institutional records supplied corroboration. A blocked publisher page was not treated as an invalid DOI. This is not independent replication of the cited studies or a comprehensive retraction-status audit. Some retained conference entries use abbreviated venue names and omit proceedings page ranges; that is a formatting/completeness issue rather than evidence of an invalid source.

## Entry-by-entry record

| Ref. | Source | Validation and citation relevance | Update |
|---|---|---|---|
| [1] | [BPR](https://arxiv.org/abs/1205.2618) | Pairwise ranking from implicit feedback. UAI 2009 and pp. 452–461 are explicitly identified despite the later 2012 arXiv upload. | Added original-manuscript link. |
| [2] | [Neural collaborative filtering](https://hexiangnan.github.io/papers/www17-ncf.pdf) | Nonlinear user–item interaction learning; authors, WWW 2017 and DOI match the original paper. | Retained. |
| [3] | [SASRec](https://arxiv.org/abs/1808.09781) | Self-attention for sequential recommendation; ICDM 2018 acceptance is stated. The simplified manuscript baseline is not described as an official reproduction. | Retained. |
| [4] | [MovieLens](https://files.grouplens.org/datasets/movielens/ml-32m-README.html) | GroupLens gives the cited 2015 paper, authors, volume 5, issue 4, article 19 and DOI as its standard dataset citation. MovieLens-20M availability was separately checked on the GroupLens dataset page. | Retained. |
| [5] | [GRU4Rec](https://arxiv.org/html/1511.06939v4) | Recurrent session recommendation; original camera-ready paper for ICLR 2016. The 2015 preprint date is not a publication-year error. | Retained. |
| [6] | [Transformer](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | Attention architecture; official NIPS 2017 proceedings, volume 30. | Replaced preprint URL with proceedings record. |
| [7] | [BERT4Rec](https://github.com/FeiSun/BERT4Rec) | Author repository identifies bidirectional masked-item modeling and supplies CIKM 2019, pp. 1441–1450 and the matching DOI. | Retained. |
| [8] | [Graph attention networks](https://arxiv.org/abs/1710.10903) | Learned attention over graph neighbors; ICLR 2018 explicitly identified. | Retained. |
| [9] | [LightGCN](https://arxiv.org/html/2002.02126v4) | User–item graph propagation; original paper identifies SIGIR 2020 and DOI 10.1145/3397271.3401063. | Added publication DOI in place of preprint URL. |
| [10] | [KGAT](https://arxiv.org/html/1905.07854v2) | Attention over higher-order knowledge-graph connections; original paper identifies KDD 2019 and matching publication DOI. | Added publication DOI in place of preprint URL. |
| [11] | [LightGCN++](https://pure.kaist.ac.kr/en/publications/revisiting-lightgcn-unexpected-inflexibility-inconsistency-and-a-/) | Author institution confirms authors, RecSys 2024, pp. 957–962 and DOI. The stated scaling, neighbor-weighting and layer-pooling discussion matches the abstract. | Retained. |
| [12] | [Deep Interest Network](https://arxiv.org/html/1706.06978v4) | Candidate-ad-specific interest representation is supported by the local activation unit description. It is appropriately distinguished from E2E-PREF slate-wide conditioning. | Retained. |
| [13] | [Personalized re-ranking](https://www.researchwithrutgers.org/en/publications/personalized-re-ranking-for-recommendation/) | Author institution confirms RecSys 2019, pp. 3–11 and DOI. Whole-list transformer interactions support the manuscript’s list-context statement. | Retained. |
| [14] | [Sentence-BERT](https://aclanthology.org/D19-1410/) | Official ACL record confirms authors, title, EMNLP-IJCNLP 2019, pp. 3982–3992 and DOI; reusable sentence embeddings are supported. | Retained. |
| [15] | [MiniLM](https://proceedings.neurips.cc/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) | Official NeurIPS 2020 record confirms authors and volume 33. Supports attention-distillation background, not by itself the exact sentence-embedding checkpoint. | Replaced preprint URL with proceedings record. |
| [16] | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md) | Exact revision exists. The official model card supports the 384-dimensional representation. This is model documentation, not a peer-reviewed research paper. | Pinned URL to exact revision; added resource type and access date. |
| [17] | [UniSRec](https://arxiv.org/html/2206.05941v1) | Original KDD 2022 paper includes DOI 10.1145/3534678.3539381 and supports transferable representations derived from item descriptions. | Retained. |
| [18] | [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) | PMLR confirms authors, ICML 2020, volume 119 and pp. 1597–1607. Supports general contrastive learning, rather than a recommendation-specific experimental claim. | Added proceedings link. |
| [19] | [SGL](https://arxiv.org/abs/2010.10783) | Author manuscript confirms SIGIR 2021 and the DOI; augmented graph views support the related-work statement. | Retained. |
| [20] | [S3-Rec](https://arxiv.org/abs/2008.07873) | Author manuscript confirms CIKM 2020 and DOI. Its auxiliary objectives connect attributes, items, subsequences and sequences. | Retained. |
| [21] | [CL4SRec](https://ofey.me/) | Author bibliography confirms ICDE 2022, pp. 1259–1273 and DOI. The original manuscript at https://arxiv.org/abs/2010.14395 supports augmented sequence contrastive learning. | Retained. |
| [22] | [Item-based collaborative filtering](https://experts.umn.edu/en/publications/item-based-collaborative-filtering-recommendation-algorithms/) | Author institution confirms WWW 2001, pp. 285–295 and DOI. Supports item-neighborhood recommendation. | Added publication DOI. |
| [23] | [Neural recommendation reproducibility](https://arxiv.org/abs/1907.06902) | Original paper confirms authors, RecSys 2019 and DOI; reproducibility and baseline-comparison concerns match its stated findings. | Retained. |
| [24] | [NCF versus matrix factorization](https://research.google/pubs/neural-collaborative-filtering-vs-matrix-factorization-revisited/) | Authors’ institutional record confirms RecSys 2020. Supports the importance of optimized matrix-factorization references. | Retained. |
| [25] | [Turning Dross Into Gold Loss](https://arxiv.org/abs/2309.07602) | Original paper verifies authors and DOI; authors’ repository https://github.com/antklen/sasrec-bert4rec-recsys23 confirms RecSys 2023 and pp. 1120–1125. Supports the claim that loss choice can alter SASRec/BERT4Rec comparisons. | Added DOI and pp. 1120–1125. |
| [26] | [Sampled recommendation metrics](https://research.google/pubs/on-sampled-metrics-for-item-recommendation/) | Authors’ institutional record confirms KDD 2020 and supports the statement that sampled and complete rankings need not preserve model comparisons. | Retained. |
| [27] | [Sampling reliability](https://doi.org/10.1145/3705328.3748086) | ACM confirms all three authors, RecSys 2025, pp. 360–369 and DOI. Supports the interaction of sampling and exposure bias in offline evaluation. | Added pp. 360–369. |
| [28] | [AdamW](https://arxiv.org/abs/1711.05101) | Original manuscript explicitly identifies ICLR 2019; decoupled weight decay supports the optimizer citation. The 2017 preprint identifier is not an incorrect publication year. | Retained. |

## Files

- Revised manuscript: E2E_PREF_references_checked_manuscript.docx
- Source manuscript: E2E_PREF_findings_focused_manuscript.docx

The scientific contribution and findings were not revised during this reference check.
