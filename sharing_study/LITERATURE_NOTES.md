# Architectural positioning checked 2026-10-04

The follow-up evaluates a design choice, not a claim that representation sharing or end-to-end recommendation was invented here.

- SASRec, Section III-D and IV-C, explicitly shares item embeddings between the sequence input and prediction layer. Source: https://arxiv.org/html/1808.09781v1 . Existing manuscript reference retained and discussion clarified.
- TGSRec combines temporal collaborative signals and sequential patterns on a continuous-time bipartite graph. E2E-PREF instead uses a static item co-occurrence graph feeding history and item-content pathways. Source: https://arxiv.org/html/2108.06625v2 ; CIKM 2021 DOI https://doi.org/10.1145/3459637.3482242 . Added to related work; no direct performance comparison claimed.
- IAFCL combines item-attribute graph fusion and contrastive sequential recommendation. Source: publisher abstract and metadata https://link.springer.com/article/10.1007/s00530-024-01486-7 . Multimedia Systems 30, article291,2024. Abstract-level comparison only; full subscriber text was not accessed. Added to related work; no direct performance comparison claimed.
- PRM models candidate-list context, and UniSRec uses item text with contrastive sequence representation learning. Existing manuscript references retained. Sources: https://yongfeng.me/attach/pei-recsys2019.pdf and https://arxiv.org/abs/2206.05941 .

This targeted comparison does not establish exhaustive priority or prove novelty. Experimental evidence can support a design choice; novelty also depends on its distinction from prior architectures.
