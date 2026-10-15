# Research revision and study selection

Current training revision: `calibrated_v3` adds training-only standardization of pooled scores,
independent head/backbone fine-tuning learning rates and automatic dataset execution. See
[the training diagnosis](TRAINING_AND_DATASET_QUEUE.md) and the updated [methodology](METHODOLOGY.md).
Older paper/code optimization settings described below are historical; actual comparisons must
use the saved current protocol, including the matched `no_standardization` ablation.

Suggested title: **RG-CBM: Radiomics-Inspired Descriptor Guidance for Annotation-Efficient Visual Concept Bottlenecks**.

The paper should ask whether image-derived descriptor guidance improves an explicitly interpretable,
annotation-free bottleneck across fine-grained and medical classification. Train a separate model on
each dataset; this experiment does not establish unsupervised domain adaptation or cross-dataset transfer.
Adding datasets alone is not sufficient evidence of novelty or acceptance by an IEEE journal.

## Papers reviewed and appropriate roles

| Supplied study | What it actually studies | Role in the revision |
|---|---|---|
| MCPNet++: Interpretable Classification Models via Multi-Level Concept Prototypes, TPAMI 2026, DOI 10.1109/TPAMI.2026.3680506 | Metadata confirms the extension; the supplied PDF exceeds the 32 MiB upload-transfer limit. Its full text could not be retrieved through the available publisher endpoint. | Intended primary prototype comparator, **pending direct full-text verification**. Do not attribute the older paper's settings or results to this extension. |
| MCPNet: An Interpretable Classifier via Multi-Level Concept Prototypes, CVPR 2024, Wang et al., DOI 10.1109/CVPR52733.2024.01035 | Public paper, arXiv 2404.08968v3, and official NVlabs/MCPNet README reviewed. Learns multi-level concept segments using CKA; extracts weighted-PCA prototypes; CCD distribution matching supplies classification. Uses AwA2, Caltech101 and CUB. CUB official split is 5994/5794; attribute annotations are not training targets. | **Primary reviewed empirical base**: comparable unsupervised prototype learning on CUB. RG-CBM retains one feature level and a sparse linear head; it does not implement CKA, weighted PCA or CCD. |
| The Decoupling Concept Bottleneck Model, TPAMI 2025, Zhang et al., DOI 10.1109/TPAMI.2024.3489597; supplied 10720263.pdf | Explicit/implicit concepts, concept insufficiency, and human interaction; uses CUB, Derm7pt, CelebA, with concept labels and different backbones/settings. | **Second reviewed conceptual base**: discuss bottleneck capacity and semantic distortion. Compare label accuracy only under matched experiments. Its supervised concept accuracy is not directly comparable with RG-CBM's descriptor correlation. |
| Learning Transferable Conceptual Prototypes for Interpretable Unsupervised Domain Adaptation, TIP 2024, Gao et al. | Source/target adaptation on Office-Home, VisDA and DomainNet, with transferable prototypes and pseudo-label consistency. | Related work and possible future extension. Direct UDA comparisons would need a new source/target protocol and considerable method changes. |
| MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion; supplied 10720262.pdf | Physics-motivated inverse-problem unfolding and multispectral/hyperspectral fusion. | Interpretability context, not a classification baseline. |
| Sensitivity Analysis-Based Explainable Diffusion Transformers for Anomaly Detection in Consumer Electronics Manufacturing; supplied 10720261.pdf | Reconstruction/detection on industrial anomaly data, including VisA; evaluates a different output/task. | Context only. Do not compare anomaly-detection scores with classification accuracy. |
| Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving; supplied 10720264.pdf | Safety and explainability in end-to-end driving. | Motivation/context, not a matched quantitative baseline. |

The comparison sources are https://github.com/NVlabs/MCPNet and https://github.com/deepopo/DCBM.
The public MCPNet text is https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_MCPNet_An_Interpretable_Classifier_via_Multi-Level_Concept_Prototypes_CVPR_2024_paper.pdf.
The exact MCPNet++ manuscript remains a specific review limitation; a compressed copy under 32 MiB
would enable the extension-specific comparison. None of its unseen numerical results were invented.

## Dataset choice

| Dataset | Classes | Official training pool / test | Main purpose | Independent explanation evidence |
|---|---:|---:|---|---|
| CUB-200-2011 | 200 | 5994 / 5794 | Fine-grained classification; overlaps reviewed MCPNet and DCBM | Official bounding boxes for foreground response; 312 attributes and parts retained for future audited semantic probes |
| Oxford-IIIT Pet | 37 | 3680 / 3669 | Animal breed classification with detailed texture/region structure | Official foreground/background/border trimaps, used **only at evaluation** |
| Kaggle Brain Tumor MRI | 4 | 5600 / 1600 | Secondary medical applicability; preserves the original task | Descriptor alignment; no expert tumor masks or verified patient IDs supplied |

CUB/Pet validation is a seeded stratified 10% subset of the official training pool: 5394/600 and
3312/368. MRI validation uses exact-image-hash groups to prevent identical images entering both
training and validation. Seed 42 with the pinned scikit-learn 1.5.2 environment gives 5038/562. Earlier smoke tests
used a newer scikit-learn release and produced 5040/560; those historical outputs remain labeled
as validation evidence. The saved dataset manifest is authoritative for each run. Official test membership is preserved in every case.
Archive mirrors are explicitly recorded in `rgcbm/data.py`; original publisher metadata determines
membership and class labels. CUB's additional CVPR-2016 text embeddings are unused and not extracted.
Do not interpret a mirror's CC0 label as replacing the original dataset's terms; retain its README.

## Findings in the submitted draft and original notebook

| Issue | Evidence | Implemented correction or research requirement |
|---|---|---|
| Draft and code report different settings | Draft weights: edge 1.0, texture/morphology 0.5, diversity/disentanglement 0.05, stability 0.10; notebook: 0.18/0.10/0.08/0.02/0.01/0.02. Draft dropout 0.15 vs code 0.10; draft LR 1e-3/1e-5 vs 4e-4/8e-6; 45 vs 90 final Keras layers. | One saved Config, fingerprinted with split and backbone. Revised method uses the notebook values. Report actual experiment configurations rather than either old prose table. |
| Descriptor names imply unsupported anatomy | Original `choose_concept_name` picks a string by concept index modulo four; a correlation with a thresholded map cannot establish lesion shape, asymmetry, tumor margins or contrast enhancement. | Names are signed measured associations, with bootstrap uncertainty and explicitly weak grounding. No artificial anatomical/part names. |
| Morphology prior is not a shape measurement | Thresholded grayscale intensities plus smoothing contain no geometric compactness/asymmetry calculation or tumor segmentation annotation. | Describe it as an **image-derived region prior**; keep the original computation to limit methodological changes. Actual morphology-specific claims require independently measured geometric descriptors. |
| Validation leakage | Audit of the original seed-42 image split found 29 exact duplicate pairs across train/validation. No exact-byte overlap between the supplied training and test pools was found. | Group exact image hashes within the MRI training pool. Near-duplicates and patient leakage still require further investigation. |
| Faithfulness conflates signs and causality | Removing the largest absolute predicted-class term can remove opposing evidence. A latent edit is not an intervention on an image feature or disease. | Primary metric removes the strongest positive predicted-versus-runner-up margin term; report random ablation on matching cases and retain the old absolute protocol separately. Use the phrase **model dependence**. |
| Grounding can evaluate its own name selection | Selecting the strongest descriptor and reporting that maximum on the same examples is optimistic. | Choose descriptor/sign on validation; report that frozen choice on test. Independently audit CUB boxes/Pet trimaps without training on them. |
| Concept collapse or insufficient capacity | All four concepts within a group are trained against the group's mean prior; they need not represent four distinct semantic concepts. Sixteen pooled concepts may underfit 200 bird classes. | Retain diversity/decorrelation, export effective rank/redundancy, test K=16/32/64 and unguided/guidance ablations. Do not claim 16 unique high-level concepts without evidence. |
| Environment-dependent APIs, forced refresh, credentials and lost optimizer state | Original Colab notebook upgrades unpinned packages, forces dataset refresh and relies on live session state/weights. | Frozen Windows wheelhouse; cached local archives; stage/epoch/batch/optimizer/scheduler/scaler/RNG checkpointing; secure optional LLM credentials. |
| LLM correctness not guaranteed | A prompt does not prove faithful prose. | JSON field, class, probability, concept-index and sign checks; explicit deterministic fallback; raw output retained for audit. Human factuality evaluation is still required. |

The draft's reported 91.06% accuracy and ablation values were read, **not reproduced or independently
verified**. The new smoke-run metrics are software checks, never replacement experimental results.
The original TensorFlow model files cannot be loaded directly into this PyTorch implementation.

## Experimental plan

1. Run full CUB, Pet and MRI experiments with seeds 42, 43 and 44. Test data must not tune any hyperparameter.
2. Compare with the runnable EfficientNet-B1 linear baseline and an unguided concept bottleneck under
   identical preprocessing, pretrained checkpoint, splits, seed, batch size, stages and selection rule.
3. Run no-edge, no-texture, no-region, no-all-guidance, no-diversity, no-decorrelation, no-stability,
   no-fine-tuning and K=32/64 ablations. Use independent run names/checkpoints.
4. Reproduce official MCPNet/MCPNet++ and DCBM baselines using their released implementation and
   documented protocol. Keep **matched reruns** separate from **published context numbers**. Different
   backbones, pretraining, concept labels, crops, splits, resolutions and budgets must be disclosed.
5. Report accuracy, macro/weighted F1, balanced accuracy, top-5 (informative for CUB/Pet), calibration,
   confusion matrices, class-wise results, parameter count, runtime and peak GPU memory. The notebook
   exports the classifier metrics; time/memory measurements need the actual target GPU runs.
6. Report positive-margin vs random ablation, legacy absolute ablation, stability, frozen test grounding,
   concept effective rank and redundancy, foreground response/area ratio and peak localization.
   Object foreground alignment supports localization, not anatomical/part semantics or segmentation quality.
7. Assess generated explanations against the structured evidence with blinded human review: unsupported
   claims, correct support/opposition, omissions and readability. Run a with/without-LLM verbalization
   comparison; classifier predictions must remain identical. No clinical benefit has been established.

The safe central claim is that RG-CBM **encourages descriptor-associated, explicitly decomposable
visual bottlenecks without concept labels**. The effectiveness/generalization claim awaits the complete
experiments. Do not call this clinical semantic discovery, causal explanation, or multi-level prototype
learning. Suitable IEEE venues depend on the resulting contribution and evidence; venue branding alone
does not repair unsupported claims or inconsistent experimental reporting.

## Added journal-facing executable audits

All three notebooks include domain-scoped evidence cards, reliability diagrams, concept redundancy,
manual latent interventions with exact logit-change verification, positive-support versus matched-random
deletion curves with paired bootstrap intervals, and hardware/latency profiling. An optional CUB
post-training semantic audit uses official certainty>=3 attributes. Validation associations require
at least 50 images, |r|>=0.2, and Bonferroni alpha=0.05/(312×K); selected attributes are evaluated with
held-out AUC. Provisional names explicitly identify **annotation-assisted association**. This audit
does not train or alter model weights, does not rename concepts from test data, and must not be
presented as annotation-free high-level semantic naming. Default descriptor names remain fixed.
