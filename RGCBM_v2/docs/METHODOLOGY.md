# Revised methodology text

## Framework

RG-CBM learns image-level classifications through a radiomics-inspired visual concept bottleneck.
The same architecture is trained independently on CUB-200-2011, Oxford-IIIT Pet and brain MRI.
Only image class labels enter supervision. Object masks, bird attributes and bounding boxes do not
enter feature extraction, cropping, concept losses or model selection. The method uses low-level
descriptor associations rather than assuming automatically discovered anatomical or part semantics.

For RGB image x in [0,1], EfficientNet-B1 produces spatial features F. A 1×1 convolution, batch
normalization and SiLU project F to d=128 dimensions. Both local features and K prototypes are
L2-normalized. Their response is A_k(i,j)=sigmoid(4·<z_hat(i,j),p_hat_k>), and u_k=mean_(i,j) A_k(i,j).
The `guided_v4` profile standardizes each pooled response with a non-affine BatchNorm1d,
then sets c_k=sigmoid((u_k−mu_k)/sqrt(v_k+epsilon)). Mini-batch statistics are used during training;
running mean/variance estimated only during training are fixed at validation/test/inference.
This addresses compressed pooled scores without adding a feature-to-class bypass. The legacy
profile uses c_k=u_k. Spatial guidance and descriptor grounding still use the original A maps.
This is feature standardization, not calibration of class probabilities. c_k=0.5 corresponds to
the estimated training-mean response; zero clamping is an extreme latent edit, not removal of anatomy.
The default CUB K=64 comprises 16 edge-guided, 16 texture-guided, 16 region/intensity-guided and 16
unguided concepts. Pet and MRI use K=16 (four per family). There are twelve distinct descriptor
targets, repeated across four learnable variants per target for CUB, not 64 unique semantic labels.
The capacity choice uses a disclosed frozen-feature validation diagnostic; K16/32/64 remain ablations.

Class logits satisfy z_q=b_q+sum_k W_qk c_k. Concept dropout applies during training only. At
inference, the signed contribution W_qk c_k reconstructs a class logit exactly. For the predicted
class q and runner-up r, c_k(W_qk−W_rk) decomposes the logit margin; the bias difference is reported
separately. Probability changes after a concept edit are nonlinear and are not sums of contributions.

## Descriptor guidance

Descriptor maps are computed in FP32 from the augmented, unnormalized image. Four targets per
family distinguish local edge-transition steepness, edge abundance, directional coherence and fine
edge detail; texture SD, neighbouring contrast, homogeneity and local entropy; and bright-region,
dark-region, local intensity-deviation and threshold-region occupancy. See CONCEPT_GUIDANCE.md for
all formulas and normalization. They are image-derived low-level proxies, not lesion segmentation,
IBSI radiomics, compactness, asymmetry or established parts. Original composite maps remain available
for the legacy group-average-guidance ablation.

Individual alignment is the channel-averaged MSE between each guided activation map and its assigned
resized target. Unlike averaging all channels before alignment, this constrains channel identity:
swapping two distinct target channels changes the objective. This is an explicit V4 refinement of
weak guidance; the model architecture and descriptor-family weights remain shared with V3.
The remaining losses are prototype off-diagonal cosine diversity, off-diagonal concept
correlation, mean absolute concept activation, L1 classifier weights, and representation stability
between gently perturbed images. Both stability views use the same inference-mode feature path.

L = weighted class CE + α(t)[0.18 L_edge + 0.10 L_texture + 0.08 L_region + 0.02 L_diversity
    + 0.01 L_decorrelation + 5e−5 L_concept-L1 + 0.02 L_stability] + 5e−6 ||W||_1.

These coefficients follow the working notebook; the manuscript's conflicting coefficients must be
replaced. α is zero for the first three head epochs and ramps over five epochs. Fine-tuning uses α=1.
Class weights are calculated from training labels only.

## Optimization and reproducibility

Images are resized to 240×240. Natural-image augmentation permits horizontal flips; MRI does not
use them. Modest affine/contrast changes are seeded per image and epoch. The head stage freezes the
backbone for up to 8 epochs with AdamW at 2e−3. The fine-tuning stage updates the last three
torchvision feature blocks for up to 30 epochs at 2e−5, with projection/prototypes/classifier at
1e−3, keeping backbone BN frozen. These explicit V3/V4 settings replace the initial
20/20-epoch shared-LR 4e−4/8e−6 profile; saved configurations specify the actual protocol. This block
selection is explicitly different from the old Keras layer count. Reduce-on-plateau and validation
early stopping are stage-local. The final guided V4 model is the checkpoint with best validation accuracy among candidates exposed
to nonzero guidance. Actual exposure counts survive checkpoint reload. A warm-up-only run is
explicitly flagged, not described as grounded; best_any.pt retains the unrestricted validation best.
The fine-tuning stage starts from the best head checkpoint with its actual exposure metadata.

The PyTorch backend enables native Windows Blackwell support. Torchvision ImageNet-B1 V2 weights
and its RGB normalization differ from the original TensorFlow checkpoint/preprocessing. Consequently,
old metrics and weights are not transferred, and every comparative experiment must be rerun.

Official test lists remain fixed. A validation subset is selected within the official training pool;
MRI exact duplicate images are grouped. Run identity includes configuration, class order, split
manifest hash and backbone checksum. Atomic checkpoints store model, optimizer, scheduler, scaler,
stage, epoch, batch position, running totals and random states. Exact CPU interrupted/resumed
weights were verified; bitwise GPU reproducibility still depends on the target CUDA/driver stack.
Periodic full-state snapshots are saved every five epochs and at stage end, retaining three per
stage by default. last.pt still updates every 50 batches and every epoch. Live training status,
CSV history and accuracy/CE curves are saved during training. The sequential dataset queue uses
separate models, standardization statistics, run folders and manifests; completed evidence is
reused only under its exact saved configuration and pipeline status. Test scores never tune the queue.

## Grounding, evaluation and text

Grounding measures per-image spatial Pearson correlation between each concept map and each descriptor
on validation. A guided concept retains its intended descriptor: a stronger alternative cannot rename
away failed guidance. Verification requires actual guidance exposure, n>=50, positive mean r>=0.10,
a 95% image-bootstrap interval excluding zero and BH-FDR q<0.05 over K x 16 tests. A reliable inverse
association is a guidance conflict. Unexposed, unknown and insufficient associations stay qualified.
Unguided channels can receive the strongest validation descriptor association under the same checks;
sign and selection are disclosed. Domain-prefixed names refer to measured image properties, not
unmeasured bird parts, fur or tumor borders. Frozen names are independently evaluated on test.

In addition to classification and calibration metrics, latent ablation tests compare removing the
strongest positive margin-supporting concept with random concept removal on the same eligible
examples. The original largest-absolute-class-contribution protocol remains separately exported.
Stability uses fixed mild photometric changes. Foreground response normalized by foreground area
and peak-in-foreground rate use held-out CUB boxes/Pet trimaps; no segmentation performance is claimed.

Structured evidence contains the fixed prediction, uncalibrated probability, bias, exact class and
margin decompositions, concept names, validation uncertainty and image descriptor response relative
to validation quartiles. Named prototype plates include full training images, response maps, peak
crops and source coordinates. Held-out case panels pair learned maps with their actual guidance
targets. Each exact margin term is allocated over its unit-mass spatial map: the resulting signed
panel sums to the concept margin, but is an allocation convention, not causal image localization.
300 dpi PNG, PDF, SVG, NPZ and caption/provenance exports accompany the figures.

Local Qwen2.5-1.5B-Instruct or optional Gemini paraphrases bounded evidence statements into explanation/limitations prose only. Structured
prediction fields are rendered deterministically. Prose gates require the strongest descriptor names
and weak-evidence qualifications and reject specified class/concept and clinical-claim errors. Invalid
generation falls back to a labeled evidence template; raw text and reason are saved. Report and
figure use exactly the same image packet. These checks cannot guarantee semantic factuality; human
auditing remains required. The verbalizer does not feed back into classification, training or naming.

## Figure caption

**Overview of revised RG-CBM.** A shared single-level prototype bottleneck is independently trained
on fine-grained animal and medical image datasets. Twelve measurable edge, texture and region/intensity targets guide
individual channels within three concept groups; the fourth remains unguided. A sparse linear classifier admits exact signed
concept contributions. Validation-only grounding audits intended targets and supplies qualified descriptor names, while held-out
classification, perturbation and independent foreground annotations assess performance and
interpretability. The explanation verbalizer consumes fixed evidence and does not influence predictions.
