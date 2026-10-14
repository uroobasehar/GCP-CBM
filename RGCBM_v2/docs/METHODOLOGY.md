# Revised methodology text

## Framework

RG-CBM learns image-level classifications through a radiomics-inspired visual concept bottleneck.
The same architecture is trained independently on CUB-200-2011, Oxford-IIIT Pet and brain MRI.
Only image class labels enter supervision. Object masks, bird attributes and bounding boxes do not
enter feature extraction, cropping, concept losses or model selection. The method uses low-level
descriptor associations rather than assuming automatically discovered anatomical or part semantics.

For RGB image x in [0,1], EfficientNet-B1 produces spatial features F. A 1×1 convolution, batch
normalization and SiLU project F to d=128 dimensions. Both local features and K prototypes are
L2-normalized. Their response is A_k(i,j)=sigmoid(4·<z_hat(i,j),p_hat_k>), and c_k=mean_(i,j) A_k(i,j).
The default K=16 comprises four edge-guided, four texture-guided, four region-prior-guided and four
unguided concepts. K=32/64 are capacity ablations, not changed default results.

Class logits satisfy z_q=b_q+sum_k W_qk c_k. Concept dropout applies during training only. At
inference, the signed contribution W_qk c_k reconstructs a class logit exactly. For the predicted
class q and runner-up r, c_k(W_qk−W_rk) decomposes the logit margin; the bias difference is reported
separately. Probability changes after a concept edit are nonlinear and are not sums of contributions.

## Descriptor guidance

Descriptor maps are computed from the augmented, unnormalized image, not the ImageNet-normalized
tensor. Each is independently min–max normalized. Edge guidance combines normalized Sobel magnitude
and absolute Laplacian responses with weights 0.7/0.3. Texture combines local standard deviation
in a 9×9 window with a neighboring-intensity contrast/homogeneity proxy, matching the notebook's
GLCM-inspired extension; it is not an exact radiomics GLCM feature. The region prior retains the
original mean/std threshold rule (mean+0.45·std; mean−0.90·std with weight 0.35), smoothing and
normalization. It is a pseudo-region map, not segmentation, compactness, asymmetry or tumor shape.
Intensity is a post-training grounding descriptor and has no separate training alignment branch.

Group alignment is MSE between the mean activation map within each guided group and the resized
descriptor. The remaining losses are prototype off-diagonal cosine diversity, off-diagonal concept
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
backbone for up to 20 epochs with AdamW at 4e−4. The fine-tuning stage updates the last three
torchvision feature blocks for up to 20 epochs at 8e−6, keeping backbone BN frozen. This block
selection is explicitly different from the old Keras layer count. Reduce-on-plateau and validation
early stopping are stage-local. The final model is the checkpoint with best validation accuracy
across both stages; the fine-tuning stage starts from the best head checkpoint.

The PyTorch backend enables native Windows Blackwell support. Torchvision ImageNet-B1 V2 weights
and its RGB normalization differ from the original TensorFlow checkpoint/preprocessing. Consequently,
old metrics and weights are not transferred, and every comparative experiment must be rerun.

Official test lists remain fixed. A validation subset is selected within the official training pool;
MRI exact duplicate images are grouped. Run identity includes configuration, class order, split
manifest hash and backbone checksum. Atomic checkpoints store model, optimizer, scheduler, scaler,
stage, epoch, batch position, running totals and random states. Exact CPU interrupted/resumed
weights were verified; bitwise GPU reproducibility still depends on the target CUDA/driver stack.

## Grounding, evaluation and text

Grounding measures per-image spatial Pearson correlation between each concept map and each descriptor
on validation. The strongest absolute mean association selects the descriptor; its sign is retained.
Image-bootstrap confidence intervals and a minimum strength/sample rule keep weak concepts explicitly
weak. Names describe the measured association and do not assign arbitrary clinical or part labels.
The selected descriptor/sign are fixed before independent test alignment is measured.

In addition to classification and calibration metrics, latent ablation tests compare removing the
strongest positive margin-supporting concept with random concept removal on the same eligible
examples. The original largest-absolute-class-contribution protocol remains separately exported.
Stability uses fixed mild photometric changes. Foreground response normalized by foreground area
and peak-in-foreground rate use held-out CUB boxes/Pet trimaps; no segmentation performance is claimed.

Structured evidence contains the fixed prediction, uncalibrated probability, bias, exact class and
margin decompositions, concept names, grounding uncertainty flags and signed contributions. A local
Qwen2.5-1.5B-Instruct model or optional Gemini verbalizes this evidence after classification. JSON
checks reject modified predictions, invalid concept IDs and reversed signs; failed generations fall
back to a deterministic template. Semantic factuality still requires human auditing. Neither the LLM
nor its generated text feeds back into classification, training or grounding.

## Figure caption

**Overview of revised RG-CBM.** A shared single-level prototype bottleneck is independently trained
on fine-grained animal and medical image datasets. Weak edge, texture and region descriptors guide
three concept groups; the fourth remains unguided. A sparse linear classifier admits exact signed
concept contributions. Validation-only grounding supplies measured descriptor names, while held-out
classification, perturbation and independent foreground annotations assess performance and
interpretability. The explanation verbalizer consumes fixed evidence and does not influence predictions.
