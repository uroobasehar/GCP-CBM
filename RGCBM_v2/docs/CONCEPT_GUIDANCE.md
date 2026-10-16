# Named, auditable concept guidance (guided_v4)

The primary concept labels are measurable image descriptors, with Bird-image, Pet-image or MRI-image
prefixes. They describe the domain in which the measurement was made. They do not assert named bird
parts, fur traits or tumor morphology without corresponding evidence. The original notebook rotated
four names within a group using the concept index; group-average alignment alone could not establish
those individual names. This revision makes each guided channel's intended measurement explicit.

## Guidance contract

The architecture remains a single prototype bottleneck with no feature-to-class bypass. The three
guided families are boundary/edge, texture and region/intensity; the fourth is unguided. For K=16,
each guided channel has one distinct target below. CUB now defaults to K=64: each target guides four
learnable prototype responses, with replicate/learned-variant suffixes. This is 12 guidance types,
not 64 independently established semantic attributes. Pet and MRI retain K=16. K16/32/64 ablations
and `legacy_group_guidance` quantify the capacity and individual-guidance changes separately.

All targets use grayscale luminance of the augmented RGB image in [0,1], before ImageNet normalization.
Sobel kernels are divided by eight. Local means use a reflect-padded 9x9 window. N denotes per-image
min-max normalization with a numerical floor. These are reproducible image-derived proxies, not
IBSI-compliant radiomics measurements or segmentation masks.

| Family | Display phrase | Measured map |
|---|---|---|
| Boundary | Edge-transition steepness | N(Sobel magnitude / (local SD + 0.03)) |
| Boundary | Local edge abundance | Local fraction above the image's 75th gradient percentile |
| Boundary | Directional edge coherence | Structure-tensor coherence times N(gradient magnitude) |
| Boundary | Fine edge detail | N(absolute discrete Laplacian) |
| Texture | Local texture variability | N(local intensity SD) |
| Texture | Neighbouring-pixel contrast | N(local mean of squared horizontal/vertical differences) |
| Texture | Local texture homogeneity | Local mean of 1/(1+16 times neighbouring squared difference) |
| Texture | Local intensity entropy | Eight-bin local histogram entropy / log(8) |
| Region | Relatively bright image regions | Local occupancy above mean + 0.45 SD |
| Region | Relatively dark image regions | Local occupancy below mean - 0.90 SD |
| Region | Local intensity deviation | N(abs(intensity - local mean)) |
| Region | Threshold-region occupancy | Local occupancy of the union of bright and dark thresholds |

Each guided map is aligned by MSE to its resized target. The three family losses average over their
channels and retain the existing 0.18/0.10/0.08 coefficients. Diversity, decorrelation, sparsity,
stability and the linear classifier remain. Descriptor calculations stay FP32 under mixed precision.
The old four descriptor maps remain available for legacy comparisons and grounding of unguided channels.

## Intended guidance is checked, not assumed

`guidance_contract.json` records configured channel-to-target assignments. Checkpoints count actual
updates exposed to nonzero guidance. The selected checkpoint, its stage/epoch and update count are
recorded. Guided V4 selects the best validation checkpoint among guidance-exposed candidates; a
warm-up-only run has a visible warning and is labeled guidance not yet applied. `best_any.pt`
separately retains the unconstrained validation best for diagnosis. Historical exposure cannot be
inferred from a final configuration file and remains unknown when absent from older checkpoints.

On validation, the spatial Pearson correlation for every concept/descriptor pair is summarized
across images. A reliable association requires at least 50 finite images, absolute mean r >= 0.10,
a 95% image-bootstrap interval excluding zero and BH-FDR q < 0.05 across K x 16 descriptor tests.
Guided concepts are assessed against their intended target even when another descriptor correlates
more strongly. Reliable positive alignment is verified; reliable inverse alignment is a guidance
conflict; other cases stay unverified. Guidance not exposed/unrecorded stays explicit. Unguided
channels may receive a validation-selected descriptor association with the same multiplicity check.
Test images never choose names. Full correlation statistics and the strongest observed alternative
are retained for audit. These tests establish descriptor association, not a causal benefit of guidance.

Examples of honest labels are `MRI-image edge-transition steepness`,
`Pet-image local texture variability — unverified guidance`, and
`Bird-image Reduced directional edge coherence — guidance conflict`.
A high edge-transition response over a whole MRI is not sufficient to claim a sharp tumor boundary;
that stronger claim requires a lesion ROI/mask and dedicated validation. Cross-domain low-level
names are deliberately conservative. The optional CUB attribute audit is a separate, disclosed,
annotation-assisted post-training analysis, never a source of training labels or hidden semantics.

## Figure and explanation outputs

Every run exports named alignment matrices, a concept dictionary/cards, per-concept prototype plates
(full training image, learned-response overlay and peak crop), and patch source paths/coordinates.
Case panels show the real held-out image, supporting/opposing named maps, prediction/reference class,
a signed margin allocation, and the corresponding descriptor-target-versus-learned-map comparison.
Paper-oriented figures have 300 dpi PNG plus PDF/SVG; smoke panels carry an explicit watermark.
`illustrative_cases.csv`, `prototype_provenance.csv`, raw NPZ arrays and captions document selection
and provenance. Cases are evenly spaced, not cherry-picked by correctness. Prototype peaks are
coarse grid response regions, not independently discovered object-part boxes.

For a case, exact margin terms are c_k(W_pred,k-W_runner,k), with classifier bias separately reported.
The signed spatial panel allocates each term over its map normalized to unit spatial mass. Its sum
reconstructs the concept portion of the margin, but the allocation convention is neither Grad-CAM
nor unique causal localization. The standardized pooled score is not the mean of the displayed map;
figure captions and saved evidence retain the distinction.

Reports use the SAME image/evidence packet as the case panel. Prediction, confidence, concept indices,
signs and contributions come from deterministic classifier evidence. Qwen/Gemini paraphrase bounded evidence statements into two prose fields. The report renders
the classifier probability, margin, bias and signed-contribution table beside those fields. Gates require the strongest support/opposition descriptor names, preserve weak-grounding
qualifiers, and reject known class/concept and clinical-claim errors. Invalid output falls back to an
explicitly labeled deterministic report; raw generation and reason are retained. Valid JSON and these
gates are not a guarantee of semantic correctness: review prose before publication. The default local
Qwen model is bundled and can run without Internet. `LLM_BACKEND='template'` gives a faster deterministic
report when the LLM is unnecessary. The LLM never participates in classification or naming.

## Accuracy diagnosis and experiment design

The reported early CUB losses near 5.3 were close to log(200), indicating little class discrimination.
A full-training-pool frozen-feature diagnostic used 5,394 train and 600 validation images, a single
cached augmentation per training image, ten head-only epochs and simplified CE + guidance. Best
validation accuracies were 72.0% for a pooled-feature classifier, 35.33% for K16, and 53.33% for K64.
No test data entered that choice. These are software/learning diagnostics, NOT complete RG-CBM or
literature benchmark results. Frozen features, omitted regularizers, one seed and a different
training budget limit interpretation. The portable diagnostic and complete epoch records are in
`diagnostics/full_cub_frozen_probe.py` and `docs/guidance_v4/full_cub_probe.json`.

The new default CUB capacity is motivated by that validation result; the two-stage protocol still
requires full GPU runs, fixed test evaluation and at least three seeds. Report classification,
calibration, intended-target alignment, verified-guidance fraction, redundancy, matched deletion
curves and independent foreground evaluation jointly. Include K16/32/64, unguided, family-removal,
legacy group guidance and no-standardization controls. No high held-out accuracy is guaranteed by
the implementation or smoke checks. Keep `RUN_TAG='guided_v4'` so older experiments remain intact.
