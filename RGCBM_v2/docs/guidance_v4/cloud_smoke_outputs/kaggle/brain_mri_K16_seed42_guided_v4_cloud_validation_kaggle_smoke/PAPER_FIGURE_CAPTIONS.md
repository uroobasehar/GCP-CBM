# Figure interpretation and export guide

All named concept/prototype and explanation panels are exported as 300 dpi PNG plus PDF/SVG.
The alignment matrix compares validation concept maps against measurable image descriptors;
outlined cells identify intended guidance, not evidence that guidance succeeded. Unverified and
conflicting targets remain labeled. The table records n, bootstrap intervals and BH-FDR q values.

Prototype plates show the original training image, model response, peak crop and grounding status.
Paths/ranks/crop coordinates are in prototype_provenance.csv. A prototype is an exemplar of model
response, not proof of a bird part, fur property or tumor anatomy.

Case panels pair named concept heatmaps with exact signed predicted-versus-runner-up contributions.
The final spatial panel allocates each concept term in proportion to its nonnegative spatial map;
its pixel sum equals the concept component of the logit margin. Bias is separate. This allocation
is a visualization convention, not unique localization, Grad-CAM or a causal image explanation.
illustrative_cases.csv links every figure to the SAME image's evidence and LLM report. Examples are
selected by evenly spaced test-list indices, not cherry-picked for successful prediction.

Guidance comparison panels show the image-derived target beside learned response; clinical regions
and anatomical parts are not annotated by these weak maps. Independent localization/annotation and
human review are required for stronger domain claims. Smoke figures are software demonstrations,
not journal experimental findings. Include full-run sample counts, seeds and held-out results.
