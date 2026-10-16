# Model evidence explanation

Image: data/brain_mri/Testing/glioma/Te-gl_1.jpg. Reference class: glioma.

The model predicted glioma with an uncalibrated probability of 0.283; the runner-up was notumor. Supporting model patterns: MRI-image fine edge detail — unverified guidance (margin contribution +0.2269); MRI-image relatively bright image regions — unverified guidance (margin contribution +0.2107); MRI-image local texture variability — unverified guidance (margin contribution +0.1596). Opposing model patterns: MRI-image relatively dark image regions — weak unguided association (-0.2776); MRI-image neighbouring-pixel contrast — unverified guidance (-0.2484); MRI-image relatively bright image regions — weak unguided association (learned variant 1) (-0.1995). Image measurements: MRI-image fine edge detail — unverified guidance: typical descriptor response relative to validation; MRI-image relatively bright image regions — unverified guidance: low descriptor response relative to validation; MRI-image local texture variability — unverified guidance: high descriptor response relative to validation. The classifier bias contributes +0.1140 to this logit margin. Some cited patterns have unverified grounding; their intended guidance must not be reported as established anatomy or a learned clinical finding.

Model-derived image-descriptor associations. No clinical, anatomical, species-part or causal image claim is established by correlation. Probability is uncalibrated unless calibration is separately evaluated.

Backend: template. Prose requires review.
