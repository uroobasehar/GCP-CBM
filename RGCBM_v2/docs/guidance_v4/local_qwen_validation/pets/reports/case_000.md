# Model evidence explanation

Image: data/pets/images/Abyssinian_201.jpg. Reference class: Abyssinian.

Classifier prediction: **Sphynx**; runner-up: **miniature_pinscher**. Uncalibrated probability: 0.0632. Logit margin: +0.3533; separate bias: -0.0043.

| Descriptor | Margin contribution | Grounding status |
|---|---:|---|
| Pet-image local intensity deviation — unverified guidance | +0.2632 | unverified |
| Pet-image neighbouring-pixel contrast — unverified guidance | +0.2323 | unverified |
| Pet-image local edge abundance — unverified guidance | +0.2152 | unverified |
| Pet-image local texture homogeneity — unverified guidance | -0.1218 | unverified |
| Pet-image relatively bright image regions — unverified guidance | -0.1142 | unverified |
| Pet-image local edge abundance — weak unguided association (learned variant 1) | -0.1067 | unguided_weak |

The model predicted Sphynx with an uncalibrated probability of 0.063; the runner-up was miniature_pinscher. Supporting model patterns: Pet-image local intensity deviation — unverified guidance (margin contribution +0.2632); Pet-image neighbouring-pixel contrast — unverified guidance (margin contribution +0.2323); Pet-image local edge abundance — unverified guidance (margin contribution +0.2152). Opposing model patterns: Pet-image local texture homogeneity — unverified guidance (-0.1218); Pet-image relatively bright image regions — unverified guidance (-0.1142); Pet-image local edge abundance — weak unguided association (learned variant 1) (-0.1067). Image measurements: Pet-image local intensity deviation — unverified guidance: high descriptor response relative to validation; Pet-image neighbouring-pixel contrast — unverified guidance: typical descriptor response relative to validation; Pet-image local edge abundance — unverified guidance: typical descriptor response relative to validation. The classifier bias contributes -0.0043 to this logit margin. Some cited patterns have unverified grounding; their intended guidance must not be reported as established anatomy or a learned clinical finding.

Model-derived image-descriptor associations. No clinical, anatomical, species-part or causal image claim is established by correlation. Probability is uncalibrated unless calibration is separately evaluated.

Backend: template. Numerical fields and table come directly from the classifier. Generated prose requires review.
