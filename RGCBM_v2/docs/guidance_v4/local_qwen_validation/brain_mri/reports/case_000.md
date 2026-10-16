# Model evidence explanation

Image: data/brain_mri/Testing/glioma/Te-gl_1.jpg. Reference class: glioma.

Classifier prediction: **glioma**; runner-up: **notumor**. Uncalibrated probability: 0.2830. Logit margin: +0.0317; separate bias: +0.1140.

| Descriptor | Margin contribution | Grounding status |
|---|---:|---|
| MRI-image fine edge detail — unverified guidance | +0.2269 | unverified |
| MRI-image relatively bright image regions — unverified guidance | +0.2107 | unverified |
| MRI-image local texture variability — unverified guidance | +0.1596 | unverified |
| MRI-image relatively dark image regions — weak unguided association | -0.2776 | unguided_weak |
| MRI-image neighbouring-pixel contrast — unverified guidance | -0.2484 | unverified |
| MRI-image relatively bright image regions — weak unguided association (learned variant 1) | -0.1995 | unguided_weak |

The model favors glioma over notumor due to MRI-image fine edge detail (supporting) and relatively dark image regions (opposing), despite some unverified associations.

Some cited descriptor associations are unverified. These are model-derived descriptor associations, not established anatomical or clinical findings. Probability is uncalibrated.

Backend: local_qwen. Numerical fields and table come directly from the classifier. Generated prose requires review.
