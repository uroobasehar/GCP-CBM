# Concept evidence cards

MRI image-label classification. Sequence/acquisition and patient independence are unverified. Brightness is not contrast enhancement; region priors are not tumor masks. Model class labels are not clinical diagnoses.

## MRI-image edge-transition steepness — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_steepness. Weak: True. Validation correlation: -0.102; 95% image-bootstrap interval: [-0.2483229827135801, 0.026341539341956648]. BH-FDR q: 0.5440379374736064.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image local edge abundance — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_density. Weak: True. Validation correlation: 0.420; 95% image-bootstrap interval: [0.34346872568130493, 0.48368531465530396]. BH-FDR q: 0.0785488315990993.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image directional edge coherence — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_coherence. Weak: True. Validation correlation: -0.038; 95% image-bootstrap interval: [-0.13796243108808995, 0.03683042526245117]. BH-FDR q: 0.6451526091466495.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image fine edge detail — unverified guidance

Group: boundary-guided. Status: unverified. Target: fine_edge_detail. Weak: True. Validation correlation: 0.240; 95% image-bootstrap interval: [0.1561932560056448, 0.3203134536743164]. BH-FDR q: 0.11213420331478119.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image local texture variability — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_variation. Weak: True. Validation correlation: -0.168; 95% image-bootstrap interval: [-0.264870285987854, -0.04498568307608356]. BH-FDR q: 0.2105836605806963.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image neighbouring-pixel contrast — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_contrast. Weak: True. Validation correlation: -0.007; 95% image-bootstrap interval: [-0.07841993868350983, 0.034968599677085876]. BH-FDR q: 0.8981261076750578.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image local texture homogeneity — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_homogeneity. Weak: True. Validation correlation: 0.200; 95% image-bootstrap interval: [0.15467846393585205, 0.27211386114358904]. BH-FDR q: 0.11213420331478119.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image local intensity entropy — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_entropy. Weak: True. Validation correlation: 0.115; 95% image-bootstrap interval: [-0.0008516889298334718, 0.17620755769312382]. BH-FDR q: 0.27528340888745856.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image relatively bright image regions — unverified guidance

Group: region-guided. Status: unverified. Target: bright_region. Weak: True. Validation correlation: 0.464; 95% image-bootstrap interval: [0.26586544439196585, 0.6616962775588038]. BH-FDR q: 0.1280982541102989.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image relatively dark image regions — unverified guidance

Group: region-guided. Status: unverified. Target: dark_region. Weak: True. Validation correlation: -0.473; 95% image-bootstrap interval: [-0.5456258058547974, -0.3312082588672638]. BH-FDR q: 0.1280982541102989.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image local intensity deviation — unverified guidance

Group: region-guided. Status: unverified. Target: local_intensity_deviation. Weak: True. Validation correlation: 0.137; 95% image-bootstrap interval: [0.04621315002441406, 0.23241576552391052]. BH-FDR q: 0.2259892357720269.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image threshold-region occupancy — unverified guidance

Group: region-guided. Status: unverified. Target: region_occupancy. Weak: True. Validation correlation: -0.173; 95% image-bootstrap interval: [-0.688687264919281, 0.2196117341518402]. BH-FDR q: 0.6631809946452601.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image neighbouring-pixel contrast — weak unguided association

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.234; 95% image-bootstrap interval: [0.09611608106642963, 0.34267449378967285]. BH-FDR q: 0.17339370990621633.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image relatively dark image regions — weak unguided association

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.308; 95% image-bootstrap interval: [0.18136020004749298, 0.5606638193130493]. BH-FDR q: 0.2721539562895098.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image relatively bright image regions — weak unguided association (learned variant 1)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.440; 95% image-bootstrap interval: [0.3438885770738125, 0.5548871159553528]. BH-FDR q: 0.11213420331478119.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## MRI-image relatively bright image regions — weak unguided association (learned variant 2)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: -0.389; 95% image-bootstrap interval: [-0.545365571975708, -0.25662705302238464]. BH-FDR q: 0.1280982541102989.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

