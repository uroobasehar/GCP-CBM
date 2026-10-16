# Concept evidence cards

Pet breed classification. Texture/region associations alone do not establish fur type, muzzle anatomy or disease. Foreground trimap alignment is localization evidence, not proof of a breed-specific semantic concept.

## Pet-image edge-transition steepness — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_steepness. Weak: True. Validation correlation: -0.244; 95% image-bootstrap interval: [-0.29460261911153796, -0.15349723398685455]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local edge abundance — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_density. Weak: True. Validation correlation: 0.293; 95% image-bootstrap interval: [0.08054538071155548, 0.49684208631515503]. BH-FDR q: 0.19507917269008365.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image directional edge coherence — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_coherence. Weak: True. Validation correlation: 0.155; 95% image-bootstrap interval: [-0.11434166878461838, 0.42564460560679457]. BH-FDR q: 0.4860988931918363.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image fine edge detail — unverified guidance

Group: boundary-guided. Status: unverified. Target: fine_edge_detail. Weak: True. Validation correlation: 0.257; 95% image-bootstrap interval: [0.051503084413707254, 0.407096803188324]. BH-FDR q: 0.19507917269008365.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local texture variability — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_variation. Weak: True. Validation correlation: 0.383; 95% image-bootstrap interval: [0.14550566952675584, 0.5626890063285828]. BH-FDR q: 0.1635171707640303.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image neighbouring-pixel contrast — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_contrast. Weak: True. Validation correlation: 0.260; 95% image-bootstrap interval: [0.14294858276844025, 0.35129809379577637]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local texture homogeneity — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_homogeneity. Weak: True. Validation correlation: 0.288; 95% image-bootstrap interval: [0.18417720161378384, 0.3671914041042328]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local intensity entropy — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_entropy. Weak: True. Validation correlation: -0.349; 95% image-bootstrap interval: [-0.533943535387516, -0.16416309773921967]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image relatively bright image regions — unverified guidance

Group: region-guided. Status: unverified. Target: bright_region. Weak: True. Validation correlation: -0.165; 95% image-bootstrap interval: [-0.28821968734264375, -0.03859945759177208]. BH-FDR q: 0.21522469307059672.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image relatively dark image regions — unverified guidance

Group: region-guided. Status: unverified. Target: dark_region. Weak: True. Validation correlation: 0.200; 95% image-bootstrap interval: [-0.040253896266222, 0.4922769367694855]. BH-FDR q: 0.3320613349835897.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local intensity deviation — unverified guidance

Group: region-guided. Status: unverified. Target: local_intensity_deviation. Weak: True. Validation correlation: 0.260; 95% image-bootstrap interval: [0.06532755652442575, 0.4540133781731129]. BH-FDR q: 0.21690862641917716.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image threshold-region occupancy — unverified guidance

Group: region-guided. Status: unverified. Target: region_occupancy. Weak: True. Validation correlation: -0.146; 95% image-bootstrap interval: [-0.34302252531051636, 0.05182208502665162]. BH-FDR q: 0.36946906113043065.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local edge abundance — weak unguided association (learned variant 1)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.367; 95% image-bootstrap interval: [0.2154497653245926, 0.47572287917137146]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local edge abundance — weak unguided association (learned variant 2)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: -0.450; 95% image-bootstrap interval: [-0.6203274875879288, -0.24978981800377362]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image local edge abundance — weak unguided association (learned variant 3)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.363; 95% image-bootstrap interval: [0.140921950340271, 0.49818938970565796]. BH-FDR q: 0.1590638584560818.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Pet-image threshold-region occupancy — weak unguided association

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: -0.350; 95% image-bootstrap interval: [-0.5766670107841492, -0.13390269875526428]. BH-FDR q: 0.19507917269008365.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

