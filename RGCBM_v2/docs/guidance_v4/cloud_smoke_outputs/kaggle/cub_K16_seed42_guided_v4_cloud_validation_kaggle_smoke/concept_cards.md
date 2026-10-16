# Concept evidence cards

Bird species classification. Edge/texture/region associations alone do not establish a wing, bill, plumage color or anatomical part. Attribute-assisted labels require the separately reported annotation audit.

## Bird-image edge-transition steepness — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_steepness. Weak: True. Validation correlation: 0.090; 95% image-bootstrap interval: [-0.103144377656281, 0.4155183732509613]. BH-FDR q: 0.6633815527951569.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local edge abundance — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_density. Weak: True. Validation correlation: 0.267; 95% image-bootstrap interval: [0.1395571529865265, 0.4773675203323364]. BH-FDR q: 0.19149098572907625.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image directional edge coherence — unverified guidance

Group: boundary-guided. Status: unverified. Target: edge_coherence. Weak: True. Validation correlation: 0.294; 95% image-bootstrap interval: [0.1474902331829071, 0.4396916329860687]. BH-FDR q: 0.14869229738102402.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image fine edge detail — unverified guidance

Group: boundary-guided. Status: unverified. Target: fine_edge_detail. Weak: True. Validation correlation: 0.131; 95% image-bootstrap interval: [0.036938510835170746, 0.22461178936064244]. BH-FDR q: 0.24181309957352895.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture variability — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_variation. Weak: True. Validation correlation: -0.472; 95% image-bootstrap interval: [-0.5725122481584549, -0.3750487178564071]. BH-FDR q: 0.08146700263023376.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image neighbouring-pixel contrast — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_contrast. Weak: True. Validation correlation: 0.304; 95% image-bootstrap interval: [0.24509230069816113, 0.35060808062553406]. BH-FDR q: 0.08146700263023376.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture homogeneity — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_homogeneity. Weak: True. Validation correlation: -0.286; 95% image-bootstrap interval: [-0.42117857933044434, -0.1507743638008832]. BH-FDR q: 0.14376351370740292.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local intensity entropy — unverified guidance

Group: texture-guided. Status: unverified. Target: texture_entropy. Weak: True. Validation correlation: 0.371; 95% image-bootstrap interval: [0.19890093803405762, 0.5434290170669556]. BH-FDR q: 0.1443168976727654.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image relatively bright image regions — unverified guidance

Group: region-guided. Status: unverified. Target: bright_region. Weak: True. Validation correlation: -0.076; 95% image-bootstrap interval: [-0.15800638496875763, 0.04134790599346161]. BH-FDR q: 0.3651807283100329.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image relatively dark image regions — unverified guidance

Group: region-guided. Status: unverified. Target: dark_region. Weak: True. Validation correlation: 0.257; 95% image-bootstrap interval: [-0.05384613201022148, 0.5688239932060242]. BH-FDR q: 0.3651807283100329.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local intensity deviation — unverified guidance

Group: region-guided. Status: unverified. Target: local_intensity_deviation. Weak: True. Validation correlation: 0.312; 95% image-bootstrap interval: [0.11878533661365509, 0.5060470700263977]. BH-FDR q: 0.19149098572907625.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image threshold-region occupancy — unverified guidance

Group: region-guided. Status: unverified. Target: region_occupancy. Weak: True. Validation correlation: 0.156; 95% image-bootstrap interval: [-0.08907711505889893, 0.3609366431832316]. BH-FDR q: 0.4284389740830168.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture variability — weak unguided association

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.290; 95% image-bootstrap interval: [0.19012888446450235, 0.4161735475063324]. BH-FDR q: 0.11925252846309117.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture response — weak unguided association (learned variant 1)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: -0.426; 95% image-bootstrap interval: [-0.5137579441070557, -0.3578246235847473]. BH-FDR q: 0.07119210064411163.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture homogeneity — weak unguided association

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: -0.292; 95% image-bootstrap interval: [-0.5097425654530525, -0.04301906004548073]. BH-FDR q: 0.24027146911621095.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

## Bird-image local texture response — weak unguided association (learned variant 2)

Group: unguided (guidance absent). Status: unguided_weak. Target: None. Weak: True. Validation correlation: 0.449; 95% image-bootstrap interval: [0.3397718071937561, 0.5710437297821045]. BH-FDR q: 0.10990607738494873.

explicit guidance target plus measured validation association; no index-based rotation of semantic names. Inspect multiple training prototype patches and held-out overlays. Confirm consistency, foreground support, and lack of background shortcuts; do not infer medical or part semantics from correlation.

