from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')
blue, teal, gray = '#edf3fa', '#e9f4ef', '#f1f2f4'
def box(x, y, w, h, text, color=blue, fontsize=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12,rounding_size=0.10', linewidth=1.1, edgecolor='#253548', facecolor=color)
    ax.add_patch(patch); ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color='#162333', linespacing=1.5)
def arrow(start, end, dashed=False, color='#253548'):
    ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>', 'lw': 1.4, 'linestyle': '--' if dashed else '-', 'color': color, 'mutation_scale': 13})

ax.text(.15, 8.65, 'RG-CBM: descriptor-grounded concept bottleneck for visual classification', fontsize=18, weight='bold', color='#162333')
ax.text(.15, 8.2, 'Shared architecture; separate trained model per dataset. No manually annotated concepts or masks in the training objective.', fontsize=11, color='#465568')
box(.15, 6.25, 2.5, 1.35, 'Images\nCUB · Oxford Pet · MRI\nRGB, 240 × 240')
box(3.15, 6.25, 2.3, 1.35, 'EfficientNet-B1\nImageNet initialization\nSpatial feature map')
box(5.95, 6.25, 2.45, 1.35, '1 × 1 projection\nBN + SiLU, d = 128\nL2-normalized features')
box(8.9, 6.25, 2.55, 1.35, 'Learnable prototypes\nCosine → sigmoid(4s)\nK spatial concept maps')
box(12, 6.25, 3.5, 1.35, 'Global average pooling\nK = 16: 4 + 4 + 4 + 4\nSparse linear classifier → class')
for a, b in [(2.7, 3.1), (5.5, 5.9), (8.45, 8.85), (11.5, 11.95)]: arrow((a, 6.93), (b, 6.93))
box(.15, 3.9, 3.15, 1.35, 'Image-derived weak priors\nEdge · local texture · region\nIntensity for grounding only', teal)
box(3.9, 3.9, 4.25, 1.35, 'Training objective\nClass CE + group alignment + diversity\nDecorrelation + sparsity + stability', teal)
box(8.9, 3.9, 2.55, 1.35, 'Two-stage optimization\nFrozen backbone →\nlast blocks fine-tuned', teal)
box(12, 3.9, 3.5, 1.35, 'Recoverable progress\nStage · epoch · batch · optimizer\nScheduler · scaler · RNG', gray)
arrow((1.4, 6.2), (1.4, 5.3), dashed=True)
arrow((3.35, 4.57), (3.85, 4.57), dashed=True)
arrow((8.2, 4.57), (8.85, 4.57), dashed=True)
arrow((10.18, 5.3), (10.18, 6.2), dashed=True)
arrow((11.5, 4.57), (11.95, 4.57), dashed=True)
ax.text(.15, 3.3, 'Dashed arrows: training only. Validation selects checkpoints and descriptor names; test data never select or rename concepts.', fontsize=10, color='#465568')
box(.15, 1.25, 4.2, 1.4, 'Validation descriptor grounding\nSigned spatial correlations + uncertainty\nWeak associations remain explicitly weak', gray)
box(4.95, 1.25, 4.2, 1.4, 'Exact model evidence\nClass logit: bias + Σ contribution\nPredicted-versus-runner-up margin', blue)
box(9.75, 1.25, 2.6, 1.4, 'Explanation verbalizer\nOffline Qwen / template\nOptional online Gemini', blue)
box(12.95, 1.25, 2.55, 1.4, 'Output review\nJSON + sign checks\nHuman review of prose', gray)
arrow((4.4, 1.95), (4.9, 1.95))
arrow((9.2, 1.95), (9.7, 1.95))
arrow((12.4, 1.95), (12.9, 1.95))
ax.plot([15.65, 15.85, 15.85, 7.05], [6.93, 6.93, 2.97, 2.97], color='#253548', lw=1.2)
arrow((7.05, 2.97), (7.05, 2.72))
ax.text(.15, .65, 'Independent evaluation: accuracy / macro-F1 / calibration · concept ablation vs random · stability · held-out alignment · CUB boxes / Pet trimaps', fontsize=10, color='#465568')
ax.text(.15, .2, 'Descriptor association is not a part annotation, tumor mask, clinical finding, or causal claim about the image. LLM output never feeds the classifier.', fontsize=10, color='#465568')
fig.tight_layout()
for extension in ['svg', 'pdf', 'png']:
    fig.savefig(root / f'methodology_diagram.{extension}', dpi=200, bbox_inches='tight')
plt.close(fig)
