from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights


@dataclass
class Config:
    dataset: str = 'cub'
    seed: int = 42
    concepts: int = 16
    projection: int = 128
    image_size: int = 240
    batch_size: int = 24
    workers: int = 0  # Safe in Windows/Jupyter; increase only in a guarded script.
    head_epochs: int = 20
    finetune_epochs: int = 20
    finetune_blocks: int = 3
    head_lr: float = 4e-4
    finetune_lr: float = 8e-6
    finetune_head_lr: float | None = None  # None preserves the original shared LR.
    concept_normalization: str = 'none'  # Legacy checkpoint-compatible default.
    validation_fraction: float = 0.1
    dropout: float = 0.10
    edge: float = 0.18
    texture: float = 0.10
    morphology: float = 0.08
    diversity: float = 0.02
    disentangle: float = 0.01
    activation_l1: float = 5e-5
    stability: float = 0.02
    classifier_l1: float = 5e-6
    classification_only_epochs: int = 3
    ramp_epochs: int = 5
    checkpoint_every_batches: int = 50
    checkpoint_every_epochs: int = 5
    keep_epoch_checkpoints: int = 3  # Per stage; 0 keeps every periodic snapshot.
    show_progress: bool = True
    patience: int = 8
    mixed_precision: bool = True
    pretrained: bool = True
    max_train_batches: int | None = None
    max_validation_batches: int | None = None

    def __post_init__(self):
        if self.concepts < 4 or self.concepts % 4:
            raise ValueError('Concept count must be a positive multiple of four.')
        if self.batch_size < 2:
            raise ValueError('Training batches need at least two images for concept decorrelation.')
        if self.head_epochs < 1 or self.finetune_epochs < 0:
            raise ValueError('Use at least one head epoch and nonnegative fine-tuning epochs.')
        if self.workers < 0 or not 0 < self.validation_fraction < 1:
            raise ValueError('Invalid loader/split configuration.')
        if self.concept_normalization not in ['none', 'running_standardize']:
            raise ValueError('Unknown concept normalization.')
        if min(self.head_lr, self.finetune_lr) <= 0 or (self.finetune_head_lr is not None and self.finetune_head_lr <= 0):
            raise ValueError('Learning rates must be positive.')
        if self.checkpoint_every_batches < 1 or self.checkpoint_every_epochs < 1 or self.keep_epoch_checkpoints < 0:
            raise ValueError('Invalid checkpoint interval/retention.')

    def to_dict(self):
        return asdict(self)


def training_config(profile='calibrated_v3', **overrides):
    if profile == 'calibrated_v3':
        values = dict(head_epochs=8, finetune_epochs=30, head_lr=2e-3, finetune_lr=2e-5,
                      finetune_head_lr=1e-3, concept_normalization='running_standardize')
    elif profile == 'legacy':
        values = {}
    else:
        raise ValueError('Use calibrated_v3 or legacy.')
    values.update(overrides)
    return Config(**values)


def normalize_map(x):
    low = x.amin((2, 3), keepdim=True)
    high = x.amax((2, 3), keepdim=True)
    return (x - low) / (high - low).clamp_min(1e-8)


def priors(images):
    """Image-derived weak priors on unnormalized RGB [0,1]; never segmentation labels."""
    x = images.float().clamp(0, 1)
    gray = (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    sx = gray.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    sy = sx.transpose(2, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    sobel = (F.conv2d(padded, sx).square() + F.conv2d(padded, sy).square() + 1e-8).sqrt()
    laplace = gray.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    edge = normalize_map(0.7 * normalize_map(sobel) + 0.3 * normalize_map(F.conv2d(padded, laplace).abs()))
    mean = F.avg_pool2d(gray, 9, stride=1, padding=4)
    var = (F.avg_pool2d(gray.square(), 9, stride=1, padding=4) - mean.square()).clamp_min(0)
    local_sd = normalize_map((var + 1e-8).sqrt())
    right = F.pad(gray[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
    down = F.pad(gray[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
    diff = 0.5 * ((gray - right).square() + (gray - down).square())
    contrast = normalize_map(F.avg_pool2d(diff, 9, stride=1, padding=4))
    homogeneity = normalize_map(F.avg_pool2d(1 / (1 + 16 * diff), 9, stride=1, padding=4))
    texture = normalize_map(0.75 * local_sd + 0.25 * normalize_map(0.75 * contrast + 0.25 * (1 - homogeneity)))
    gm, gs = gray.mean((2, 3), keepdim=True), gray.std((2, 3), keepdim=True, correction=0)
    pseudo = (gray > gm + 0.45 * gs).float() + 0.35 * (gray < gm - 0.90 * gs).float()
    morphology = normalize_map(F.avg_pool2d(pseudo, 7, stride=1, padding=3))
    return {'edge': edge.detach(), 'texture': texture.detach(), 'morphology': morphology.detach(), 'intensity': normalize_map(gray).detach()}


class RGCBM(nn.Module):
    def __init__(self, num_classes, config: Config, asset_dir=None):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        backbone = efficientnet_b1(weights=None)
        if config.pretrained:
            weights_path = Path(asset_dir or 'assets') / 'efficientnet_b1_imagenet.pth'
            if not weights_path.is_file():
                raise FileNotFoundError(f'Missing offline backbone asset: {weights_path}. Run prepare_assets.py online first.')
            backbone.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        self.backbone = backbone.features
        self.projection = nn.Sequential(nn.Conv2d(1280, config.projection, 1, bias=False), nn.BatchNorm2d(config.projection), nn.SiLU())
        self.prototypes = nn.Parameter(torch.randn(config.concepts, config.projection) * 0.02)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.concepts, num_classes)
        if config.concept_normalization == 'running_standardize':
            self.concept_norm = nn.BatchNorm1d(config.concepts, affine=False)
        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.stage = 'head'
        self.set_stage('head')

    def set_stage(self, stage):
        self.stage = stage
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if stage == 'finetune':
            for block in list(self.backbone.children())[-self.config.finetune_blocks:]:
                for p in block.parameters():
                    p.requires_grad_(True)
        elif stage != 'head':
            raise ValueError(stage)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for p in module.parameters():
                    p.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'backbone'):
            if self.stage == 'head':
                self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def concepts_from(self, images):
        z = self.projection(self.backbone((images - self.image_mean) / self.image_std))
        features = F.normalize(z, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        maps = torch.sigmoid(4 * torch.einsum('bdhw,kd->bkhw', features, p))
        concepts = maps.mean((2, 3))
        if self.config.concept_normalization == 'running_standardize':
            # Training-only running statistics; fixed at evaluation, no class bypass.
            concepts = torch.sigmoid(self.concept_norm(concepts))
        return maps, concepts

    def forward(self, images):
        maps, concepts = self.concepts_from(images)
        logits = self.classifier(self.dropout(concepts))
        return {'maps': maps, 'concepts': concepts, 'logits': logits, 'probs': logits.softmax(1)}

    def predict_from_concepts(self, concepts):
        logits = self.classifier(concepts)
        return logits, logits.softmax(1)


class BackboneClassifier(RGCBM):
    """Matched EfficientNet-B1 + pooled-feature linear baseline, with no concept bottleneck."""
    def __init__(self, num_classes, config, asset_dir=None):
        super().__init__(num_classes, config, asset_dir)
        self.projection = nn.Identity()
        self.register_parameter('prototypes', None)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, images):
        features = self.backbone((images - self.image_mean) / self.image_std).mean((2, 3))
        logits = self.classifier(self.dropout(features))
        return {'logits': logits, 'probs': logits.softmax(1)}


def loss_terms(model, images, targets, output, class_weights, scale):
    """Same loss families as original RG-CBM; consistent inference-mode stability views."""
    cfg = model.config
    maps, c = output['maps'].float(), output['concepts'].float()
    descriptor = priors(images)
    group = cfg.concepts // 4
    losses = {'classification': F.cross_entropy(output['logits'].float(), targets, weight=class_weights)}
    for i, name in enumerate(['edge', 'texture', 'morphology']):
        target_map = F.interpolate(descriptor[name], size=maps.shape[-2:], mode='bilinear', align_corners=False)
        losses[name] = F.mse_loss(maps[:, i * group:(i + 1) * group].mean(1, keepdim=True), target_map)
    p = F.normalize(model.prototypes.float(), dim=1)
    off_diag = 1 - torch.eye(cfg.concepts, device=p.device)
    losses['diversity'] = ((p @ p.T) * off_diag).square().mean()
    centered = c - c.mean(0, keepdim=True)
    cov = centered.T @ centered / max(c.shape[0] - 1, 1)
    std = (cov.diag() + 1e-6).sqrt()
    corr = cov / (std[:, None] * std[None, :] + 1e-6)
    losses['disentangle'] = (corr * off_diag).square().mean()
    losses['activation_l1'] = c.abs().mean()
    losses['classifier_l1'] = model.classifier.weight.abs().sum()
    losses['stability'] = c.new_zeros(())
    if cfg.stability and scale:
        mode = model.training
        model.eval()
        # Both sides retain gradients through the same inference path.
        _, base = model.concepts_from(images)
        _, changed = model.concepts_from((images * 1.02 + 3 / 255).clamp(0, 1))
        model.train(mode)
        losses['stability'] = F.mse_loss(base.float(), changed.float())
    auxiliary = sum(getattr(cfg, name) * losses[name] for name in ['edge', 'texture', 'morphology', 'diversity', 'disentangle', 'activation_l1', 'stability'])
    total = losses['classification'] + scale * auxiliary + cfg.classifier_l1 * losses['classifier_l1']
    return total, losses
