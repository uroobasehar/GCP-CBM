"""Measurable image descriptors, guidance contracts and domain-scoped concept names."""
import torch
from torch.nn import functional as F

ANCHORS = {
    'edge_steepness': ('boundary', 'edge-transition steepness'),
    'edge_density': ('boundary', 'local edge abundance'),
    'edge_coherence': ('boundary', 'directional edge coherence'),
    'fine_edge_detail': ('boundary', 'fine edge detail'),
    'texture_variation': ('texture', 'local texture variability'),
    'texture_contrast': ('texture', 'neighbouring-pixel contrast'),
    'texture_homogeneity': ('texture', 'local texture homogeneity'),
    'texture_entropy': ('texture', 'local intensity entropy'),
    'bright_region': ('region', 'relatively bright image regions'),
    'dark_region': ('region', 'relatively dark image regions'),
    'local_intensity_deviation': ('region', 'local intensity deviation'),
    'region_occupancy': ('region', 'threshold-region occupancy'),
}
KEYS = list(ANCHORS)
LEGACY = {'edge': ('boundary', 'image-edge response'), 'texture': ('texture', 'local texture response'),
          'morphology': ('region', 'threshold-region response'), 'intensity': ('intensity', 'relative image brightness')}
DOMAIN = {'cub': 'Bird-image', 'pets': 'Pet-image', 'brain_mri': 'MRI-image'}


def normalize(x):
    return (x - x.amin((2, 3), keepdim=True)) / (x.amax((2, 3), keepdim=True) - x.amin((2, 3), keepdim=True)).clamp_min(1e-8)


@torch.no_grad()
def descriptor_bank(images):
    # Quantiles, local variance and entropy must stay in FP32 even under CUDA AMP.
    with torch.autocast(device_type=images.device.type, enabled=False):
        return _descriptor_bank_float(images.float())


def _descriptor_bank_float(images):
    """All maps are image-derived proxies on resized RGB, never lesion/part annotations.

    Steepness is gradient magnitude relative to local SD, not clinical margin sharpness.
    Coherence is a structure-tensor measure, not contour closure/irregularity.
    """
    from .core import priors
    x = images.float().clamp(0, 1)
    g = (x * x.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)
    sx = g.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None] / 8
    gx = F.conv2d(F.pad(g, (1, 1, 1, 1), mode='reflect'), sx)
    gy = F.conv2d(F.pad(g, (1, 1, 1, 1), mode='reflect'), sx.transpose(2, 3))
    strength = (gx.square() + gy.square() + 1e-8).sqrt()
    pool = lambda z: F.avg_pool2d(F.pad(z, (4, 4, 4, 4), mode='reflect'), 9, stride=1)
    mean = pool(g); sd = (pool(g.square()) - mean.square()).clamp_min(0).add(1e-8).sqrt()
    xx, yy, xy = pool(gx.square()), pool(gy.square()), pool(gx * gy)
    coherence = ((xx - yy).square() + 4 * xy.square()).sqrt() / (xx + yy + 1e-6)
    threshold = torch.quantile(strength.flatten(2), .75, dim=2)[..., None, None]
    edge_pixels = (strength > threshold).float()
    dx = (g - F.pad(g[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')).square()
    dy = (g - F.pad(g[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')).square()
    difference = (dx + dy) / 2
    entropy = torch.zeros_like(g)
    # Exact local 8-bin histogram proxy, reflect padding avoids zero-padding edge artifacts.
    bins = (g * 8).long().clamp_max(7)
    for b in range(8):
        p = pool((bins == b).float())
        entropy -= p * p.clamp_min(1e-8).log()
    gm, gs = g.mean((2, 3), keepdim=True), g.std((2, 3), keepdim=True, correction=0)
    bright = (g > gm + .45 * gs).float()
    dark = (g < gm - .90 * gs).float()
    laplace = g.new_tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])[None, None]
    maps = [normalize(strength / (sd + .03)), pool(edge_pixels), coherence.clamp(0, 1) * normalize(strength),
            normalize(F.conv2d(F.pad(g, (1, 1, 1, 1), mode='reflect'), laplace).abs()),
            normalize(sd), normalize(pool(difference)), pool(1 / (1 + 16 * difference)),
            (entropy / torch.log(g.new_tensor(8.))).clamp(0, 1),
            pool(bright), pool(dark), normalize((g - mean).abs()), pool((bright + dark).clamp_max(1))]
    return {**priors(x), **dict(zip(KEYS, maps))}


def guidance_contract(config):
    group = config.concepts // 4
    rows = []
    for k in range(config.concepts):
        family = ['boundary', 'texture', 'region', 'unguided'][k // group]
        target = None
        if family != 'unguided':
            j = k // group
            target = KEYS[j * 4 + (k % group) % 4] if config.guidance_mode == 'individual_descriptors_v4' else ['edge', 'texture', 'morphology'][j]
        enabled = family != 'unguided' and getattr(config, {'boundary': 'edge', 'texture': 'texture', 'region': 'morphology'}.get(family, 'edge')) > 0
        rows.append({'concept_index': k, 'family': family, 'target_descriptor': target,
                     'guidance_applied': enabled, 'mode': config.guidance_mode,
                     'replicate': (k % group) // 4 + 1})
    return rows


def readable_name(dataset, descriptor, direction, status, family, replicate=1):
    phrase = {**LEGACY, **ANCHORS}[descriptor][1]
    prefix = 'Reduced ' if direction == 'inverse' and status in ['conflict','unguided_verified'] else ''
    suffix = {'verified': '', 'unverified': ' — unverified guidance', 'conflict': ' — guidance conflict',
              'not_exposed':' — guidance not yet applied', 'exposure_unknown':' — guidance exposure unrecorded',
              'unguided_verified': ' — unguided association', 'unguided_weak': ' — weak unguided association'}[status]
    return f'{DOMAIN[dataset]} {prefix}{phrase}{suffix}' + (f' (replicate {replicate})' if replicate > 1 else '')
