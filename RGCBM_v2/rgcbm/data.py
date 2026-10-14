from pathlib import Path
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

SOURCES = {
    'cub': {'slug': 'wenewone/cub2002011', 'official': 'https://data.caltech.edu/records/65de6-vp158', 'classes': 200, 'description': 'CUB-200-2011; official image-level train/test split; attributes/boxes evaluation only'},
    'pets': {'slug': 'julinmaloof/the-oxfordiiit-pet-dataset', 'official': 'https://www.robots.ox.ac.uk/~vgg/data/pets/', 'classes': 37, 'description': 'Oxford-IIIT Pet; official trainval/test split; trimaps evaluation only'},
    'brain_mri': {'slug': 'masoudnickparvar/brain-tumor-mri-dataset', 'official': 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset', 'classes': 4, 'description': 'Kaggle MRI; supplied image-level split, not verified patient-independent'},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def prepare_data(root, dataset, online=False):
    """Retain and reuse complete source archives; never redownload by default."""
    root = Path(root)
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    archive = data / f'{dataset}.zip'
    target = data / dataset
    marker = target / '.extracted.json'
    if marker.is_file():
        return target
    if not archive.is_file():
        if not online:
            raise FileNotFoundError(f'Missing {archive}. Copy the bundled dataset archive or run online preparation.')
        url = 'https://www.kaggle.com/api/v1/datasets/download/' + SOURCES[dataset]['slug']
        partial = archive.with_suffix('.partial')
        print(f'Downloading {dataset} from the declared Kaggle mirror...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as output:
            if response.headers.get('content-type') != 'application/zip':
                raise RuntimeError('Dataset endpoint did not return ZIP data.')
            shutil.copyfileobj(response, output, 1024 * 1024)
        os.replace(partial, archive)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe archive member path.')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise RuntimeError('Symbolic links in dataset archive are not supported.')
        if source.testzip() is not None:
            raise RuntimeError('Dataset ZIP failed CRC validation.')
        target.mkdir(parents=True, exist_ok=True)
        # This mirror also contains 5.8 GB of unrelated CVPR-2016 text embeddings.
        # RG-CBM uses the original images, split files, attributes, parts and masks only.
        members = [m for m in source.infolist() if not (dataset == 'cub' and m.filename.startswith('cvpr2016_cub/'))]
        source.extractall(target, members=members)
    atomic_json(marker, {'source': SOURCES[dataset], 'archive_sha256': sha256(archive)})
    return target


def _unique_file(root, name):
    matches = list(Path(root).rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one {name}, found {len(matches)} under {root}')
    return matches[0]


def _mapping(path, cast=str):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            key, value = line.split(maxsplit=1)
            result[int(key)] = cast(value)
    return result


def build_manifest(root, dataset, seed=42, validation_fraction=0.1):
    root = Path(root)
    base = prepare_data(root, dataset, online=False)
    rows = []
    def relative(path):
        return Path(path).relative_to(root).as_posix()
    if dataset == 'cub':
        parent = _unique_file(base, 'images.txt').parent
        images = _mapping(parent / 'images.txt')
        labels = _mapping(parent / 'image_class_labels.txt', int)
        split = _mapping(parent / 'train_test_split.txt', int)
        names = _mapping(parent / 'classes.txt')
        boxes = _mapping(parent / 'bounding_boxes.txt', lambda v: list(map(float, v.split())))
        classes = [names[i] for i in sorted(names)]
        for image_id, path in sorted(images.items()):
            rows.append({'path': relative(parent / 'images' / path), 'label': labels[image_id] - 1, 'class_name': names[labels[image_id]], 'split': 'train' if split[image_id] == 1 else 'test', 'image_id': image_id, 'bbox': boxes[image_id]})
        if sum(r['split'] == 'train' for r in rows) != 5994 or sum(r['split'] == 'test' for r in rows) != 5794:
            raise RuntimeError('CUB archive does not match the official 5994/5794 split.')
    elif dataset == 'pets':
        annotations = _unique_file(base, 'trainval.txt').parent
        # Mirrors normally place images next to annotations; discover unambiguously.
        candidate = annotations.parent / 'images'
        if not candidate.is_dir():
            candidate = _unique_file(base, 'Abyssinian_1.jpg').parent
        label_names = {}
        for which, filename in [('train', 'trainval.txt'), ('test', 'test.txt')]:
            for line in (annotations / filename).read_text().splitlines():
                if not line.strip() or line.startswith('#'):
                    continue
                stem, lab, *_ = line.split()
                name = stem.rsplit('_', 1)[0]
                label_names[int(lab)] = name
                row = {'path': relative(candidate / (stem + '.jpg')), 'label': int(lab) - 1, 'class_name': name, 'split': which}
                mask = annotations / 'trimaps' / (stem + '.png')
                if mask.is_file():
                    row['mask'] = relative(mask)
                rows.append(row)
        classes = [label_names[i] for i in sorted(label_names)]
        if sum(r['split'] == 'train' for r in rows) != 3680 or sum(r['split'] == 'test' for r in rows) != 3669:
            raise RuntimeError('Pet archive does not match the official 3680/3669 split.')
    elif dataset == 'brain_mri':
        parents = list(base.rglob('Training'))
        if len(parents) != 1:
            raise RuntimeError('Expected one Training folder.')
        parent = parents[0].parent
        classes = sorted(p.name for p in (parent / 'Training').iterdir() if p.is_dir())
        if classes != ['glioma', 'meningioma', 'notumor', 'pituitary']:
            raise RuntimeError('Unexpected MRI classes.')
        for which, folder in [('train', 'Training'), ('test', 'Testing')]:
            for lab, name in enumerate(classes):
                for path in sorted((parent / folder / name).rglob('*')):
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        rows.append({'path': relative(path), 'label': lab, 'class_name': name, 'split': which})
    else:
        raise ValueError(dataset)
    if len(classes) != SOURCES[dataset]['classes']:
        raise RuntimeError('Unexpected class count.')
    missing = [r['path'] for r in rows if not (root / r['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing dataset images, e.g. {missing[:3]}')
    original_train = [i for i, r in enumerate(rows) if r['split'] == 'train']
    if dataset == 'brain_mri':
        # Exact duplicate images must not inflate validation/model-selection estimates.
        groups = [sha256(root / rows[i]['path']) for i in original_train]
        splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / validation_fraction)), shuffle=True, random_state=seed)
        _, local_validation = next(splitter.split(original_train, [rows[i]['label'] for i in original_train], groups))
        validation = [original_train[j] for j in local_validation]
    else:
        _, validation = train_test_split(original_train, test_size=validation_fraction, random_state=seed, stratify=[rows[i]['label'] for i in original_train])
    for i in validation:
        rows[i]['split'] = 'validation'
    marker = json.loads((base / '.extracted.json').read_text())
    manifest = {'dataset': dataset, 'source': marker, 'seed': seed, 'validation_fraction': validation_fraction, 'validation_policy': 'stratified exact-hash groups (approximately requested fraction)' if dataset == 'brain_mri' else 'stratified image split within official training pool', 'classes': classes, 'rows': rows}
    manifest['manifest_hash'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest['counts'] = {s: sum(r['split'] == s for r in rows) for s in ['train', 'validation', 'test']}
    atomic_json(root / 'data' / f'{dataset}_manifest_seed{seed}.json', manifest)
    return manifest


class ImageDataset(Dataset):
    def __init__(self, root, manifest, split, config, augment=False):
        self.root, self.config = Path(root), config
        self.rows = [r for r in manifest['rows'] if r['split'] == split]
        self.augment, self.epoch = augment, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as source:
            image = source.convert('RGB')
        image = TF.resize(image, [self.config.image_size] * 2, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.augment:
            gen = torch.Generator().manual_seed(self.config.seed + self.epoch * 1000003 + index)
            random = torch.rand(5, generator=gen).tolist()
            angle = (random[0] - 0.5) * 18
            translation = [int((random[1] - 0.5) * 0.04 * self.config.image_size), int((random[2] - 0.5) * 0.04 * self.config.image_size)]
            image = TF.affine(image, angle, translation, 0.96 + 0.08 * random[3], 0, interpolation=InterpolationMode.BILINEAR)
            image = TF.adjust_contrast(image, 0.94 + 0.12 * random[4])
            if self.config.dataset != 'brain_mri' and random[1] < 0.5:
                image = TF.hflip(image)
        return TF.to_tensor(image), int(row['label']), index


def loader(dataset, shuffle=False, epoch=0):
    dataset.epoch = epoch
    config = dataset.config
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, generator=generator, num_workers=config.workers, pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(dataset) % config.batch_size == 1)


def leakage_audit(root, manifest):
    """Report exact-byte duplicates; neither proves nor repairs patient independence."""
    seen, overlaps = {}, []
    for row in manifest['rows']:
        digest = sha256(Path(root) / row['path'])
        if digest in seen and seen[digest]['split'] != row['split']:
            overlaps.append({'sha256': digest, 'first': seen[digest]['path'], 'first_split': seen[digest]['split'], 'second': row['path'], 'second_split': row['split']})
        else:
            seen[digest] = row
    result = {'dataset': manifest['dataset'], 'cross_split_exact_duplicates': overlaps, 'patient_independence': 'not established' if manifest['dataset'] == 'brain_mri' else 'not applicable', 'note': 'Do not change official test membership silently. Resolve leakage before publishing affected estimates.'}
    atomic_json(Path(root) / 'data' / f'{manifest["dataset"]}_leakage_audit.json', result)
    return result
