"""Read-only audit of the bundled Windows lock; no resolver or network requests.

Run with --installed on the Windows target to additionally verify its actual prefix.
"""
from pathlib import Path, PurePosixPath
from collections import defaultdict
from importlib import metadata
import argparse
import email
import fnmatch
import hashlib
import json
import os
import platform
import struct
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def windows_tags():
    return set(cpython_tags((3, 11), abis=['cp311'], platforms=['win_amd64'])) | set(compatible_tags((3, 11), interpreter='cp311', platforms=['win_amd64']))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(root, installed=False):
    record = json.loads((root / 'wheel_validation.json').read_text())
    packages = {p['name']: p for p in record['packages']}
    runtime = json.loads((root / 'conda_manifest.json').read_text())
    conda = {p['name']: p for p in runtime['packages']}
    target = dict(record['target'])
    target['implementation_version'] = target['python_full_version'] = conda['python']['version']
    tags = windows_tags()
    # PyArrow 26's NumPy >=2 requirement is enforced in native initialization,
    # but omitted from its wheel's Requires-Dist metadata.
    if Version(packages['pyarrow']['version']) >= Version('26'):
        require(Version(packages['numpy']['version']) >= Version('2'), 'PyArrow >=26 requires NumPy >=2 at import time, despite missing wheel metadata.')
    lock = {}
    for line in (root / 'lock-win-py311.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        text, digest = line.split(' --hash=sha256:')
        req = Requirement(text)
        name = canonicalize_name(req.name)
        require(name not in lock, 'Duplicate lock entry: ' + name)
        lock[name] = (req, digest)
    require(set(lock) == set(packages), 'Lock and wheel report contain different packages.')
    active = 0
    dlls, destinations = defaultdict(list), defaultdict(list)
    for name, package in packages.items():
        path = root / 'wheelhouse' / package['wheel']
        require(sha(path) == package['sha256'] == lock[name][1], 'Wheel hash mismatch: ' + name)
        wheel_name, version, _, wheel_tags = parse_wheel_filename(path.name)
        require(canonicalize_name(wheel_name) == name and str(version) == package['version'], 'Wheel identity mismatch: ' + name)
        require(lock[name][0].specifier.contains(version, prereleases=True), 'Lock version mismatch: ' + name)
        require(bool(tags.intersection(wheel_tags)), 'Wrong target ABI/platform: ' + path.name)
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, 'Wheel CRC failed: ' + path.name)
            info = email.message_from_bytes(archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))))
            require(not info['Requires-Python'] or SpecifierSet(info['Requires-Python']).contains(target['python_full_version']), 'Wrong Python version: ' + name)
            for text in info.get_all('Requires-Dist') or []:
                req = Requirement(text)
                if req.marker and not any(req.marker.evaluate({**target, 'extra': extra}) for extra in set(package['extras_required']) | {''}):
                    continue
                child = packages.get(canonicalize_name(req.name))
                require(child is not None and req.specifier.contains(child['version'], prereleases=True), f'Incompatible dependency: {name}: {text}')
                require(set(req.extras).issubset(child['extras_required']), f'Missing extras: {name}: {text}')
                active += 1
            for item in archive.infolist():
                if item.is_dir():
                    continue
                parts = PurePosixPath(item.filename).parts
                if parts[0].endswith('.data'):
                    location = '/'.join((['site-packages'] if parts[1] in ['purelib', 'platlib'] else [parts[1]]) + list(parts[2:]))
                else:
                    location = 'site-packages/' + item.filename
                destinations[location.lower()].append((path, item.filename))
                if item.filename.lower().endswith('.dll'):
                    dlls[parts[-1].lower()].append((path, item.filename))
    for line in (root / 'requirements-win-py311.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        req = Requirement(line)
        if not req.marker or req.marker.evaluate(target):
            require(req.specifier.contains(packages[canonicalize_name(req.name)]['version'], prereleases=True), 'Root requirement mismatch: ' + line)
    conda_dependencies = 0
    for package in conda.values():
        require(sha(root / 'conda-packages' / package['fn']) == package['sha256'], 'Conda hash mismatch: ' + package['name'])
        for text in package['depends'] + package.get('constrains', []):
            pieces = text.split()
            name = pieces[0]
            if name not in conda and text in package.get('constrains', []):
                continue  # Conda constraints are conditional, not dependencies.
            require(name in conda, 'Missing conda dependency: ' + text)
            if len(pieces) > 1:
                expression = pieces[1]
                alternatives = expression.split('|')
                require(any(SpecifierSet(','.join(term if term[0] in '<>=!~' else '==' + term for term in alt.split(','))).contains(conda[name]['version'], prereleases=True) for alt in alternatives), 'Conda version conflict: ' + text)
            if len(pieces) > 2:
                require(fnmatch.fnmatch(conda[name]['build'], pieces[2]), 'Conda build conflict: ' + text)
            conda_dependencies += 1
    def duplicate_records(inventory, enforce):
        output = []
        for key, entries in inventory.items():
            if len(entries) < 2:
                continue
            values = []
            for wheel, member in entries:
                with zipfile.ZipFile(wheel) as archive:
                    digest = hashlib.sha256(archive.read(member)).hexdigest()
                values.append({'wheel': wheel.name, 'member': member, 'sha256': digest})
            identical = len({v['sha256'] for v in values}) == 1
            if enforce:
                require(identical, 'Conflicting installed file: ' + key)
            output.append({'name': key, 'identical': identical, 'files': values})
        return output
    result = {'status': 'passed', 'scope': 'Windows artifacts checked statically; native DLL loading/GPU execution require target acceptance.', 'target': target, 'wheels': len(packages), 'active_wheel_dependencies': active, 'conda_packages': len(conda), 'conda_dependency_constraints': conda_dependencies, 'checks': ['wheel/conda SHA-256', 'wheel CRC', 'CPython ABI/platform tags', 'Requires-Python', 'root/transitive requirements and extras', 'conda dependencies/builds', 'installed destination collisions'], 'duplicate_installed_files': duplicate_records(destinations, True), 'duplicate_dll_basenames': duplicate_records(dlls, False), 'dll_count': sum(map(len, dlls.values()))}
    if installed:
        require(os.name == 'nt' and struct.calcsize('P') == 8 and sys.version_info[:2] == (3, 11), '--installed requires the target 64-bit Windows CPython 3.11 environment.')
        require(platform.python_version() == conda['python']['version'], 'Python runtime differs from the bundled conda manifest.')
        prefix = Path(sys.prefix).resolve()
        seen = defaultdict(list)
        for dist in metadata.distributions():
            seen[canonicalize_name(dist.metadata['Name'])].append(dist)
        for name, package in packages.items():
            require(len(seen[name]) == 1, 'Missing or duplicate installed distribution: ' + name)
            dist = seen[name][0]
            require(dist.version == package['version'], 'Installed version differs from lock: ' + name)
            require(Path(dist.locate_file('')).resolve().is_relative_to(prefix), 'Package leaked from outside the target prefix: ' + name)
        result['installed'] = {'status': 'passed', 'prefix': str(prefix), 'packages': len(packages), 'python': platform.python_version()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output or root / ('installed_dependency_audit.json' if args.installed else 'dependency_audit.json')
    try:
        result = audit(root, args.installed)
    except Exception as exc:
        output.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2), encoding='utf-8')
        raise
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Dependency audit passed: {result['wheels']} wheels, {result['active_wheel_dependencies']} active requirements, {result['conda_packages']} conda packages. Report: {output}")


if __name__ == '__main__':
    main()
