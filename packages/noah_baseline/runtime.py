"""Import and artifact boundaries for the independently tested development tree."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
from pathlib import Path
import re
import sys


class IsolationError(RuntimeError):
    """A runtime attempted to use another source or artifact namespace."""


def inside(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


class SourceBoundary(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path):
        self.root = root.resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'src', 'scripts'}:
            raise IsolationError(f'Legacy module namespace is disabled: {fullname}')
        if fullname == 'noah_baseline' or fullname.startswith('noah_baseline.'):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or not spec.origin or not inside(Path(spec.origin), self.root/'packages/noah_baseline'):
                raise IsolationError(f'Package origin is outside this worktree: {fullname}')
        return None


def install_source_boundary(root: Path) -> None:
    if not sys.flags.isolated:
        raise IsolationError('Use python -I tools/run.py; ambient Python paths are not accepted')
    if any(name == 'src' or name.startswith('src.') for name in sys.modules):
        raise IsolationError('Legacy modules were loaded before the source boundary')
    sys.meta_path.insert(0, SourceBoundary(root))


def assert_local_modules(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name == 'src' or name.startswith('src.'):
            raise IsolationError('Legacy module loaded: '+name)
        if name == 'noah_baseline' or name.startswith('noah_baseline.'):
            origin = getattr(module, '__file__', None)
            if not origin or not inside(Path(origin), root/'packages/noah_baseline'):
                raise IsolationError('Nonlocal package origin: '+name)


class ArtifactNamespace:
    """New outputs are worktree-local; legacy caches have no fallback loader."""
    def __init__(self, root: Path, profile: str):
        if profile not in {'paper_reproduction', 'strict_prospective', 'reduced_pilot'}:
            raise IsolationError('An explicit output profile is required')
        self.worktree = root.resolve()
        self.root = root/'artifacts'/profile
        if not inside(self.root, self.worktree):
            raise IsolationError('Artifact namespace escapes the clean worktree')

    def owned(self, path: Path) -> Path:
        if not inside(path, self.root):
            raise IsolationError('Legacy, external or cross-profile artifact path rejected')
        return path

    def cache_path(self, *, recipe_sha256: str, dataset_sha256: str,
                   encoder_checkpoint_sha256: str, item_sha256: str) -> Path:
        hashes = (recipe_sha256, dataset_sha256, encoder_checkpoint_sha256, item_sha256)
        if any(not re.fullmatch('[0-9a-f]{64}', value) for value in hashes):
            raise IsolationError('Cache requires complete recipe, dataset, checkpoint and content identities')
        path = self.root/'cache'/recipe_sha256/dataset_sha256/encoder_checkpoint_sha256/(item_sha256+'.bin')
        return self.owned(path)
