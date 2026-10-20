"""Adversarial source/cache isolation and namespace migration checks."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from noah_baseline.runtime import ArtifactNamespace, IsolationError, assert_local_modules


ROOT = Path(__file__).resolve().parents[1]


def test_no_legacy_import_or_ambient_pythonpath(tmp_path):
    for name in ('src', 'src.models.noah', 'scripts.train_pilot'):
        with pytest.raises(IsolationError):
            importlib.import_module(name)
    rogue = tmp_path/'noah_baseline'
    rogue.mkdir()
    (rogue/'__init__.py').write_text("raise RuntimeError('ambient package imported')")
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable,'-I',str(ROOT/'tools/run.py'),'status'],
                            cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['package_root'] == str(ROOT/'packages/noah_baseline')
    assert_local_modules(ROOT)


def test_nonisolated_launcher_and_expensive_commands_are_blocked():
    for args in ([], ['-I']):
        result = subprocess.run([sys.executable,*args,str(ROOT/'tools/run.py'),'train'],capture_output=True,text=True)
        assert result.returncode == 2
        assert 'BLOCKED' in result.stderr


def test_no_external_cache_or_cross_profile_fallback(tmp_path):
    namespace = ArtifactNamespace(tmp_path,'reduced_pilot')
    external = tmp_path.parent/'legacy_cache'
    external.mkdir(exist_ok=True)
    namespace.root.mkdir(parents=True)
    (namespace.root/'cache').symlink_to(external,target_is_directory=True)
    with pytest.raises(IsolationError):
        namespace.cache_path(recipe_sha256='a'*64,dataset_sha256='b'*64,
                             encoder_checkpoint_sha256='c'*64,item_sha256='d'*64)
    with pytest.raises(IsolationError):
        namespace.owned(tmp_path/'artifacts/paper_reproduction/model.pt')
    with pytest.raises(IsolationError):
        namespace.cache_path(recipe_sha256='unknown',dataset_sha256='b'*64,
                             encoder_checkpoint_sha256='c'*64,item_sha256='d'*64)


def test_hashes_and_imports_match_reviewed_migration():
    ledger=json.loads((ROOT/'migration_inventory.json').read_text())
    for item in ledger:
        path=ROOT/item['clean_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['clean_sha256']
        if item['clean_path'].startswith('packages/'):
            normalized=path.read_text().replace('noah_baseline.','src.')
            assert hashlib.sha256(normalized.encode()).hexdigest()==item['legacy_sha256']
    for path in (ROOT/'packages').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith(('src.','scripts.'))


def test_profiles_are_templates_with_no_methodological_defaults():
    for name in ('paper_reproduction','strict_prospective','reduced_pilot'):
        profile=json.loads((ROOT/'configs/profiles'/f'{name}.json').read_text())
        assert profile['status']=='PENDING_METHOD_APPROVAL'
        assert profile['execution_enabled'] is False
        assert profile['decisions']=={}
    pilot=json.loads((ROOT/'configs/profiles/reduced_pilot.json').read_text())
    assert pilot['evaluation_mode'] is None


def test_migrated_candidate_parameter_inventory_without_allocating_weights():
    import torch
    from noah_baseline.models.noah import NOAH
    with torch.device('meta'):
        model=NOAH(d_model=768,d_latent=384,n_layers=12,n_heads=12,head_dim=64,ffn_ratio=8)
    counts={name:sum(p.numel() for p in getattr(model,name).parameters())
            for name in ('projector','backbone','prior','posterior','decoder')}
    assert counts==dict(projector=2378496,backbone=141697536,prior=1181184,
                        posterior=1771008,decoder=1929620)
    assert sum(counts.values())==148957844
    assert sum(counts.values())!=150322970
