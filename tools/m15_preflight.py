"""Read-only M1 integrity, development-target inventory and inference timing; no training."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    if not sys.flags.isolated:
        raise RuntimeError('Invoke python -I tools/m15_preflight.py')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / 'packages'))
    from noah_baseline.runtime import install_source_boundary, assert_local_modules
    install_source_boundary(root)
    import numpy as np
    import torch
    from noah_baseline.models.noah import NOAH
    from noah_baseline.data.schema import CATEGORY_NAMES
    from noah_baseline.m1.training import seed_all

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m1-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.m1_run.resolve()
    output = args.output.resolve()
    namespace = (root / 'artifacts/reduced_pilot/m15').resolve()
    if namespace not in output.parents or output.exists():
        raise RuntimeError('Require a new M1.5 artifact path; overwrite forbidden')
    inventory = json.loads((run / 'artifact_inventory.json').read_text())
    for name, expected in inventory.items():
        path = run / name
        if not path.is_file() or sha(path) != expected or path.stat().st_mode & 0o222:
            raise RuntimeError('M1 immutability/integrity failed: ' + name)
    cohort = json.loads((run / 'index_cohort.json').read_text())
    windows = torch.load(run / 'windows.pt', map_location='cpu', weights_only=False)
    # Only train/validation targets are inspected. Test windows remain quarantined.
    split_by_subject = {row['subject_id']: row['split'] for row in cohort}
    stats = {}
    for split in ('train', 'validation'):
        source_split = 'val' if split == 'validation' else split
        selected = [w for w in windows if split_by_subject[w['subject_id']] == source_split]
        counts = torch.zeros(31, dtype=torch.int64)
        gaps, zero, unknown, numeric, targets = [], 0, 0, 0, []
        lengths = []
        transitions = torch.zeros((31, 31), dtype=torch.int64)
        for window in selected:
            b = window['batch']
            mask = b['loss_mask'][:, 1:] & b['attention_mask'][:, 1:] & b['attention_mask'][:, :-1]
            positions = mask.nonzero().tolist()
            assert len(positions) == len(window['owned_target_ids'])
            targets.extend(window['owned_target_ids'])
            lengths.append(int(b['attention_mask'].sum()))
            counts += torch.bincount(b['category_ids'][:, 1:][mask], minlength=31)
            numeric += int((mask & b['has_value'][:, 1:]).sum())
            for row, pos in positions:
                transitions[int(b['category_ids'][row, pos]), int(b['category_ids'][row, pos + 1])] += 1
                if not b['has_next_gap'][row, pos + 1]:
                    unknown += 1
                else:
                    gap = float(b['delta_next'][row, pos + 1])
                    assert gap >= 0
                    if gap == 0:
                        zero += 1
                    else:
                        gaps.append(gap)
        assert len(set(targets)) == len(targets) == int(counts.sum()) == int(transitions.sum())
        assert len(targets) == unknown + zero + len(gaps)
        stats[split] = dict(
            patients=len(set(w['subject_id'] for w in selected)), windows=len(selected),
            owned_targets=len(targets), category_counts={CATEGORY_NAMES[i]: int(n) for i, n in enumerate(counts) if n},
            category_count_vector=counts.tolist(), numeric_targets=numeric,
            time_unknown=unknown, time_observed_zero=zero, time_observed_positive=len(gaps),
            positive_gap_hours_quantiles=dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'],
                np.quantile(gaps, [0, .25, .5, .75, .95, 1]).tolist())) if gaps else {},
            positive_log1p_gap_mean=float(np.log1p(gaps).mean()) if gaps else None,
            attention_lengths=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
            owned_target_ledger_sha256=hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode()).hexdigest(),
            transition_count_sha256=hashlib.sha256(transitions.numpy().tobytes()).hexdigest())
    train_subjects = {w['subject_id'] for w in windows if split_by_subject[w['subject_id']] == 'train'}
    val_subjects = {w['subject_id'] for w in windows if split_by_subject[w['subject_id']] == 'val'}
    assert train_subjects and val_subjects and not train_subjects & val_subjects
    config = json.loads((run / 'config.json').read_text())
    seed_all(config['seed'])
    model = NOAH(**config['model']).eval()
    train_windows = [w for w in windows if w['subject_id'] in train_subjects]
    times = []
    with torch.inference_mode():
        for w in train_windows[:4]:
            model(**w['batch'], beta=1.)
        for w in train_windows[:64]:
            start = time.perf_counter()
            model(**w['batch'], beta=1.)
            times.append(time.perf_counter() - start)
    assert_local_modules(root)
    report = dict(schema_version=1, purpose='M1.5 proposal only; zero optimizer updates',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        preflight_source_sha256=sha(Path(__file__)),
        m1_run=str(run.relative_to(root)), verified_readonly_m1_files=len(inventory),
        m1_artifact_inventory_sha256=sha(run / 'artifact_inventory.json'),
        pinned_inputs={name: inventory[name] for name in ['windows.pt', 'index_cohort.json', 'patient_splits.csv',
            'source_manifest.json', 'source.tar', 'config.json', 'model.pt', 'experiment_manifest.json']},
        model_parameters=sum(p.numel() for p in model.parameters()),
        module_parameters={name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()},
        development=stats, test_target_inspection=False, optimizer_updates_executed=0,
        inference_timing=dict(threads=torch.get_num_threads(), timed_windows=len(times),
            median_seconds=statistics.median(times), mean_seconds=statistics.mean(times),
            p95_seconds=float(np.quantile(times, .95)), peak_process_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            caveat='CPU inference only; not a measured training-throughput benchmark'),
        environment=dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
