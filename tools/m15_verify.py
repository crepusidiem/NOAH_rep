"""Real-data M1/M1.5 operation equivalence fixture, not a retained training run."""
import json
import os
from pathlib import Path
import sys


def main():
    if not sys.flags.isolated:
        raise RuntimeError('Invoke python -I tools/m15_verify.py')
    root=Path(__file__).resolve().parents[1]
    os.chdir(root);sys.path.insert(0,str(root/'packages'))
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    from noah_baseline.runtime import install_source_boundary,assert_local_modules
    install_source_boundary(root)
    from noah_baseline.m15.controller import load_data,dump
    from noah_baseline.m15.operations import LearningTrainer,evaluate,state,exact
    from noah_baseline.m1.training import Trainer
    from noah_baseline.evaluation.contracts import sha256_file
    config=json.loads((root/'configs/m15.json').read_text())
    parts,ordinals=load_data(root,config)
    identity=dict(train_subjects=[w.subject_id for w in parts['train']],controller='real operation fixture')
    reference=Trainer(config['candidate'],parts['train'],identity)
    expected_trace=reference.update();expected=state(reference)
    actual=LearningTrainer(config['candidate'],parts['train'],identity,5260)
    record=actual.update()
    assert {k:record[k] for k in expected_trace}==expected_trace
    assert exact(expected,state(actual))
    before=state(actual)
    for split in parts:
        evaluate(actual.model,parts[split][:2],ordinals[split][:2],0.)
    assert exact(before,state(actual))
    assert_local_modules(root)
    result=dict(status='PASS',m1_source_and_artifacts='immutable hash/byte match',
        train_targets=2335,validation_targets=436,test_evaluated=False,
        real_reference_operation='one update bitwise equal to M1 including optimizer/RNG',
        evaluation_state_invariance='PASS',physical_fixture_optimizer_calls=2,retained_experiment_updates=0,
        source_sha256=sha256_file(Path(__file__)),resolved_config_sha256=sha256_file(root/'configs/m15.json'))
    dump(root/'artifacts/reduced_pilot/m15/real-operation-fixture.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
