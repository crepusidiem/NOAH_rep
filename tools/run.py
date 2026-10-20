"""Run isolated checks and explicitly authorized, CPU-bounded M1 integration."""
from pathlib import Path
import json
import os
import sys


def main():
    if not sys.flags.isolated:
        print('BLOCKED: invoke python -I tools/run.py', file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root/'packages'))
    from noah_baseline.runtime import install_source_boundary, assert_local_modules
    install_source_boundary(root)
    command = sys.argv[1] if len(sys.argv)>1 else 'status'
    if command == 'test':
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        import pytest
        result = pytest.main(['tests', *sys.argv[2:]])
    elif command in {'m1','m1-check'}:
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
        if command == 'm1':
            from noah_baseline.m1.pipeline import main as action
        else:
            import runpy
            action=runpy.run_path(str(root/'tools/m1_fixture_checks.py'))['main']
        sys.argv=[sys.argv[0],*sys.argv[2:]]
        result=action()
    elif command == 'release':
        from noah_baseline.evaluation.cli import main as release
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        result = release()
    elif command == 'status':
        print(json.dumps({'package_root':str(root/'packages/noah_baseline'),
                          'clinical_pipeline':'M1_BOUNDED_IMPLEMENTED', 'training':'M1_CPU_MAX16_STEPS',
                          'alpha_sweep':'HOLD', 'methodology':'M1 authorized scope; later phases remain gated'}))
        result = 0
    else:
        print('BLOCKED: baseline supports test, release and status only; no training or alpha launcher',file=sys.stderr)
        result = 2
    assert_local_modules(root)
    return int(result)


if __name__ == '__main__':
    raise SystemExit(main())
