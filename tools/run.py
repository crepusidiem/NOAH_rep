"""Run baseline checks through an isolated interpreter; never launch training."""
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
    elif command == 'release':
        from noah_baseline.evaluation.cli import main as release
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        result = release()
    elif command == 'status':
        print(json.dumps({'package_root':str(root/'packages/noah_baseline'),
                          'clinical_pipeline':'NOT_IMPLEMENTED', 'training':'NOT_LAUNCHED',
                          'alpha_sweep':'HOLD', 'methodology':'M01-M12 remain pending'}))
        result = 0
    else:
        print('BLOCKED: baseline supports test, release and status only; no training or alpha launcher',file=sys.stderr)
        result = 2
    assert_local_modules(root)
    return int(result)


if __name__ == '__main__':
    raise SystemExit(main())
