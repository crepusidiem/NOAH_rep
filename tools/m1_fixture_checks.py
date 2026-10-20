"""Run actual M1 fixture cases and bind their results to the package/test source."""
import argparse
import json
from pathlib import Path
import re

from noah_baseline.evaluation.contracts import ContractError, sha256_file
from noah_baseline.runtime import ArtifactNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=ArtifactNamespace(root,'reduced_pilot').owned(root/args.output)
    if output.exists():raise ContractError('Fixture evidence overwrite forbidden')
    import pytest
    class Recorder:
        def __init__(self):self.results={};self.nodes=[]
        def pytest_runtest_logreport(self,report):
            match=re.search('test_m1_(\\d{2})_',report.nodeid)
            if match and (report.when=='call' or report.failed):
                name='M1-'+match.group(1)
                self.results[name]='FAIL' if report.failed or report.skipped else 'PASS'
                self.nodes.append(dict(node=report.nodeid,phase=report.when,outcome=report.outcome))
    recorder=Recorder()
    code=pytest.main(['tests/test_m1.py','-q','-o','cache_dir=/tmp/noah_m1_fixture_cache'],plugins=[recorder])
    files={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'packages').rglob('*.py'))}
    files.update({str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'tests').glob('*.py'))})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(exit_code=int(code),cases=recorder.results,nodes=recorder.nodes,source_files=files),indent=2)+'\n')
    if code or len(recorder.results)!=8:raise ContractError('M1 fixture suite failed or acceptance case missing')
    return 0
