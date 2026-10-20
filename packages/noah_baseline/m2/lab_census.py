"""Supplement the per-file-limited census with one explicitly90s-bounded read-only query."""
import json
from pathlib import Path
import threading
import time
import duckdb
from noah_baseline.m2.inventory import dump,sha


def lab_census(root, *, out=None, source=None):
    config=json.loads((root/'configs/m2.json').read_text());a=source or root/'artifacts/reduced_pilot/m2/m2-a-20261020-01'
    out=out or root/'artifacts/reduced_pilot/m2/m2-lab-census-20261020-01';out.mkdir(exist_ok=False)
    path=Path(config['roots']['hosp'])/'labevents.csv'
    c=duckdb.connect(':memory:');c.execute('SET threads=2');c.execute("SET memory_limit='4GB'")
    sql='''SELECT c.split,count(*) source_rows,count(DISTINCT l.subject_id) contributing_patients,
        count(DISTINCT l.hadm_id) linked_hospital_admissions,count(DISTINCT itemid) distinct_specific_items
        FROM read_csv_auto(?,ignore_errors=false) l JOIN read_csv_auto(?) c USING(subject_id) GROUP BY c.split ORDER BY c.split'''
    (out/'query.sql').write_text(sql+'\n');start=time.monotonic();timer=threading.Timer(90,c.interrupt);timer.start()
    try:
        rows=c.execute(sql,[str(path),str(a/'patient_splits.csv')]).fetchdf().to_dict('records')
        result=dict(status='COUNTED',counts=rows)
    except Exception as exc:result=dict(status='QUERY_INTERRUPTED_OR_SOURCE_ERROR',error=type(exc).__name__)
    finally:timer.cancel()
    result.update(elapsed_seconds=time.monotonic()-start,source_bytes=path.stat().st_size,
        source_sha256=sha(Path(__file__)),scope='supplemental bounded source-row census, not lab preprocessing',cpu_limit=2,wall_limit_seconds=90)
    dump(out/'aggregate.json',result);print(json.dumps(result,indent=2))
