"""Incremental cohort/event/embedding/window accounting, reusing M2 validation.

This builds a manifest from actual canonical outputs. It neither maps raw C1
rows to events nor launches extraction/encoders. Assigned patients remain in
the manifest even when no valid outputs have been registered for them. These
counts describe ingested outputs, not completeness of raw-source processing.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

from noah_baseline.m2.dataset import validate_entry
from noah_baseline.m2.events import EmbeddingCache
from noah_baseline.m2.stream_index import atomic_json
from noah_baseline.runtime import ArtifactNamespace


class CohortManifest:
    def __init__(self,root,out,splits,*,dataset_sha256,minimum_targets=64):
        ArtifactNamespace(root,'reduced_pilot').owned(out)
        if type(minimum_targets) is not int or minimum_targets<1 or not splits or not set(splits.values())<={'train','val','test'}:
            raise ValueError('Explicit patient split and target criterion required')
        if len(dataset_sha256)!=64 or any(c not in '0123456789abcdef' for c in dataset_sha256):
            raise ValueError('Explicit dataset identity required')
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        self.out=out;self.cache=EmbeddingCache(root);self.dataset_sha256=dataset_sha256;self.minimum=minimum_targets
        self.split_sha256=hashlib.sha256(json.dumps(sorted(splits.items()),separators=(',',':')).encode()).hexdigest()
        self.db=sqlite3.connect(out/'accounting.sqlite');self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''CREATE TABLE patients(subject INTEGER PRIMARY KEY,split TEXT NOT NULL);
            CREATE TABLE events(id TEXT PRIMARY KEY NOT NULL,subject INTEGER NOT NULL REFERENCES patients,category INTEGER,modality TEXT,
                encounter INTEGER,scorable INTEGER,source_sha TEXT NOT NULL);
            CREATE TABLE owners(id TEXT PRIMARY KEY NOT NULL REFERENCES events,window TEXT NOT NULL);
            CREATE TABLE windows(id TEXT PRIMARY KEY NOT NULL,subject INTEGER NOT NULL REFERENCES patients);
            CREATE TABLE encodings(source TEXT NOT NULL,kind TEXT,subject INTEGER NOT NULL REFERENCES patients,status TEXT,reason TEXT,reference TEXT,
                PRIMARY KEY(source,kind));
            CREATE TABLE deferred(source TEXT PRIMARY KEY NOT NULL,subject INTEGER NOT NULL REFERENCES patients,reason TEXT NOT NULL);
            CREATE TABLE artifacts(path TEXT PRIMARY KEY,sha TEXT NOT NULL,role TEXT NOT NULL);''')
        self.db.executemany('INSERT INTO patients VALUES (?,?)',sorted(splits.items()));self.db.commit()

    def event(self,record,*,source_file_sha256,scorable=True):
        if len(source_file_sha256)!=64 or any(c not in '0123456789abcdef' for c in source_file_sha256):
            raise ValueError('Canonical event requires source-file hash separately from value-content hash')
        event=validate_entry(record,self.cache,expected_dataset_sha256=self.dataset_sha256)
        try:
            self.db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',(event.event_id,event.subject_id,event.category,
                event.modality,event.encounter_id,int(scorable),source_file_sha256))
        except sqlite3.IntegrityError as e:raise ValueError('Duplicate canonical identity or patient outside frozen split') from e

    def encoding(self,*,source_key,kind,subject_id,status,reason=None,reference=None):
        # Categorical inputs use learned model lookups, not an external cache.
        if kind not in {'numeric','specifics','text','cxr','ecg','echo'}:raise ValueError('Unknown encoding kind')
        if status not in {'encoded','cached','missing','unavailable','failed','deferred'}:raise ValueError('Explicit encoding outcome required')
        if status in {'encoded','cached'}:
            if reference is None or reference['kind']!=kind or reference['dataset_sha256']!=self.dataset_sha256:
                raise ValueError('Successful encoding needs actual dataset-bound value evidence')
            self.cache.get(reference)
        elif not reason or reference is not None:raise ValueError('Failed/missing encoding needs a reason and no invented value')
        try:self.db.execute('INSERT INTO encodings VALUES (?,?,?,?,?,?)',
            (source_key,kind,subject_id,status,reason,json.dumps(reference,sort_keys=True) if reference else None))
        except sqlite3.IntegrityError as e:raise ValueError('Repeated source encoding or patient outside split') from e

    def defer(self,*,source_key,subject_id,reason):
        if not reason:raise ValueError('Deferred source records must remain explicit')
        try:self.db.execute('INSERT INTO deferred VALUES (?,?,?)',(source_key,subject_id,reason))
        except sqlite3.IntegrityError as e:raise ValueError('Repeated deferred source or patient outside split') from e

    def window(self,window,*,window_id):
        batch=window['batch'];ids=window['token_event_ids'];subject=window['subject_id']
        attention=batch['attention_mask'][0].bool();loss=batch['loss_mask'][0].bool()
        if len(ids)!=len(attention) or (loss & ~attention).any():raise ValueError('Invalid window padding/masks')
        expected=[ids[i+1] for i in range(len(ids)-1) if loss[i+1] and attention[i+1] and attention[i]]
        if None in expected or expected!=window['owned_target_ids']:raise ValueError('Shifted target identity mismatch')
        if any(loss[i] and ids[i] is None for i in range(len(ids))):raise ValueError('Prefix/padding owns clinical loss')
        if not expected:raise ValueError('A trainable window requires actual shifted targets')
        # Keep previously registered events after a rejected window transaction.
        self.db.commit()
        try:
            with self.db:
                self.db.execute('INSERT INTO windows VALUES (?,?)',(window_id,subject))
                for event_id in expected:
                    row=self.db.execute('SELECT subject,scorable FROM events WHERE id=?',(event_id,)).fetchone()
                    if row!=(subject,1):raise ValueError('Window target is unscorable, absent or belongs to another patient')
                    self.db.execute('INSERT INTO owners VALUES (?,?)',(event_id,window_id))
        except sqlite3.IntegrityError as e:raise ValueError('Duplicate shifted target/window ownership') from e

    def artifact(self,path,*,role):
        p=Path(path)
        with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        self.db.execute('INSERT INTO artifacts VALUES (?,?,?)',(str(p),digest,role))

    def finish(self,*,require_windows=True):
        missing=self.db.execute('SELECT count(*) FROM events e LEFT JOIN owners o USING(id) WHERE e.scorable=1 AND o.id IS NULL').fetchone()[0]
        if require_windows and missing:raise ValueError('Scored canonical targets missing independent window ownership')
        def counts(sql):return self.db.execute(sql).fetchall()
        cohort=counts('''SELECT p.split,count(*),sum(n.events>0),sum(n.scored>=?) FROM patients p
            JOIN (SELECT p.subject,count(e.id) events,sum(CASE WHEN o.id IS NOT NULL THEN 1 ELSE 0 END) scored
            FROM patients p LEFT JOIN events e ON e.subject=p.subject LEFT JOIN owners o ON o.id=e.id GROUP BY p.subject) n
            ON n.subject=p.subject GROUP BY p.split'''.replace('?',str(self.minimum)))
        window_count=self.db.execute('SELECT count(*) FROM windows').fetchone()[0]
        result=dict(schema_version=1,status='ENCODED_EVENTS_AND_OWNED_WINDOWS' if not missing and window_count else 'WINDOWS_PENDING',
            dataset_sha256=self.dataset_sha256,patient_split_assignment_sha256=self.split_sha256,minimum_owned_targets=self.minimum,
            split_patients={s:dict(assigned=a,with_canonical_event=e,eligible_owned_targets=n) for s,a,e,n in cohort},
            category_support=[dict(split=s,category=c,events=n,patients=p) for s,c,n,p in counts('SELECT p.split,e.category,count(*),count(DISTINCT e.subject) FROM events e JOIN patients p ON p.subject=e.subject GROUP BY p.split,e.category')],
            modality_support=[dict(split=s,modality=m,events=n,patients=p) for s,m,n,p in counts('SELECT p.split,e.modality,count(*),count(DISTINCT e.subject) FROM events e JOIN patients p ON p.subject=e.subject GROUP BY p.split,e.modality')],
            encounter_attribution=dict(counts("SELECT CASE WHEN encounter IS NULL THEN 'longitudinal_unknown_encounter' ELSE 'encounter_attributed' END,count(*) FROM events GROUP BY 1")),
            encoding_outcomes=[dict(kind=k,status=s,source_records=n) for k,s,n in counts('SELECT kind,status,count(*) FROM encodings GROUP BY kind,status')],
            deferred_reasons=dict(counts('SELECT reason,count(*) FROM deferred GROUP BY reason')),
            trainable_windows=dict(counts('SELECT p.split,count(*) FROM windows w JOIN patients p ON w.subject=p.subject GROUP BY p.split')),
            valid_owned_targets=dict(counts('SELECT p.split,count(*) FROM owners o JOIN events e USING(id) JOIN patients p ON p.subject=e.subject GROUP BY p.split')),
            missing_window_ownership=missing,artifacts=[dict(path=p,sha256=h,role=r) for p,h,r in counts('SELECT * FROM artifacts')],
            input_coverage='ONLY_REGISTERED_OUTPUTS; source-processing completeness requires separate evidence',
            scientific_profile='declared occurrence-time engineering candidate; not exact paper or strict prospective validity',paper_fidelity='NOT_CERTIFIED',strict_prospective='NOT_CERTIFIED',full_m2_gate='HOLD')
        self.db.commit()
        with (self.out/'accounting.sqlite').open('rb') as f:ledger_sha=hashlib.file_digest(f,'sha256').hexdigest()
        result['accounting_ledger']=dict(path='accounting.sqlite',sha256=ledger_sha,
            contents='event/source identities, patient splits, encoding references and independent shifted ownership')
        atomic_json(self.out/'manifest.json',result)
        return result

    def close(self):self.db.close()
