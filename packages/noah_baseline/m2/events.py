"""Provenance-preserving event/cache contracts; no default scientific aggregation span."""
from dataclasses import dataclass,asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from noah_baseline.runtime import ArtifactNamespace

DIMENSIONS={'numeric':44,'text':768,'cxr':768,'ecg':192,'echo':512,'specifics':768}


@dataclass(frozen=True)
class Event:
    event_id:str
    subject_id:int
    encounter_id:int|None
    linkage_status:str
    dataset:str
    dataset_version:str
    source_row_id:str
    source_path:str
    occurred_at:float
    available_at:float|None
    availability_source:str
    modality:str
    category:int
    specific_item:str|None
    units:str|None
    transform_revision:str
    content_sha256:str|None=None
    encoder_checkpoint_sha256:str|None=None
    cache_recipe_sha256:str|None=None
    embedding_ref:str|None=None

    def validate(self):
        for k in ('event_id','dataset','dataset_version','source_row_id','source_path','transform_revision'):
            if not getattr(self,k):raise ValueError('Missing source provenance: '+k)
        if not math.isfinite(self.occurred_at) or self.available_at is not None and not math.isfinite(self.available_at):
            raise ValueError('Invalid timestamp')
        if not self.availability_source:raise ValueError('Availability provenance must be explicit')
        if (self.available_at is None)!=(self.availability_source=='unknown'):
            raise ValueError('Unknown availability cannot become an occurrence-time proxy')
        if self.linkage_status not in {'source','unique_interval','ambiguous','unlinked'}:raise ValueError('Unknown linkage status')
        if (self.encounter_id is None)!=(self.linkage_status in {'ambiguous','unlinked'}):raise ValueError('Encounter/linkage mismatch')
        if self.modality not in {'categorical',*DIMENSIONS.keys()-{'specifics'}} or not 0<=self.category<31:
            raise ValueError('Invalid modality/category')
        if self.embedding_ref and not all((self.content_sha256,self.encoder_checkpoint_sha256,self.cache_recipe_sha256)):
            raise ValueError('Embedding lacks content/encoder/recipe provenance')
        return self


def cutoff_view(events,cutoff,profile):
    if profile not in {'paper_reproduction','occurrence_diagnostic','strict_prospective'}:raise ValueError('Explicit profile required')
    kept=[]
    for event in events:
        event.validate()
        if event.occurred_at>cutoff:continue
        if profile=='strict_prospective' and (event.available_at is None or event.available_at>cutoff):continue
        kept.append(event)
    return sorted(kept,key=lambda e:(e.occurred_at,e.event_id))


class EmbeddingCache:
    def __init__(self,root):self.namespace=ArtifactNamespace(root,'reduced_pilot')

    def put(self,array,*,kind,content_sha256,dataset_sha256,checkpoint_sha256,recipe):
        a=np.asarray(array,dtype=np.float32)
        if kind not in DIMENSIONS or a.shape!=(DIMENSIONS[kind],) or not np.isfinite(a).all() or not np.any(a):
            raise ValueError('Invalid modality content vector; no missing/failed zero fallback')
        if not recipe or 'transform_revision' not in recipe:raise ValueError('Complete versioned recipe required')
        recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        p=self.namespace.cache_path(recipe_sha256=recipe_sha,dataset_sha256=dataset_sha256,
            encoder_checkpoint_sha256=checkpoint_sha256,item_sha256=content_sha256).with_suffix('.npy')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            if not np.array_equal(np.load(p,allow_pickle=False),a):raise ValueError('Cache collision or nondeterministic encoder')
        else:
            with p.open('xb') as f:np.save(f,a,allow_pickle=False)
        file_sha=hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(path=str(p),kind=kind,shape=list(a.shape),file_sha256=file_sha,content_sha256=content_sha256,
            dataset_sha256=dataset_sha256,encoder_checkpoint_sha256=checkpoint_sha256,recipe_sha256=recipe_sha,recipe=recipe)

    def get(self,reference):
        path=self.namespace.owned(Path(reference['path']))
        recipe_sha=hashlib.sha256(json.dumps(reference['recipe'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        expected=self.namespace.cache_path(recipe_sha256=recipe_sha,dataset_sha256=reference['dataset_sha256'],
            encoder_checkpoint_sha256=reference['encoder_checkpoint_sha256'],item_sha256=reference['content_sha256']).with_suffix('.npy')
        if recipe_sha!=reference['recipe_sha256'] or path.resolve()!=expected.resolve():
            raise ValueError('Cache reference identity/recipe/path mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['file_sha256']:
            raise ValueError('Cache content hash mismatch')
        value=np.load(path,allow_pickle=False)
        if value.dtype!=np.float32 or value.shape!=(DIMENSIONS[reference['kind']],) or not np.isfinite(value).all():
            raise ValueError('Cached modality representation mismatch')
        return value


def aggregate_minmax(records,*,origin,span,stride):
    if not all(math.isfinite(v) for v in (origin,span,stride)) or span<=0 or stride<=0:
        raise ValueError('Finite positive explicit span and stride required')
    if not records:return []
    result=[];start=origin;last=max(r['occurred_at'] for r in records)
    while start<=last:
        group=[r for r in records if start<=r['occurred_at']<start+span]
        if group:
            if len({(r['subject_id'],r['encounter_id'],r['item'],r['units']) for r in group})!=1:
                raise ValueError('Aggregation cannot mix patients, encounters, items or units')
            if not all(math.isfinite(r['value']) for r in group):raise ValueError('Nonfinite source observation')
            source_availability=None if any(r['available_at'] is None for r in group) else max(r['available_at'] for r in group)
            latest=max(r['occurred_at'] for r in group)
            availability=None if source_availability is None else max(source_availability,latest)
            ids=sorted(r['source_row_id'] for r in group)
            result.append(dict(contributor_ids=ids,occurred_at=sum(r['occurred_at'] for r in group)/len(group),
                available_at=availability,availability_source='unknown' if availability is None else 'derived_contributor_max',
                source_available_at_max=source_availability,last_contributor_at=latest,
                minimum=min(r['value'] for r in group),maximum=max(r['value'] for r in group),window_start=start,span=span,stride=stride))
        start+=stride
    return result
