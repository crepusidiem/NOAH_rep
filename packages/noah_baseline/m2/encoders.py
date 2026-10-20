"""Small real-source frozen-encoder checks only; no legacy cache or random-weight fallback."""
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from noah_baseline.m2.events import EmbeddingCache
from noah_baseline.m2.inventory import dump,sha


def smoke(root, *, out=None, source=None):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoTokenizer,AutoImageProcessor,__version__ as transformer_version
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Encoder verification requires an allocated SLURM GPU; no CPU fallback')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=json.loads((root/'configs/m2.json').read_text());start=time.monotonic()
    source=source or root/'artifacts/reduced_pilot/m2/m2-a-20261020-01'
    out=out or root/'artifacts/reduced_pilot/m2/m2-b-20261020-02'
    samples=json.loads((source/'small_source_samples.json').read_text());out.mkdir(exist_ok=False)
    capture=out/'executed-source';capture.mkdir()
    for name in ('encoders.py','events.py'):(capture/name).write_bytes((Path(__file__).parent/name).read_bytes())
    dataset_sha=sha(source/'source_inventory.json');cache=EmbeddingCache(root)
    result={'source_dataset_inventory_sha256':dataset_sha,'config_sha256':sha(root/'configs/m2.json'),
        'encoder_source_sha256':sha(Path(__file__)),'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'transformers':transformer_version,'modalities':{},'records':[]}
    for kind,asset in [('text','biolord'),('cxr','rad_dino')]:
        path=Path(config['assets'][asset]);weight=path/'model.safetensors'
        info={'asset_revision':path.name,'weight_sha256':sha(weight),'config_sha256':sha(path/'config.json')}
        try:
            model,loading=AutoModel.from_pretrained(str(path),local_files_only=True,output_loading_info=True)
            if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                raise ValueError('Pretrained checkpoint failed strict loading-key verification')
            for p in model.parameters():p.requires_grad_(False)
            model=model.cuda().eval();info['parameters']=sum(p.numel() for p in model.parameters());assert not any(p.requires_grad for p in model.parameters())
            if kind=='text':
                tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
                inputs=[dict(text=r['row']['text'],source=r) for r in samples if r['dataset']=='note']
                dictionary=Path(config['roots']['hosp'])/'d_labitems.csv'
                with dictionary.open(newline='') as f:
                    for row in list(csv.DictReader(f))[:2]:
                        inputs.append(dict(text=row['label'],source=dict(dataset='hosp_dictionary',source_path=str(dictionary),row={'itemid':row['itemid']})))
                for record in inputs:
                    text=record['text']
                    if not text or not text.strip():raise ValueError('Empty real text is missing, not a zero vector')
                    length=len(tokenizer(text,truncation=False)['input_ids'])
                    encoded=tokenizer(text,return_tensors='pt',truncation=True,max_length=128).to('cuda')
                    def infer():
                        h=model(**encoded).last_hidden_state;mask=encoded['attention_mask'].unsqueeze(-1)
                        return torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),p=2,dim=-1)[0]
                    with torch.inference_mode():a=infer();b=infer()
                    if not torch.equal(a,b):raise ValueError('Text inference replay differs')
                    recipe=dict(transform_revision='m2-biolord-candidate-v1',pooling='attention-mask mean',l2=True,
                        max_length=128,long_policy='right truncation, recorded; not asserted author policy',
                        asset_revision=path.name,tokenizer_sha256=sha(path/'tokenizer.json'),transformers=transformer_version)
                    content_sha=hashlib.sha256(text.encode()).hexdigest()
                    role='specifics' if record['source']['dataset']=='hosp_dictionary' else 'text'
                    ref=cache.put(a.cpu().numpy(),kind=role,content_sha256=content_sha,dataset_sha256=dataset_sha,checkpoint_sha256=info['weight_sha256'],recipe=recipe)
                    result['records'].append(dict(kind=role,source=record['source'],token_count=length,truncated=length>128,embedding=ref,replay='bitwise'))
                info.update(status='VERIFIED',scope='frozen real-source candidate encoding; long-note truncation is an assumption',examples=len(inputs))
                result['modalities']['categorical']=dict(status='PARTIALLY VERIFIED',examples=2,scope='real item labels encoded; learned category lookup belongs to NOAH; full taxonomy adapters remain incomplete')
            else:
                processor=AutoImageProcessor.from_pretrained(str(path),local_files_only=True)
                inputs=[r for r in samples if r['dataset']=='cxr']
                for record in inputs:
                    row=record['row'];image_path=(Path(config['roots']['cxr'])/row['path']).with_suffix('.jpg')
                    with Image.open(image_path) as im:
                        source_size=im.size;source_mode=im.mode;image=im.convert('RGB')
                    pixel=processor(images=image,return_tensors='pt').to('cuda')
                    with torch.inference_mode():a=model(**pixel).last_hidden_state[0,0];b=model(**pixel).last_hidden_state[0,0]
                    if not torch.equal(a,b):raise ValueError('CXR inference replay differs')
                    recipe=dict(transform_revision='m2-rad-dino-candidate-v1',asset_revision=path.name,pooling='last hidden CLS',
                        image_processor_sha256=sha(path/'preprocessor_config.json'),input_mode='PIL RGB, one processor rescale',
                        source_resize_provenance='original JPEG source2.0.0; index hashes matched; not author-exact recipe',transformers=transformer_version)
                    ref=cache.put(a.cpu().numpy(),kind='cxr',content_sha256=sha(image_path),dataset_sha256=dataset_sha,checkpoint_sha256=info['weight_sha256'],recipe=recipe)
                    result['records'].append(dict(kind='cxr',source=record,image_path=str(image_path),source_size=source_size,source_mode=source_mode,embedding=ref,replay='bitwise'))
                info.update(status='PARTIALLY VERIFIED',examples=len(inputs),encoder_check='VERIFIED',
                    limitation='candidate CLS/processor recipe, unique encounter/time integration and complete dataset remain to be validated')
            if info['examples']==0:raise ValueError('No real examples; shape tests cannot verify modality')
            del model;torch.cuda.empty_cache()
        except Exception as exc:
            dump(out/(kind+'-private-error.json'),dict(type=type(exc).__name__,detail=str(exc)))
            info.update(status='BLOCKED',error=type(exc).__name__)
        result['modalities'][kind]=info
        dump(out/(kind+'-check.json'),info)
    from noah_baseline.data.numeric import DyadicFourierEncoder
    numeric=[r for r in samples if r['dataset']=='ed' and r['table']=='vitalsign' and r['row'].get('heartrate')]
    for record in numeric:
        vector=DyadicFourierEncoder().encode(float(record['row']['heartrate']))
        assert vector.shape==(44,) and np.isfinite(vector).all()
        result['records'].append(dict(kind='numeric',source=record,units='bpm',fourier_dimension=44,
            source_availability='unknown',candidate='unchanged M1 22-scale concatenated sin/cos; no unit normalization'))
    result['modalities']['numeric']=dict(status='PARTIALLY VERIFIED',examples=len(numeric),
        scope='real heart-rate basis verified; aggregation span/stride and broader units remain explicit unset choices')
    import wfdb
    ecg=[]
    for record in [r for r in samples if r['dataset']=='ecg']:
        path=Path(config['roots']['ecg'])/record['row']['path']
        try:
            waveform,metadata=wfdb.rdsamp(str(path))
            ecg.append(dict(source=record,shape=list(waveform.shape),sample_rate=metadata['fs'],leads=metadata['sig_name'],units=metadata['units'],finite=bool(np.isfinite(waveform).all())))
        except Exception as exc:ecg.append(dict(source=record,error=type(exc).__name__))
    result['ecg_source_checks']=ecg
    result['modalities']['ecg']=dict(status='BLOCKED',source_examples=len(ecg),reason='No configured verified pretrained OTIS192-D/1x24 checkpoint; no inference attempted')
    echo=[]
    for record in [r for r in samples if r['dataset']=='echo']:
        path=Path(config['roots']['echo'])/record['row']['dicom_filepath']
        echo.append(dict(source=record,file_exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else None))
    result['echo_source_checks']=echo
    result['modalities']['echo']=dict(status='BLOCKED',source_examples=len(echo),reason='EchoPrime checkpoint absent from configured assets; DICOM decoder not installed',pydicom_available=bool(importlib.util.find_spec('pydicom')))
    result.update(elapsed_seconds=time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        training_launched=False,bulk_encoding_launched=False)
    dump(out/'private_results.json',result)
    safe={k:v for k,v in result.items() if k not in {'records','ecg_source_checks','echo_source_checks'}}
    dump(out/'aggregate.json',safe)
    print(json.dumps(safe,indent=2))
