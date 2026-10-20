"""Bounded genuine ECG/ECHO preprocessing and frozen GPU inference evidence."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import wfdb
from torchvision.models.video import mvit_v2_s

from noah_baseline.m2.echo import preprocess_echo
from noah_baseline.m2.events import EmbeddingCache
from noah_baseline.m2.inventory import dump, sha
from noah_baseline.m2.otis import OTISEncoder, normalize_ecg


def media_smoke(root, *, out, source):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('A Slurm GPU allocation is mandatory; no CPU fallback')
    started = time.monotonic(); out.mkdir(exist_ok=False)
    recipe = json.loads((root/'configs/m2_integration.json').read_text())
    config = json.loads((root/'configs/m2.json').read_text())
    catalog = json.loads((root/'configs/encoder_assets.json').read_text())
    samples = json.loads((source/'small_source_samples.json').read_text())
    capture = out/'executed-source'; capture.mkdir()
    for name in ['packages/noah_baseline/m2/media_smoke.py','packages/noah_baseline/m2/otis.py',
                 'packages/noah_baseline/m2/echo.py','packages/noah_baseline/m2/events.py',
                 'configs/m2.json','configs/m2_integration.json','configs/encoder_assets.json','tools/m2_run.py','sbatch/m2_media_smoke.sh']:
        p = capture/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((root/name).read_bytes())
    torch.set_num_threads(2); torch.manual_seed(recipe['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cache = EmbeddingCache(root); dataset_sha = sha(source/'source_inventory.json')
    result = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  config_sha256=sha(root/'configs/m2_integration.json'),job_id=os.environ['SLURM_JOB_ID'],
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,seed=recipe['seed'],records=[],modalities={})
    for kind in ['ecg','echo']:
        asset = catalog['assets']['otis' if kind=='ecg' else 'echoprime']; path = Path(asset['path'])
        if sha(path) != asset['weight_sha256']:
            raise ValueError('Encoder checkpoint identity changed')
        examples = [s for s in samples if s['dataset']==kind][:recipe[kind]['max_records']]
        try:
            if kind == 'ecg':
                preprocessed=[]
                for number,record in enumerate(examples):
                    wavepath=Path(config['roots']['ecg'])/record['row']['path']
                    start=time.monotonic();wave,meta=wfdb.rdsamp(str(wavepath))
                    if set(meta['units']) != {'mV'}:
                        raise ValueError('ECG source physical units require an explicit new recipe')
                    a=normalize_ecg(wave,meta['sig_name'],lead_order=recipe['ecg']['lead_order'],sample_rate=meta['fs'])
                    content=hashlib.sha256((sha(wavepath.with_suffix('.hea'))+sha(wavepath.with_suffix('.dat'))).encode()).hexdigest()
                    preprocessed.append((number,record,a,content,time.monotonic()-start))
                variants={}
                for policy in recipe['ecg']['position_policies']:
                    model=OTISEncoder.from_checkpoint(path,position_policy=policy).cuda().eval()
                    assert not any(p.requires_grad for p in model.parameters())
                    variants[policy]=[]
                    for number,record,a,content,preprocessing_seconds in preprocessed:
                        x=torch.from_numpy(a).unsqueeze(0).cuda();torch.cuda.synchronize();start=time.monotonic()
                        with torch.inference_mode():
                            patches=model.patches(x);first=patches.mean(1)[0];second=model(x)[0]
                        torch.cuda.synchronize();seconds=time.monotonic()-start
                        if patches.shape!=(1,2496,192) or not torch.equal(first,second):
                            raise ValueError('Final patch mean or deterministic ECG replay failed')
                        cache_recipe=dict(transform_revision='m2-otis-real-v1',**recipe['ecg'],position_policy=policy,
                            source_units='mV',checkpoint_sha256=asset['weight_sha256'],domain_offset=model.domain_offset,
                            trained_encoder_load='strict; all expected keys loaded',torch=torch.__version__)
                        ref=cache.put(first.cpu().numpy(),kind='ecg',content_sha256=content,dataset_sha256=dataset_sha,
                            checkpoint_sha256=asset['weight_sha256'],recipe=cache_recipe)
                        np.testing.assert_array_equal(cache.get(ref),first.cpu().numpy())
                        result['records'].append(dict(kind=kind,source=record,variant=policy,embedding=ref,
                            preprocessing_seconds=preprocessing_seconds,inference_and_replay_seconds=seconds,
                            patch_shape=[2496,192],replay='bitwise',excluded_pretraining_keys=model.excluded_pretraining_keys))
                        variants[policy].append(first.cpu().numpy())
                    del model;torch.cuda.empty_cache()
                pairs=[]
                for first,second in zip(*[variants[p] for p in recipe['ecg']['position_policies']]):
                    pairs.append(dict(cosine=float(np.dot(first,second)/(np.linalg.norm(first)*np.linalg.norm(second))),
                                      mean_absolute_difference=float(np.abs(first-second).mean())))
                result['modalities'][kind]=dict(status='PARTIALLY VERIFIED',encoder_check='VERIFIED',records=len(examples),
                    position_sensitivity=pairs,production_position_policy=None,
                    limitation='explicit alternate temporal tables and normalization are candidates; production selection not made')
            else:
                model=mvit_v2_s(weights=None);model.head[-1]=torch.nn.Linear(model.head[-1].in_features,512)
                model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
                model=model.cuda().eval().requires_grad_(False)
                for number,record in enumerate(examples):
                    dicom=Path(config['roots']['echo'])/record['row']['dicom_filepath'];start=time.monotonic()
                    tensor,summary=preprocess_echo(dicom,out/f'echo-{number}.avi',recipe['echo'])
                    preprocess_seconds=time.monotonic()-start;x=tensor.unsqueeze(0).cuda();torch.cuda.synchronize();start=time.monotonic()
                    with torch.inference_mode():first=model(x)[0];second=model(x)[0]
                    torch.cuda.synchronize();seconds=time.monotonic()-start
                    if first.shape!=(512,) or not torch.equal(first,second):
                        raise ValueError('EchoPrime output dimension or deterministic replay failed')
                    import av,pydicom
                    cache_recipe=dict(transform_revision='m2-echoprime-real-v1',**recipe['echo'],
                        torchvision=__import__('torchvision').__version__,av=av.__version__,pydicom=pydicom.__version__,
                        checkpoint_sha256=asset['weight_sha256'],output='video512, no study view features')
                    ref=cache.put(first.cpu().numpy(),kind='echo',content_sha256=sha(dicom),dataset_sha256=dataset_sha,
                        checkpoint_sha256=asset['weight_sha256'],recipe=cache_recipe)
                    np.testing.assert_array_equal(cache.get(ref),first.cpu().numpy())
                    result['records'].append(dict(kind=kind,source=record,embedding=ref,preprocessing=summary,
                        preprocessing_seconds=preprocess_seconds,inference_and_replay_seconds=seconds,replay='bitwise'))
                result['modalities'][kind]=dict(status='PARTIALLY VERIFIED',encoder_check='VERIFIED',records=len(examples),
                    limitation='metadata crop/AVI/frame path executed; annotation-removal candidate needs visual review and production freeze')
                del model;torch.cuda.empty_cache()
        except Exception as error:
            dump(out/(kind+'-private-error.json'),dict(type=type(error).__name__,detail=str(error)))
            result['modalities'][kind]=dict(status='BLOCKED',error=type(error).__name__)
        dump(out/(kind+'-status.json'),result['modalities'][kind])
    # Bind real source item strings to the same frozen BioLORD candidate used by M2-B.
    from transformers import AutoModel,AutoTokenizer,__version__ as transformers_version
    source_events=json.loads((root/'artifacts/reduced_pilot/m2/m2-source-smoke-20261020-03/events.json').read_text())
    items=sorted({e['specific_item'] for e in source_events if e['specific_item']})
    text_path=Path(config['assets']['biolord']);weight_sha=sha(text_path/'model.safetensors')
    if weight_sha!='f8c969586cc6b0fd393faa7d879de93d6cc532123956041fefd3474194322050':
        raise ValueError('Specifics BioLORD differs from the verified M2 candidate')
    model,loading=AutoModel.from_pretrained(str(text_path),local_files_only=True,output_loading_info=True)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise ValueError('Strict pretrained specifics model load failed')
    model=model.cuda().eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(text_path),local_files_only=True)
    for item in items:
        tokens=tokenizer(item,return_tensors='pt',truncation=True,max_length=128).to('cuda')
        def infer_item():
            h=model(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
            return torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)[0]
        with torch.inference_mode():first=infer_item();second=infer_item()
        if not torch.equal(first,second):raise ValueError('Specifics inference replay differs')
        item_recipe=dict(transform_revision='m2-source-specifics-v1',pooling='attention-mask mean',l2=True,
            max_length=128,asset_revision=text_path.name,tokenizer_sha256=sha(text_path/'tokenizer.json'),
            transformers=transformers_version)
        reference=cache.put(first.cpu().numpy(),kind='specifics',content_sha256=hashlib.sha256(item.encode()).hexdigest(),
            dataset_sha256=dataset_sha,checkpoint_sha256=weight_sha,recipe=item_recipe)
        result['records'].append(dict(kind='specifics',specific_item=item,embedding=reference,replay='bitwise'))
    result['auxiliary_specifics']=dict(status='VERIFIED',source_item_strings=len(items),dimension=768)
    del model;torch.cuda.empty_cache()
    result.update(elapsed_seconds=time.monotonic()-started,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  training_launched=False,bulk_encoding_launched=False)
    dump(out/'private_results.json',result)
    aggregate={k:v for k,v in result.items() if k!='records'};dump(out/'aggregate.json',aggregate)
    print(json.dumps(aggregate,indent=2))
