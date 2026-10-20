"""Independent OTIS operations and fail-closed ECHO preprocessing contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from noah_baseline.m2.otis import Attention, PatchEmbed, normalize_ecg, temporal_positions
from noah_baseline.m2.echo import ultrasound_region, clip_tensor, avi_roundtrip, motion_hull_mask


def test_otis_double_projection_against_explicit_attention():
    torch.manual_seed(77)
    model=Attention().double().eval();x=torch.randn(2,7,192,dtype=torch.float64)
    qkv=x@model.qkv.weight.T+model.qkv.bias
    q,k,v=qkv.chunk(3,-1)
    matrices=model.mha.in_proj_weight.chunk(3,0);biases=model.mha.in_proj_bias.chunk(3,0)
    q,k,v=[(z@w.T+b).reshape(2,7,3,64).transpose(1,2) for z,w,b in zip((q,k,v),matrices,biases)]
    attention=torch.softmax((q@k.transpose(-1,-2))/8.,dim=-1)
    h=(attention@v).transpose(1,2).reshape(2,7,192)
    h=h@model.mha.out_proj.weight.T+model.mha.out_proj.bias
    reference=h@model.proj.weight.T+model.proj.bias
    torch.testing.assert_close(model(x),reference,atol=1e-12,rtol=1e-11)


def test_otis_patch_tail_and_explicit_position_choices():
    patch=PatchEmbed().eval();x=torch.randn(1,1,12,5000);changed=x.clone();changed[...,4992:]=1000
    with torch.inference_mode():a=patch(x);b=patch(changed)
    assert a.shape==(1,192,12,208) and torch.equal(a,b)
    width=96;grid=np.arange(42,dtype=np.float32);frequency=1./10000**(np.arange(48,dtype=np.float32)/48.)
    phase=np.outer(grid,frequency)
    old=torch.from_numpy(np.vstack((np.zeros((1,width)),np.hstack((np.sin(phase),np.cos(phase)))))).float()[None]
    analytic=temporal_positions(old,208,policy='analytic_extension')
    torch.testing.assert_close(analytic[:,:43],old,atol=1e-6,rtol=1e-6)
    interpolated=temporal_positions(old,208,policy='upstream_linear')
    assert torch.equal(interpolated[:,:1],old[:,:1]) and not torch.equal(interpolated,analytic)
    with pytest.raises(ValueError):temporal_positions(old,208,policy='hidden_default')
    with pytest.raises(ValueError):temporal_positions(torch.ones_like(old),208,policy='analytic_extension')


def test_ecg_lead_order_missingness_and_normalization():
    leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    raw=np.arange(60000,dtype=np.float64).reshape(5000,12)%503
    first=normalize_ecg(raw,leads,lead_order=leads,sample_rate=500)
    permutation=list(reversed(range(12)))
    second=normalize_ecg(raw[:,permutation],[leads[i] for i in permutation],lead_order=leads,sample_rate=500)
    np.testing.assert_array_equal(first,second)
    assert first.shape==(1,12,5000) and np.isfinite(first).all() and np.max(np.abs(first))<=4
    for invalid,rate in [(np.zeros_like(raw),500),(raw,250),(raw[:-1],500)]:
        with pytest.raises(ValueError):normalize_ecg(invalid,leads,lead_order=leads,sample_rate=rate)
    with pytest.raises(ValueError):normalize_ecg(raw,leads[:-1]+['V5'],lead_order=leads,sample_rate=500)


def test_echo_metadata_region_is_mandatory_and_inclusive(monkeypatch):
    region=dict(RegionSpatialFormat=1,RegionDataType=1,RegionLocationMinX0=2,RegionLocationMinY0=3,
        RegionLocationMaxX1=10,RegionLocationMaxY1=12)
    assert ultrasound_region([region],20,20)==(2,3,11,13)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'artifacts/reduced_pilot/m2/runtime-deps'))
    from pydicom.dataset import Dataset
    source_like=Dataset()
    for k,v in region.items():setattr(source_like,k,v)
    assert ultrasound_region([source_like],20,20)==(2,3,11,13)
    with pytest.raises(ValueError):ultrasound_region([],20,20)
    with pytest.raises(ValueError):ultrasound_region([dict(region,RegionLocationMaxX1=20)],20,20)
    with pytest.raises(ValueError):motion_hull_mask(np.zeros((4,32,32,3),dtype=np.uint8))


def test_echo_frame_indices_color_and_normalized_padding():
    frames=np.stack([np.full((8,8,3),i,dtype=np.uint8) for i in range(4)])
    recipe=dict(zoom_fraction_per_edge=0.,mean=[0,0,0],std=[1,1,1],clip_start=0,frames=4,stride=2)
    tensor,receipt=clip_tensor(frames,recipe)
    assert tensor.shape==(3,4,224,224)
    assert receipt['source_frame_indices']==[0,2,None,None] and receipt['padded_frames']==2
    assert torch.all(tensor[:,1]==2) and torch.all(tensor[:,2:]==0)
    with pytest.raises(ValueError):clip_tensor(frames.astype(np.float32),recipe)


def test_lossless_avi_preserves_non_gray_pixels_and_order(tmp_path,monkeypatch):
    dependencies=Path(__file__).resolve().parents[1]/'artifacts/reduced_pilot/m2/runtime-deps'
    monkeypatch.syspath_prepend(str(dependencies))
    frames=np.stack([np.full((224,224,3),[10+i,40+i,70+i],dtype=np.uint8) for i in range(3)])
    np.testing.assert_array_equal(avi_roundtrip(frames,tmp_path/'fixture.avi'),frames)
