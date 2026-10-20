"""Metadata-first, fail-closed ECHO preprocessing with explicit pilot candidates.

No pixel-mask fallback replaces DICOM ultrasound-region metadata. Annotation
removal is a temporal-motion convex-hull candidate requiring visual validation.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay


def ultrasound_region(regions, width, height):
    candidates = []
    for region in regions:
        def value(key, default=None):
            item=region.get(key, default)
            return getattr(item,'value',item)
        if int(value('RegionSpatialFormat', 0)) != 1 or int(value('RegionDataType', 0)) != 1:
            continue
        keys = ('RegionLocationMinX0','RegionLocationMinY0','RegionLocationMaxX1','RegionLocationMaxY1')
        if any(value(k) is None for k in keys):
            continue
        x0, y0, x1, y1 = [int(value(k)) for k in keys]
        # DICOM max coordinates are inclusive; array slicing is exclusive.
        x1 += 1; y1 += 1
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Ultrasound metadata region outside source image')
        candidates.append((-(x1-x0)*(y1-y0), x0, y0, x1, y1))
    if not candidates:
        raise ValueError('No observed2D tissue ultrasound region; no image-only fallback')
    _, *box = min(candidates)
    return tuple(box)


def motion_hull_mask(frames):
    gray = frames.astype(np.float32).mean(axis=-1)
    moving = np.std(gray, axis=0) > 1.
    moving = ndimage.binary_opening(moving, iterations=1)
    labels, count = ndimage.label(moving)
    if count == 0:
        raise ValueError('No genuine motion region; cannot silently return unmasked clip')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    component = labels == sizes.argmax()
    component = ndimage.binary_dilation(component, iterations=5)
    points = np.column_stack(np.nonzero(component))
    if len(points) < 100:
        raise ValueError('Insufficient tissue motion support for annotation mask')
    hull = ConvexHull(points); vertices = points[hull.vertices]
    yy, xx = np.indices(component.shape)
    return (Delaunay(vertices).find_simplex(np.column_stack((yy.ravel(), xx.ravel()))) >= 0).reshape(component.shape)


def clip_tensor(frames, recipe):
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
        raise ValueError('Expected nonempty decoded RGB uint8 video')
    edge = float(recipe['zoom_fraction_per_edge'])
    if not 0 <= edge < .5:
        raise ValueError('Explicit valid zoom fraction required')
    h, w = frames.shape[1:3]; y, x = int(h*edge), int(w*edge)
    crop = frames[:, y:h-y, x:w-x]
    resized = np.stack([np.array(Image.fromarray(frame).resize((224,224), Image.Resampling.BILINEAR)) for frame in crop])
    mean = np.asarray(recipe['mean'], dtype=np.float32)[None,None,None,:]
    std = np.asarray(recipe['std'], dtype=np.float32)[None,None,None,:]
    if np.any(std <= 0):
        raise ValueError('Invalid pinned normalization')
    normalized = (resized.astype(np.float32)-mean)/std
    indices = recipe['clip_start'] + np.arange(recipe['frames'])*recipe['stride']
    selected = np.zeros((recipe['frames'],224,224,3), dtype=np.float32)
    valid = indices < len(normalized)
    selected[valid] = normalized[indices[valid]]
    tensor = torch.from_numpy(selected.transpose(3,0,1,2).copy())
    return tensor, dict(source_frame_indices=[int(i) if ok else None for i,ok in zip(indices,valid)],
                        padded_frames=int((~valid).sum()),zoom_crop=[x,y,w-x,h-y])


def avi_roundtrip(frames, path):
    import av
    with av.open(str(path), 'w', format='avi') as output:
        stream = output.add_stream('ffv1', rate=30)
        stream.width = 224; stream.height = 224; stream.pix_fmt = 'bgr0'
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as video:
        decoded = np.stack([f.to_ndarray(format='rgb24') for f in video.decode(video=0)])
    if not np.array_equal(frames, decoded):
        raise ValueError('Lossless224AVI staging changed pixels or frame order')
    return decoded


def preprocess_echo(path: Path, avi_path: Path, recipe):
    import pydicom
    ds = pydicom.dcmread(path)
    count = int(ds.get('NumberOfFrames', 1))
    if not 2 <= count <= 300 or int(ds.BitsAllocated) != 8:
        raise ValueError('Bounded smoke requires2..300 frames and explicit8-bit input')
    pixels = ds.pixel_array  # pydicom returns RGB for YBR; no second color conversion.
    if pixels.ndim == 3:
        pixels = np.repeat(pixels[...,None], 3, axis=-1)
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError('Unsupported DICOM color/range; no implicit rescale')
    x0,y0,x1,y1 = ultrasound_region(ds.get('SequenceOfUltrasoundRegions', []),pixels.shape[2],pixels.shape[1])
    cropped = pixels[:,y0:y1,x0:x1]
    mask = motion_hull_mask(cropped)
    # Candidate for grayscale tissue clips: remove static chromatic annotations,
    # rather than deleting dynamic color information. Never silently apply to color-flow regions.
    if any(int(getattr(r,'RegionDataType',0))==2 for r in ds.get('SequenceOfUltrasoundRegions',[])):
        raise ValueError('Color-flow clip needs a separately validated annotation policy')
    chromatic=(cropped.max(axis=-1).astype(np.int16)-cropped.min(axis=-1).astype(np.int16))>32
    static_chromatic=ndimage.binary_dilation(chromatic.mean(axis=0)>=.9,iterations=1)
    masked = cropped.copy(); masked[:,~mask | static_chromatic] = 0
    staged = np.stack([np.array(Image.fromarray(f).resize((224,224),Image.Resampling.BILINEAR)) for f in masked])
    decoded = avi_roundtrip(staged, avi_path)
    tensor, summary = clip_tensor(decoded, recipe)
    Image.fromarray(staged[0]).save(avi_path.with_suffix('.review.png'))
    summary.update(source_shape=list(pixels.shape),photometric=str(ds.PhotometricInterpretation),
        transfer_syntax=str(ds.file_meta.TransferSyntaxUID),metadata_region=[x0,y0,x1,y1],
        annotation_mask='candidate_motion_hull_plus_static_chromatic; visual review required',
        static_chromatic_mask_fraction=float(static_chromatic.mean()),
        tissue_mask_fraction=float(mask.mean()),avi_roundtrip='bitwise',source_frames=count,
        decoder=f'pydicom{pydicom.__version__}',color_conversion='pydicom YBR-to-RGB once',
        preprocessing_status='PARTIALLY_VERIFIED_ANNOTATION_POLICY_CANDIDATE')
    if tensor.shape != (3,16,224,224) or not torch.isfinite(tensor).all():
        raise ValueError('Invalid EchoPrime input tensor')
    return tensor, summary
