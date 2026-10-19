"""
Pre-caches RAD-DINO (768-D CLS) embeddings for all cohort chest radiographs in batch.

Optimized for high GPU throughput:
- Uses PyTorch DataLoader with multi-worker image pre-fetching.
- Batched forward passes through microsoft/rad-dino.
- Deterministic O(1) filesystem path resolution.
- Idempotent: skips already cached embeddings.
"""

import argparse
from pathlib import Path
import time
from typing import List, Tuple
import duckdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


class CXRDataset(Dataset):
    def __init__(self, items: List[Tuple[str, Path]], processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        dicom_id, img_path = self.items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            # Process image to tensor
            processed = self.processor(images=img, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)  # [3, 518, 518]
            return dicom_id, pixel_values, True
        except Exception as e:
            # Corrupt image fallback: return dummy tensor and False flag
            return dicom_id, torch.zeros(3, 518, 518, dtype=torch.float32), False


def collate_cxr_batch(batch):
    dicom_ids = [b[0] for b in batch]
    valid_flags = [b[2] for b in batch]
    pixel_tensors = torch.stack([b[1] for b in batch])
    return dicom_ids, pixel_tensors, valid_flags


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-cache RAD-DINO embeddings for CXRs.")
    parser.add_argument("--manifest", type=str, default="data/pilot_cohort_manifest.csv")
    parser.add_argument("--cxr_base_dir", type=str, default="/home/wey371/rajpurkarlab/datasets/cxr/MIMIC-CXR/raw_jpg")
    parser.add_argument("--cache_dir", type=str, default="data/cache/cxr")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional test limit")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cxr_base_dir = Path(args.cxr_base_dir)

    print("=" * 70)
    print("NOAH CXR RAD-DINO PRE-CACHING PIPELINE")
    print(f"Manifest:     {args.manifest}")
    print(f"CXR Base Dir: {cxr_base_dir}")
    print(f"Cache Dir:    {cache_dir}")
    print(f"Device:       {args.device}")
    print(f"Batch Size:   {args.batch_size}")
    print("=" * 70)

    # 1. Query all relevant CXRs from metadata
    meta_path = cxr_base_dir / "mimic-cxr-2.0.0-metadata.csv.gz"
    if not meta_path.exists():
        meta_path = cxr_base_dir / "mimic-cxr-2.0.0-metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found in {cxr_base_dir}")

    con = duckdb.connect()
    read_meta = f"read_csv('{meta_path}', header=true, quote='\"', escape='\"', null_padding=true, ignore_errors=true, parallel=false)"
    read_man = f"read_csv('{args.manifest}', header=true)"

    query = f"""
        SELECT m.dicom_id, m.subject_id, m.study_id
        FROM {read_meta} m
        JOIN {read_man} man ON CAST(m.subject_id AS BIGINT) = CAST(man.subject_id AS BIGINT)
    """
    rows = con.execute(query).fetchall()
    con.close()

    print(f"Found {len(rows):,} total CXR records matching cohort manifest.")

    # 2. Check disk existence and skip already cached files
    pending_items: List[Tuple[str, Path]] = []
    already_cached = 0
    missing_on_disk = 0

    for did, sid, stid in rows:
        cache_file = cache_dir / f"{did}.npy"
        if cache_file.exists():
            already_cached += 1
            continue

        rel_path = f"files/p{str(sid)[:2]}/p{sid}/s{stid}/{did}.jpg"
        img_path = cxr_base_dir / rel_path
        if img_path.exists():
            pending_items.append((str(did), img_path))
        else:
            missing_on_disk += 1

    print(f"-> Already cached: {already_cached:,}")
    print(f"-> Missing on disk: {missing_on_disk:,}")
    print(f"-> Pending encoding: {len(pending_items):,}")

    if args.limit is not None and args.limit > 0:
        pending_items = pending_items[: args.limit]
        print(f"-> Limited to first {len(pending_items)} items for test.")

    if not pending_items:
        print("All CXR images are already cached! Nothing to do.")
        return

    # 3. Load RAD-DINO
    if args.device == "cuda" or "cuda" in args.device:
        try:
            test_conv = torch.nn.Conv2d(3, 16, 3).to(args.device)
            test_x = torch.zeros(1, 3, 14, 14, device=args.device)
            _ = test_conv(test_x)
        except Exception as e:
            print(f"[Warning] cuDNN initialization check failed: {e}. Falling back to native CUDA kernels (cudnn.enabled=False).")
            torch.backends.cudnn.enabled = False

    print("\nLoading microsoft/rad-dino model and processor...")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino").to(args.device)
    model.eval()

    # 4. DataLoader
    dataset = CXRDataset(pending_items, processor)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cxr_batch,
        pin_memory=(args.device == "cuda"),
        persistent_workers=False,
    )

    print(f"Starting batch inference across {len(loader)} batches...")
    t_start = time.time()
    saved_count = 0

    with torch.no_grad():
        for batch_dids, pixel_values, valid_flags in tqdm(loader, desc="Encoding CXRs"):
            pixel_values = pixel_values.to(args.device)
            out = model(pixel_values=pixel_values)
            # CLS token is at index 0: [B, 768]
            cls_embs = out.last_hidden_state[:, 0, :].cpu().to(torch.float32).numpy()

            for did, emb, is_valid in zip(batch_dids, cls_embs, valid_flags):
                if is_valid:
                    np.save(cache_dir / f"{did}.npy", emb)
                    saved_count += 1

    t_elapsed = time.time() - t_start
    throughput = saved_count / max(t_elapsed, 0.001)
    print("=" * 70)
    print(f"CXR PRE-CACHING COMPLETED!")
    print(f"Successfully cached: {saved_count:,} new CXR embeddings")
    print(f"Total time elapsed:  {t_elapsed:.2f} s ({throughput:.2f} images/s)")
    print(f"Total cached files:  {len(list(cache_dir.glob('*.npy'))):,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
