"""
End-to-End Data Curation Pipeline for NOAH Reproduction (N-05).

Executes:
1. Multi-threaded DuckDB extraction across all 31 categories from MIMIC-IV.
2. Offline caching and assignment of:
   - Clinical text specifics (BioLORD-2023, 768-D)
   - Chest radiographs (RAD-DINO, 768-D)
   - ECG waveforms (OTIS 1x24 patch-mean, 192-D)
3. Window construction with 6 demographic prefixes, context L <= 1024, 256 overlap,
   and exact loss ownership masking (each clinical event scored exactly once).
4. Patient-disjoint partitioning (Train, Val, Test) matching pilot_cohort_manifest.csv.
5. Export of training-ready torch tensors and dataset summary ledger.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from src.data.schema import (
    EventCategory,
    Modality,
    DemographicPrefix,
    EventRecord,
    PatientJourney,
    ModelWindow,
)
from src.data.numeric import DyadicFourierEncoder
from src.data.duckdb_extractor import DuckDBEventExtractor
from src.data.embeddings_cache import TextEmbeddingCache, CXREmbeddingCache, ECGEmbeddingCache
from src.data.window_builder import WindowBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Run NOAH Data Curation Pipeline")
    parser.add_argument("--manifest", type=str, default="data/pilot_cohort_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--overlap_len", type=int, default=256)
    parser.add_argument("--min_events", type=int, default=64)
    parser.add_argument("--max_patients", type=int, default=None, help="Limit number of patients for quick tests")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_media_encoding", action="store_true", help="Skip heavy media encoding during test runs")
    parser.add_argument("--encode_demographics", action="store_true", default=True, help="Encode demographic prefix with text embeddings")
    parser.add_argument("--no_encode_demographics", dest="encode_demographics", action="store_false", help="Disable demographic text embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NOAH DATA CURATION PIPELINE (N-05)")
    print(f"Manifest:       {args.manifest}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Max Seq Len:    {args.max_seq_len} (Overlap: {args.overlap_len})")
    print(f"Min Events:     {args.min_events}")
    print(f"Device:         {args.device}")
    print("=" * 70)

    t_start = time.time()

    # 1. Initialize Extractor
    print("\n[Step 1/5] Initializing DuckDB extractor...")
    extractor = DuckDBEventExtractor(manifest_path=Path(args.manifest))

    # Read manifest splits
    import csv
    patient_splits: Dict[int, str] = {}
    with open(args.manifest, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            patient_splits[sid] = row["split"].strip().lower()

    all_sids = list(patient_splits.keys())
    if args.max_patients is not None:
        all_sids = all_sids[: args.max_patients]
        print(f"Subsetting to first {len(all_sids)} patients for this run.")

    target_sids = set(all_sids)

    # 2. Extract Events across all tables
    print(f"\n[Step 2/5] Extracting events across all MIMIC tables for {len(target_sids)} patients...")
    t_extract = time.time()
    journeys = extractor.extract_all(target_sids=target_sids)
    print(f"-> Extraction completed in {time.time() - t_extract:.2f} s")

    # 3. Filter Eligibility
    eligible_journeys: Dict[int, PatientJourney] = {}
    ineligible_count = 0
    total_raw_events = 0

    for sid, journey in journeys.items():
        total_raw_events += journey.total_events
        if journey.is_eligible(min_events=args.min_events):
            eligible_journeys[sid] = journey
        else:
            ineligible_count += 1

    print(f"-> Total raw events extracted: {total_raw_events:,}")
    print(f"-> Eligible patients (>={args.min_events} events): {len(eligible_journeys)} / {len(target_sids)}")
    print(f"-> Ineligible patients filtered: {ineligible_count}")

    # Immediately release DuckDB memory buffers and unused objects
    extractor.close()
    del extractor
    del journeys
    import gc
    gc.collect()

    # 4. Multi-modal Feature Embedding & Caching
    print("\n[Step 3/5] Resolving and caching multimodal embeddings...")
    text_cache = TextEmbeddingCache(
        cache_db_path=cache_dir / "text_embeddings.sqlite",
        device=args.device,
        batch_size=64,
    )
    cxr_cache = CXREmbeddingCache(
        cache_dir=cache_dir / "cxr",
        cxr_base_dir=Path("/home/wey371/rajpurkarlab/datasets/cxr/MIMIC-CXR/raw_jpg"),
        device=args.device,
    )
    ecg_cache = ECGEmbeddingCache(cache_dir=cache_dir / "ecg", device=args.device)

    # Collect all unique text strings across eligible patients
    all_texts: Set[str] = set()
    for j in eligible_journeys.values():
        for ev in j.events:
            if ev.has_specifics and ev.specifics_text:
                all_texts.add(ev.specifics_text.strip())

    demo_texts: Set[str] = set()
    if args.encode_demographics:
        for j in eligible_journeys.values():
            for _, demo in j.demographic_history:
                demo_texts.add(f"Date of birth: {demo.birth_year}")
                demo_texts.add(f"Race: {demo.race}")
                demo_texts.add(f"Gender: {demo.gender}")
                demo_texts.add(f"Insurance: {demo.insurance}")
                demo_texts.add(f"Language: {demo.language}")
                demo_texts.add(f"Marital status: {demo.marital_status}")
            demo = j.demographics
            demo_texts.add(f"Date of birth: {demo.birth_year}")
            demo_texts.add(f"Race: {demo.race}")
            demo_texts.add(f"Gender: {demo.gender}")
            demo_texts.add(f"Insurance: {demo.insurance}")
            demo_texts.add(f"Language: {demo.language}")
            demo_texts.add(f"Marital status: {demo.marital_status}")
        for cat in ["Race", "Insurance", "Language", "Marital status"]:
            demo_texts.add(f"{cat}: Unknown")
        all_texts.update(demo_texts)

    print(f"-> Found {len(all_texts):,} unique clinical and demographic text strings. Embedding with BioLORD-2023...")
    t_text = time.time()
    text_vectors = text_cache.get_embeddings(all_texts)
    print(f"-> Text strings embedded in {time.time() - t_text:.2f} s")

    demo_embeddings_dict = {t: text_vectors[t] for t in demo_texts if t in text_vectors} if args.encode_demographics else None
    if demo_embeddings_dict:
        print(f"-> Resolved {len(demo_embeddings_dict):,} demographic prefix text embeddings.")

    # Release BioLORD model from memory if loaded
    if hasattr(text_cache, "_model") and text_cache._model is not None:
        del text_cache._model
        del text_cache._tokenizer
        text_cache._model = None
        text_cache._tokenizer = None
        gc.collect()

    # Pre-encode unique CXR images in batches
    if not args.skip_media_encoding:
        all_cxrs: List[Tuple[str, Optional[str]]] = []
        for j in eligible_journeys.values():
            for ev in j.events:
                if ev.modality_id == Modality.CXR:
                    did = ev.source_row_id.replace("cxr_", "")
                    rel_p = ev.value_raw
                    all_cxrs.append((did, rel_p))
        if all_cxrs:
            print(f"-> Pre-encoding {len(all_cxrs):,} CXR images with RAD-DINO...")
            t_cxr = time.time()
            cxr_cache.batch_encode(all_cxrs, batch_size=32 if "cuda" in args.device else 8)
            print(f"-> CXR images encoded/verified in {time.time() - t_cxr:.2f} s")
            if hasattr(cxr_cache, "_model") and cxr_cache._model is not None:
                del cxr_cache._model
                del cxr_cache._processor
                cxr_cache._model = None
                cxr_cache._processor = None
                gc.collect()

    # Assign embeddings to event records
    print("-> Attaching embeddings to patient event records...")
    cxr_count = 0
    ecg_count = 0
    for j in tqdm(eligible_journeys.values(), desc="Attaching embeddings"):
        for ev in j.events:
            # 1. Text specifics
            if ev.has_specifics and ev.specifics_text:
                cleaned_text = ev.specifics_text.strip()
                if cleaned_text in text_vectors:
                    ev.specifics_vec = text_vectors[cleaned_text]

            # 2. CXR images
            if ev.modality_id == Modality.CXR and not args.skip_media_encoding:
                dicom_id = ev.source_row_id.replace("cxr_", "")
                rel_path = ev.value_raw
                cxr_vec = cxr_cache.get_embedding(dicom_id, relative_path=rel_path)
                if cxr_vec is not None:
                    ev.value_vec = cxr_vec
                    cxr_count += 1

            # 3. ECG waveforms
            if ev.modality_id == Modality.ECG and not args.skip_media_encoding:
                study_id = ev.source_row_id.replace("ecg_", "")
                rel_path = ev.value_raw
                ecg_vec = ecg_cache.get_embedding(study_id, rel_path)
                if ecg_vec is not None:
                    ev.value_vec = ecg_vec
                    ecg_count += 1

    # 5. Build Rolling Context Windows & Masks
    print(f"\n[Step 4/5] Building rolling windows (L={args.max_seq_len}, Overlap={args.overlap_len})...")
    builder = WindowBuilder(
        max_seq_len=args.max_seq_len,
        overlap_len=args.overlap_len,
        prefix_len=6,
        min_events_per_patient=args.min_events,
        max_val_dim=768,
        demo_embeddings_dict=demo_embeddings_dict,
    )

    num_eligible_patients = len(eligible_journeys)

    # Sharded saving configuration (avoids monolithic pickle memory blowup)
    shard_size = 100
    split_dirs = {
        "train": output_dir / "train",
        "val": output_dir / "val",
        "test": output_dir / "test",
    }
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    buffers: Dict[str, List[ModelWindow]] = {"train": [], "val": [], "test": []}
    shard_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    window_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    def flush_shard(split_name: str):
        if not buffers[split_name]:
            return
        shard_path = split_dirs[split_name] / f"shard_{shard_counts[split_name]:04d}.pt"
        torch.save(buffers[split_name], shard_path)
        shard_counts[split_name] += 1
        buffers[split_name].clear()
        gc.collect()

    total_windows = 0
    total_scored_tokens = 0
    category_counts: Dict[str, int] = {}
    modality_counts: Dict[str, int] = {}

    eligible_sids = list(eligible_journeys.keys())
    for sid in tqdm(eligible_sids, desc="Building & saving windows"):
        journey = eligible_journeys.pop(sid)
        split = patient_splits.get(sid, "train")
        pts_windows = builder.build_patient_windows(journey)

        for w in pts_windows:
            total_windows += 1
            window_counts[split] += 1
            total_scored_tokens += int(w.loss_mask.sum())

            # Accumulate scored category counts
            scored_cats = w.category_ids[w.loss_mask]
            for cat_id in scored_cats:
                cat_name = EventCategory(cat_id).name
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

            scored_mods = w.modality_ids[w.loss_mask]
            for mod_id in scored_mods:
                mod_name = Modality(mod_id).name
                modality_counts[mod_name] = modality_counts.get(mod_name, 0) + 1

            buffers[split].append(w)
            if len(buffers[split]) >= shard_size:
                flush_shard(split)

    # Flush remaining window buffers
    for split_name in list(buffers.keys()):
        flush_shard(split_name)

    print("\n[Step 5/5] Sharded dataset saved successfully:")
    for split_name in ["train", "val", "test"]:
        print(f"-> {split_name.upper()}: {window_counts[split_name]:,} windows across {shard_counts[split_name]} shards in {split_dirs[split_name]}")

    # Export Summary Ledger
    summary = {
        "total_requested_patients": len(target_sids),
        "eligible_patients": num_eligible_patients,
        "ineligible_patients": ineligible_count,
        "total_raw_events": total_raw_events,
        "total_windows": total_windows,
        "total_scored_tokens": total_scored_tokens,
        "windows_per_split": window_counts,
        "shards_per_split": shard_counts,
        "scored_category_distribution": category_counts,
        "scored_modality_distribution": modality_counts,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    summary_path = output_dir / "curation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CURATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Windows:       {total_windows:,}")
    print(f"Total Scored Tokens: {total_scored_tokens:,}")
    print(f"Saved to:            {output_dir}")
    print(f"Summary written to:  {summary_path}")
    print(f"Total Wall Time:     {summary['elapsed_seconds']} s")
    print("=" * 70)


if __name__ == "__main__":
    main()
