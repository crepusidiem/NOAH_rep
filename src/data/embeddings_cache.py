"""
High-Throughput Offline Embedding Cache for Multimodal Records.

Handles offline feature extraction and local caching for:
1. Text Specifics & Clinical Notes: FremyCompany/BioLORD-2023 (768-D)
2. Chest Radiographs (CXR): microsoft/rad-dino CLS (768-D)
3. Electrocardiograms (ECG): OTIS 1x24 patch-mean (192-D)
4. Echocardiograms (ECHO): Modular hook for EchoPrime global CLS (512-D)
"""

import os
from pathlib import Path
import sqlite3
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import torch
from PIL import Image
import wfdb

# Transformers and vision models
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor


class TextEmbeddingCache:
    """
    Caches BioLORD-2023 (768-D) text embeddings by text string in a local SQLite database.
    Since clinical item names, drugs, and categories repeat frequently,
    caching by unique string eliminates ~99% of redundant encoder forward passes.
    """
    def __init__(
        self,
        cache_db_path: Path = Path("data/cache/text_embeddings.sqlite"),
        model_id: str = "FremyCompany/BioLORD-2023",
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 128,
    ):
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._init_db()

        self._tokenizer = None
        self._model = None

    def _init_db(self):
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS text_cache (
                    text_content TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def _lazy_load_model(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def get_embeddings(self, texts: Iterable[str]) -> Dict[str, np.ndarray]:
        """
        Retrieves cached embeddings for texts, and batches/computes missing ones.
        """
        unique_texts = list(set(t.strip() for t in texts if t and t.strip()))
        if not unique_texts:
            return {}

        results: Dict[str, np.ndarray] = {}
        missing_texts: List[str] = []

        # 1. Query SQLite cache
        with sqlite3.connect(self.cache_db_path) as conn:
            # Chunk queries to stay well within SQLite variable limits
            chunk_size = 500
            for i in range(0, len(unique_texts), chunk_size):
                chunk = unique_texts[i : i + chunk_size]
                placeholders = ",".join("?" * len(chunk))
                cursor = conn.execute(
                    f"SELECT text_content, embedding FROM text_cache WHERE text_content IN ({placeholders})",
                    chunk,
                )
                for row_text, blob in cursor.fetchall():
                    results[row_text] = np.frombuffer(blob, dtype=np.float32)

        for t in unique_texts:
            if t not in results:
                missing_texts.append(t)

        # 2. Compute missing texts in batches
        if missing_texts:
            self._lazy_load_model()
            new_rows: List[Tuple[str, bytes]] = []

            for i in range(0, len(missing_texts), self.batch_size):
                batch_texts = missing_texts[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                with torch.no_grad():
                    out = self._model(**inputs)
                    # Mean pooling over valid attention tokens
                    attn_mask = inputs["attention_mask"].unsqueeze(-1)
                    embs = (out.last_hidden_state * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
                    # L2 normalize
                    embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

                batch_np = embs.cpu().to(torch.float32).numpy()
                for text_str, emb_vec in zip(batch_texts, batch_np):
                    results[text_str] = emb_vec
                    new_rows.append((text_str, emb_vec.tobytes()))

            # Insert into database
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.executemany("INSERT OR REPLACE INTO text_cache VALUES (?, ?)", new_rows)
                conn.commit()

        return results


class CXREmbeddingCache:
    """
    Caches RAD-DINO (768-D CLS) embeddings for chest radiographs.
    Stores arrays in a local directory keyed by dicom_id.
    Uses deterministic O(1) filesystem path resolution without recursive globbing.
    """
    def __init__(
        self,
        cache_dir: Path = Path("data/cache/cxr"),
        cxr_base_dir: Path = Path("/home/wey371/rajpurkarlab/datasets/cxr/MIMIC-CXR/raw_jpg"),
        model_id: str = "microsoft/rad-dino",
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cxr_base_dir = Path(cxr_base_dir)
        self.model_id = model_id
        self.device = device

        self._processor = None
        self._model = None
        self._index: Optional[Dict[str, str]] = None

    def _lazy_load_model(self):
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def _build_index_if_needed(self):
        """Builds an in-memory dictionary mapping dicom_id -> relative path in <1s using metadata."""
        if self._index is not None:
            return
        self._index = {}
        meta_candidates = [
            self.cxr_base_dir / "mimic-cxr-2.0.0-metadata.csv.gz",
            self.cxr_base_dir / "mimic-cxr-2.0.0-metadata.csv",
            self.cxr_base_dir / "metadata.csv",
        ]
        meta_file = next((f for f in meta_candidates if f.exists()), None)
        if meta_file is None:
            return

        try:
            import duckdb
            con = duckdb.connect()
            rows = con.execute(f"""
                SELECT dicom_id, subject_id, study_id
                FROM read_csv('{meta_file}', header=true, quote='\"', escape='\"', null_padding=true, ignore_errors=true, parallel=false)
            """).fetchall()
            for did, sid, stid in rows:
                if did and sid and stid:
                    self._index[str(did)] = f"files/p{str(sid)[:2]}/p{sid}/s{stid}/{did}.jpg"
            con.close()
        except Exception as e:
            # Fallback gzip/csv parser
            import gzip, csv
            opener = gzip.open if str(meta_file).endswith(".gz") else open
            try:
                with opener(meta_file, "rt") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        did = row.get("dicom_id")
                        sid = row.get("subject_id")
                        stid = row.get("study_id")
                        if did and sid and stid:
                            self._index[str(did)] = f"files/p{str(sid)[:2]}/p{sid}/s{stid}/{did}.jpg"
            except Exception as e2:
                print(f"Warning: Failed to index CXR metadata from {meta_file}: {e2}")

    def resolve_image_path(self, dicom_id: str, relative_path: Optional[str] = None) -> Optional[Path]:
        """
        Deterministically resolves the image file path in O(1) time without filesystem globbing.
        """
        # 1. Direct relative path if provided
        if relative_path:
            cand = self.cxr_base_dir / relative_path
            if cand.exists():
                return cand

        # 2. In case dicom_id itself contains relative path
        if "/" in dicom_id or "\\" in dicom_id:
            cand = self.cxr_base_dir / dicom_id
            if cand.exists():
                return cand

        # 3. Lookup in metadata index
        self._build_index_if_needed()
        if self._index and dicom_id in self._index:
            cand = self.cxr_base_dir / self._index[dicom_id]
            if cand.exists():
                return cand

        return None

    def get_embedding(self, dicom_id: str, relative_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Retrieves cached 768-D embedding or computes it from the JPG image.
        """
        clean_id = Path(dicom_id).stem if "/" in dicom_id else dicom_id
        cache_file = self.cache_dir / f"{clean_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        img_path = self.resolve_image_path(clean_id, relative_path)
        if img_path is None or not img_path.exists():
            return None

        # Process and encode
        self._lazy_load_model()
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model(**inputs)
                cls_emb = out.last_hidden_state[:, 0, :].cpu().to(torch.float32).numpy().squeeze(0)

            np.save(cache_file, cls_emb)
            return cls_emb
        except Exception as e:
            print(f"Warning: Failed to encode CXR {clean_id} at {img_path}: {e}")
            return None

    def batch_encode(
        self,
        items: List[Tuple[str, Optional[str]]],
        batch_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        High-throughput batch encoding for a list of (dicom_id, relative_path) tuples.
        Skips already cached embeddings and performs batched GPU/CPU forward passes.
        """
        results: Dict[str, np.ndarray] = {}
        pending_items: List[Tuple[str, Path]] = []

        for did, rel_path in items:
            clean_id = Path(did).stem if "/" in did else did
            cache_file = self.cache_dir / f"{clean_id}.npy"
            if cache_file.exists():
                results[clean_id] = np.load(cache_file)
            else:
                p = self.resolve_image_path(clean_id, rel_path)
                if p is not None and p.exists():
                    pending_items.append((clean_id, p))

        if not pending_items:
            return results

        self._lazy_load_model()
        for i in range(0, len(pending_items), batch_size):
            chunk = pending_items[i : i + batch_size]
            imgs = []
            valid_ids = []
            for clean_id, p in chunk:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(img)
                    valid_ids.append(clean_id)
                except Exception as e:
                    print(f"Warning: Corrupt CXR image at {p}: {e}")

            if not imgs:
                continue

            try:
                inputs = self._processor(images=imgs, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self._model(**inputs)
                    embs = out.last_hidden_state[:, 0, :].cpu().to(torch.float32).numpy()

                for clean_id, emb in zip(valid_ids, embs):
                    np.save(self.cache_dir / f"{clean_id}.npy", emb)
                    results[clean_id] = emb
            except Exception as e:
                print(f"Warning: Batch encoding failure on batch {i}: {e}")

        return results


class ECGEmbeddingCache:
    """
    Caches OTIS (192-D patch mean) embeddings for 12-lead ECG waveforms.
    Loads raw 500Hz signals via wfdb and runs OTIS with 1x24 patch adapter.
    """
    def __init__(
        self,
        cache_dir: Path = Path("data/cache/ecg"),
        ecg_base_dir: Path = Path("/home/wey371/rajpurkarlab/datasets/mimic-iv-ecg/1.0"),
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ecg_base_dir = Path(ecg_base_dir)
        self.device = device
        self._model = None

    def _lazy_load_model(self):
        if self._model is None:
            import sys
            otis_path = str(Path("vendor/otis").resolve())
            if otis_path not in sys.path:
                sys.path.insert(0, otis_path)
            from models_vit import vit_baseDeep_patchX
            domains = {"ecg": (1, 12, 5000)}
            self._model = vit_baseDeep_patchX(
                img_size=(1, 12, 5000),
                domains=domains,
                patch_size=(1, 24),
                global_pool=True,
            ).to(self.device)
            self._model.eval()

    def get_embedding(self, study_id: str, relative_path: str) -> Optional[np.ndarray]:
        """
        Retrieves cached 192-D ECG embedding or loads WFDB waveform and runs OTIS.
        """
        cache_file = self.cache_dir / f"{study_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        # Remove extensions if any
        clean_rel = relative_path.replace(".hea", "").replace(".dat", "")
        wfdb_record_path = self.ecg_base_dir / clean_rel
        if not (self.ecg_base_dir / f"{clean_rel}.hea").exists():
            return None

        self._lazy_load_model()
        try:
            record = wfdb.rdrecord(str(wfdb_record_path))
            signal = record.p_signal  # [time=5000, leads=12]
            
            # Robust verification & padding/trimming to exactly 5000 samples and 12 leads
            if signal.shape[0] < 5000:
                pad_len = 5000 - signal.shape[0]
                signal = np.pad(signal, ((0, pad_len), (0, 0)), mode="constant")
            elif signal.shape[0] > 5000:
                signal = signal[:5000, :]

            # Clean NaNs or Infs
            signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

            # Shape for OTIS: [B=1, C=1, V=12, T=5000]
            sig_tensor = torch.from_numpy(signal.T).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # Positional embedding for leads: 1..12
            Tp = 5000 // 24 # 208
            idx = torch.arange(12, device=self.device) + 1
            pos_embed_y = idx.view(1, 12, 1).expand(1, 12, Tp).contiguous()

            with torch.no_grad():
                emb = self._model.forward_features(sig_tensor, pos_embed_y)
                emb_np = emb.cpu().to(torch.float32).numpy().squeeze(0)

            np.save(cache_file, emb_np)
            return emb_np
        except Exception as e:
            print(f"Warning: Failed to load/encode ECG {study_id} at {wfdb_record_path}: {e}")
            return None


class EchoEmbeddingCache:
    """
    Modular placeholder for Echocardiography Video Cache (EchoPrime 512-D global CLS).
    Plug-and-play hook ready to be activated when MIMIC-IV-ECHO is mounted.
    """
    def __init__(self, cache_dir: Path = Path("data/cache/echo")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding(self, study_id: str, video_path: Optional[str] = None) -> Optional[np.ndarray]:
        cache_file = self.cache_dir / f"{study_id}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        # When ECHO videos are mounted, load EchoPrime model and embed [N, 3, 16, 224, 224]
        return None
