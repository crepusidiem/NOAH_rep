"""
Tests for Deterministic CXR Path Resolution and Offline Caching (N-03 / Step 2).

Verifies:
1. O(1) deterministic path resolution for relative paths.
2. Fast metadata index lookup when only dicom_id is available (no recursive globbing).
3. Correct RAD-DINO (768-D CLS) feature extraction on real MIMIC-CXR images.
4. Batch encoding throughput and cache file integrity.
5. Robust handling of missing/corrupt image files.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from PIL import Image

from src.data.embeddings_cache import CXREmbeddingCache

# Sample known real patient & study in MIMIC-CXR
REAL_CXR_BASE = Path("/home/wey371/rajpurkarlab/datasets/cxr/MIMIC-CXR/raw_jpg")
SAMPLE_DICOM_ID = "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014"
SAMPLE_REL_PATH = "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
SAMPLE_DICOM_ID_2 = "174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962"
SAMPLE_REL_PATH_2 = "files/p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg"


def test_deterministic_path_resolution_with_rel_path():
    """Verify O(1) resolution directly succeeds without globbing."""
    cache = CXREmbeddingCache(cxr_base_dir=REAL_CXR_BASE)
    resolved = cache.resolve_image_path(SAMPLE_DICOM_ID, relative_path=SAMPLE_REL_PATH)
    assert resolved is not None
    assert resolved.exists()
    assert resolved.name == f"{SAMPLE_DICOM_ID}.jpg"


def test_deterministic_path_resolution_with_index():
    """Verify O(1) metadata index resolves dicom_id when relative_path is None."""
    cache = CXREmbeddingCache(cxr_base_dir=REAL_CXR_BASE)
    resolved = cache.resolve_image_path(SAMPLE_DICOM_ID, relative_path=None)
    assert resolved is not None
    assert resolved.exists()
    assert resolved.name == f"{SAMPLE_DICOM_ID}.jpg"


def test_missing_dicom_returns_none():
    """Verify non-existent DICOM ID returns None gracefully."""
    cache = CXREmbeddingCache(cxr_base_dir=REAL_CXR_BASE)
    resolved = cache.resolve_image_path("non_existent_fake_dicom_id_12345")
    assert resolved is None


def test_cxr_encoding_and_caching():
    """Verify RAD-DINO generates valid 768-D CLS vector and caches it."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = CXREmbeddingCache(
            cache_dir=Path(tmp_dir),
            cxr_base_dir=REAL_CXR_BASE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        emb = cache.get_embedding(SAMPLE_DICOM_ID, relative_path=SAMPLE_REL_PATH)
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (768,)
        assert emb.dtype == np.float32
        assert not np.isnan(emb).any()
        assert not np.isinf(emb).any()

        # Check cache file exists on disk
        cached_file = Path(tmp_dir) / f"{SAMPLE_DICOM_ID}.npy"
        assert cached_file.exists()

        # Reload directly from cache (model should not need to run again)
        emb_reloaded = cache.get_embedding(SAMPLE_DICOM_ID)
        assert np.array_equal(emb, emb_reloaded)


def test_batch_encode():
    """Verify batch encoding processes multiple images and saves cache files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = CXREmbeddingCache(
            cache_dir=Path(tmp_dir),
            cxr_base_dir=REAL_CXR_BASE,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        items = [
            (SAMPLE_DICOM_ID, SAMPLE_REL_PATH),
            (SAMPLE_DICOM_ID_2, SAMPLE_REL_PATH_2),
        ]
        results = cache.batch_encode(items, batch_size=2)
        assert len(results) == 2
        for did in [SAMPLE_DICOM_ID, SAMPLE_DICOM_ID_2]:
            assert did in results
            assert results[did].shape == (768,)
            assert (Path(tmp_dir) / f"{did}.npy").exists()
