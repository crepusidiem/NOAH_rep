# Server report: 2026-10-14_n01_baseline

- Checked at / timezone: 2026-10-14 13:30 EDT (America/New_York)
- Stage: inspection
- Repo / branch / source commit: `/home/wey371/rajpurkarlab/home/wey371/NOAH_rep`, branch `main`, commit `12af95a5a29024e8b2e4e9800307c81b4d0bbef8`
- Dirty tree at execution: no (clean working tree)
- Interpreter / environment: `/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python` (Python 3.12.15, PyTorch 2.14.1, TorchVision 0.29.1, Polars 2.0.0, Pandas 3.0.6, PyArrow 26.0.0, SciPy 1.18.1, PyTest 9.1.1)
- Data root / scope: `/home/wey371/rajpurkarlab/datasets`
  - MIMIC-IV v2.2 (`hosp`, `icu`): verified present (27 GB in `hosp` directory)
  - MIMIC-IV-ED v2.2: verified present
  - MIMIC-IV-Note v2.2: verified present
  - MIMIC-IV-ECG v1.0: verified present (800k+ WFDB records)
  - MIMIC-CXR-resized v2.0.0: verified present
  - MIMIC-IV-ECHO v1.0: NOT FOUND (missing on filesystem; agreed pilot scope uses 5 modalities)
- Slurm job / state / exit code: NOT_RUN (interactive node execution for baseline verification; background extraction completed with exit code 0)
- Command / config / seed:
  - `PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/pytest tests/` (exit code 0, 6 passed)
  - `PYTHONPATH=. /home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python scripts/test_pilot_extraction.py` (exit code 0)
  - Cohort manifest: `data/pilot_cohort_manifest.csv` (1,000 subjects, seed=42)

## Results
- Cohort Manifest: 1,000 unique subjects (700 train / 150 val / 150 test).
- Test Extraction Sample (2 subjects):
  - Subject `18003894`: 550 events extracted, scorable eligibility check (>=64 events) passed.
  - Subject `10109710`: 332 events extracted, scorable eligibility check (>=64 events) passed.
- Unit Tests: 6/6 tests passed in `test_temporal_oracle.py`.
- Foundation Encoders: BioLORD-2023, RAD-DINO, OTIS weights not yet downloaded in `scratch/cache/huggingface` or `scratch/cache/torch`.

## Outputs
- `reports/server/2026-10-14_n01_baseline/REPORT.md`: This execution snapshot.
- `data/pilot_cohort_manifest.csv`: Existing 1,000-candidate split manifest.

## Failures and limitations
- Uncompressed 13 GB `labevents.csv` scan in Python `csv.reader` took ~5 minutes for 2 patients. Full cohort extraction must be optimized with columnar or indexed reader (e.g., DuckDB / Polars).
- MIMIC-IV-ECHO is missing; pilot restricted to 5 modalities by PI agreement.

## Next action
- Proceed to P0 / N-02 (Freeze initial pilot scope and minimal configurations in `configs/pilot_config.yaml`).
- Proceed to P0 / N-03 (Verify frozen encoder candidates BioLORD, RAD-DINO, OTIS 1x24 on sample records).
