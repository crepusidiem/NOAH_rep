# NOAH data interface and preparation plan

Updated: 2026-10-19. Current evidence: [architecture audit](ARCHITECTURE_AUDIT.md). See [PROJECT_STATUS.md](PROJECT_STATUS.md) for progress and [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for encoding steps. The 1,000-candidate manifest has 700/150/150 assignments; event readers are partially implemented. The near-term target is a five-modality small-scale training path. The full six-modality target also requires ECHO. Inventory statements below come from the researcher-provided server report; record actual reading and encoding results by source.

## 1. The paper's data interface

The interface is NOAH's unified input representation, rather than a required network API. The paper does not supply a complete data-file schema or table-to-event implementation.

Organize raw records into timestamped patient event sequences, then project each event into a shared-width model token:

1. Source layer: hospital/ICU/ED relational tables, clinical text, chest radiographs, ECG waveforms and ECHO videos.
2. Event layer: available lifetime records linked by `subject_id`, including multiple encounters and outpatient records; tests, administrations and transfers retain distinct times/types.
3. Representation layer: trainable event-category embeddings, frozen text specifics and content/numeric representations, projected to width 768.
4. Training layer: time features, six demographic prefixes, windows, attention masks and loss masks.

| Event information | Purpose | Information to retain during preparation |
|---|---|---|
| subject_id | Cross-source joins and patient-disjoint splits | Source patient ID, table and row identifier |
| hadm_id / stay_id / study_id | Encounter/study linkage; not present for every event | Allow missing associations; preserve outpatient events without admission IDs |
| Event time | Ordering, age and directional gaps | Original timestamp and separate availability/storage time for information-boundary checks |
| High-level category | 31 categories, including admission, tests and process starts/ends | Explicit source-table/item mappings |
| Type specifics | Drug, test or diagnosis names | Original item dictionaries and descriptive text |
| Content modality/value | Categorical, numeric, text, CXR, ECG or ECHO | Values/units, text, media paths or frozen embeddings |
| Missing/end/source metadata | Semantics and provenance | Distinguish missing from zero; emit process-end events at actual termination |

This table proposes an engineering interface, not a verbatim paper schema. Shared model width 768 does not imply uniform external dimensions: ECG=192, ECHO=512, text/CXR=768. Numeric 44-D follows a candidate sin/cos pair for each of 22 scales; the complete formula/convention is not explicitly published.

## 2. Reported server inventory and remaining checks

Reported dataset root:

```text
/home/wey371/rajpurkarlab/datasets
```

| Source | Paper version | Supplied report | Current evidence level | Required checks |
|---|---|---|---|---|
| MIMIC-IV | 2.2 | hosp/icu, about 30 GB, reported complete | Reported stored | Tables/subdirectories, version files, compression basis, rows, permissions and sample reads |
| MIMIC-IV-ED | 2.2 | About 704 MB, reported complete | Reported stored | Tables, version, patient joins and readability |
| MIMIC-IV-Note | 2.2 | About 6.3 GB, reported complete | Reported stored | Note tables, timestamps, actual text reads, version and authorization |
| MIMIC-IV-ECG | 1.0 | 800,035 WFDB records, reported complete | Waveforms reported stored | Index-to-.hea/.dat pairing, file existence, sampled decoding and failures |
| MIMIC-CXR-JPG | 2.0.0; paper calls it 2.0 | Images/metadata, reported complete | Images reported stored | Image/metadata correspondence, patient/study IDs, decoding and version |
| MIMIC-IV-ECHO | 1.0 | Researcher supplied1.0.1 path | Directory exists; video decoding/encoding NOT_RUN | `/n/data1/hms/dbmi/rajpurkar/lab/datasets/mimic-iv-echo/1.0.1/`; verify version difference, mapping and DICOM pipeline |

Capacity and directory names alone do not establish completeness. Index existence, media presence, decoding, user authorization and usable training inputs are separate checks. Mounts/free space, symlinks, ACL/group permissions, checksums and read-success rates remain to be recorded.

Record six source states independently: path exists, version matches, files/tables complete, current user can read, authorization confirmed, and prepared events/embeddings usable.

## 3. Data semantics and scope

Configure time units, birth-year proxies, aggregation windows and numeric Fourier layouts separately; see [ASSUMPTIONS.md](ASSUMPTIONS.md). ECHO event share and patient coverage are different statistics. The full six-modality target requires actual videos and their encoder path.

Sources and modalities are not one-to-one. Relational tables produce both categorical and numeric content. The five reported sources can cover categorical, numeric, text, CXR and ECG; this is not a complete six-modality dataset.

MIMIC-IV sets `anchor_age=91` for patients older than 89 and uses patient-specific date shifts. A proxy birth year is not a true DOB; shifted calendar years cannot directly compare patients' true calendar time. See [MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/).

## 4. Preparation work packages

Distinguish paper requirements from engineering acceptance choices. Record configurations and preserve original files.

| Package | Processing | Output and acceptance | Paper evidence / unresolved choice |
|---|---|---|---|
| A. Inventory/version/reading | List source tables/files, versions and checksums; distinguish missing, corrupt, unreadable and unauthorized | Inventory, sample-read results/failures and reproducible counts | Engineering acceptance; reported GB is not a completeness criterion |
| B. Patient/event mapping | Join by subject_id; retain encounter/study links; map 31 categories; check one-to-many duplication | Event dictionary, source mappings/row IDs and pre/post-join counts | pp.15,17; exact SQL/dictionaries not published |
| C. Numeric/time semantics | Parse numeric/text states, units and missingness; sort/tie-break; preserve starts/ends; inspect event versus availability time | Traceable timelines, unit rules, duplicate/anomaly policies | Units, ties and birth-year proxies are assumptions; no blanket filling/clipping |
| D. Vital aggregation | Sliding-window min/max for dense measurements; mean-time stamps; retain contributing events | Pre/post counts/distributions and availability tests | p.15; span/stride/min-max representation unspecified, distinct from NEWS2's 1h window |
| E. Text/CXR content | Retain clinical text/time; associate image metadata; frozen BioLORD/RAD-DINO extraction | Embeddings, model/processor hashes and failures | pp.15-17; long-text policy, image details and exact checkpoints remain |
| F. ECG | Validate .hea/.dat and record_list; leads, rate, length, nonfinite values and decoding; frozen OTIS final-patch means | Traceable 192-D record embeddings | p.15; 12 leads, 500Hz, 10s, 2,496 patches; tail unspecified |
| G. Windows/masks | Freeze 70/15/15 patient split; check at least 64 scored tokens; 2048 windows/256 overlap and six prefixes | Patient/window manifests and exactly-once clinical target scoring | p.15; prefix budget, tails and metadata snapshot require choices |
| H. Cache/trainable inputs | Align content projection interfaces, modality/presence masks, time and loss ownership | Loadable batch passing dimensions, target-shift and future-information checks | pp.20-21; raw storage alone is not training readiness |

Retain real text, images and waveforms. Diagnosis labels or automated ECG measurements do not replace the specified content path. WFDB records contain .hea/.dat pairs; device timestamps may differ from other records. Check cross-source consistency rather than deleting every ECG outside an admission interval. See [ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/).

### Scientific choices for PI guidance

- Time units, age/birth-time proxies and simultaneous-event ordering.
- Dense-vital aggregation span/stride and min/max representation.
- Literal window-end metadata versus a separate strict as-of causal audit.
- Future-gap handling at representation cutoffs and in free rollout.
- Five-modality pilot scope and timing of complete six-modality work.

These choices change time features, samples and information boundaries; record them beyond a generic cleaning label.

## 5. Acquisition versus preparation

Public access-policy check date: 2026-10-13. These describe source policies, not verified personal account permissions.

| Data/resource | Source situation | Project action |
|---|---|---|
| MIMIC-IV / ED / Note / CXR-JPG | Restricted access; credentialing, required training and project DUA | Prepare an authorized readable copy; otherwise resolve access |
| MIMIC-IV-ECG 1.0 waveforms | Open access under its license; not a blanket statement about text reports/other platforms | Verify pairing/decoding of reported files; obtain missing files for that version if needed |
| MIMIC-IV-ECHO | Supplied1.0.1 directory exists; paper1.0 | Verify DICOM/video/index reads and preprocessing; record version difference |
| BioLORD / RAD-DINO / OTIS / EchoPrime assets | Frozen encoders specified; server asset status not established | Inventory weights, versions, conditions and inference environment; these are model assets, not new patients |
| Prepared events and embedding caches | Raw data reported; complete NOAH inputs not established | Preprocess, encode offline and validate |

ECHO may follow the prototype under an agreed scope; the full target still requires video content, not only ultrasound measurements. The paper uses 1.0; record any version substitution explicitly.

Sources: [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/), [ED](https://physionet.org/content/mimic-iv-ed/2.2/), [Note](https://physionet.org/content/mimic-iv-note/2.2/), [CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), [ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), [ECHO](https://physionet.org/content/mimic-iv-echo/1.0/).

## 6. Execution order and PI discussion

1. Verify source reads/assets and complete two patients' events, real encoding and windows.
2. Verify synthetic temporal alignment, dual reconstruction, masks and prior-only rollout.
3. Expand to the fixed 1,000 candidates, report retained counts and complete small-scale training, validation and recovery.
4. Plan 4-6 weeks for the first complete run under the accepted one-engineer, usable-data/weights/GPU assumption.

Seek PI guidance on pilot modalities, ECHO timing and key temporal semantics. Base full-scale compute requests on measured pilot throughput, memory and cache costs. See [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md) for dependencies and gates.
