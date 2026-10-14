# Understanding MIMIC data and NOAH preparation

Updated: 2026-10-14. This introduction explains source data and the existing code baseline `30dd669`. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for current implementation and execution evidence.

## 1. What MIMIC contains

MIMIC contains deidentified clinical records. A hospital admission generates many records: admission times, ward transfers, lab tests/results, medication orders/administrations, clinical notes, imaging and ECGs. They are stored in separate tables/files by source system.

Think of separate hospital registers: `patients` contains patient information, `admissions` hospital encounters, `labevents` lab results, and `prescriptions`/`emar` different medication information. IDs connect them; one file is not a complete patient record.

MIMIC-IV `hosp` covers hospital data and `icu` intensive-care systems. ED is emergency care; Note is clinical text; CXR is chest radiography; ECG records electrical cardiac signals; ECHO is echocardiography. A patient in one source need not appear in another. See [MIMIC-IV 2.2](https://physionet.org/content/mimiciv/2.2/).

| Common ID | Meaning | Join considerations |
|---|---|---|
| subject_id | Patient | Links multiple encounters of the same patient |
| hadm_id | Hospital admission | Several per patient; outpatient events may lack it |
| stay_id | Stay in a source module | ICU and ED namespaces/semantics remain distinct |
| study_id | Examination/study identifier | Matching numbers alone do not link ECG, CXR and ECHO |
| itemid | Measurement/clinical item | Use the corresponding dictionary for names, units and semantics |

Dates are deidentified. Within-patient intervals are preserved; shifted calendar years across patients are not the same real calendar. `anchor_year`/`anchor_age` support an age proxy, not exact DOB recovery. `anchor_age=91` represents patients older than 89. See [official time/age information](https://physionet.org/content/mimiciv/2.2/).

## 2. Sources versus modalities

A source identifies the database/system; a modality identifies the form of an event's content. Six sources do not map one-to-one to six modalities.

| NOAH modality | Interpretation | Raw example | Paper content representation |
|---|---|---|---|
| Categorical | Clinical/administrative state or action | Admission, transfer, medication start | Trainable category representation with optional text specifics |
| Numeric | Clinically meaningful scalar with units | Heart rate, blood pressure, lab result | Fourier representation, then projection |
| Text | Clinical narrative | Discharge summary, radiology report | Frozen BioLORD, 768-D |
| CXR | Static image | Chest radiograph pixels | Frozen RAD-DINO, 768-D |
| ECG | Electrical signal over time | Multilead voltage waveform | Frozen OTIS, 192-D |
| ECHO | Cardiac motion video | Ultrasound frames | Frozen EchoPrime, 512-D |

Dimension means the number of values in a feature vector, not pixels, leads or disease classes. An embedding is an encoder's numerical representation of content, not a diagnosis or a guarantee that raw content can be reconstructed.

Relational tables can produce both categorical and numeric events. Text can provide content or type specifics. Medication start is an event category; drug name is a specific; dosage can be numeric. Category describes what happened, while modality describes how content is represented.

## 3. Why NOAH requires preparation

Hospital tables are organized by workflow; NOAH consumes patient timelines. Preparation makes type, unit, time and missingness conventions explicit.

Entirely fictional teaching example:

| Time | Raw record | Unified event |
|---|---|---|
| 09:00 | Admission table row | Categorical admission event, with admission-type specifics |
| 09:15 | Heart rate 82 bpm | Numeric vital event, heart-rate specifics, value 82 |
| 10:00 | ECG index path | ECG event; decode waveform and extract OTIS features |
| 11:00 | CXR study/image IDs | CXR event; decode pixels and extract RAD-DINO features |
| 12:00 | Clinical/imaging report | Text event; extract BioLORD features |

Each patient has a distinct sequence. Inputs combine category, specifics, content, event time and directional gaps.

Separate process starts and ends. For medication starting at 09:00 and ending at 11:00, put the end event at 11:00. Adding its known two-hour duration to the 09:00 input would expose future information.

## 4. Practical preparation order

### 4.1 Establish what is readable

Path existence, completeness, decoding, access authorization and prepared samples are different checks. ECG `record_list.csv` supplies an index; read paired .hea/.dat files to establish signal loading. CXR metadata does not establish image completeness.

Code records processing rules. Execution output establishes a particular run. Cache files/checksums establish reusable artifacts. Keep these evidence levels distinct.

### 4.2 Link patients and clinical items

Join by `subject_id`; preserve admission/study IDs and source rows. Connect lab items to names/units so that clinical concepts such as creatinine remain traceable. Check pre/post-join counts to prevent one-to-many event duplication.

### 4.3 Interpret values, units and missingness

The same measurement can use different units; conversion needs declared rules. True zero, null and nonnumeric text differ. Do not fill all with zero or discard all textual lab results: a result such as negative can be clinically informative and needs a defined representation.

The current lab reader retains parseable `valuenum` and uses `item_<id>` specifics without a name/unit dictionary. This is partial coverage, not complete lab semantics.

### 4.4 Ordering and temporal features

Define event and availability times. `charttime` often indicates clinical recording time; `storetime` indicates system entry, with source-specific meanings. They may differ.

Current code uses hourly gaps, age in years and category/source-ID tie ordering. These are code choices, not necessarily paper settings.

The final observed event does not imply a simultaneous successor. Unknown next gaps differ from true zero. Current final `delta_next=0` needs boundary availability/target masks during training.

### 4.5 Aggregate dense vital signs

ICU records can be frequent enough to dominate tokens. The paper uses sliding-window min/max, stamped at mean time.

For heart rates 80,82,100, min=80 and max=100 preserve a range. A maximum from later in the window requires an availability check when predicting earlier time points.

The helper defaults to a 2h span and 1h stride. These are assumptions and are not connected to the current two-patient extraction entry. The paper does not supply those values; they differ from NEWS2's 1h label window.

### 4.6 Read and encode each modality

| Content | Preparation | Encoder relationship |
|---|---|---|
| Text | Preserve content/time; define long/empty-text handling | Frozen BioLORD, 768-D; note IDs alone are insufficient |
| CXR | Link patient/study/image, decode and version transforms | RAD-DINO consumes pixels, not labels or metadata |
| ECG | Validate header/signal pairs, leads, sample rate/count and anomalies | OTIS consumes waveforms, not automated-measurement CSVs |
| ECHO | Decode, extract region/remove overlays, sample/normalize frames | EchoPrime consumes videos, not only ultrasound measurements |
| Numeric | Preserve clinical item, units/presence and consistent Fourier basis | No external pretrained encoder; record phase/scaling |

WFDB ECG records pair .hea and .dat files. Device timestamps may differ from other clinical records; an outpatient ECG outside an admission interval is not automatically erroneous. See [ECG 1.0](https://physionet.org/content/mimic-iv-ecg/1.0/).

Frozen means external parameters are not updated by NOAH training. Offline precomputation runs encoders first and caches vectors for training. Record exact weights and preprocessing; a same-named release need not produce identical features. See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed recipes.

### 4.7 Build model windows

The paper uses 2048-token windows, 256 overlap and six demographic prefixes. Overlap supplies context without scoring its clinical targets twice.

Attention masks determine visible positions; loss masks determine scored targets. They are distinct.

Split 70/15/15 by patient, not by encounters. A patient's admissions cannot be distributed across training/test. The manifest's 700/150/150 assignments establish this ratio, not complete event extraction or sample readiness.

## 5. Current repository progress

| Artifact | Established content | Remaining evidence/work |
|---|---|---|
| `pilot_cohort_manifest.csv` | 1,000 unique IDs; 700 train/150 val/150 test | Media readability, complete events and at least 64 valid scored events |
| `subcohort.py` | Adult admission/ECG/CXR intersection, seed=42 | This enriched pilot differs from the paper's full cohort distribution |
| `extract_events.py` | Admission/discharge/death, ECG/CXR metadata and numeric lab code | Remaining categories, text, medications and ED/ICU details |
| `numeric.py` | Candidate 44-D dyadic sine/cosine encoding | Author-exact phase/scaling remain unspecified |
| `vitals.py` | Min/max helper, defaults 2h/1h | Connection and execution in the main pilot path |
| `test_pilot_extraction.py` | Four reader methods for the first two patients, with printed sequences | 1,000-patient ETL/windows/embedding caches |
| `run_pilot_gpu.sh` | Resource request/environment setup and nvidia-smi entry | Actual allocation, encoder execution and training |

The patient manifest and partial event code are available. Complete clinical content, actual reading/encoding and acceptance remain before trainable samples are ready. Current CXR paths point to `mimic-cxr-resized/2.0.0`; verify source, resolution and transforms against the paper's CXR-JPG path.

## 6. Topics for PI guidance

1. Whether to validate the pipeline on the enriched 1,000-candidate pilot before returning to natural modality missingness.
2. Selected event categories, dictionaries, text and media to integrate next.
3. Time units, age proxy, vital windows and metadata snapshots as declared assumptions.
4. ECHO acquisition/shared-copy search and availability of all four external encoder assets.
5. Data-readiness evidence: event provenance/counts, real-content success rates, caches, windows and masks.

Data preparation does not resolve every architecture ambiguity. Record five-modality pilots, full six-modality reproduction and information-boundary audits separately.

## 7. Evidence sources

Paper: [NOAH.pdf](../paper/NOAH.pdf), pp.15-17,20,49-50; see [PAPER_TRACEABILITY.md](PAPER_TRACEABILITY.md).

Code baseline `30dd669`: `data/pilot_cohort_manifest.csv`; `src/data/schema.py`, `subcohort.py`, `extract_events.py`, `numeric.py`, `vitals.py`; `scripts/test_pilot_extraction.py`; `sbatch/run_pilot_gpu.sh`.

Current implementation/execution evidence: [PROJECT_STATUS.md](PROJECT_STATUS.md).
