# M2 public encoder weight acquisition

Executed 2026-10-20 (America/New_York) in the clean development worktree. Source HEAD before acquisition: `2b07e5ccf66868f04c43a628d938b9bb2af09300`, with the new acquisition script captured separately by SHA-256. The researcher explicitly authorized downloading OTIS/EchoPrime into the shared model store. This phase acquires and registers assets; it does not select new preprocessing rules, execute real-record encoder inference, retrain NOAH or start bulk extraction.

## Registered assets

Model store: `/n/data1/hms/dbmi/rajpurkar/lab/home/wey371/models`. Paths/checksums are versioned in [encoder_assets.json](../../../configs/encoder_assets.json); current M2 paths are configured in [m2.json](../../../configs/m2.json). Both weights and their acquisition receipts are read-only. Source definitions, licenses and loading instructions are retained beside them under `upstream-source/`, with repository revision/file hashes. Weights remain outside Git.

| Asset | Relative path under the model store | Bytes | SHA-256 |
|---|---|---:|---|
| OTIS base | `otis/author-base-2024/otis_base.pth` | 108,005,017 | `e80a532ab83d9f8a4c4f080c4359d5b867aadc16a83a1d12882fefa6de876f55` |
| EchoPrime video encoder | `echoprime/v1.0.0/echo_prime_encoder.pt` | 138,642,379 | `7ca32e8bfde248bd6d8c7e46fdb7440385169af4dc2f416b5de840bdc2e64f3b` |

OTIS comes from the public Google Drive folder linked by the [official README](https://github.com/oetu/otis/blob/30282e9e2fd98511f9e33d90c2b9d86d5ae874e0/README.md). File ID `1AZVqME8nX5V2RjvF3moBPregb1lUDnxP`; source folder ID `1sMxJwvyZY7M2Z_gykcjLgIXa13iYDCEf`. The published folder size of 108,005,017 bytes matched the download. No upstream checksum was published; the local SHA-256 binds this acquired copy. The larger OTIS model was not downloaded because the base checkpoint supplies the required 192-D candidate.

EchoPrime comes from the [official v1.0.0 release](https://github.com/echonet/EchoPrime/releases/tag/v1.0.0), following its [official README](https://github.com/echonet/EchoPrime/blob/03874a5f203695f38968068c21584656d475b6b1/README.md). The 1,338,009,590-byte `model_data.zip` matched GitHub's published SHA-256 `b29362b6b40e8695b138d191f874ad92417b6da9fcdb2f0764ed1440be5272b3`. Only `model_data/weights/echo_prime_encoder.pt` was extracted, with ZIP member CRC `6e98630b` checked. The acquisition's temporary archive was removed automatically; text/view/retrieval models and report candidates were not retained. Final weight payload is 246,647,396 bytes, plus small receipts/source files; raw clinical datasets were not accessed or copied by acquisition.

## Executed compatibility checks

See [inspection.json](inspection.json) for executed tensor evidence. CPU inspection used the existing Slurm allocation 56795452, with an exact 2-CPU/8GiB/3-minute step request. No GPU inference or model-weight optimization ran.

- Restricted `torch.load(weights_only=True, map_location='meta')` inspected both files. OTIS contained the known `argparse.Namespace` metadata class, explicitly allowlisted; no arbitrary checkpoint globals were accepted. EchoPrime required no extra globals. No unrestricted pickle loading was used.
- OTIS has 292 state tensors, CLS width 192 and convolution patch tensor `[192,1,1,24]`. Saved arguments also specify patch `[1,24]`. This confirms the checkpoint's width and patch kernel without dropping or reinitializing weights. The saved temporal position table `[1,43,96]` and training `time_steps=1008` do not establish a full 5,000-sample adapter. Temporal/domain positions, full patch extraction and final patch-mean inference remain unverified.
- EchoPrime has 397 tensors. Its state strictly loads `torchvision.models.video.mvit_v2_s(weights=None)` with final `Linear(768,512)`, yielding 34,623,872 parameters. This is the 512-D video encoder, not the 523-D study representation. Strict structural loading does not verify DICOM regions, overlay removal, frame/crop/normalization policies or real-video output.
- Re-running acquisition returned `EXISTING_HASH_VERIFIED` for both assets without downloading or modifying either weight. This checks idempotence and rejects an existing registered weight if its checksum differs.

The model identities are public upstream **ASSUMED candidates**, not independently established NOAH author checkpoints. M2 real-source ECG/ECHO inference remains unverified. The current M2 smoke diagnostic records configured assets and the actual remaining adapter/preprocessing boundary, rather than reporting that these downloaded files are absent. Earlier executed smoke receipts are unchanged.

## Reproduction and preservation

From the clean development worktree:

```bash
/home/wey371/rajpurkarlab/home/wey371/.venvs/noah/bin/python -I tools/download_encoder_weights.py --asset all
```

`--asset otis` and `--asset echoprime` support independent acquisition; `--root` explicitly selects another model store. The downloader checks declared size, the published EchoPrime archive checksum, member CRC, final weight checksum and existing receipts. It uses exclusive writes and refuses conflicting existing files. It downloads no model code for execution and launches no inference or training. Each download has a 15-minute elapsed guard and 45-second socket timeout; temporary files are cleaned on normal exception/exit.

Per-asset immutable `asset_manifest.json` records acquisition time, official URL/file identity, source repository revision, checksum assurance and the fact that pipeline verification remains pending. Inspection source, config/source copies and executed checks are preserved privately in `artifacts/reduced_pilot/m2/m2-assets-20261020-01/`. Safe aggregate evidence alone is versioned here. No M1/M1.5 or earlier M2 source capture, patient manifest, checkpoint, prediction or cache was overwritten. See [verification.json](verification.json) for the current suite and preservation checks.

The next M2-B dependency is genuine-record ECG/ECHO preprocessing and frozen inference under explicitly recorded candidate policies. Existing M2-C/D, bulk-resource and alpha gates remain unchanged.
