# W&B → SFTP Backup Pipeline

Backs up every run in the `theta-tech-ai/semler-qfhd` Weights & Biases project
(metadata, config, summary, history, and run files) to Semler Scientific's
SFTP server, with a fully restart-safe, resumable design. Originally
targeted Google Drive as the destination; that's now legacy (see
[Legacy: Google Drive migration](#legacy-google-drive-migration) below).

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install wandb paramiko

# W&B auth - either export WANDB_API_KEY, or `wandb login` once
# (writes to ~/.netrc)

# SFTP credentials - JSON file, chmod 600, NOT committed to this repo:
#   {"host": "files.semlerscientific.net", "port": 22,
#    "username": "...", "password": "..."}
# Path is set by SFTP_CREDENTIALS_PATH in sftp_backup_lib.py
# (defaults to /home/ubuntu/.semler_sftp_credentials.json)

python3 wandb_backup_parallel.py
```

Safe to stop (Ctrl+C) and rerun at any time — it picks up exactly where it
left off.

## Scripts

| Script | Purpose |
|---|---|
| `wandb_backup_parallel.py` | **Main entry point.** Full project backup, 12 concurrent workers. See [Design](#design) below. |
| `wandb_backup_local.py` | Same backup logic, sequential (single-threaded). Useful for debugging without concurrency noise. |
| `wandb_backup_colab.py` | Same again, tuned for running in a Google Colab notebook cell. |
| `wandb_ckpt_backup.py` | Checkpoint-only pass: downloads/uploads `model`-type logged artifacts (`.ckpt` files) for runs listed in `ckpt_target_run_ids.txt`. Every other run is only checked for *whether* it has checkpoints (recorded in its own manifest), never downloaded. |
| `sftp_backup_lib.py` | Shared SFTP helpers (connection pooling, retry, atomic upload, remote directory resolution) used by the backup scripts above. |
| `migrate_drive_to_sftp.py` | One-time migration: copies the old Google-Drive-backed tree to SFTP without re-touching W&B. |
| `drive_sync.py` / `exchange_drive_token.py` | Legacy Google Drive sync + OAuth token exchange, superseded by the SFTP destination. Kept for reference / in case Drive is ever needed again. |
| `rename_run_folders.py` | One-off utility to rename old manifest-tracked folders to their run ID. |

## `ckpt_target_run_ids.txt`

A priority list of run IDs, one per line, that get treated specially by
both `wandb_backup_parallel.py` and `wandb_ckpt_backup.py`:

- Moved to the **front of the processing queue**
- Held to a **stricter "verify before skip" standard** — instead of trusting
  the manifest's word that a run is already backed up, the actual bytes are
  checked (locally, or on the SFTP server if the local copy was already
  cleaned up)
- Uploaded **individually** as their own browsable folder on the server,
  rather than being bundled into a bulk zip batch (see below) — this is
  what makes that verification possible in the first place

Currently 1,860 IDs: a manually-curated priority list, plus the top ~10% of
runs by `val/epoch/80th_percentile_balanced_accuracy_score` (highest) and
`val/epoch/80th_percentile_mae` (lowest), unioned together.

## Design

### Restart safety

Every run's backup state lives in `backup_manifest.json`, keyed by run ID.
Progress is saved after every run, atomically (write to a thread-suffixed
temp file, then rename). Stopping the script at any point and rerunning it
picks up exactly where it left off — completed runs are skipped, failed
files are retried.

### Concurrency

`wandb_backup_parallel.py` runs 12 worker threads (`MAX_WORKERS`) via
`ThreadPoolExecutor`. Each worker owns its own SFTP connection
(`paramiko` clients aren't thread-safe); the manifest is protected by a
lock and only ever mutated by the thread that owns a given run's record.

### Bulk zip batching

Uploading tens of thousands of small files one at a time is dominated by
SFTP round-trip latency. Non-priority ("bulk") runs are downloaded
normally, then staged locally; once `BULK_ZIP_BATCH_SIZE` (10) of them have
accumulated, they're zipped into a single archive and uploaded as one file
under `<project_remote_dir>/bulk_batches/`, instead of many small per-file
transfers. Priority-list runs are exempt from this — they stay as
individually browsable folders so the strict verification above keeps
working.

Batching is resume-safe: a run only gets marked "uploaded" in the manifest
once its *batch's* zip is actually confirmed on the server. An interrupted
partial batch just gets re-staged into a fresh batch on the next run — no
run is ever silently dropped.

`bulk_batch_registry.json` is a local-only lookup index (never uploaded)
mapping `run_id ↔ batch_zip_name` in both directions, so you don't have to
scan the full manifest to find out which batch a given run landed in.

### Tracking-file backup

After every successful upload (a priority run's own upload, or a bulk
batch's zip upload), the current `backup_manifest.json` and
`bulk_batch_registry.json` are also pushed to the SFTP project root — so a
remote copy of the backup's own bookkeeping survives even if the machine
running the script doesn't.

### Exclusions

Some media is skipped for every run regardless of priority status:
occlusion maps, Grad-CAM visualizations, activation dumps, augmentation
previews (name-substring match, case-insensitive), and `output.log`
(near-identical run to run, not worth the storage). Scatter plot images
(`prediction_scatter`, `best_scatter`) are kept only for priority-list runs.

## Known limitations

- **W&B Artifacts are not yet backed up** by the main pipeline — only run
  files, config, summary, and history. `wandb_ckpt_backup.py` covers
  *checkpoint* artifacts specifically as a separate pass; other artifact
  types are still a gap.
- The SFTP account currently requires **TOTP two-factor authentication** in
  addition to a password, which blocks fully unattended runs unless a
  service account / app-specific password is set up on Semler's side, or
  the TOTP secret (not just a rotating code) is captured.

## Legacy: Google Drive migration

Backups originally went to a Google Drive Shared Drive. That's since moved
to the Semler SFTP server; `migrate_drive_to_sftp.py` was a one-time script
to copy the already-backed-up Drive content over without re-hitting the W&B
API. `drive_sync.py` and `exchange_drive_token.py` remain for reference.
