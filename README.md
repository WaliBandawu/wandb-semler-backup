# W&B → SFTP Backup Pipeline

Backs up the curated [target-list runs](#ckpt_target_run_idstxt) from the
`theta-tech-ai/semler-qfhd` Weights & Biases project (metadata, config,
summary, history, and run files) to Semler Scientific's SFTP server, with a
fully restart-safe, resumable design. There's also a script for backing up
a Google Drive **folder** to the same server (see [Google Drive folder
backup](#google-drive-folder-backup) below) - unrelated to how W&B backups
originally targeted Google Drive as their destination before moving to
SFTP, which is now legacy (see [Legacy: Google Drive
migration](#legacy-google-drive-migration) below).

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install wandb paramiko pyotp google-api-python-client google-auth google-auth-oauthlib

cp .env.example .env
# then fill in .env with real values:
#   WANDB_API_KEY=...            (or skip this and `wandb login` once instead)
#   SEMLER_SFTP_HOST=files.semlerscientific.net
#   SEMLER_SFTP_PORT=22
#   SEMLER_SFTP_USERNAME=...
#   SEMLER_SFTP_PASSWORD=...
#   SEMLER_SFTP_TOTP_SECRET=...  (the account's permanent TOTP seed - see below)
# .env is gitignored - it never gets committed.

python3 wandb_backup_parallel.py
```

Safe to stop (Ctrl+C) and rerun at any time — it picks up exactly where it
left off.

If `SEMLER_SFTP_*` isn't set, `sftp_backup_lib.py` falls back to reading a
JSON credentials file (`{"host", "port", "username", "password", "totp_secret"}`)
from `SFTP_CREDENTIALS_PATH` — the existing method on the server this already
runs on. `.env` is the easier path for a fresh checkout.

### SFTP two-factor authentication

The `SEMLER_SFTP_*` account requires TOTP 2FA in addition to a password —
after password auth, the server holds the connection open in a
`keyboard-interactive` challenge (`"Two Factor Authentication" / "Enter your
TOTP two factor value."`) before it'll open any channel, SFTP included.

`SEMLER_SFTP_TOTP_SECRET` must be the account's **permanent TOTP seed** — the
base32 string shown once next to the QR code when 2FA was enrolled (most
authenticator apps can also export/reveal it for an existing entry) — not a
rotating 6-digit code, which expires in ~30 seconds and can't be reused.
With the seed set, `sftp_backup_lib.py` generates a fresh valid code for
every connection automatically (via `pyotp`), including the many
per-worker-thread reconnects a full backup makes — no human types a code,
ever. Without it, any connection attempt fails immediately with an error
explaining what's missing, rather than hanging.

If nobody has the seed saved, it can only be recovered by re-enrolling 2FA
on the account (invalidating the old one), or by asking Semler for a
service account/app-specific credential exempt from interactive 2FA.

## Scripts

| Script | Purpose |
|---|---|
| `wandb_backup_parallel.py` | **Main entry point.** Backs up only the [target-list runs](#ckpt_target_run_idstxt) (not the full project) — metadata, config, summary, history, run files, and checkpoints — concurrent workers scaled to the machine it runs on. See [Design](#design) below. |
| `wandb_backup_local.py` | Same backup logic, sequential (single-threaded). Useful for debugging without concurrency noise. Still does a full-project backup - not yet updated to the target-list-only scope above. |
| `wandb_backup_colab.py` | Same again, tuned for running in a Google Colab notebook cell. Still does a full-project backup - not yet updated to the target-list-only scope above. |
| `wandb_ckpt_backup.py` | Separate, broader checkpoint pass: like `wandb_backup_parallel.py`, downloads/uploads `model`-type logged artifacts for target-list runs, but also walks every *other* run in the full project and checks (never downloads) whether it has checkpoints, recording that fact in its own manifest. Redundant with `wandb_backup_parallel.py` for target-list runs specifically; still useful for that full-project checkpoint-presence sweep. |
| `sftp_backup_lib.py` | Shared SFTP helpers (connection pooling, retry, atomic upload, remote directory resolution) used by the backup scripts above. |
| `drive_backup_to_sftp.py` | Backs up a Google Drive **folder** (not a W&B project) to SFTP, under `HD_Data/asset_archive/<drive folder's own name>/`. See [Google Drive folder backup](#google-drive-folder-backup) below. |
| `gdrive_lib.py` | Shared Google Drive OAuth/API-client helper used by `drive_backup_to_sftp.py`, `migrate_drive_to_sftp.py`, and `drive_sync.py`. |
| `exchange_drive_token.py` | One-time OAuth helper: exchanges an authorization code for the long-lived refresh token the Drive scripts above need. See [Google Drive folder backup](#google-drive-folder-backup) below. |
| `migrate_drive_to_sftp.py` | One-time migration: copies the old Google-Drive-backed W&B-runs tree to SFTP without re-touching W&B. |
| `drive_sync.py` | Legacy Google Drive sync (the other direction: local → Drive), superseded by the SFTP destination. Kept for reference. |
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

`wandb_backup_parallel.py` runs multiple worker threads (`MAX_WORKERS`) via
`ThreadPoolExecutor`. This work is I/O-bound (network waits, not CPU), so
`MAX_WORKERS` is computed from the machine it runs on —
`min(32, os.cpu_count() + 4)`, the same formula Python's own
`ThreadPoolExecutor` uses by default for exactly this case — rather than a
number tuned for one specific box. Each worker owns its own SFTP connection
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

Since `wandb_backup_parallel.py` now only ever processes target-list runs
(see [Scripts](#scripts) above), every run it sees counts as priority, so
this path is currently unreachable in practice - dead code kept for when
`wandb_backup_local.py` / `wandb_backup_colab.py` (which still do full
non-priority backups) need it.

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

- **Only `model`-type W&B Artifacts (checkpoints) are backed up.** The main
  pipeline downloads these itself now (`checkpoints/<artifact>/...` inside
  each run's folder, uploaded as part of that run's normal SFTP upload —
  see [Design](#design)). Other artifact types are still a gap.
  `wandb_ckpt_backup.py` still exists separately for its broader
  full-project pass (checking *every* run in the project, not just the
  target list, for whether it has checkpoint artifacts).
- The SFTP account requires **TOTP two-factor authentication** in addition
  to a password. This is now handled automatically (see [SFTP two-factor
  authentication](#sftp-two-factor-authentication) above) as long as
  `SEMLER_SFTP_TOTP_SECRET` is set — unattended runs are blocked only until
  that seed is captured and configured.

## Google Drive folder backup

`drive_backup_to_sftp.py` backs up an arbitrary Google Drive **folder**
(not a W&B project) to the same SFTP server, landing at
`HD_Data/asset_archive/<the Drive folder's own name>/` — a sibling of the
`W&B Runs` tree the other scripts use, under the same `asset_archive` root.
It currently points at one folder, configured as `DRIVE_FOLDER_ID` near the
top of the script:

```
https://drive.google.com/drive/folders/1zgbI4HtAh4L5VsbEGVeF9fSUylAVhvLx
```

(Currently resolves to a folder named "QFHD".) Like the W&B pipeline, it's
restart-safe (`drive_folder_backup_manifest.json`) and uses [bulk zip
batching](#bulk-zip-batching) — downloaded files accumulate locally and get
zipped into batches of `BULK_ZIP_BATCH_SIZE` (10) before upload, instead of
one SFTP round-trip per (often small) file.

```bash
python3 drive_backup_to_sftp.py
```

### One-time setup: Google Drive OAuth

The Drive scripts (`drive_backup_to_sftp.py`, `migrate_drive_to_sftp.py`,
`drive_sync.py`) authenticate via a long-lived OAuth refresh token stored
locally, never committed to this repo:

- `~/.gdrive_oauth_client.json` — `{"client_id", "client_secret", "token_uri"}`
- `~/.gdrive_token.json` — the refresh token, produced by the exchange below

To set these up from scratch:

1. **Create a Google Cloud OAuth client.**
   - [console.cloud.google.com](https://console.cloud.google.com) → pick or
     create a project.
   - **APIs & Services → Library** → search "Google Drive API" → **Enable**.
   - **APIs & Services → OAuth consent screen** — if not already configured,
     set it to "External," fill in the minimum (app name, your email), and
     add your own Google account as a **test user** (the app can stay in
     "Testing" status — that's fine for personal/internal use).
   - **APIs & Services → Credentials → Create Credentials → OAuth client
     ID**.
     - Application type: **Web application** (must be Web, not Desktop —
       needed to set a fixed custom redirect URI next).
     - Under **Authorized redirect URIs**, add exactly:
       `https://developers.google.com/oauthplayground`
     - Save it, and copy the **Client ID** and **Client Secret**.

2. **Save the client config locally:**

   ```bash
   cat > ~/.gdrive_oauth_client.json << 'EOF'
   {"client_id": "YOUR_CLIENT_ID", "client_secret": "YOUR_CLIENT_SECRET", "token_uri": "https://oauth2.googleapis.com/token"}
   EOF
   chmod 600 ~/.gdrive_oauth_client.json
   ```

3. **Get a one-time authorization code via OAuth Playground:**
   - Go to [developers.google.com/oauthplayground](https://developers.google.com/oauthplayground).
   - Click the **gear icon** (top right) → check **"Use your own OAuth
     credentials"** → paste in your Client ID and Client Secret.
   - If there's an **"Auto-exchange authorization code for tokens"**
     checkbox, **uncheck it** — otherwise Playground consumes the code
     automatically and you can't grab it.
   - In the left panel (Step 1), under "Input your own scopes," enter:
     `https://www.googleapis.com/auth/drive` → click **Authorize APIs**.
   - Log in with the Google account that has access to the target Drive
     folder, and grant access.
   - You'll land on Step 2, showing the raw authorization code — copy it.
     **It expires in a few minutes**, so move to the next step right away.

4. **Exchange the code for a refresh token:**

   ```bash
   echo -n "PASTE_THE_CODE_HERE" > ~/.gdrive_auth_code.txt
   chmod 600 ~/.gdrive_auth_code.txt
   python3 exchange_drive_token.py
   # on success, saves ~/.gdrive_token.json and prints "Refresh token saved: yes"
   rm ~/.gdrive_auth_code.txt   # one-time use, no longer needed
   ```

The refresh token doesn't expire from normal use, so this is a one-time
setup — `gdrive_lib.get_drive_service()` refreshes the access token
automatically on every run after this.

## Legacy: Google Drive migration

Backups originally went to a Google Drive Shared Drive. That's since moved
to the Semler SFTP server; `migrate_drive_to_sftp.py` was a one-time script
to copy the already-backed-up Drive content over without re-hitting the W&B
API. `drive_sync.py` remains for reference. (`exchange_drive_token.py`
itself isn't legacy - see [Google Drive folder
backup](#google-drive-folder-backup) above, which still needs it.)
