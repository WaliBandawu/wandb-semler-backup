# ============================================================
# GOOGLE DRIVE FOLDER -> SEMLER SFTP BACKUP
#
# Backs up everything under a single Google Drive folder
# (DRIVE_FOLDER_ID below) to the Semler SFTP server, landing at
#   HD_Data/asset_archive/<drive folder's own name>/...
# - a sibling of the "W&B Runs" tree the other backup scripts use,
# under the same asset_archive root.
#
# Downloads stream to a local temp file, then get bundled: once
# BULK_ZIP_BATCH_SIZE files have downloaded, they're zipped into one
# archive and uploaded as a single file, instead of one SFTP
# round-trip per (often small) file - the same "Bulk zip batching"
# approach wandb_backup_parallel.py uses for its own bulk runs.
#
# Restart-safe via its own manifest (drive_folder_backup_manifest.json).
# A file only gets marked "backed_up" once its *batch's* zip is
# actually confirmed on the server; an interrupted partial batch just
# gets re-staged into a fresh batch on the next run.
# ============================================================

import copy
import io
import json
import os
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload

import gdrive_lib
import sftp_backup_lib as sftp_lib

# ============================================================
# CONFIGURATION
# ============================================================

# https://drive.google.com/drive/folders/1zgbI4HtAh4L5VsbEGVeF9fSUylAVhvLx
DRIVE_FOLDER_ID = "1zgbI4HtAh4L5VsbEGVeF9fSUylAVhvLx"

BACKUP_BASE = sftp_lib.resolve_backup_base()
MANIFEST_FILE = BACKUP_BASE / "drive_folder_backup_manifest.json"
LOG_FILE = BACKUP_BASE / "drive_folder_backup_errors.log"

# Per-file downloads land here transiently before being zipped into a
# batch (see BULK_ZIP_BATCH_SIZE below).
STAGING_DIR = BACKUP_BASE / "_drive_folder_backup_staging"

# Batch archives themselves are staged here before upload, deleted
# immediately after (whether the upload succeeded or not).
ZIP_STAGING = BACKUP_BASE / "_drive_folder_backup_zip_staging"

# Number of downloaded files bundled into one zip before upload -
# same default as wandb_backup_parallel.py's bulk batching.
BULK_ZIP_BATCH_SIZE = 10

# I/O-bound (Drive download + SFTP upload, not CPU), so more threads
# than cores is fine - scales with whatever machine this runs on.
# Same formula Python's own ThreadPoolExecutor uses by default,
# capped at 32 so a big machine doesn't hammer the single SFTP
# server harder than it can handle.
MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)

MANIFEST_LOCK = threading.RLock()
PRINT_LOCK = threading.Lock()
ERRORS_LOCK = threading.Lock()
ERRORS = []


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] {message}"
    with ERRORS_LOCK:
        ERRORS.append(error_message)
    log(f"⚠ {error_message}")
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ERRORS_LOCK, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(error_message + "\n")
    except Exception:
        log("⚠ Could not write to error log")


def load_manifest():
    if not MANIFEST_FILE.exists():
        return {"version": 1, "created_at": datetime.now().isoformat(), "files": {}}
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("files", {})
        return manifest
    except Exception as e:
        log_error(f"Could not read backup manifest: {e}")
        return {"version": 1, "created_at": datetime.now().isoformat(), "files": {}}


def save_manifest(manifest):
    with MANIFEST_LOCK:
        manifest["updated_at"] = datetime.now().isoformat()
        snapshot = copy.deepcopy(manifest)

    temp_file = MANIFEST_FILE.with_suffix(f".tmp.{threading.get_ident()}")
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    temp_file.replace(MANIFEST_FILE)


# ============================================================
# GOOGLE DRIVE (read-only side of the backup)
# ============================================================


def get_drive_folder_name(service, folder_id):
    info = service.files().get(
        fileId=folder_id, fields="name", supportsAllDrives=True
    ).execute()
    return info["name"]


def list_drive_tree_flat(service, folder_id, prefix=""):
    """Recursively list every FILE (not folder) under folder_id.
    Returns a list of {id, path, size} with path relative to the
    starting folder. Uses corpora="allDrives" so this works whether
    the folder lives in a Shared Drive or someone's personal Drive -
    unlike the W&B-runs migration script, this isn't scoped to one
    known Shared Drive."""
    out = []
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            spaces="drive", corpora="allDrives",
            includeItemsFromAllDrives=True, supportsAllDrives=True,
            pageToken=page_token,
        ).execute()

        for f in results.get("files", []):
            rel_path = f"{prefix}{f['name']}"
            if f["mimeType"] == "application/vnd.google-apps.folder":
                out.extend(list_drive_tree_flat(service, f["id"], prefix=f"{rel_path}/"))
            else:
                size = f.get("size")
                out.append({
                    "id": f["id"],
                    "path": rel_path,
                    "size": int(size) if size is not None else None,
                })

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return out


def download_drive_file(service, file_id, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.FileIO(local_path, "wb")
    try:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    finally:
        fh.close()


def sanitize_relative_path(rel_path):
    """Apply the same per-segment sanitization used by direct SFTP
    uploads, so backed-up files land under the same kind of remote
    paths as everything else on the server."""
    return "/".join(sftp_lib.safe_name(part) for part in rel_path.split("/"))


# ============================================================
# BULK ZIP BATCHING
#
# Downloaded files accumulate here (thread-safe) until
# BULK_ZIP_BATCH_SIZE is reached; whichever thread's download happens
# to fill the batch does the zip + upload + manifest write for the
# whole batch, not just its own file. Mirrors
# wandb_backup_parallel.py's bulk-run batching.
# ============================================================

_bulk_remote_dir = None  # resolved once, before workers start
_pending_batch = []      # list of (drive_path, temp_local_path, size)
_batch_lock = threading.Lock()
_batch_counter = 0


def resolve_bulk_remote_dir(dest_remote_dir):
    """Resolve (and cache) the shared <dest_remote_dir>/bulk_batches
    directory once, sequentially, before any worker threads start."""
    global _bulk_remote_dir
    _bulk_remote_dir = sftp_lib.resolve_bulk_batch_remote_dir(dest_remote_dir)
    return _bulk_remote_dir


def _zip_batch(batch_items, zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for drive_path, temp_local, _ in batch_items:
            zf.write(temp_local, arcname=sanitize_relative_path(drive_path))


def _flush_batch(batch_items, manifest):
    """Zip + upload one batch of downloaded files as a single archive,
    then delete their local temp copies and record all of them in the
    manifest under one shared "backed_up" status."""
    global _batch_counter

    if not batch_items:
        return

    with _batch_lock:
        _batch_counter += 1
        batch_number = _batch_counter

    zip_name = f"drive_batch_{batch_number:05d}.zip"
    zip_path = ZIP_STAGING / zip_name
    paths = [item[0] for item in batch_items]

    log(f"\n[BULK ZIP] Building {zip_name} from {len(batch_items)} files: {paths}")

    try:
        _zip_batch(batch_items, zip_path)
    except Exception as e:
        log_error(f"Bulk batch {zip_name}: zipping failed: {e}")
        # Local temp files are left in place - none have a "backed_up"
        # manifest entry yet, so they'll be re-staged into a fresh
        # batch and retried the next time this script runs.
        return

    success, error = sftp_lib.upload_bulk_zip(zip_path, _bulk_remote_dir, zip_name)

    if success:
        log(f"[BULK ZIP] ✓ Uploaded {zip_name}")
        backed_up_at = datetime.now().isoformat()
        remote_zip_path = sftp_lib.remote_join(_bulk_remote_dir, zip_name)

        with MANIFEST_LOCK:
            for drive_path, _, size in batch_items:
                manifest["files"][drive_path] = {
                    "status": "backed_up",
                    "size": size,
                    "backed_up_at": backed_up_at,
                    "batch_zip": remote_zip_path,
                }
            save_manifest(manifest)

        for _, temp_local, _ in batch_items:
            temp_local.unlink(missing_ok=True)
    else:
        log_error(f"Bulk batch {zip_name}: upload failed: {error}")
        # Local temp files are left in place for the same reason as
        # the zip-failure case above.

    try:
        zip_path.unlink(missing_ok=True)
    except Exception:
        pass


def stage_file_for_batch(drive_path, temp_local, size, manifest):
    batch_to_flush = None

    with _batch_lock:
        _pending_batch.append((drive_path, temp_local, size))

        if len(_pending_batch) >= BULK_ZIP_BATCH_SIZE:
            batch_to_flush = list(_pending_batch)
            _pending_batch.clear()

    if batch_to_flush:
        _flush_batch(batch_to_flush, manifest)


def flush_remaining_batch(manifest):
    """Call once after all files have been downloaded, to upload
    whatever partial batch (fewer than BULK_ZIP_BATCH_SIZE files) is
    still pending."""
    with _batch_lock:
        remaining = list(_pending_batch)
        _pending_batch.clear()

    if remaining:
        log(f"\n[BULK ZIP] Flushing final partial batch of {len(remaining)} files")
        _flush_batch(remaining, manifest)


# ============================================================
# PER-FILE DOWNLOAD
# ============================================================


def download_one_file(entry, manifest, index, total):
    drive_path = entry["path"]
    size = entry["size"]

    with MANIFEST_LOCK:
        existing = manifest["files"].get(drive_path)
    if existing and existing.get("status") == "backed_up":
        return {"path": drive_path, "status": "backed_up", "skipped": True}

    temp_local = STAGING_DIR / f"{threading.get_ident()}_{Path(drive_path).name}"

    try:
        log(f"[{index}/{total}] ↓ Drive → local: {drive_path}")
        service = gdrive_lib.get_drive_service()
        download_drive_file(service, entry["id"], temp_local)

        local_size = temp_local.stat().st_size
        if size is not None and local_size != size:
            log(f"[{index}/{total}] ⚠ size mismatch after download ({drive_path}): expected {size}, got {local_size}")

        stage_file_for_batch(drive_path, temp_local, local_size, manifest)
        return {"path": drive_path, "status": "staged"}

    except Exception as e:
        log_error(f"{drive_path}: download failed: {e}")
        record = {"status": "failed", "error": str(e), "attempted_at": datetime.now().isoformat()}
        with MANIFEST_LOCK:
            manifest["files"][drive_path] = record
        try:
            temp_local.unlink(missing_ok=True)
        except Exception:
            pass
        return {"path": drive_path, "status": "failed"}


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 70)
    print("BACKING UP GOOGLE DRIVE FOLDER -> SFTP")
    print(f"Source: https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
    print("=" * 70)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_STAGING.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    print(f"\nPreviously backed up (this job's manifest): "
          f"{sum(1 for r in manifest['files'].values() if r.get('status') == 'backed_up')}")

    print("\nResolving Drive source folder...")
    service = gdrive_lib.get_drive_service()

    try:
        folder_name = get_drive_folder_name(service, DRIVE_FOLDER_ID)
    except Exception as e:
        print(f"✗ Could not resolve Drive folder {DRIVE_FOLDER_ID}: {e}")
        return

    print(f"✓ Drive source folder: \"{folder_name}\"")

    print("\nResolving SFTP destination folder...")
    dest_remote_dir = sftp_lib.resolve_remote_dir_under_root(folder_name)
    resolve_bulk_remote_dir(dest_remote_dir)
    print(f"✓ SFTP destination ready: {dest_remote_dir}")
    print(f"  (batches of {BULK_ZIP_BATCH_SIZE} files uploaded to {_bulk_remote_dir})")

    print("\nListing every file under the Drive source folder (this can take a while)...")
    start = time.time()
    entries = list_drive_tree_flat(service, DRIVE_FOLDER_ID)
    print(f"✓ Found {len(entries):,} files in {time.time() - start:.1f}s")

    if not entries:
        print("Nothing to back up.")
        return

    total = len(entries)
    staged = 0
    failed = []

    print(f"\nUsing {MAX_WORKERS} concurrent workers.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_one_file, entry, manifest, i, total): entry
            for i, entry in enumerate(entries, start=1)
        }

        completed = 0
        for future in as_completed(futures):
            entry = futures[future]
            completed += 1

            try:
                result = future.result()
                if result["status"] == "failed":
                    failed.append(result["path"])
                else:
                    staged += 1
            except Exception as e:
                failed.append(entry["path"])
                log_error(f"{entry['path']}: unexpected error: {e}")
                traceback.print_exc()

            if completed % 50 == 0 or completed == total:
                print(f"\nProgress: {completed:,}/{total:,} "
                      f"(downloaded/staged {staged:,}, failed {len(failed):,})")

    flush_remaining_batch(manifest)
    save_manifest(manifest)

    backed_up = sum(1 for r in manifest["files"].values() if r.get("status") == "backed_up")

    try:
        for f in STAGING_DIR.glob("*"):
            f.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("BACKUP COMPLETE")
    print("=" * 70)
    print(f"Files backed up : {backed_up:,}/{total:,}")
    print(f"Files failed    : {len(failed):,}")
    print(f"\nManifest: {MANIFEST_FILE}")
    if ERRORS:
        print(f"Errors this session: {len(ERRORS)} (see {LOG_FILE})")
    if failed or backed_up < total:
        print("\nRun this script again to retry only the failed/incomplete files.")


if __name__ == "__main__":
    main()
