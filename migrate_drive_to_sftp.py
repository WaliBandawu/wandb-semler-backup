# ============================================================
# ONE-TIME MIGRATION: Google Drive -> Semler SFTP
#
# Copies the existing "W&B Runs/<entity>/<project>" tree from the
# old Google Drive backup destination to the new SFTP server,
# WITHOUT touching W&B at all (the local copies of these runs were
# already deleted after their original Drive upload, so this reads
# straight from Drive instead of re-downloading from W&B).
#
# Streams file-by-file: download to a small local temp file,
# upload to SFTP, delete the temp file. Never holds more than
# MAX_WORKERS files on local disk at once.
#
# Restart-safe via its own manifest (drive_to_sftp_manifest.json).
# Before re-downloading a file from Drive, it first does a cheap
# SFTP stat to see if that file is already correctly there (covers
# the case where a previous run uploaded a file but crashed before
# saving the manifest).
# ============================================================

import io
import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build as build_drive_service
from googleapiclient.http import MediaIoBaseDownload

import sftp_backup_lib as sftp_lib

# ============================================================
# CONFIGURATION
# ============================================================

WANDB_ENTITY = "theta-tech-ai"
WANDB_PROJECT = "semler-qfhd"

DRIVE_CLIENT_CFG_PATH = "/home/ubuntu/.gdrive_oauth_client.json"
DRIVE_TOKEN_PATH = "/home/ubuntu/.gdrive_token.json"
DRIVE_SHARED_DRIVE_ID = "0ANwW-EAug9vmUk9PVA"

BACKUP_BASE = Path("/home/ubuntu/wandb_backups")
MANIFEST_FILE = BACKUP_BASE / "drive_to_sftp_manifest.json"
LOG_FILE = BACKUP_BASE / "drive_to_sftp_errors.log"
STAGING_DIR = BACKUP_BASE / "_drive_to_sftp_staging"

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

_drive_thread_local = threading.local()


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
        log_error(f"Could not read migration manifest: {e}")
        return {"version": 1, "created_at": datetime.now().isoformat(), "files": {}}


def save_manifest(manifest):
    import copy

    with MANIFEST_LOCK:
        manifest["updated_at"] = datetime.now().isoformat()
        snapshot = copy.deepcopy(manifest)

    temp_file = MANIFEST_FILE.with_suffix(f".tmp.{threading.get_ident()}")
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    temp_file.replace(MANIFEST_FILE)


# ============================================================
# GOOGLE DRIVE (read-only side of the migration)
# ============================================================


def get_drive_service():
    if not hasattr(_drive_thread_local, "service"):
        with open(DRIVE_CLIENT_CFG_PATH) as f:
            cfg = json.load(f)
        with open(DRIVE_TOKEN_PATH) as f:
            token = json.load(f)

        creds = Credentials(
            token=token.get("access_token"),
            refresh_token=token["refresh_token"],
            token_uri=cfg["token_uri"],
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        creds.refresh(Request())

        _drive_thread_local.service = build_drive_service(
            "drive", "v3", credentials=creds, cache_discovery=False
        )
    return _drive_thread_local.service


def find_drive_folder(service, name, parent_id):
    safe = name.replace("\\", "\\\\").replace("'", "\\'")
    query = (
        f"name = '{safe}' and '{parent_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(
        q=query, fields="files(id)", spaces="drive", corpora="drive",
        driveId=DRIVE_SHARED_DRIVE_ID, includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def resolve_drive_project_folder():
    """Walk W&B Runs/<entity>/<project> on the shared drive. Returns
    None if any segment doesn't exist (nothing to migrate)."""
    service = get_drive_service()
    parent_id = DRIVE_SHARED_DRIVE_ID
    for part in ["W&B Runs", WANDB_ENTITY, WANDB_PROJECT]:
        parent_id = find_drive_folder(service, part, parent_id)
        if parent_id is None:
            return None
    return parent_id


def list_drive_tree_flat(service, folder_id, prefix=""):
    """Recursively list every FILE (not folder) under folder_id.
    Returns a list of {id, path, size} with path relative to the
    starting folder."""
    out = []
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, size)",
            spaces="drive", corpora="drive", driveId=DRIVE_SHARED_DRIVE_ID,
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


# ============================================================
# PER-FILE MIGRATION
# ============================================================


def sanitize_relative_path(rel_path):
    """Apply the same per-segment sanitization used by direct SFTP
    uploads, so migrated files and future direct uploads land under
    identical remote paths."""
    return "/".join(sftp_lib.safe_name(part) for part in rel_path.split("/"))


def migrate_one_file(entry, manifest, project_remote_dir, index, total):
    drive_path = entry["path"]
    size = entry["size"]

    with MANIFEST_LOCK:
        existing = manifest["files"].get(drive_path)
    if existing and existing.get("status") == "migrated":
        return {"path": drive_path, "status": "migrated", "skipped": True}

    remote_rel_path = sanitize_relative_path(drive_path)
    remote_path = sftp_lib.remote_join(project_remote_dir, remote_rel_path)
    remote_dir = remote_path.rsplit("/", 1)[0] if "/" in remote_path else project_remote_dir

    try:
        # Cheap first: if it's already correctly on the SFTP server
        # (e.g. an interrupted prior run), skip the Drive download.
        existing_size = sftp_lib.with_sftp_retry(sftp_lib.remote_file_size, remote_path)

        if size is not None and existing_size == size:
            log(f"[{index}/{total}] ↪ SKIP (already on SFTP): {drive_path}")
            record = {"status": "migrated", "size": size, "migrated_at": datetime.now().isoformat(), "note": "already present"}
            with MANIFEST_LOCK:
                manifest["files"][drive_path] = record
            return {"path": drive_path, "status": "migrated"}

        temp_local = STAGING_DIR / f"{threading.get_ident()}_{Path(drive_path).name}"

        log(f"[{index}/{total}] ↓ Drive → local: {drive_path}")
        service = get_drive_service()
        download_drive_file(service, entry["id"], temp_local)

        local_size = temp_local.stat().st_size
        if size is not None and local_size != size:
            log(f"[{index}/{total}] ⚠ size mismatch after download ({drive_path}): expected {size}, got {local_size}")

        log(f"[{index}/{total}] ↑ local → SFTP: {remote_rel_path}")

        def _upload(sftp):
            sftp_lib.ensure_remote_dir(sftp, remote_dir)
            sftp_lib.upload_file(sftp, temp_local, remote_dir, remote_name=Path(remote_rel_path).name)

        sftp_lib.with_sftp_retry(_upload)

        temp_local.unlink(missing_ok=True)

        record = {
            "status": "migrated",
            "size": local_size,
            "migrated_at": datetime.now().isoformat(),
        }
        with MANIFEST_LOCK:
            manifest["files"][drive_path] = record

        log(f"[{index}/{total}] ✓ COMPLETE: {drive_path}")
        return {"path": drive_path, "status": "migrated"}

    except Exception as e:
        log_error(f"{drive_path}: migration failed: {e}")
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
    print("MIGRATING GOOGLE DRIVE -> SFTP")
    print(f"Source: Shared Drive / W&B Runs / {WANDB_ENTITY} / {WANDB_PROJECT}")
    print("=" * 70)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    print(f"\nPreviously migrated (this job's manifest): "
          f"{sum(1 for r in manifest['files'].values() if r.get('status') == 'migrated')}")

    print("\nResolving Drive source folder...")
    service = get_drive_service()
    drive_folder_id = resolve_drive_project_folder()

    if drive_folder_id is None:
        print("✗ Source folder not found on Drive - nothing to migrate.")
        return

    print("✓ Drive source folder resolved")

    print("\nResolving SFTP destination folder...")
    project_remote_dir = sftp_lib.resolve_project_remote_dir(WANDB_ENTITY, WANDB_PROJECT)
    print(f"✓ SFTP destination ready: {project_remote_dir}")

    print("\nListing every file under the Drive source folder (this can take a while)...")
    start = time.time()
    entries = list_drive_tree_flat(service, drive_folder_id)
    print(f"✓ Found {len(entries):,} files in {time.time() - start:.1f}s")

    if not entries:
        print("Nothing to migrate.")
        return

    total = len(entries)
    migrated = []
    failed = []

    print(f"\nUsing {MAX_WORKERS} concurrent workers.\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(migrate_one_file, entry, manifest, project_remote_dir, i, total): entry
            for i, entry in enumerate(entries, start=1)
        }

        completed = 0
        for future in as_completed(futures):
            entry = futures[future]
            completed += 1

            try:
                result = future.result()
                if result["status"] == "migrated":
                    migrated.append(result["path"])
                else:
                    failed.append(result["path"])
            except Exception as e:
                failed.append(entry["path"])
                log_error(f"{entry['path']}: unexpected error: {e}")
                traceback.print_exc()

            if completed % 50 == 0 or completed == total:
                save_manifest(manifest)
                print(f"\nProgress: {completed:,}/{total:,} "
                      f"(migrated {len(migrated):,}, failed {len(failed):,})")

    save_manifest(manifest)

    try:
        for f in STAGING_DIR.glob("*"):
            f.unlink(missing_ok=True)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"Files migrated : {len(migrated):,}/{total:,}")
    print(f"Files failed   : {len(failed):,}")
    print(f"\nManifest: {MANIFEST_FILE}")
    if ERRORS:
        print(f"Errors this session: {len(ERRORS)} (see {LOG_FILE})")
    if failed:
        print("\nRun this script again to retry only the failed/incomplete files.")


if __name__ == "__main__":
    main()
