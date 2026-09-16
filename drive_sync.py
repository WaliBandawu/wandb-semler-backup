import json
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

CLIENT_CFG_PATH = "/home/ubuntu/.gdrive_oauth_client.json"
TOKEN_PATH = "/home/ubuntu/.gdrive_token.json"

LOCAL_ROOT = Path("/home/ubuntu/wandb_backups")

# Destination Shared Drive. Everything under LOCAL_ROOT is mirrored
# directly under this Shared Drive's root.
SHARED_DRIVE_ID = "0ANwW-EAug9vmUk9PVA"

SYNC_MANIFEST_PATH = LOCAL_ROOT / "drive_sync_manifest.json"

EXCLUDE_NAMES = {"drive_sync_manifest.json", "drive_sync.log"}

# I/O-bound (Drive API calls, not CPU), so more threads than cores
# is fine - scales with whatever machine this runs on. Same formula
# Python's own ThreadPoolExecutor uses by default, capped at 32.
MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)

MANIFEST_LOCK = threading.RLock()
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def load_credentials():
    with open(CLIENT_CFG_PATH) as f:
        cfg = json.load(f)

    with open(TOKEN_PATH) as f:
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

    return creds


def load_manifest():
    if SYNC_MANIFEST_PATH.exists():
        try:
            with open(SYNC_MANIFEST_PATH) as f:
                return json.load(f)
        except Exception:
            pass

    return {"folders": {}, "files": {}}


def save_manifest(manifest):
    import copy

    with MANIFEST_LOCK:
        snapshot = copy.deepcopy(manifest)

    tmp = SYNC_MANIFEST_PATH.with_suffix(f".tmp.{threading.get_ident()}")
    with open(tmp, "w") as f:
        json.dump(snapshot, f, indent=2)
    tmp.replace(SYNC_MANIFEST_PATH)


def get_or_create_folder(service, manifest, rel_path_parts):
    key = "/".join(rel_path_parts) if rel_path_parts else ""

    with MANIFEST_LOCK:
        cached = manifest["folders"].get(key)
    if cached:
        return cached

    if not rel_path_parts:
        folder_id = SHARED_DRIVE_ID
    else:
        parent_id = get_or_create_folder(service, manifest, rel_path_parts[:-1])
        name = rel_path_parts[-1]

        safe_name = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (
            f"name = '{safe_name}' "
            f"and '{parent_id}' in parents "
            f"and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )

        results = service.files().list(
            q=query,
            fields="files(id, name)",
            spaces="drive",
            corpora="drive",
            driveId=SHARED_DRIVE_ID,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        existing = results.get("files", [])

        if existing:
            folder_id = existing[0]["id"]
        else:
            metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
            folder = service.files().create(
                body=metadata, fields="id", supportsAllDrives=True
            ).execute()
            folder_id = folder["id"]

    with MANIFEST_LOCK:
        manifest["folders"][key] = folder_id
        save_manifest(manifest)

    return folder_id


def upload_or_update_file(service, manifest, local_path, parent_id, rel_key):
    stat = local_path.stat()
    size = stat.st_size
    mtime = stat.st_mtime

    with MANIFEST_LOCK:
        record = manifest["files"].get(rel_key)

    if record and record.get("size") == size and record.get("mtime") == mtime:
        return "skipped"

    media = MediaFileUpload(str(local_path), resumable=True)

    if record and record.get("id"):
        try:
            service.files().update(
                fileId=record["id"], media_body=media, supportsAllDrives=True
            ).execute()
            action = "updated"
            file_id = record["id"]
        except HttpError as e:
            if e.resp.status == 404:
                created = service.files().create(
                    body={"name": local_path.name, "parents": [parent_id]},
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
                file_id = created["id"]
                action = "created"
            else:
                raise
    else:
        created = service.files().create(
            body={"name": local_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        file_id = created["id"]
        action = "created"

    with MANIFEST_LOCK:
        manifest["files"][rel_key] = {"id": file_id, "size": size, "mtime": mtime}
        save_manifest(manifest)

    return action


def sync_once():
    creds = load_credentials()
    manifest = load_manifest()

    # Each worker thread gets its own service/session to avoid
    # sharing one httplib2/http connection across threads.
    thread_local = threading.local()

    def get_service():
        if not hasattr(thread_local, "service"):
            thread_local.service = build(
                "drive", "v3", credentials=creds, cache_discovery=False
            )
        return thread_local.service

    main_service = get_service()

    # Ensure the root is reachable before doing anything else.
    get_or_create_folder(main_service, manifest, [])

    counts = {"created": 0, "updated": 0, "skipped": 0, "errors": 0}
    counts_lock = threading.Lock()

    upload_jobs = []

    for dirpath, dirnames, filenames in os.walk(LOCAL_ROOT):
        dirnames.sort()
        filenames.sort()

        rel_dir = Path(dirpath).relative_to(LOCAL_ROOT)
        folder_key_parts = [] if rel_dir == Path(".") else list(rel_dir.parts)

        drive_parent_id = get_or_create_folder(main_service, manifest, folder_key_parts)

        for filename in filenames:
            if filename in EXCLUDE_NAMES or filename.endswith(".tmp"):
                continue

            local_path = Path(dirpath) / filename
            rel_key = str(rel_dir / filename)
            upload_jobs.append((local_path, drive_parent_id, rel_key))

    log(f"Folders ready. {len(upload_jobs)} local files to check.")

    def run_job(job):
        local_path, parent_id, rel_key = job
        service = get_service()
        try:
            result = upload_or_update_file(service, manifest, local_path, parent_id, rel_key)
            with counts_lock:
                counts[result] = counts.get(result, 0) + 1
        except Exception as e:
            with counts_lock:
                counts["errors"] += 1
            log(f"ERROR uploading {rel_key}: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, job) for job in upload_jobs]
        for f in as_completed(futures):
            pass

    return counts


if __name__ == "__main__":
    started = time.time()
    log(f"\n=== Drive sync pass started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    counts = sync_once()
    elapsed = time.time() - started
    log(
        f"Created: {counts.get('created', 0)}  "
        f"Updated: {counts.get('updated', 0)}  "
        f"Skipped: {counts.get('skipped', 0)}  "
        f"Errors: {counts.get('errors', 0)}  "
        f"({elapsed:.1f}s)"
    )
