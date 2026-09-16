# ============================================================
# W&B → LOCAL DISK BACKUP
# STANDALONE PYTHON (run with the venv's python, e.g. in background)
#
# TARGET-LIST BACKUP
# - Every run listed in ckpt_target_run_ids.txt (not the full project)
# - Metadata / config / summary / history
# - W&B run files
# - Restart-safe local backup
# - Skips verified files already downloaded
# - Retries incomplete/failed files
#
# NOTE:
# This version does NOT yet download W&B Artifacts.
# Artifact backup should be added as the next layer for a
# genuinely complete W&B archive.
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import csv
import copy
import time
import shutil
import traceback
import hashlib
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta

import wandb

import sftp_backup_lib as sftp_lib

MANIFEST_LOCK = threading.RLock()
PRINT_LOCK = threading.Lock()

# Number of runs to back up concurrently. This work is I/O-bound
# (network waits on W&B/SFTP, not CPU), so more threads than cores
# is fine - scales with whatever machine this runs on rather than
# a number tuned for one specific box. Same formula Python's own
# ThreadPoolExecutor uses by default for exactly this reason,
# capped at 32 so a big machine doesn't hammer the single SFTP
# server harder than it can handle.
MAX_WORKERS = min(32, (os.cpu_count() or 4) + 4)

# ============================================================
# SEMLER SFTP
#
# Target/priority runs are uploaded per-run, immediately after
# each finishes downloading; local copy is deleted once its
# upload is fully confirmed. This keeps them as individually
# browsable folders on the server, which the stricter
# verify-before-skip check (3b below) depends on.
#
# Bulk (non-target) runs instead go through BULK ZIP BATCHING
# further down - uploading thousands of small files one at a
# time is dominated by SFTP round-trip latency, so they're
# grouped locally and uploaded as single zip archives instead.
# ============================================================

_sftp_project_remote_dir = None  # resolved once, before workers start


def resolve_sftp_project_folder():
    """Resolve (and cache) the shared <root>/entity/project remote
    directory chain once, sequentially, before any worker threads
    start. This is the only remote directory shared across
    concurrent runs, so it must not be created redundantly by a
    race between threads."""
    global _sftp_project_remote_dir

    _sftp_project_remote_dir = sftp_lib.resolve_project_remote_dir(
        WANDB_ENTITY, WANDB_PROJECT
    )
    return _sftp_project_remote_dir


def upload_tracking_files():
    """Best-effort: push the current backup_manifest.json and
    bulk_batch_registry.json up to the project's SFTP root too, so a
    remote copy of the backup's own bookkeeping survives even if
    this machine's local disk doesn't. Called after every individual
    upload event succeeds (a target run's own upload, or a bulk
    batch's zip upload) - not on failure, and not part of the batch
    zip itself (that stays zip-only, see 3c below)."""
    for local_path in (MANIFEST_FILE, BULK_BATCH_REGISTRY_FILE):
        if not local_path.exists():
            continue

        success, error = sftp_lib.upload_single_file(
            local_path, _sftp_project_remote_dir, local_path.name
        )

        if not success:
            log_error(f"Could not upload {local_path.name} to SFTP: {error}")

# ============================================================
# 2. CONFIGURATION
# ============================================================

WANDB_ENTITY = "theta-tech-ai"
WANDB_PROJECT = "semler-qfhd"

# None = ALL RUNS
MAX_RUNS = None

BACKUP_BASE = sftp_lib.resolve_backup_base()

PROJECT_FOLDER = (
    BACKUP_BASE / f"{WANDB_ENTITY}_{WANDB_PROJECT}"
)

# Local staging area for bulk-batch zips, deleted immediately after
# each batch's upload is confirmed - never holds more than one
# batch's worth of zips at a time in normal operation.
BULK_ZIP_STAGING = BACKUP_BASE / "_bulk_zip_staging"

# Number of bulk (non-target) runs bundled into one zip before
# upload. Target/priority runs are exempt - see SEMLER SFTP above.
BULK_ZIP_BATCH_SIZE = 10

LOG_FILE = BACKUP_BASE / "backup_errors.log"
MANIFEST_FILE = PROJECT_FOLDER / "backup_manifest.json"

# Caches each target-list run's metadata/config/summary fetched from
# W&B, so restarting the script doesn't re-fetch runs it already
# knows about. Only ever grows: on each run, just the IDs newly
# added to ckpt_target_run_ids.txt since the last cache get fetched
# and appended - see GET TARGET-LIST RUNS below.
RUN_LIST_CACHE_FILE = PROJECT_FOLDER / "run_list_cache.json"

# Local-only lookup index: which bulk batch (if any) each run ended
# up in. Never uploaded to SFTP - the manifest above already tracks
# each run's own status/batch_zip field for resume purposes; this
# is just a fast run_id <-> batch_zip lookup without needing to
# scan the whole (much larger) manifest.
BULK_BATCH_REGISTRY_FILE = PROJECT_FOLDER / "bulk_batch_registry.json"

# Target-list run IDs (same file the checkpoint-only backup uses)
# are processed first, and are held to a stricter "verify before
# skip" standard instead of trusting the manifest at face value.
#
# Lives alongside this script (ckpt_target_run_ids.txt) so it
# resolves correctly whether run locally or from wherever this
# project is deployed on the server.
TARGET_RUN_IDS_FILE = Path(__file__).resolve().parent / "ckpt_target_run_ids.txt"

# Number of retries for a failed W&B file download
MAX_FILE_RETRIES = 3

# Media files whose name contains any of these (case-insensitive)
# are skipped entirely for every run - not downloaded, and
# therefore never uploaded to SFTP either.
EXCLUDED_FILENAME_SUBSTRINGS = ("occlusion", "gradcam", "activations", "augmentations")

# Exact filenames (basename only) skipped for every run. The
# training log is effectively identical run to run and adds no
# extra information, so it's never worth backing up.
ALWAYS_EXCLUDED_EXACT_FILENAMES = {"output.log"}

# Scatter plot images are only kept for the "top run" list (the
# same spreadsheet-derived run IDs already used for stricter
# checkpoint verification, see TARGET_RUN_IDS_FILE below) - for
# every other run they aren't worth the storage/upload cost.
SCATTER_FILENAME_SUBSTRINGS = ("prediction_scatter", "best_scatter")


def is_excluded_media_file(filename, is_target=False):
    lowered = filename.lower()

    if any(
        substring in lowered
        for substring in EXCLUDED_FILENAME_SUBSTRINGS
    ):
        return True

    basename = lowered.rsplit("/", 1)[-1]

    if basename in ALWAYS_EXCLUDED_EXACT_FILENAMES:
        return True

    if not is_target and any(
        substring in lowered
        for substring in SCATTER_FILENAME_SUBSTRINGS
    ):
        return True

    return False

# ============================================================
# 3. HELPERS
# ============================================================

ERRORS = []


def format_seconds(seconds):
    seconds = max(0, seconds)

    if seconds < 60:
        return f"{seconds:.1f} seconds"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes} min {remaining_seconds} sec"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    return f"{hours} hr {remaining_minutes} min"


ERRORS_LOCK = threading.Lock()


def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] {message}"

    with ERRORS_LOCK:
        ERRORS.append(error_message)

    print(f"\n⚠ {error_message}")

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with ERRORS_LOCK, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(error_message + "\n")

    except Exception:
        print("⚠ Could not write to error log")


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_manifest():
    if not MANIFEST_FILE.exists():
        return {
            "version": 1,
            "project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "runs": {}
        }

    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "runs" not in manifest:
            manifest["runs"] = {}

        return manifest

    except Exception as e:
        log_error(f"Could not read manifest: {e}")

        # Keep the old manifest rather than destroying it.
        return {
            "version": 1,
            "project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "runs": {}
        }


def _atomic_json_save(data, target_path):
    """Deepcopy under MANIFEST_LOCK, write to a thread-suffixed temp
    file, then atomically replace the target. Shared by save_manifest
    and save_bulk_batch_registry - same tracking-file persistence
    strategy, different data."""
    with MANIFEST_LOCK:
        snapshot = copy.deepcopy(data)

    temp_file = target_path.with_suffix(f".tmp.{threading.get_ident()}")

    save_json(snapshot, temp_file)

    # Replace atomically where possible.
    temp_file.replace(target_path)


def save_manifest(manifest):
    with MANIFEST_LOCK:
        manifest["updated_at"] = datetime.now().isoformat()

    _atomic_json_save(manifest, MANIFEST_FILE)


def load_bulk_batch_registry():
    if not BULK_BATCH_REGISTRY_FILE.exists():
        return {"batches": {}, "runs": {}}

    try:
        with open(BULK_BATCH_REGISTRY_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)

        registry.setdefault("batches", {})
        registry.setdefault("runs", {})
        return registry

    except Exception as e:
        log_error(f"Could not read bulk batch registry: {e}")
        return {"batches": {}, "runs": {}}


def save_bulk_batch_registry(registry):
    _atomic_json_save(registry, BULK_BATCH_REGISTRY_FILE)


def log(message):
    with PRINT_LOCK:
        print(message)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """
    Calculate SHA-256 for a local file.

    Used for our own backup integrity verification.
    """
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_name(name):
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in name
    )


def get_file_size(path):
    try:
        return path.stat().st_size if path.exists() else 0
    except Exception:
        return 0


def load_target_run_ids():
    if not TARGET_RUN_IDS_FILE.exists():
        return []

    seen = set()
    ordered = []

    for line in TARGET_RUN_IDS_FILE.read_text().splitlines():
        rid = line.strip()

        if rid and rid not in seen:
            seen.add(rid)
            ordered.append(rid)

    return ordered


# ============================================================
# 3b. TARGET-RUN VERIFICATION (stricter "verify before skip")
#
# For every other run, a manifest saying status=complete /
# sftp.status=uploaded is trusted outright (that's what makes a
# 23k-run backup practical). For target-list runs we don't take
# that on faith: before skipping a target run's download and/or
# upload, we check that everything W&B says should exist is
# actually present - locally, or on the SFTP server if the local
# copy was already cleaned up after a prior successful upload.
# ============================================================

def expected_run_contents(run):
    """Every path (relative to the run folder) that must exist for
    this run's backup to be considered complete, mapped to its
    expected size in bytes (None = presence/non-empty is required,
    but the exact size isn't known ahead of time)."""
    expected = {
        "metadata.json": None,
        "config.json": None,
        "summary.json": None,
    }

    try:
        if list(run.scan_history()):
            expected["history.csv"] = None
    except Exception:
        # Can't tell whether history exists - require it, since
        # skipping a run that actually needs it is worse than a
        # redundant redownload.
        expected["history.csv"] = None

    for wandb_file in run.files():
        if is_excluded_media_file(wandb_file.name, is_target=True):
            continue
        size = getattr(wandb_file, "size", None)
        expected[f"files/{wandb_file.name}"] = (
            int(size) if size not in (None, 0, "0") else None
        )

    return expected


def local_run_complete(run_folder, expected):
    for rel_path, size in expected.items():
        path = run_folder / rel_path

        if not path.exists() or not path.is_file():
            return False

        local_size = get_file_size(path)

        if local_size <= 0:
            return False

        if size is not None and local_size != size:
            return False

    return True


def sftp_run_complete_safe(run_id, expected):
    """Best-effort SFTP-side verification. Any failure (connection,
    folder not found) is treated as NOT verified, so the caller
    falls back to redoing the work rather than trusting a check
    that couldn't actually run."""
    if _sftp_project_remote_dir is None:
        return False

    return sftp_lib.sftp_run_complete_safe(
        run_id, _sftp_project_remote_dir, expected, log_error=log_error
    )


# ============================================================
# 3c. BULK ZIP BATCHING (non-target runs only)
#
# Target/priority runs upload individually (see SEMLER SFTP
# above) so the strict verify-before-skip check keeps working.
# Everything else is staged locally after download and grouped
# into batches of BULK_ZIP_BATCH_SIZE; once a batch fills, it's
# zipped into one archive and uploaded as a single file instead
# of many small per-file round trips.
#
# Batching is best-effort and resume-safe: a run only gets an
# "sftp: uploaded" manifest entry once its batch's zip is
# actually confirmed on the server. If the script is interrupted
# while a partial batch is still pending, those runs are simply
# re-staged into a fresh batch on the next run - their local
# folders are never deleted until their batch succeeds.
# ============================================================

_bulk_batch_remote_dir = None  # resolved once, before workers start
_pending_bulk_batch = []       # list of (run_id, run_folder, run_record)
_bulk_batch_lock = threading.RLock()
_bulk_batch_counter = 0

# Loaded once at startup (see LOAD MANIFEST below), updated under
# MANIFEST_LOCK inside _flush_bulk_batch alongside the manifest
# itself.
_bulk_batch_registry = {"batches": {}, "runs": {}}


def resolve_bulk_batch_remote_folder():
    """Resolve (and cache) the shared bulk_batches/ remote directory
    once, sequentially, before any worker threads start."""
    global _bulk_batch_remote_dir

    _bulk_batch_remote_dir = sftp_lib.resolve_bulk_batch_remote_dir(
        _sftp_project_remote_dir
    )
    return _bulk_batch_remote_dir


def _zip_batch(batch_items, zip_path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for run_id, run_folder, _ in batch_items:
            for path in run_folder.rglob("*"):
                if path.is_file():
                    zf.write(
                        path,
                        arcname=f"{run_id}/{path.relative_to(run_folder)}"
                    )


def _flush_bulk_batch(batch_items, manifest):
    """Zip + upload one batch of bulk (non-target) runs as a single
    archive, then delete their local folders and record all of them
    in the manifest under one shared SFTP status."""
    global _bulk_batch_counter

    if not batch_items:
        return

    with _bulk_batch_lock:
        _bulk_batch_counter += 1
        batch_number = _bulk_batch_counter

    zip_name = f"bulk_batch_{batch_number:05d}.zip"
    zip_path = BULK_ZIP_STAGING / zip_name
    run_ids = [item[0] for item in batch_items]

    log(f"\n[BULK ZIP] Building {zip_name} from {len(batch_items)} runs: {run_ids}")

    try:
        _zip_batch(batch_items, zip_path)
    except Exception as e:
        log_error(f"Bulk batch {zip_name}: zipping failed: {e}")
        # Local run folders are left in place - their manifest
        # entries have no "sftp" field yet, so they'll be re-staged
        # and retried the next time this script runs.
        return

    success, error = sftp_lib.upload_bulk_zip(
        zip_path, _bulk_batch_remote_dir, zip_name
    )

    if success:
        log(f"[BULK ZIP] ✓ Uploaded {zip_name}")
        uploaded_at = datetime.now().isoformat()
        remote_zip_path = sftp_lib.remote_join(_bulk_batch_remote_dir, zip_name)

        with MANIFEST_LOCK:
            for run_id, run_folder, run_record in batch_items:
                run_record["sftp"] = {
                    "status": "uploaded",
                    "uploaded_at": uploaded_at,
                    "batch_zip": remote_zip_path
                }
                run_record["completed_at"] = uploaded_at
                manifest["runs"][run_id] = run_record

            save_manifest(manifest)

            # Registry update: same lock, same moment, so the two
            # files never disagree about which batch a run landed in.
            for run_id, run_folder, run_record in batch_items:
                if run_id in _bulk_batch_registry["runs"]:
                    # Should be unreachable - the skip logic in
                    # backup_run() never lets an already-uploaded run
                    # reach staging again. Not a failure, just a
                    # sanity-check note if it ever does happen.
                    log(
                        f"⚠ Run {run_id}: already in bulk batch registry "
                        f"under {_bulk_batch_registry['runs'][run_id]} - "
                        f"overwriting with {zip_name}"
                    )
                _bulk_batch_registry["runs"][run_id] = zip_name

            _bulk_batch_registry["batches"][zip_name] = {
                "run_ids": run_ids,
                "remote_path": remote_zip_path,
                "uploaded_at": uploaded_at
            }

            save_bulk_batch_registry(_bulk_batch_registry)

        upload_tracking_files()

        for run_id, run_folder, _ in batch_items:
            try:
                shutil.rmtree(run_folder)
            except Exception as e:
                log_error(
                    f"Run {run_id}: uploaded in {zip_name} but could "
                    f"not delete local folder: {e}"
                )
    else:
        log_error(f"Bulk batch {zip_name}: SFTP upload failed: {error}")
        # Local run folders stay in place too - same reasoning as
        # the zipping-failed branch above.

    try:
        zip_path.unlink()
    except Exception:
        pass


def stage_bulk_run(run_id, run_folder, run_record, manifest):
    """Add one downloaded (non-target) run to the pending bulk-zip
    batch. Whichever thread's run happens to fill the batch to
    BULK_ZIP_BATCH_SIZE does the zip + upload + manifest write for
    the whole batch, not just its own run."""
    batch_to_flush = None

    with _bulk_batch_lock:
        _pending_bulk_batch.append((run_id, run_folder, run_record))

        if len(_pending_bulk_batch) >= BULK_ZIP_BATCH_SIZE:
            batch_to_flush = list(_pending_bulk_batch)
            _pending_bulk_batch.clear()

    if batch_to_flush:
        _flush_bulk_batch(batch_to_flush, manifest)


def flush_remaining_bulk_batch(manifest):
    """Call once after all runs have been processed, to upload
    whatever partial batch (fewer than BULK_ZIP_BATCH_SIZE runs) is
    still pending."""
    with _bulk_batch_lock:
        remaining = list(_pending_bulk_batch)
        _pending_bulk_batch.clear()

    if remaining:
        log(f"\n[BULK ZIP] Flushing final partial batch of {len(remaining)} runs")
        _flush_bulk_batch(remaining, manifest)


# ============================================================
# 4. HEADER
# ============================================================

print("=" * 70)
print("W&B → LOCAL DISK FULL BACKUP")
print("=" * 70)


# ============================================================
# 5. PREPARE DIRECTORIES
# ============================================================

try:
    BACKUP_BASE.mkdir(
        parents=True,
        exist_ok=True
    )

    PROJECT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n✓ Backup directory:")
    print(PROJECT_FOLDER)

except Exception as e:
    print(f"✗ Could not create backup directory: {e}")
    raise


# ============================================================
# 6. LOAD MANIFEST
# ============================================================

manifest = load_manifest()

print("\n✓ Backup manifest loaded")

print(
    f"Previously tracked runs: "
    f"{len(manifest.get('runs', {}))}"
)

_bulk_batch_registry = load_bulk_batch_registry()

print(
    f"✓ Bulk batch registry loaded "
    f"({len(_bulk_batch_registry['batches'])} batches, "
    f"{len(_bulk_batch_registry['runs'])} runs indexed)"
)


# ============================================================
# 7. CONNECT TO W&B
# ============================================================

print("\n" + "=" * 70)
print("CONNECTING TO W&B")
print("=" * 70)

try:
    api = wandb.Api()

    print("✓ W&B API initialized")

except Exception as e:
    log_error(f"W&B API initialization failed: {e}")
    traceback.print_exc()
    raise


# ============================================================
# 8. GET TARGET-LIST RUNS
#
# Only the runs curated in ckpt_target_run_ids.txt are backed up by
# this pipeline - not the full project. Each is fetched individually
# (api.run(), not the much slower full-project api.runs() listing)
# and cached to disk, so a restart only has to fetch whatever run
# IDs were newly added to the target list since the last cache.
# ============================================================


class _CachedRunProxy:
    """Stands in for a wandb Run built from RUN_LIST_CACHE_FILE.

    id/name/state/url/created_at/config/summary come straight from
    the cache, no network involved. Anything else (run.files(),
    run.scan_history(), ...) fetches the real wandb Run on first use
    and delegates to it from then on - which only happens for runs
    actually being processed, not the whole target list on every
    restart."""

    __slots__ = (
        "_api", "_live", "_path", "config", "created_at",
        "id", "name", "state", "summary", "url",
    )

    def __init__(self, api_handle, path, cached):
        self.id = cached["id"]
        self.name = cached["name"]
        self.state = cached["state"]
        self.url = cached["url"]
        self.created_at = cached["created_at"]
        self.config = cached["config"]
        self.summary = cached["summary"]
        self._api = api_handle
        self._path = path
        self._live = None

    def __getattr__(self, item):
        if self._live is None:
            self._live = self._api.run(self._path)
        return getattr(self._live, item)


def _serialize_run_for_cache(run):
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
    }


def _load_run_cache():
    try:
        with open(RUN_LIST_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("entity") == WANDB_ENTITY and data.get("project") == WANDB_PROJECT:
            return data.get("runs", {})
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠ Could not read run cache, ignoring it: {e}")
    return {}


def _save_run_cache(runs_by_id):
    try:
        RUN_LIST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = RUN_LIST_CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entity": WANDB_ENTITY,
                    "project": WANDB_PROJECT,
                    "runs": runs_by_id,
                },
                f,
                default=str,
            )
        tmp_path.replace(RUN_LIST_CACHE_FILE)
    except Exception as e:
        print(f"⚠ Could not write run cache: {e}")


print("\n" + "=" * 70)
print("GETTING TARGET-LIST RUNS")
print("=" * 70)

print(
    f"\nProject: "
    f"{WANDB_ENTITY}/{WANDB_PROJECT}"
)

target_ids = load_target_run_ids()
target_set = set(target_ids)

print(
    f"\n✓ Target run IDs loaded: {len(target_ids)} "
    f"(from {TARGET_RUN_IDS_FILE})"
)

if not target_ids:
    print("\n✗ No target run IDs found - nothing to back up.")
    raise SystemExit

cached_runs = _load_run_cache()
to_fetch = [rid for rid in target_ids if rid not in cached_runs]

print(
    f"✓ {len(target_ids) - len(to_fetch)} run(s) loaded from cache "
    f"({RUN_LIST_CACHE_FILE.name})"
)

missing_targets = []

if to_fetch:
    print(f"Fetching {len(to_fetch)} run(s) not yet cached...")

    try:
        start_get_runs = time.time()

        def _fetch_one(rid):
            try:
                run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}")
                return rid, _serialize_run_for_cache(run), None
            except Exception as e:
                return rid, None, e

        fetched_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_fetch_one, rid) for rid in to_fetch]

            for future in as_completed(futures):
                rid, entry, err = future.result()
                fetched_count += 1

                if entry is not None:
                    cached_runs[rid] = entry
                else:
                    missing_targets.append(rid)

                if fetched_count % 100 == 0 or fetched_count == len(to_fetch):
                    print(f"  {fetched_count}/{len(to_fetch)} fetched...")

        elapsed = time.time() - start_get_runs

        print(
            f"✓ Fetched {len(to_fetch) - len(missing_targets)} run(s) "
            f"in {format_seconds(elapsed)}"
        )

        _save_run_cache(cached_runs)
        print(f"✓ Run cache updated ({RUN_LIST_CACHE_FILE.name})")

    except Exception as e:
        log_error(f"Could not fetch target-list runs: {e}")
        traceback.print_exc()
        raise

    if missing_targets:
        print(
            f"⚠ {len(missing_targets)} target run IDs could not be fetched "
            f"from W&B (deleted, or no access?): {missing_targets[:10]}"
            f"{' ...' if len(missing_targets) > 10 else ''}"
        )
else:
    print("✓ All target runs already cached, nothing new to fetch")

runs = [
    _CachedRunProxy(api, f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}", cached_runs[rid])
    for rid in target_ids
    if rid in cached_runs
]

print(f"\n✓ {len(runs):,} target runs ready to process")


# ============================================================
# 9. OPTIONAL LIMIT
# ============================================================

if MAX_RUNS is not None:
    runs = runs[:MAX_RUNS]

    print(
        f"\n⚠ TEST LIMIT ENABLED: "
        f"{MAX_RUNS} runs"
    )

else:
    print(
        "\n✓ TARGET-LIST BACKUP MODE: "
        "all target-list runs selected"
    )


if not runs:
    print("\n✗ No runs found.")
    raise SystemExit


# ============================================================
# 10. RUN FOLDER
# ============================================================

def get_run_folder(run):
    return (
        PROJECT_FOLDER
        / run.id
    )


# ============================================================
# 11. CHECK EXISTING FILE
# ============================================================

def is_existing_file_valid(path, expected_size=None):
    """
    Determine whether an existing local file is safe to skip.

    If W&B exposes a file size, compare it.
    Otherwise, a non-empty existing file can be reused,
    but it will not be called cryptographically verified.
    """

    if not path.exists():
        return False

    if not path.is_file():
        return False

    local_size = get_file_size(path)

    if local_size <= 0:
        return False

    # W&B sometimes reports size 0 for files (e.g. artifact
    # reference manifests) whose real size it hasn't recorded.
    # Treat that as "unknown" rather than an exact expectation.
    if expected_size not in (None, 0, "0"):

        try:
            expected_size = int(expected_size)

            if local_size != expected_size:
                return False

        except Exception:
            pass

    return True


# ============================================================
# 12. DOWNLOAD ONE W&B FILE
# ============================================================

def download_wandb_file(
    wandb_file,
    files_folder,
    run_id
):
    filename = wandb_file.name

    # W&B file names may contain folders.
    destination = (
        files_folder / Path(filename)
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Try to get the expected W&B size.
    expected_size = getattr(
        wandb_file,
        "size",
        None
    )

    # --------------------------------------------------------
    # SKIP EXISTING VERIFIED/SIZE-MATCHING FILE
    # --------------------------------------------------------

    if is_existing_file_valid(
        destination,
        expected_size
    ):

        print(
            f"  ↪ SKIP: {filename}"
            " (already downloaded)"
        )

        return {
            "status": "skipped",
            "path": str(destination),
            "size": get_file_size(destination),
            "sha256": None
        }

    # --------------------------------------------------------
    # DOWNLOAD WITH RETRIES
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_FILE_RETRIES + 1
    ):

        print(
            f"  ↓ Downloading: {filename}"
            f" [attempt {attempt}/{MAX_FILE_RETRIES}]"
        )

        start = time.time()

        try:

            # W&B handles the remote download.
            # Existing incomplete local files are NOT trusted;
            # they are replaced by a fresh valid W&B download.
            wandb_file.download(
                root=str(files_folder),
                replace=True
            )

            elapsed = (
                time.time() - start
            )

            if not destination.exists():

                raise FileNotFoundError(
                    f"Expected downloaded file "
                    f"not found: {destination}"
                )

            local_size = get_file_size(
                destination
            )

            if local_size <= 0:
                raise IOError(
                    "Downloaded file is empty."
                )

            # Verify expected size if W&B provided a real one.
            # A reported size of 0 means W&B doesn't actually know
            # the size (common for artifact reference manifests),
            # not that the file should be empty.
            if expected_size not in (None, 0, "0"):

                try:
                    if local_size != int(
                        expected_size
                    ):
                        raise IOError(
                            f"Size mismatch. "
                            f"Expected {expected_size}, "
                            f"got {local_size}"
                        )
                except (TypeError, ValueError):
                    pass

            # Calculate local checksum.
            checksum = sha256_file(
                destination
            )

            print(
                f"  ✓ COMPLETE: {filename}"
                f" ({format_seconds(elapsed)})"
            )

            return {
                "status": "downloaded",
                "path": str(destination),
                "size": local_size,
                "sha256": checksum,
                "elapsed_seconds": elapsed
            }

        except Exception as e:

            print(
                f"  ✗ Attempt {attempt} failed: {e}"
            )

            if attempt < MAX_FILE_RETRIES:

                # Short backoff before retry.
                sleep_seconds = min(
                    5 * attempt,
                    30
                )

                print(
                    f"  Retrying in "
                    f"{sleep_seconds} seconds..."
                )

                time.sleep(
                    sleep_seconds
                )

            else:

                log_error(
                    f"Run {run_id}, file "
                    f"{filename}: "
                    f"all download attempts failed: "
                    f"{e}"
                )

                return {
                    "status": "failed",
                    "path": str(destination),
                    "size": get_file_size(
                        destination
                    ),
                    "sha256": None,
                    "error": str(e)
                }


# ============================================================
# 13. BACKUP ONE RUN
# ============================================================

def backup_run(
    run,
    run_number,
    total_runs,
    manifest,
    target_set
):

    run_start = time.time()

    print("\n")
    print("=" * 70)
    print(
        f"RUN {run_number}/{total_runs}"
    )
    print("=" * 70)

    print(f"Name : {run.name}")
    print(f"ID   : {run.id}")
    print(f"State: {run.state}")
    print(f"URL  : {run.url}")

    with MANIFEST_LOCK:
        existing = manifest["runs"].get(run.id)

    is_target = run.id in target_set

    already_downloaded = bool(
        existing and existing.get("status") == "complete"
    )
    already_uploaded = bool(
        existing and (
            existing.get("sftp", {}).get("status") == "uploaded"
            # Legacy: runs already archived to Google Drive before the
            # switch to SFTP. Their local copy was already deleted, so
            # they're still treated as "done" here rather than
            # redownloaded from W&B - migrating that Drive content to
            # SFTP is handled separately by migrate_drive_to_sftp.py.
            or existing.get("drive", {}).get("status") == "uploaded"
        )
    )

    run_folder = get_run_folder(run)
    files_folder = run_folder / "files"

    # Target-list runs don't get to skip on the manifest's word
    # alone: fetch what W&B says should exist so we can verify
    # actual bytes below, instead of trusting stale status flags.
    expected = None

    if is_target and (already_downloaded or already_uploaded):
        try:
            expected = expected_run_contents(run)
        except Exception as e:
            log_error(
                f"Run {run.id}: could not determine expected contents "
                f"for target verification: {e}"
            )
            expected = None  # unknown -> can't verify -> don't skip

    if already_downloaded and already_uploaded:
        if is_target:
            print(f"↪ Target run {run.id} marked complete — verifying before skipping...")

            verified = expected is not None and (
                (run_folder.exists() and local_run_complete(run_folder, expected))
                or sftp_run_complete_safe(run.id, expected)
            )

            if verified:
                print(f"✓ Verified complete, skipping (run {run.id})")
                return {
                    "elapsed": 0.0,
                    "status": "complete"
                }

            print(
                f"⚠ Target run {run.id}: manifest said complete but "
                f"verification FAILED — redoing"
            )
            already_downloaded = False
            already_uploaded = False

        else:
            print(f"✓ Fully done (downloaded + uploaded), skipping (run {run.id})")
            return {
                "elapsed": 0.0,
                "status": "complete"
            }

    target_local_incomplete = (
        is_target
        and already_downloaded
        and run_folder.exists()
        and expected is not None
        and not local_run_complete(run_folder, expected)
    )

    if already_downloaded and run_folder.exists() and not target_local_incomplete:
        print(f"↪ Already downloaded, resuming SFTP upload only (run {run.id})")
        run_record = dict(existing)

    else:
        if target_local_incomplete:
            print(
                f"⚠ Target run {run.id}: local folder incomplete vs W&B "
                f"— repairing before upload"
            )
        run_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        files_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # This run's own working copy. Only merged into the shared
        # manifest (under lock) once, at the end of this function,
        # so concurrent runs never touch each other's dict entries
        # mid-mutation.
        run_record = {
            "run_id": run.id,
            "run_name": run.name,
            "status": "in_progress",
            "files": {}
        }

        run_record["run_url"] = run.url
        run_record["folder"] = str(run_folder)
        run_record["last_attempt"] = (
            datetime.now().isoformat()
        )

        backup_run_download(run, run_folder, files_folder, run_record, is_target)

    # ========================================================
    # UPLOAD TO SFTP
    #
    # Target/priority runs upload individually and delete their
    # local copy immediately (unchanged behavior - keeps them
    # verifiable via sftp_run_complete_safe). Bulk (non-target)
    # runs instead stage into a shared zip batch; their local
    # copy is deleted only once their batch's upload is confirmed
    # (see 3c. BULK ZIP BATCHING above), which may happen inside
    # a different run's call to backup_run().
    # ========================================================

    run_elapsed = (
        time.time() - run_start
    )

    run_record[
        "elapsed_seconds"
    ] = run_elapsed

    if run_record.get("status") == "complete" and run_folder.exists():

        if is_target:
            print(f"\n[SFTP] Uploading run {run.id}...")

            success, error = sftp_lib.upload_run_to_sftp(
                run_folder, _sftp_project_remote_dir
            )

            if success:
                run_record["sftp"] = {
                    "status": "uploaded",
                    "uploaded_at": datetime.now().isoformat()
                }

                try:
                    shutil.rmtree(run_folder)
                    print(f"[SFTP] ✓ Uploaded and deleted local copy (run {run.id})")
                except Exception as e:
                    log_error(
                        f"Run {run.id}: uploaded to SFTP but could not "
                        f"delete local folder: {e}"
                    )
            else:
                run_record["sftp"] = {
                    "status": "failed",
                    "error": error
                }
                log_error(f"Run {run.id}: SFTP upload failed: {error}")

            run_record["completed_at"] = datetime.now().isoformat()

            with MANIFEST_LOCK:
                manifest["runs"][run.id] = run_record
                save_manifest(manifest)

            if success:
                upload_tracking_files()

        else:
            print(f"\n[BULK ZIP] Staging run {run.id} for batch upload...")

            # Record the download-complete state right away so a
            # restart never re-downloads this run, even if its
            # batch hasn't filled/uploaded yet.
            with MANIFEST_LOCK:
                manifest["runs"][run.id] = run_record
                save_manifest(manifest)

            # May trigger this batch's zip + upload immediately (if
            # this run fills it) or just stage it for later - either
            # way, the manifest write above already made the
            # download durable.
            stage_bulk_run(run.id, run_folder, run_record, manifest)

    else:
        # Download itself was incomplete/failed.
        run_record["completed_at"] = datetime.now().isoformat()

        with MANIFEST_LOCK:
            manifest["runs"][run.id] = run_record
            save_manifest(manifest)

    print(
        f"Run time: "
        f"{format_seconds(run_elapsed)}"
    )

    return {
        "elapsed": run_elapsed,
        "status": run_record["status"]
    }


def _save_run_component(run, run_record, component, path, build_data):
    """Build one JSON component (metadata/config/summary), save it,
    and record success/failure into run_record[component]. Shared by
    the three near-identical steps at the top of backup_run_download
    below - history has its own shape (CSV, dynamic columns, empty-
    vs-nonempty branching) so it stays separate rather than being
    forced into this."""
    try:
        save_json(build_data(), path)
        run_record[component] = {"status": "complete", "path": str(path)}
        print(f"✓ {path.name}")
    except Exception as e:
        run_record[component] = {"status": "failed", "error": str(e)}
        log_error(f"Run {run.id}: {component} failed: {e}")


def backup_run_download(run, run_folder, files_folder, run_record, is_target=False):
    # ========================================================
    # METADATA
    # ========================================================

    print("\n[1/5] Metadata")

    # Rebuilt every time (not just on first download) because it
    # carries the latest backup timestamp.
    _save_run_component(
        run, run_record, "metadata", run_folder / "metadata.json",
        lambda: {
            "run_id": run.id,
            "run_name": run.name,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
            "group": run.group,
            "job_type": run.job_type,
            "tags": list(run.tags),
            "backup_time": datetime.now().isoformat()
        }
    )

    # ========================================================
    # CONFIG
    # ========================================================

    print("\n[2/5] Configuration")

    _save_run_component(
        run, run_record, "config", run_folder / "config.json",
        lambda: dict(run.config)
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n[3/5] Summary metrics")

    _save_run_component(
        run, run_record, "summary", run_folder / "summary.json",
        lambda: dict(run.summary)
    )

    # ========================================================
    # HISTORY
    # ========================================================

    print("\n[4/5] Run history")

    history_path = (
        run_folder / "history.csv"
    )

    try:

        # If the history file already exists, we still
        # refresh it because W&B history can contain
        # the complete run history.
        history = list(
            run.scan_history()
        )

        if history:

            columns = set()

            for row in history:
                columns.update(
                    row.keys()
                )

            columns = sorted(columns)

            temp_history = (
                run_folder / "history.csv.tmp"
            )

            with open(
                temp_history,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=columns,
                    extrasaction="ignore"
                )

                writer.writeheader()

                for row in history:
                    writer.writerow(row)

            temp_history.replace(
                history_path
            )

            run_record["history"] = {
                "status": "complete",
                "rows": len(history),
                "path": str(history_path)
            }

            print(
                f"✓ history.csv "
                f"({len(history):,} rows)"
            )

        else:

            run_record["history"] = {
                "status": "complete",
                "rows": 0,
                "path": str(history_path)
            }

            print(
                "- No history data found"
            )

    except Exception as e:

        run_record["history"] = {
            "status": "failed",
            "error": str(e)
        }

        log_error(
            f"Run {run.id}: "
            f"history failed: {e}"
        )

    # ========================================================
    # W&B RUN FILES
    # ========================================================

    print("\n[5/5] W&B run files")

    successful_files = 0
    skipped_files = 0
    failed_files = 0

    try:

        run_files = list(
            run.files()
        )

        total_files = len(
            run_files
        )

        print(
            f"Found {total_files} W&B files"
        )

        for file_number, wandb_file in enumerate(
            run_files,
            start=1
        ):

            filename = wandb_file.name

            print(
                f"\n  File "
                f"{file_number}/{total_files}"
            )

            if is_excluded_media_file(filename, is_target=is_target):

                print(
                    f"  ↪ SKIP: {filename}"
                    " (excluded: occlusion/gradcam/log/non-top-run scatter)"
                )

                result = {
                    "status": "excluded",
                    "path": None,
                    "size": None,
                    "sha256": None
                }

                run_record["files"][
                    filename
                ] = result

                continue

            result = download_wandb_file(
                wandb_file,
                files_folder,
                run.id
            )

            # Save progress immediately.
            run_record["files"][
                filename
            ] = result

            if result["status"] == "downloaded":
                successful_files += 1

            elif result["status"] == "skipped":
                skipped_files += 1

            else:
                failed_files += 1

        print(
            f"\nFiles downloaded: "
            f"{successful_files}"
        )

        print(
            f"Files skipped: "
            f"{skipped_files}"
        )

        print(
            f"Files failed: "
            f"{failed_files}"
        )

    except Exception as e:

        log_error(
            f"Run {run.id}: "
            f"could not retrieve run files: {e}"
        )

        run_record["files_error"] = str(e)

    # ========================================================
    # DETERMINE RUN STATUS
    # ========================================================

    failed_components = []

    for component in [
        "metadata",
        "config",
        "summary",
        "history"
    ]:

        component_record = run_record.get(
            component,
            {}
        )

        if component_record.get(
            "status"
        ) != "complete":

            failed_components.append(
                component
            )

    file_records = run_record.get(
        "files",
        {}
    )

    if any(
        record.get("status") == "failed"
        for record in file_records.values()
    ):
        failed_components.append(
            "run_files"
        )

    if failed_components:

        run_record["status"] = (
            "incomplete"
        )

        run_record[
            "failed_components"
        ] = failed_components

        print(
            "\n⚠ RUN INCOMPLETE"
        )

    else:

        run_record["status"] = (
            "complete"
        )

        run_record.pop(
            "failed_components",
            None
        )

        print(
            "\n✓ RUN COMPLETE"
        )


# ============================================================
# 14. START FULL BACKUP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FULL W&B BACKUP")
print("=" * 70)

print(
    f"\nRuns to process: {len(runs):,}"
)

print(
    "\nRestart-safe behavior:"
)

print(
    "  ✓ Existing complete files → SKIP"
)

print(
    "  ✓ Failed files → RETRY"
)

print(
    "  ✓ Progress → saved to manifest"
)

print(
    "  ✓ Interrupted backup → rerun safely"
)

print(
    "\nNOTE: W&B Artifact contents are not yet "
    "included in this version."
)

overall_start = time.time()

successful_runs = []
incomplete_runs = []
failed_runs = []
run_times = []


# ============================================================
# 15. PROCESS ALL RUNS (CONCURRENTLY)
# ============================================================

print("\nResolving SFTP destination folder...")
resolve_sftp_project_folder()
print(f"✓ SFTP destination ready: {_sftp_project_remote_dir}")

print("Resolving bulk-batch destination folder...")
resolve_bulk_batch_remote_folder()
print(f"✓ Bulk-batch destination ready: {_bulk_batch_remote_dir}")

print(f"\nUsing {MAX_WORKERS} concurrent workers.\n")

progress_lock = threading.Lock()
completed_count = 0

try:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_run = {
            executor.submit(
                backup_run, run, index, len(runs), manifest, target_set
            ): run
            for index, run in enumerate(runs, start=1)
        }

        for future in as_completed(future_to_run):
            run = future_to_run[future]

            try:
                result = future.result()

                with progress_lock:
                    run_times.append(result["elapsed"])

                    if result["status"] == "complete":
                        successful_runs.append(run.id)
                    else:
                        incomplete_runs.append(run.id)

                    completed_count += 1
                    remaining_count = len(runs) - completed_count

                    if remaining_count > 0 and run_times:
                        average_time = sum(run_times) / len(run_times)
                        # Concurrent workers finish roughly MAX_WORKERS
                        # at a time, so scale the naive estimate down.
                        estimated_remaining = (
                            average_time * remaining_count / MAX_WORKERS
                        )
                        estimated_finish = datetime.now() + timedelta(
                            seconds=estimated_remaining
                        )

                        print(
                            f"\nProgress: {completed_count:,}/{len(runs):,} "
                            f"(remaining {remaining_count:,}) — "
                            f"est. remaining {format_seconds(estimated_remaining)}, "
                            f"est. finish {estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    else:
                        print("\n✓ All selected runs processed.")

            except Exception as e:
                with progress_lock:
                    failed_runs.append(run.id)
                    completed_count += 1

                log_error(f"RUN {run.id}: unexpected error: {e}")
                traceback.print_exc()

except KeyboardInterrupt:
    print("\n\n⚠ BACKUP INTERRUPTED BY USER")
    print("Progress has been saved to the manifest.")
    print("Any pending bulk-zip batch (< BULK_ZIP_BATCH_SIZE runs) is")
    print("left staged locally and will be picked up on the next run.")
    save_manifest(manifest)
    raise

# Whatever bulk (non-target) runs didn't fill a full batch on their
# own get uploaded now, as one final partial batch.
flush_remaining_bulk_batch(manifest)


# ============================================================
# 16. FINAL REPORT
# ============================================================

total_elapsed = (
    time.time() - overall_start
)

print("\n")
print("=" * 70)
print("FULL BACKUP COMPLETE")
print("=" * 70)

print(
    f"\nRuns selected : "
    f"{len(runs):,}"
)

print(
    f"Completed      : "
    f"{len(successful_runs):,}"
)

print(
    f"Incomplete     : "
    f"{len(incomplete_runs):,}"
)

print(
    f"Failed         : "
    f"{len(failed_runs):,}"
)

print(
    f"Total time     : "
    f"{format_seconds(total_elapsed)}"
)

if run_times:

    average_run_time = (
        sum(run_times)
        / len(run_times)
    )

    print(
        f"Average/run    : "
        f"{format_seconds(average_run_time)}"
    )

print(
    "\nBackup location:"
)

print(PROJECT_FOLDER)

print(
    "\nManifest:"
)

print(MANIFEST_FILE)


# ============================================================
# 17. ERROR REPORT
# ============================================================

print("\n")
print("=" * 70)
print("ERROR REPORT")
print("=" * 70)

if ERRORS:

    print(
        f"\nErrors/warnings this session: "
        f"{len(ERRORS)}"
    )

    for error in ERRORS:
        print(error)

    print(
        f"\nFull error log:"
        f"\n{LOG_FILE}"
    )

else:

    print(
        "\n✓ No errors/warnings recorded "
        "during this session."
    )


# ============================================================
# 18. INCOMPLETE RUNS
# ============================================================

if incomplete_runs:

    print("\n")
    print("=" * 70)
    print("INCOMPLETE RUNS")
    print("=" * 70)

    for run_id in incomplete_runs:
        print(f"  ⚠ {run_id}")

    print(
        "\nThese runs will be retried automatically "
        "when the notebook is run again."
    )


# ============================================================
# 19. FAILED RUNS
# ============================================================

if failed_runs:

    print("\n")
    print("=" * 70)
    print("FAILED RUNS")
    print("=" * 70)

    for run_id in failed_runs:
        print(f"  ✗ {run_id}")


# ============================================================
# 20. CALCULATE BACKUP SIZE
# ============================================================

print("\n")
print("=" * 70)
print("BACKUP SIZE")
print("=" * 70)

try:

    total_backup_bytes = 0
    total_backup_files = 0

    for path in PROJECT_FOLDER.rglob("*"):

        if path.is_file():

            total_backup_bytes += (
                path.stat().st_size
            )

            total_backup_files += 1

    total_backup_mb = (
        total_backup_bytes
        / 1024
        / 1024
    )

    total_backup_gb = (
        total_backup_mb
        / 1024
    )

    print(
        f"Total files: "
        f"{total_backup_files:,}"
    )

    print(
        f"Total size: "
        f"{total_backup_mb:.2f} MB"
    )

    if total_backup_gb >= 1:

        print(
            f"Total size: "
            f"{total_backup_gb:.2f} GB"
        )

except Exception as e:

    log_error(
        f"Could not calculate backup size: {e}"
    )


# ============================================================
# 21. FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)

if (
    len(incomplete_runs) == 0
    and len(failed_runs) == 0
    and len(successful_runs) == len(runs)
):

    print(
        "✓ SUCCESS — ALL RUNS BACKED UP"
    )

else:

    print(
        "⚠ BACKUP COMPLETED "
        "BUT SOME ITEMS REQUIRE ATTENTION"
    )

print("=" * 70)

print(
    "\nThe backup is restart-safe."
)

print(
    "If Colab stops, run this notebook again."
)

print(
    "Existing completed files will be skipped "
    "and incomplete/failed work will be retried."
)

print(
    "\nIMPORTANT:"
)

print(
    "Artifact backup still needs to be added "
    "for a complete W&B archive."
)
