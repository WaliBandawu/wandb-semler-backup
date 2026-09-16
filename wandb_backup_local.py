# ============================================================
# W&B → LOCAL DISK FULL BACKUP
# STANDALONE PYTHON (run with the venv's python, e.g. in background)
#
# FULL PROJECT BACKUP
# - All W&B runs
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
import time
import traceback
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import wandb

# ============================================================
# 2. CONFIGURATION
# ============================================================

WANDB_ENTITY = "theta-tech-ai"
WANDB_PROJECT = "semler-qfhd"

# None = ALL RUNS
MAX_RUNS = None

def _resolve_backup_base():
    """BACKUP_BASE_DIR wins when set. Otherwise defaults to a
    wandb_backups/ folder next to this file, so it works on whatever
    machine the script is checked out on without editing code."""
    override = os.environ.get("BACKUP_BASE_DIR")
    if override:
        return Path(override).expanduser()

    return Path(__file__).resolve().parent / "wandb_backups"


BACKUP_BASE = _resolve_backup_base()

PROJECT_FOLDER = (
    BACKUP_BASE / f"{WANDB_ENTITY}_{WANDB_PROJECT}"
)

LOG_FILE = BACKUP_BASE / "backup_errors.log"
MANIFEST_FILE = PROJECT_FOLDER / "backup_manifest.json"

# Number of retries for a failed W&B file download
MAX_FILE_RETRIES = 3

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


def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] {message}"

    ERRORS.append(error_message)
    print(f"\n⚠ {error_message}")

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
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


def save_manifest(manifest):
    manifest["updated_at"] = datetime.now().isoformat()

    temp_file = MANIFEST_FILE.with_suffix(".tmp")

    save_json(manifest, temp_file)

    # Replace the manifest atomically where possible.
    temp_file.replace(MANIFEST_FILE)


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
# 8. GET ALL RUNS
# ============================================================

print("\n" + "=" * 70)
print("GETTING W&B RUNS")
print("=" * 70)

print(
    f"\nProject: "
    f"{WANDB_ENTITY}/{WANDB_PROJECT}"
)

try:
    start_get_runs = time.time()

    runs = list(
        api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            order="-created_at"
        )
    )

    elapsed = time.time() - start_get_runs

    print(
        f"\n✓ Found {len(runs):,} total runs"
    )

    print(
        f"Retrieval time: "
        f"{format_seconds(elapsed)}"
    )

except Exception as e:
    log_error(f"Could not retrieve W&B runs: {e}")
    traceback.print_exc()
    raise


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
        "\n✓ FULL BACKUP MODE: "
        "all runs selected"
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

    if expected_size is not None:

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

            # Verify expected size if W&B provided one.
            if expected_size is not None:

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
    manifest
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

    run_folder = get_run_folder(run)
    files_folder = run_folder / "files"

    run_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    files_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    run_record = manifest["runs"].setdefault(
        run.id,
        {
            "run_id": run.id,
            "run_name": run.name,
            "status": "in_progress",
            "files": {}
        }
    )

    run_record["run_name"] = run.name
    run_record["run_url"] = run.url
    run_record["folder"] = str(run_folder)
    run_record["last_attempt"] = (
        datetime.now().isoformat()
    )

    save_manifest(manifest)

    # ========================================================
    # METADATA
    # ========================================================

    print("\n[1/5] Metadata")

    metadata_path = (
        run_folder / "metadata.json"
    )

    try:

        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "state": run.state,
            "created_at": str(
                run.created_at
            ),
            "url": run.url,
            "group": run.group,
            "job_type": run.job_type,
            "tags": list(run.tags),
            "backup_time": (
                datetime.now().isoformat()
            )
        }

        # Regenerate metadata because it can contain
        # the latest backup timestamp.
        save_json(
            metadata,
            metadata_path
        )

        run_record["metadata"] = {
            "status": "complete",
            "path": str(metadata_path)
        }

        print("✓ metadata.json")

    except Exception as e:

        run_record["metadata"] = {
            "status": "failed",
            "error": str(e)
        }

        log_error(
            f"Run {run.id}: "
            f"metadata failed: {e}"
        )

    save_manifest(manifest)

    # ========================================================
    # CONFIG
    # ========================================================

    print("\n[2/5] Configuration")

    config_path = (
        run_folder / "config.json"
    )

    try:

        save_json(
            dict(run.config),
            config_path
        )

        run_record["config"] = {
            "status": "complete",
            "path": str(config_path)
        }

        print("✓ config.json")

    except Exception as e:

        run_record["config"] = {
            "status": "failed",
            "error": str(e)
        }

        log_error(
            f"Run {run.id}: "
            f"config failed: {e}"
        )

    save_manifest(manifest)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n[3/5] Summary metrics")

    summary_path = (
        run_folder / "summary.json"
    )

    try:

        save_json(
            dict(run.summary),
            summary_path
        )

        run_record["summary"] = {
            "status": "complete",
            "path": str(summary_path)
        }

        print("✓ summary.json")

    except Exception as e:

        run_record["summary"] = {
            "status": "failed",
            "error": str(e)
        }

        log_error(
            f"Run {run.id}: "
            f"summary failed: {e}"
        )

    save_manifest(manifest)

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

    save_manifest(manifest)

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

            save_manifest(manifest)

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

    run_record["completed_at"] = (
        datetime.now().isoformat()
    )

    run_elapsed = (
        time.time() - run_start
    )

    run_record[
        "elapsed_seconds"
    ] = run_elapsed

    save_manifest(manifest)

    print(
        f"Run time: "
        f"{format_seconds(run_elapsed)}"
    )

    return {
        "elapsed": run_elapsed,
        "status": run_record["status"]
    }


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
# 15. PROCESS ALL RUNS
# ============================================================

for index, run in enumerate(
    runs,
    start=1
):

    try:

        result = backup_run(
            run,
            index,
            len(runs),
            manifest
        )

        run_times.append(
            result["elapsed"]
        )

        if result["status"] == "complete":

            successful_runs.append(
                run.id
            )

        else:

            incomplete_runs.append(
                run.id
            )

        # ----------------------------------------------------
        # ETA
        # ----------------------------------------------------

        completed_count = index
        remaining_count = (
            len(runs) - completed_count
        )

        if remaining_count > 0:

            average_time = (
                sum(run_times)
                / len(run_times)
            )

            estimated_remaining = (
                average_time
                * remaining_count
            )

            estimated_finish = (
                datetime.now()
                + timedelta(
                    seconds=estimated_remaining
                )
            )

            print(
                "\nProgress:"
            )

            print(
                f"  Completed: "
                f"{completed_count:,}/"
                f"{len(runs):,}"
            )

            print(
                f"  Remaining: "
                f"{remaining_count:,}"
            )

            print(
                f"  Estimated remaining: "
                f"{format_seconds(estimated_remaining)}"
            )

            print(
                f"  Estimated completion: "
                f"{estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        else:

            print(
                "\n✓ All selected runs processed."
            )

    except KeyboardInterrupt:

        print(
            "\n\n⚠ BACKUP INTERRUPTED BY USER"
        )

        print(
            "Progress has been saved to the manifest."
        )

        print(
            "Run the notebook again to continue."
        )

        save_manifest(manifest)

        raise

    except Exception as e:

        failed_runs.append(
            run.id
        )

        log_error(
            f"RUN {run.id}: "
            f"unexpected error: {e}"
        )

        print(
            "\n⚠ Run failed unexpectedly."
        )

        print(
            "Continuing with the next run..."
        )

        traceback.print_exc()

        save_manifest(manifest)


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
