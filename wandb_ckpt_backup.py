# ============================================================
# W&B CHECKPOINT-ONLY BACKUP (all runs, targeted list prioritized)
#
# Walks every run in the project. Run IDs listed in
# ckpt_target_run_ids.txt are processed first (priority) and get
# their "model" type logged artifacts (checkpoint .ckpt files)
# downloaded and uploaded to the SAME SFTP destination tree used
# by wandb_backup_parallel.py:
#   W&B Runs/<entity>/<project>/<run_id>/checkpoints/<artifact>/...
#
# Every other run is only checked (via the W&B API) for whether it
# has checkpoint artifacts - that fact is recorded in the manifest,
# but the checkpoint files themselves are NEVER downloaded and NEVER
# uploaded to SFTP for a non-targeted run.
#
# Does NOT touch the main backup manifest (backup_manifest.json).
# Restart-safe via its own manifest (ckpt_manifest.json).
# ============================================================

import copy
import json
import os
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

import wandb

import sftp_backup_lib as sftp_lib

MANIFEST_LOCK = threading.RLock()
PRINT_LOCK = threading.Lock()

# Higher than the main backup's (os.cpu_count() + 4) workers: this
# job only moves small checkpoint files (not full run file trees),
# so more concurrent network + hashing work is safe. Scales with
# the machine like the main backup does, just roughly double -
# capped higher too so it stays proportionally ahead on big boxes.
MAX_WORKERS = min(48, ((os.cpu_count() or 4) + 4) * 2)

WANDB_ENTITY = "theta-tech-ai"
WANDB_PROJECT = "semler-qfhd"

BACKUP_BASE = sftp_lib.resolve_backup_base()
PROJECT_FOLDER = BACKUP_BASE / f"{WANDB_ENTITY}_{WANDB_PROJECT}"
CKPT_STAGING = BACKUP_BASE / "_ckpt_staging"
MANIFEST_FILE = PROJECT_FOLDER / "ckpt_manifest.json"
LOG_FILE = BACKUP_BASE / "ckpt_backup_errors.log"
# Lives alongside this script (ckpt_target_run_ids.txt) so it
# resolves correctly whether run locally or from wherever this
# project is deployed on the server.
RUN_IDS_FILE = Path(__file__).resolve().parent / "ckpt_target_run_ids.txt"

_sftp_project_remote_dir = None

ERRORS = []
ERRORS_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_message = f"[{timestamp}] {message}"
    with ERRORS_LOCK:
        ERRORS.append(error_message)
    log(f"\n⚠ {error_message}")
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ERRORS_LOCK, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(error_message + "\n")
    except Exception:
        log("⚠ Could not write to error log")


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
            "runs": {},
        }
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest.setdefault("runs", {})
        return manifest
    except Exception as e:
        log_error(f"Could not read ckpt manifest: {e}")
        return {
            "version": 1,
            "project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "runs": {},
        }


def save_manifest(manifest):
    with MANIFEST_LOCK:
        manifest["updated_at"] = datetime.now().isoformat()
        snapshot = copy.deepcopy(manifest)

    temp_file = MANIFEST_FILE.with_suffix(f".tmp.{threading.get_ident()}")
    save_json(snapshot, temp_file)
    temp_file.replace(MANIFEST_FILE)


# ============================================================
# SEMLER SFTP (same destination tree as the main backup)
# ============================================================


def resolve_sftp_project_folder():
    global _sftp_project_remote_dir
    _sftp_project_remote_dir = sftp_lib.resolve_project_remote_dir(
        WANDB_ENTITY, WANDB_PROJECT
    )
    return _sftp_project_remote_dir


def upload_run_checkpoints_to_sftp(run_id, staging_dir):
    """staging_dir/<run_id>/checkpoints/<artifact_dirs...> uploaded under
    W&B Runs/entity/project/<run_id>/checkpoints/..."""
    run_remote_dir = sftp_lib.remote_join(
        _sftp_project_remote_dir, sftp_lib.safe_name(run_id)
    )
    sftp_lib.with_sftp_retry(sftp_lib.ensure_remote_dir, run_remote_dir)
    sftp_lib.with_sftp_retry(sftp_lib.upload_directory_tree, staging_dir, run_remote_dir)
    return run_remote_dir


# ============================================================
# PER-RUN CHECKPOINT BACKUP
# ============================================================


def backup_run_checkpoints(run_id, index, total, manifest):
    """Full path: download every checkpoint artifact for this (targeted)
    run and upload it to SFTP."""
    with MANIFEST_LOCK:
        existing = manifest["runs"].get(run_id)
    if existing and existing.get("status") == "complete":
        log(f"[{index}/{total}] {run_id}: already backed up, skipping")
        return {"run_id": run_id, "status": "complete", "skipped": True}

    log(f"\n{'=' * 60}\n[{index}/{total}] RUN {run_id} (TARGET)\n{'=' * 60}")

    record = {
        "run_id": run_id,
        "target": True,
        "status": "in_progress",
        "artifacts": {},
        "last_attempt": datetime.now().isoformat(),
    }

    try:
        api = wandb.Api()
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    except Exception as e:
        log_error(f"Run {run_id}: not found / could not fetch run: {e}")
        record["status"] = "failed"
        record["error"] = f"run fetch failed: {e}"
        with MANIFEST_LOCK:
            manifest["runs"][run_id] = record
            save_manifest(manifest)
        return {"run_id": run_id, "status": "failed"}

    run_staging = CKPT_STAGING / run_id / "checkpoints"
    run_staging.mkdir(parents=True, exist_ok=True)

    try:
        artifacts_record, any_failed = sftp_lib.download_model_artifacts(
            run, run_staging,
            existing_artifacts=(existing or {}).get("artifacts", {})
        )
    except RuntimeError as e:
        log_error(f"Run {run_id}: {e}")
        record["status"] = "failed"
        record["error"] = str(e)
        with MANIFEST_LOCK:
            manifest["runs"][run_id] = record
            save_manifest(manifest)
        return {"run_id": run_id, "status": "failed"}

    record["artifacts"] = artifacts_record

    if not artifacts_record:
        log(f"[{index}/{total}] {run_id}: no checkpoint (model) artifacts found")
        record["status"] = "complete"
        record["note"] = "no model artifacts on this run"
        with MANIFEST_LOCK:
            manifest["runs"][run_id] = record
            save_manifest(manifest)
        return {"run_id": run_id, "status": "complete"}

    for artifact_name, info in artifacts_record.items():
        if info.get("status") == "failed":
            log_error(
                f"Run {run_id}: artifact {artifact_name} download "
                f"failed: {info.get('error')}"
            )

    if run_staging.exists() and any(run_staging.iterdir()):
        log(f"  [SFTP] uploading checkpoints for {run_id}...")
        try:
            upload_run_checkpoints_to_sftp(run_id, run_staging)
            record["sftp"] = {"status": "uploaded", "uploaded_at": datetime.now().isoformat()}
            shutil.rmtree(CKPT_STAGING / run_id, ignore_errors=True)
            log(f"  [SFTP] ✓ uploaded + local staging cleaned for {run_id}")
        except Exception as e:
            log_error(f"Run {run_id}: SFTP upload failed: {e}")
            record["sftp"] = {"status": "failed", "error": str(e)}
            any_failed = True

    record["status"] = "incomplete" if any_failed else "complete"
    record["completed_at"] = datetime.now().isoformat()

    with MANIFEST_LOCK:
        manifest["runs"][run_id] = record
        save_manifest(manifest)

    return {"run_id": run_id, "status": record["status"]}


def check_run_has_checkpoints(run_id, index, total, manifest):
    """Lightweight path for non-targeted runs: check via the API whether
    the run has any checkpoint ("model") artifacts and record that fact.
    Never downloads the artifact files and never uploads to SFTP."""
    with MANIFEST_LOCK:
        existing = manifest["runs"].get(run_id)
    if existing and existing.get("status") == "complete":
        return {"run_id": run_id, "status": "complete", "skipped": True}

    log(f"[{index}/{total}] {run_id}: checking (not in target list, no download/upload)")

    record = {
        "run_id": run_id,
        "target": False,
        "status": "in_progress",
        "last_attempt": datetime.now().isoformat(),
    }

    try:
        api = wandb.Api()
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
    except Exception as e:
        log_error(f"Run {run_id}: not found / could not fetch run: {e}")
        record["status"] = "failed"
        record["error"] = f"run fetch failed: {e}"
        with MANIFEST_LOCK:
            manifest["runs"][run_id] = record
            save_manifest(manifest)
        return {"run_id": run_id, "status": "failed"}

    try:
        model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    except Exception as e:
        log_error(f"Run {run_id}: could not list artifacts: {e}")
        record["status"] = "failed"
        record["error"] = f"artifact list failed: {e}"
        with MANIFEST_LOCK:
            manifest["runs"][run_id] = record
            save_manifest(manifest)
        return {"run_id": run_id, "status": "failed"}

    record["has_ckpt_artifacts"] = bool(model_artifacts)
    record["artifact_names"] = [a.name for a in model_artifacts]
    record["sftp"] = {"status": "skipped_not_in_target_list"}
    record["status"] = "complete"
    record["completed_at"] = datetime.now().isoformat()

    with MANIFEST_LOCK:
        manifest["runs"][run_id] = record
        save_manifest(manifest)

    return {"run_id": run_id, "status": "complete"}


def load_target_run_ids():
    seen = set()
    ordered = []
    for line in RUN_IDS_FILE.read_text().splitlines():
        rid = line.strip()
        if rid and rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


def fetch_all_run_ids():
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    return [r.id for r in runs]


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 70)
    print("W&B CHECKPOINT BACKUP (all runs, target list prioritized)")
    print("=" * 70)

    target_ids = load_target_run_ids()
    target_set = set(target_ids)
    print(f"\nTarget run IDs (full download + SFTP upload): {len(target_ids)} (from {RUN_IDS_FILE})")

    BACKUP_BASE.mkdir(parents=True, exist_ok=True)
    PROJECT_FOLDER.mkdir(parents=True, exist_ok=True)
    CKPT_STAGING.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    print(f"Previously tracked (this ckpt job): {len(manifest.get('runs', {}))}")

    print("\nResolving SFTP destination folder...")
    resolve_sftp_project_folder()
    print(f"✓ SFTP destination ready: {_sftp_project_remote_dir}/<run_id>/checkpoints")

    print("\nFetching full run list from W&B project (this can take a while for large projects)...")
    all_ids = fetch_all_run_ids()
    other_ids = [rid for rid in all_ids if rid not in target_set]
    ordered_ids = target_ids + other_ids
    total = len(ordered_ids)
    print(f"✓ {len(all_ids)} total runs in project")
    print(
        f"Processing order: {len(target_ids)} target runs first (download + SFTP upload), "
        f"then {len(other_ids)} other runs (checkpoint-existence check only - "
        f"NEVER downloaded or uploaded to SFTP)"
    )

    print(f"\nUsing {MAX_WORKERS} concurrent workers.\n")

    completed = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {}
        for i, rid in enumerate(ordered_ids, start=1):
            if rid in target_set:
                future = executor.submit(backup_run_checkpoints, rid, i, total, manifest)
            else:
                future = executor.submit(check_run_has_checkpoints, rid, i, total, manifest)
            future_to_id[future] = rid

        for future in as_completed(future_to_id):
            rid = future_to_id[future]
            try:
                result = future.result()
                if result["status"] == "complete":
                    completed.append(rid)
                else:
                    failed.append(rid)
            except Exception as e:
                failed.append(rid)
                log_error(f"Run {rid}: unexpected error: {e}")
                traceback.print_exc()

    target_failed = [rid for rid in failed if rid in target_set]
    other_failed = [rid for rid in failed if rid not in target_set]

    with MANIFEST_LOCK:
        other_with_ckpts = sum(
            1
            for rid, rec in manifest["runs"].items()
            if rid not in target_set and rec.get("has_ckpt_artifacts")
        )

    print("\n" + "=" * 70)
    print("CHECKPOINT BACKUP COMPLETE")
    print("=" * 70)
    print(f"Target runs (downloaded + uploaded to SFTP): {len(target_ids) - len(target_failed)}/{len(target_ids)}")
    print(f"Target runs failed: {len(target_failed)} -> {target_failed}")
    print(f"Other runs checked (no download/upload): {len(other_ids) - len(other_failed)}/{len(other_ids)}")
    print(f"Other runs failed: {len(other_failed)}")
    print(f"Other runs with checkpoints that were correctly NOT uploaded to SFTP: {other_with_ckpts}")
    print(f"\nManifest: {MANIFEST_FILE}")
    if ERRORS:
        print(f"\nErrors this session: {len(ERRORS)} (see {LOG_FILE})")


if __name__ == "__main__":
    main()
