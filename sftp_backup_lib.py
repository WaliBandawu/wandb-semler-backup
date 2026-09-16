# ============================================================
# SEMLER SFTP BACKUP DESTINATION
#
# Shared helpers used by wandb_backup_parallel.py and
# wandb_ckpt_backup.py to upload backup files to the Semler SFTP
# server, replacing the old Google Drive destination.
#
# Credentials come from environment variables (SEMLER_SFTP_HOST,
# SEMLER_SFTP_PORT, SEMLER_SFTP_USERNAME, SEMLER_SFTP_PASSWORD - see
# .env.example), loaded from a local .env file if one exists. Falls
# back to the JSON file at SFTP_CREDENTIALS_PATH for the existing
# server deployment, which already has that file in place. Neither
# form is ever committed to this repo.
# ============================================================

import hashlib
import json
import os
import posixpath
import stat as _stat
import threading
import time
from pathlib import Path

import paramiko
import pyotp

SFTP_CREDENTIALS_PATH = str(Path.home() / ".semler_sftp_credentials.json")


def _load_dotenv(path=".env"):
    """Minimal .env loader: KEY=VALUE per line, '#' comments, blank
    lines ignored. Doesn't overwrite variables already set in the
    real environment - explicit `export`s always win over the file."""
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            os.environ.setdefault(key, value)


_load_dotenv()

def resolve_backup_base():
    """Local staging directory the backup scripts read/write to.

    BACKUP_BASE_DIR wins when set. Otherwise defaults to a
    wandb_backups/ folder next to this file, so it works on whatever
    machine the script is checked out on without editing code."""
    override = os.environ.get("BACKUP_BASE_DIR")
    if override:
        return Path(override).expanduser()

    return Path(__file__).resolve().parent / "wandb_backups"


# Base remote directory backups are written under. Relative (no
# leading slash) so it resolves against whatever home/default
# directory this SFTP account lands in.
#
# Confirmed layout on files.semlerscientific.net: backups live
# under HD_Data/asset_archive (sibling to HD_Data/clinical_trial_tests).
SFTP_REMOTE_ROOT = "HD_Data/asset_archive"

_thread_local = threading.local()

# One shared cache of "this remote directory is known to exist" so
# concurrent workers don't all redundantly stat/mkdir the same
# shared parent directories (entity/project level).
_known_dirs = set()
_known_dirs_lock = threading.Lock()


def _load_credentials():
    host = os.environ.get("SEMLER_SFTP_HOST")
    username = os.environ.get("SEMLER_SFTP_USERNAME")
    password = os.environ.get("SEMLER_SFTP_PASSWORD")

    if host and username and password:
        return {
            "host": host,
            "port": int(os.environ.get("SEMLER_SFTP_PORT", "22")),
            "username": username,
            "password": password,
            # The permanent base32 seed behind the account's TOTP 2FA
            # (not a rotating 6-digit code) - lets every connection
            # generate its own valid code with no human involved.
            "totp_secret": os.environ.get("SEMLER_SFTP_TOTP_SECRET"),
        }

    try:
        with open(SFTP_CREDENTIALS_PATH) as f:
            creds = json.load(f)
            creds.setdefault("totp_secret", None)
            return creds
    except FileNotFoundError:
        raise FileNotFoundError(
            "No SFTP credentials found. Set SEMLER_SFTP_HOST/"
            "SEMLER_SFTP_USERNAME/SEMLER_SFTP_PASSWORD (e.g. via a "
            f".env file - see .env.example) or provide {SFTP_CREDENTIALS_PATH}."
        )


def _connect():
    creds = _load_credentials()
    transport = paramiko.Transport((creds["host"], creds.get("port", 22)))
    transport.banner_timeout = 30
    transport.start_client()

    # Plain password auth only gets partial success on this server -
    # it then demands a TOTP code via keyboard-interactive before a
    # channel (e.g. SFTP) can be opened. Drive that second factor
    # ourselves instead of leaving connect() to silently half-auth.
    try:
        remaining = transport.auth_password(
            creds["username"], creds["password"]
        )
    except paramiko.AuthenticationException as e:
        transport.close()
        raise paramiko.AuthenticationException(
            f"SFTP password rejected for {creds['username']}@{creds['host']}: {e}"
        )

    if not transport.is_authenticated() and "keyboard-interactive" in remaining:
        totp_secret = creds.get("totp_secret")

        if not totp_secret:
            transport.close()
            raise RuntimeError(
                f"SFTP account {creds['username']}@{creds['host']} requires "
                "a TOTP code (Two Factor Authentication) after the password, "
                "but no SEMLER_SFTP_TOTP_SECRET is configured. Set it in "
                ".env to the account's permanent TOTP seed (the base32 "
                "string shown when 2FA was enrolled - not a rotating "
                "6-digit code) so every connection can generate its own "
                "valid code automatically."
            )

        totp = pyotp.TOTP(totp_secret)

        def _totp_handler(title, instructions, prompt_list):
            return [totp.now() for _ in prompt_list]

        transport.auth_interactive(creds["username"], _totp_handler)

    if not transport.is_authenticated():
        transport.close()
        raise paramiko.AuthenticationException(
            f"SFTP authentication did not complete for "
            f"{creds['username']}@{creds['host']}"
        )

    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get_channel().settimeout(120)
    return transport, sftp


def get_sftp_client():
    """Thread-local SFTP client. Each worker thread gets its own
    connection (paramiko SFTP clients are not safe to share across
    threads). Reconnects automatically if the connection has died."""
    client = getattr(_thread_local, "sftp", None)

    if client is not None:
        try:
            client.getcwd()
            return client
        except Exception:
            try:
                _thread_local.transport.close()
            except Exception:
                pass
            _thread_local.sftp = None

    transport, sftp = _connect()
    _thread_local.transport = transport
    _thread_local.sftp = sftp
    return sftp


RETRY_BACKOFF_SECONDS_PER_ATTEMPT = 5
SFTP_RETRY_BACKOFF_CAP_SECONDS = 20


def with_sftp_retry(fn, *args, retries=3, **kwargs):
    """Run fn(sftp, *args, **kwargs), retrying once more with a fresh
    connection if the first attempt fails on a connection-shaped error."""
    last_error = None

    for attempt in range(1, retries + 1):
        sftp = get_sftp_client()

        try:
            return fn(sftp, *args, **kwargs)
        except Exception as e:
            last_error = e
            # Force a reconnect on the next get_sftp_client() call.
            try:
                _thread_local.transport.close()
            except Exception:
                pass
            _thread_local.sftp = None

            if attempt < retries:
                time.sleep(min(
                    RETRY_BACKOFF_SECONDS_PER_ATTEMPT * attempt,
                    SFTP_RETRY_BACKOFF_CAP_SECONDS
                ))

    raise last_error


def safe_name(name):
    """Sanitize a single path component for a remote filesystem that
    may be Windows-backed (illegal chars: \\ / : * ? " < > |)."""
    return "".join(
        c if c not in '\\/:*?"<>|' else "_"
        for c in name
    )


def remote_join(*parts):
    return posixpath.join(*parts)


def ensure_remote_dir(sftp, remote_path):
    """mkdir -p semantics for a remote directory, tolerant of races
    between concurrent worker threads creating the same shared parent."""
    with _known_dirs_lock:
        if remote_path in _known_dirs:
            return

    parts = [p for p in remote_path.split("/") if p]
    current = ""

    for part in parts:
        current = f"{current}/{part}" if current else part

        with _known_dirs_lock:
            if current in _known_dirs:
                continue

        try:
            sftp.stat(current)
        except FileNotFoundError:
            try:
                sftp.mkdir(current)
            except OSError:
                # Lost a race with another thread/process - fine as
                # long as it exists now.
                sftp.stat(current)

        with _known_dirs_lock:
            _known_dirs.add(current)

    with _known_dirs_lock:
        _known_dirs.add(remote_path)


def remote_file_size(sftp, remote_path):
    try:
        return sftp.stat(remote_path).st_size
    except FileNotFoundError:
        return None


def upload_file(sftp, local_path, remote_dir, remote_name=None):
    """Idempotent file upload: if a same-named file of the same size
    already exists at the destination, it's reused instead of
    re-uploaded. Returns True if a network upload actually happened."""
    remote_name = remote_name or safe_name(local_path.name)
    remote_path = remote_join(remote_dir, remote_name)

    local_size = local_path.stat().st_size
    existing_size = remote_file_size(sftp, remote_path)

    if existing_size is not None and existing_size == local_size:
        return False

    temp_remote_path = f"{remote_path}.part-{threading.get_ident()}"

    sftp.put(str(local_path), temp_remote_path)

    try:
        sftp.remove(remote_path)
    except FileNotFoundError:
        pass

    sftp.posix_rename(temp_remote_path, remote_path)
    return True


def upload_directory_tree(sftp, local_dir, remote_dir):
    """Mirror upload_directory_to_drive: create remote_dir/<local_dir.name>,
    upload every file directly inside it, THEN recurse into
    subdirectories one at a time (depth-first, files before dirs), so a
    remote folder is never left partially populated mid-crash."""
    this_remote_dir = remote_join(remote_dir, safe_name(local_dir.name))
    ensure_remote_dir(sftp, this_remote_dir)

    entries = sorted(local_dir.iterdir())

    for entry in entries:
        if entry.is_file():
            upload_file(sftp, entry, this_remote_dir)

    for entry in entries:
        if entry.is_dir():
            upload_directory_tree(sftp, entry, this_remote_dir)

    return this_remote_dir


def list_remote_tree(sftp, remote_dir, prefix=""):
    """Recursively list every file under remote_dir, keyed by its path
    relative to remote_dir -> size in bytes."""
    tree = {}

    try:
        entries = sftp.listdir_attr(remote_dir)
    except FileNotFoundError:
        return tree

    for entry in entries:
        rel_path = f"{prefix}{entry.filename}"
        full_path = remote_join(remote_dir, entry.filename)

        if _stat.S_ISDIR(entry.st_mode):
            tree.update(list_remote_tree(sftp, full_path, prefix=f"{rel_path}/"))
        else:
            tree[rel_path] = entry.st_size

    return tree


def upload_run_to_sftp(run_folder, project_remote_dir):
    """Upload one run's whole local folder tree to the SFTP server,
    depth-first and fully sequential within this call. Returns
    (success: bool, error: str or None)."""
    try:
        with_sftp_retry(upload_directory_tree, run_folder, project_remote_dir)
        return True, None
    except Exception as e:
        return False, str(e)


def upload_single_file(local_path, remote_dir, remote_name):
    """Upload one file with retry. Returns (success: bool, error: str
    or None). Generic - used for bulk-batch zips as well as small
    tracking files (manifest, registry)."""
    try:
        with_sftp_retry(upload_file, local_path, remote_dir, remote_name=remote_name)
        return True, None
    except Exception as e:
        return False, str(e)


def upload_bulk_zip(zip_path, remote_dir, remote_name):
    """Upload one bulk-batch zip (many runs bundled together) as a
    single file. Returns (success: bool, error: str or None)."""
    return upload_single_file(zip_path, remote_dir, remote_name)


def sftp_run_complete_safe(run_id, project_remote_dir, expected, log_error=None):
    """Best-effort SFTP-side verification. Any failure (connection,
    missing folder) is treated as NOT verified, so the caller falls
    back to redoing the work rather than trusting a check that
    couldn't actually run."""
    try:
        run_remote_dir = remote_join(project_remote_dir, safe_name(run_id))
        tree = with_sftp_retry(list_remote_tree, run_remote_dir)

        if not tree:
            return False

        for rel_path, size in expected.items():
            if rel_path not in tree:
                return False

            remote_size = tree[rel_path]

            if size is not None and remote_size is not None and remote_size != size:
                return False

        return True

    except Exception as e:
        if log_error:
            log_error(f"Run {run_id}: SFTP verification failed: {e}")
        return False


def resolve_remote_dir_under_root(*name_parts):
    """Resolve (and cache) <SFTP_REMOTE_ROOT>/<name_parts...>, sanitizing
    each part. Generic version of resolve_project_remote_dir below,
    for backups that aren't shaped like entity/project (e.g. a single
    named folder mirrored from Google Drive)."""
    sftp = get_sftp_client()
    remote_dir = remote_join(SFTP_REMOTE_ROOT, *(safe_name(p) for p in name_parts))
    ensure_remote_dir(sftp, remote_dir)
    return remote_dir


def resolve_project_remote_dir(entity, project):
    """Resolve (and cache) the shared <root>/entity/project directory
    chain once, before any worker threads start."""
    return resolve_remote_dir_under_root(entity, project)


def resolve_bulk_batch_remote_dir(project_remote_dir):
    """Resolve (and cache) the shared <project_remote_dir>/bulk_batches
    directory, once, before any worker threads start. Bulk (non-target)
    runs are zipped in groups and uploaded here as single archives,
    instead of as individually browsable per-run folders."""
    sftp = get_sftp_client()
    remote_dir = remote_join(project_remote_dir, "bulk_batches")
    ensure_remote_dir(sftp, remote_dir)
    return remote_dir


# ============================================================
# W&B CHECKPOINT ARTIFACT DOWNLOAD
#
# Not SFTP-specific, but shared by wandb_backup_parallel.py and
# wandb_ckpt_backup.py - the only module both already import - so
# this "download every model artifact for a run" logic lives in one
# place instead of two near-identical copies.
# ============================================================

ARTIFACT_DOWNLOAD_MAX_RETRIES = 3
ARTIFACT_DOWNLOAD_BACKOFF_SECONDS_PER_ATTEMPT = 5
ARTIFACT_DOWNLOAD_BACKOFF_CAP_SECONDS = 30


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Calculate SHA-256 for a local file - used for backup integrity
    verification."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def download_model_artifacts(run, dest_root, existing_artifacts=None):
    """Downloads every 'model'-type logged artifact for `run` into
    dest_root/<artifact-dirname>/..., retrying failed downloads and
    skipping artifacts already fully present on disk (restart-safe).
    Every downloaded file is hashed for backup integrity checks.

    Returns (artifacts_record, any_failed). Raises RuntimeError if the
    artifact list itself couldn't be fetched - the caller decides how
    that failure should be recorded (manifest shape differs between
    callers, so that's left to them).

    `existing_artifacts` (optional): a previous run's artifact record
    dict, reused as-is for artifacts found already downloaded so their
    aliases/file hashes aren't lost across restarts."""
    existing_artifacts = existing_artifacts or {}

    try:
        model_artifacts = [a for a in run.logged_artifacts() if a.type == "model"]
    except Exception as e:
        raise RuntimeError(f"could not list checkpoint artifacts: {e}") from e

    artifacts_record = {}
    any_failed = False

    for artifact in model_artifacts:
        artifact_dirname = artifact.name.replace(":", "_v")
        dest_dir = dest_root / artifact_dirname

        if dest_dir.exists() and any(p.is_file() for p in dest_dir.rglob("*")):
            print(f"  ↪ SKIP: {artifact.name} (already downloaded)")
            artifacts_record[artifact.name] = existing_artifacts.get(
                artifact.name,
                {"status": "downloaded", "aliases": artifact.aliases}
            )
            continue

        downloaded = False
        last_error = None

        for attempt in range(1, ARTIFACT_DOWNLOAD_MAX_RETRIES + 1):
            print(
                f"  ↓ Downloading checkpoint: {artifact.name}"
                f" [attempt {attempt}/{ARTIFACT_DOWNLOAD_MAX_RETRIES}]"
            )

            try:
                artifact.download(root=str(dest_dir))
                downloaded = True
                break
            except Exception as e:
                last_error = e
                print(f"  ✗ Attempt {attempt} failed: {e}")

                if attempt < ARTIFACT_DOWNLOAD_MAX_RETRIES:
                    time.sleep(min(
                        ARTIFACT_DOWNLOAD_BACKOFF_SECONDS_PER_ATTEMPT * attempt,
                        ARTIFACT_DOWNLOAD_BACKOFF_CAP_SECONDS
                    ))

        if not downloaded:
            artifacts_record[artifact.name] = {
                "status": "failed",
                "error": str(last_error)
            }
            any_failed = True
            continue

        files_info = {}
        for f in sorted(dest_dir.rglob("*")):
            if f.is_file():
                files_info[str(f.relative_to(dest_dir))] = {
                    "size": f.stat().st_size,
                    "sha256": sha256_file(f),
                }

        artifacts_record[artifact.name] = {
            "status": "downloaded",
            "aliases": artifact.aliases,
            "files": files_info,
        }

        print(f"  ✓ COMPLETE: {artifact.name}")

    return artifacts_record, any_failed
