# ============================================================
# SEMLER SFTP BACKUP DESTINATION
#
# Shared helpers used by wandb_backup_parallel.py and
# wandb_ckpt_backup.py to upload backup files to the Semler SFTP
# server, replacing the old Google Drive destination.
#
# Credentials are read from SFTP_CREDENTIALS_PATH (JSON file,
# chmod 600, not committed anywhere) rather than hardcoded here.
# ============================================================

import json
import posixpath
import stat as _stat
import threading
import time

import paramiko

SFTP_CREDENTIALS_PATH = "/home/ubuntu/.semler_sftp_credentials.json"

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
    with open(SFTP_CREDENTIALS_PATH) as f:
        return json.load(f)


def _connect():
    creds = _load_credentials()
    transport = paramiko.Transport((creds["host"], creds.get("port", 22)))
    transport.banner_timeout = 30
    transport.connect(username=creds["username"], password=creds["password"])
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
                time.sleep(min(5 * attempt, 20))

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


def resolve_project_remote_dir(entity, project):
    """Resolve (and cache) the shared <root>/entity/project directory
    chain once, before any worker threads start."""
    sftp = get_sftp_client()
    remote_dir = remote_join(SFTP_REMOTE_ROOT, safe_name(entity), safe_name(project))
    ensure_remote_dir(sftp, remote_dir)
    return remote_dir


def resolve_bulk_batch_remote_dir(project_remote_dir):
    """Resolve (and cache) the shared <project_remote_dir>/bulk_batches
    directory, once, before any worker threads start. Bulk (non-target)
    runs are zipped in groups and uploaded here as single archives,
    instead of as individually browsable per-run folders."""
    sftp = get_sftp_client()
    remote_dir = remote_join(project_remote_dir, "bulk_batches")
    ensure_remote_dir(sftp, remote_dir)
    return remote_dir
