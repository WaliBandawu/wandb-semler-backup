# ============================================================
# GOOGLE DRIVE ACCESS
#
# Shared OAuth + API client helper for scripts that read from Google
# Drive (migrate_drive_to_sftp.py, drive_sync.py, drive_backup_to_sftp.py).
#
# Credentials come from a one-time OAuth exchange (see
# exchange_drive_token.py) saved locally as DRIVE_CLIENT_CFG_PATH /
# DRIVE_TOKEN_PATH - never committed to this repo.
# ============================================================

import json
import threading
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build as build_drive_service

DRIVE_CLIENT_CFG_PATH = str(Path.home() / ".gdrive_oauth_client.json")
DRIVE_TOKEN_PATH = str(Path.home() / ".gdrive_token.json")

_thread_local = threading.local()


def get_drive_service():
    """Thread-local Drive API client, built from locally-stored OAuth
    credentials. Each thread builds and caches its own client on first
    use (the underlying HTTP client isn't guaranteed thread-safe)."""
    if not hasattr(_thread_local, "service"):
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

        _thread_local.service = build_drive_service(
            "drive", "v3", credentials=creds, cache_discovery=False
        )

    return _thread_local.service
