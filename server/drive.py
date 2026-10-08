"""Optional Google Drive upload for evidence (local-first).

Activates only when env BAQYLAU_DRIVE_FOLDER_ID + credentials file
(GOOGLE_APPLICATION_CREDENTIALS or ./drive_credentials.json) exist AND
``googleapiclient`` is installed. Otherwise returns "" — evidence stays
local, which is the default for the hackathon case.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def upload_if_configured(local_path: str) -> str:
    """Upload file to Drive folder, return web link or ''."""
    try:
        folder = os.environ.get("BAQYLAU_DRIVE_FOLDER_ID", "")
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "drive_credentials.json")
        if not folder or not os.path.exists(creds) or not os.path.exists(local_path):
            return ""
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        scopes = ["https://www.googleapis.com/auth/drive.file"]
        cred = service_account.Credentials.from_service_account_file(creds, scopes=scopes)
        svc = build("drive", "v3", credentials=cred)
        meta = {"name": os.path.basename(local_path), "parents": [folder]}
        media = MediaFileUpload(local_path, resumable=False)
        f = svc.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
        return str(f.get("webViewLink", ""))
    except Exception as exc:
        logger.debug("drive upload skipped: %s", exc)
        return ""
