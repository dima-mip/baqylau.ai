"""Baqylau student client: auth token, session poll, exam, heartbeat, evidence.

All calls best-effort with short timeouts — network hiccups must never kill
the local proctoring loop. Raises nothing; returns {} / None on failure.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


class BaqylauClient:
    """Thin JSON client for the Baqylau Web panel."""

    def __init__(self, base_url: str, timeout: float = 4.0) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.token = ""
        self.name = ""

    # -- low level ------------------------------------------------------
    def _req(self, method: str, path: str, data=None, headers=None):
        try:
            body = None
            hdrs = dict(headers or {})
            if data is not None and not isinstance(data, bytes):
                body = json.dumps(data).encode()
                hdrs["Content-Type"] = "application/json"
            else:
                body = data
            if self.token:
                hdrs["X-Token"] = self.token
            req = urllib.request.Request(self.base + path, data=body, headers=hdrs,
                                         method=method)
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
            try:
                return json.loads(raw.decode())
            except Exception:
                return {}
        except Exception as exc:
            logger.debug("net %s %s failed: %s", method, path, exc)
            return None

    def _upload(self, path: str, fields: dict, file_field: str,
                filename: str, filebytes: bytes):
        """Multipart upload. Never raises; returns parsed JSON or None."""
        try:
            boundary = "----baqylau%d" % abs(hash(filename))
            buf = io.BytesIO()
            for k, v in fields.items():
                buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
            ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            buf.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
            buf.write(filebytes)
            buf.write(f"\r\n--{boundary}--\r\n".encode())
            hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
            if self.token:
                hdrs["X-Token"] = self.token
            req = urllib.request.Request(self.base + path, data=buf.getvalue(),
                                         headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as exc:
            logger.debug("upload failed: %s", exc)
            return None

    # -- API ------------------------------------------------------------
    def login(self, username: str, password: str) -> bool:
        r = self._req("POST", "/api/login", {"username": username, "password": password})
        if r and r.get("token"):
            self.token = r["token"]
            self.name = r.get("name", username)
            return True
        return False

    def session(self) -> dict:
        return self._req("GET", "/api/session") or {}

    def exam(self) -> dict:
        return self._req("GET", "/api/exam") or {}

    def submit(self, answers: list) -> dict:
        return self._req("POST", "/api/submit", {"answers": answers}) or {}

    def image(self, path: str) -> bytes:
        """Download question image bytes. Returns b'' on failure."""
        try:
            hdrs = {}
            if self.token:
                hdrs["X-Token"] = self.token
            req = urllib.request.Request(self.base + path, headers=hdrs, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = r.read()
            return data if len(data) <= 4 * 1024 * 1024 else b""
        except Exception as exc:
            logger.debug("image download failed: %s", exc)
            return b""

    def heartbeat(self, risk: float, level: str, violations: list, thumb_jpg: bytes) -> bool:
        try:
            b64 = base64.b64encode(thumb_jpg).decode() if thumb_jpg else ""
        except Exception:
            b64 = ""
        r = self._req("POST", "/api/heartbeat",
                      {"risk": risk, "level": level, "violations": violations, "thumb": b64})
        return bool(r and r.get("ok"))

    def evidence(self, filebytes: bytes, filename: str, kind: str,
                 risk: float, session_id: int = 0) -> dict:
        return self._upload("/api/evidence",
                            {"kind": kind, "risk": risk, "session_id": session_id},
                            "file", filename, filebytes) or {}
