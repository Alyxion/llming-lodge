"""Upload manager for session-isolated file storage."""
from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)

# Allowed MIME types for upload
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
}

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@dataclass
class FileAttachment:
    name: str
    path: Path
    size: int
    mime_type: str
    file_id: str
    consumed: bool = False
    text_content: Optional[str] = None


class UploadManager:
    """Per-session file upload manager with user isolation."""

    _sessions: ClassVar[dict[str, "UploadManager"]] = {}

    def __init__(self, session_id: str, user_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.files: list[FileAttachment] = []
        self._base_dir = Path(tempfile.gettempdir()) / "llming_lodge_uploads" / session_id
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls, session_id: str) -> Optional["UploadManager"]:
        return cls._sessions.get(session_id)

    @classmethod
    def create(cls, session_id: str, user_id: str) -> "UploadManager":
        mgr = cls(session_id, user_id)
        cls._sessions[session_id] = mgr
        return mgr

    async def store_file(self, filename: str, content: bytes, user_id: str) -> FileAttachment:
        if user_id != self.user_id:
            raise PermissionError("User ID mismatch")
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {len(content)} bytes (max {MAX_FILE_SIZE})")

        file_id = uuid.uuid4().hex[:12]
        # Sanitize filename: keep only the basename, no path traversal
        safe_name = Path(filename).name
        dest = self._base_dir / f"{file_id}_{safe_name}"
        dest.write_bytes(content)

        mime_type = self._guess_mime(safe_name)

        attachment = FileAttachment(
            name=safe_name,
            path=dest,
            size=len(content),
            mime_type=mime_type,
            file_id=file_id,
        )
        self.files.append(attachment)
        logger.info(f"[UPLOAD] Stored {safe_name} ({len(content)} bytes) as {file_id}")
        return attachment

    def remove_file(self, file_id: str, user_id: str) -> None:
        if user_id != self.user_id:
            raise PermissionError("User ID mismatch")
        for f in self.files:
            if f.file_id == file_id:
                if f.path.exists():
                    f.path.unlink()
                self.files.remove(f)
                logger.info(f"[UPLOAD] Removed {f.name} ({file_id})")
                return
        logger.warning(f"[UPLOAD] File not found: {file_id}")

    def get_pending(self) -> list[FileAttachment]:
        return [f for f in self.files if not f.consumed]

    def cleanup(self) -> None:
        import shutil
        if self._base_dir.exists():
            shutil.rmtree(self._base_dir, ignore_errors=True)
        self.files.clear()
        self._sessions.pop(self.session_id, None)
        logger.info(f"[UPLOAD] Cleaned up session {self.session_id}")

    @staticmethod
    def _guess_mime(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return mime_map.get(ext, "application/octet-stream")
