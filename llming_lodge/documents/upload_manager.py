"""Upload manager for session-isolated in-memory file storage."""
from __future__ import annotations

import logging
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

# Defaults — overridden at startup via UploadManager.configure() from ChatAppConfig
_DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB per single file
_DEFAULT_MAX_SESSION_SIZE = 10 * 1024 * 1024  # 10 MB total per conversation


@dataclass
class FileAttachment:
    name: str
    size: int
    mime_type: str
    file_id: str
    consumed: bool = False
    text_content: Optional[str] = None
    raw_data: Optional[bytes] = None


class UploadManager:
    """Per-session file upload manager — all data stays in memory."""

    _sessions: ClassVar[dict[str, "UploadManager"]] = {}
    max_file_size: ClassVar[int] = _DEFAULT_MAX_FILE_SIZE
    max_session_size: ClassVar[int] = _DEFAULT_MAX_SESSION_SIZE

    def __init__(self, session_id: str, user_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.files: list[FileAttachment] = []

    @classmethod
    def configure(cls, *, max_file_size: int | None = None, max_session_size: int | None = None) -> None:
        """Set class-level upload limits (call once at startup from ChatAppConfig)."""
        if max_file_size is not None:
            cls.max_file_size = max_file_size
        if max_session_size is not None:
            cls.max_session_size = max_session_size
        logger.info(
            "[UPLOAD] Configured limits: max_file=%d MB, max_session=%d MB",
            cls.max_file_size // (1024 * 1024),
            cls.max_session_size // (1024 * 1024),
        )

    @classmethod
    def get(cls, session_id: str) -> Optional["UploadManager"]:
        return cls._sessions.get(session_id)

    @classmethod
    def create(cls, session_id: str, user_id: str) -> "UploadManager":
        mgr = cls(session_id, user_id)
        cls._sessions[session_id] = mgr
        return mgr

    @property
    def total_size(self) -> int:
        """Total bytes of all files currently stored in this session."""
        return sum(f.size for f in self.files)

    async def store_file(self, filename: str, content: bytes, user_id: str) -> FileAttachment:
        if user_id != self.user_id:
            raise PermissionError("User ID mismatch")
        if len(content) > self.max_file_size:
            raise ValueError(f"File too large: {len(content)} bytes (max {self.max_file_size})")
        if self.total_size + len(content) > self.max_session_size:
            used_mb = self.total_size / (1024 * 1024)
            raise ValueError(
                f"Conversation file limit exceeded: {used_mb:.1f} MB used, "
                f"adding {len(content) / (1024 * 1024):.1f} MB would exceed "
                f"{self.max_session_size / (1024 * 1024):.0f} MB limit"
            )

        file_id = uuid.uuid4().hex[:12]
        safe_name = Path(filename).name

        mime_type = self._guess_mime(safe_name)

        attachment = FileAttachment(
            name=safe_name,
            size=len(content),
            mime_type=mime_type,
            file_id=file_id,
            raw_data=content,
        )
        self.files.append(attachment)
        logger.info(f"[UPLOAD] Stored {safe_name} ({len(content)} bytes) as {file_id}")
        return attachment

    def remove_file(self, file_id: str, user_id: str) -> None:
        if user_id != self.user_id:
            raise PermissionError("User ID mismatch")
        for f in self.files:
            if f.file_id == file_id:
                self.files.remove(f)
                logger.info(f"[UPLOAD] Removed {f.name} ({file_id})")
                return
        logger.warning(f"[UPLOAD] File not found: {file_id}")

    def get_pending(self) -> list[FileAttachment]:
        return [f for f in self.files if not f.consumed]

    def cleanup(self) -> None:
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
