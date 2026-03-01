"""Abstract email service — draft, update, send.

Each method that creates or sends a message returns the provider's
message ID so the frontend can track it and issue follow-up actions
(update draft, send existing draft, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmailService(ABC):

    @abstractmethod
    async def save_draft(
        self,
        subject: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        body_html: str,
        **kw,
    ) -> dict:
        """Create a new draft.

        Returns ``{"ok": True, "message_id": "..."}``
        or ``{"ok": False, "error": "..."}``.
        """

    @abstractmethod
    async def update_draft(
        self,
        message_id: str,
        subject: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        body_html: str,
        **kw,
    ) -> dict:
        """Update an existing draft by *message_id*.

        Returns ``{"ok": True, "message_id": "..."}``
        or ``{"ok": False, "error": "..."}``.
        """

    @abstractmethod
    async def send_email(
        self,
        subject: str,
        to: list[str],
        cc: list[str],
        bcc: list[str],
        body_html: str,
        draft_id: Optional[str] = None,
        **kw,
    ) -> dict:
        """Send an email.

        If *draft_id* is given, send the existing draft instead of
        composing a new message.

        Returns ``{"ok": True, "message_id": "..."}``
        or ``{"ok": False, "error": "..."}``.
        """
