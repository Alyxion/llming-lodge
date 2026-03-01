"""Pluggable service interfaces for llming-lodge.

Each service domain (directory, email, chat, storage, …) defines a small
abstract base class.  Host apps supply concrete implementations (O365,
Google Workspace, …) via ``ChatUserConfig`` fields.  The chat WebSocket
handlers call the methods defined here — no provider-specific code lives
in llming-lodge itself.
"""

from llming_lodge.services.directory_service import DirectoryService
from llming_lodge.services.email_service import EmailService

__all__ = ["DirectoryService", "EmailService"]
