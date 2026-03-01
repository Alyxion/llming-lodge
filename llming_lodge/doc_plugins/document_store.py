"""Document model and session-scoped document store."""

import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document created within a chat conversation."""
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    conversation_id: str = ""
    type: str = ""  # plotly, latex, table, text_doc, presentation, html, email_draft
    name: str = ""
    data: Any = None  # JSON-serializable document data
    version: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class DocumentSessionStore:
    """Thread-safe in-memory document store scoped to a session.

    Documents live for the duration of the chat session and are
    synced to the frontend via WebSocket for IDB persistence.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._docs: Dict[str, Document] = {}
        self._notify_callback: Optional[Any] = None

    def set_notify_callback(self, callback) -> None:
        """Set a callback(event_type, document) for WebSocket notifications."""
        self._notify_callback = callback

    def _notify(self, event_type: str, doc: Document) -> None:
        if self._notify_callback:
            try:
                self._notify_callback(event_type, doc)
            except Exception:
                pass

    def create(self, type: str, name: str, data: Any,
               conversation_id: str = "") -> Document:
        doc = Document(
            type=type,
            name=name,
            data=data,
            conversation_id=conversation_id,
        )
        with self._lock:
            self._docs[doc.id] = doc
        self._notify("doc_created", doc)
        return doc

    def get(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._docs.get(doc_id)

    def list_all(self) -> List[Document]:
        with self._lock:
            return list(self._docs.values())

    def list_by_type(self, doc_type: str) -> List[Document]:
        with self._lock:
            return [d for d in self._docs.values() if d.type == doc_type]

    def update(self, doc_id: str, data: Any = None,
               name: Optional[str] = None) -> Optional[Document]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if not doc:
                return None
            if data is not None:
                doc.data = data
            if name is not None:
                doc.name = name
            doc.version += 1
            doc.updated_at = time.time()
        self._notify("doc_updated", doc)
        return doc

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            doc = self._docs.pop(doc_id, None)
        if doc:
            self._notify("doc_deleted", doc)
            return True
        return False

    # Backward compat: old type names → new
    _TYPE_ALIASES = {"word": "text_doc", "powerpoint": "presentation"}

    def restore_from_list(self, docs_data: List[dict]) -> None:
        """Restore documents from frontend IDB on reconnect."""
        with self._lock:
            for d in docs_data:
                doc = Document(**d)
                doc.type = self._TYPE_ALIASES.get(doc.type, doc.type)
                self._docs[doc.id] = doc
