"""EmailDraftMCP -- in-process MCP server for composing email drafts."""

import json
import logging
from typing import Any, Dict, List

from llming_lodge.tools.mcp import InProcessMCPServer
from llming_lodge.doc_plugins.document_store import DocumentSessionStore

logger = logging.getLogger(__name__)


class EmailDraftMCP(InProcessMCPServer):
    """MCP server that lets the LLM compose and edit email draft documents.

    Email drafts are stored as documents with type ``email_draft`` and data:
    ``{subject, to, cc, bcc, body_html, attachments: [{ref, name}]}``.
    """

    def __init__(self, store: DocumentSessionStore) -> None:
        self._store = store

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "email_get_draft",
                "displayName": "Get Draft",
                "displayDescription": "Get the full email draft",
                "icon": "mail",
                "description": (
                    "Get the full email draft document. Returns subject, "
                    "recipients (to/cc/bcc), body_html, and attachments."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "email_update_subject",
                "displayName": "Update Subject",
                "displayDescription": "Update the email subject line",
                "icon": "title",
                "description": (
                    "Update the subject line of an email draft."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                        "subject": {
                            "type": "string",
                            "description": "New subject line",
                        },
                    },
                    "required": ["document_id", "subject"],
                },
            },
            {
                "name": "email_update_recipients",
                "displayName": "Update Recipients",
                "displayDescription": "Update to/cc/bcc recipients",
                "icon": "group",
                "description": (
                    "Update the recipients of an email draft. Provide one or "
                    "more of: to, cc, bcc. Each is a list of email addresses."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                        "to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Primary recipients (email addresses)",
                        },
                        "cc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "CC recipients (email addresses)",
                        },
                        "bcc": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "BCC recipients (email addresses)",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "email_update_body",
                "displayName": "Update Body",
                "displayDescription": "Update the email body (HTML)",
                "icon": "edit",
                "description": (
                    "Replace the body of an email draft with new HTML content. "
                    "Use clean, professional HTML suitable for email clients. "
                    "Avoid JavaScript. Keep styling inline."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                        "body_html": {
                            "type": "string",
                            "description": "New body HTML content",
                        },
                    },
                    "required": ["document_id", "body_html"],
                },
            },
            {
                "name": "email_add_attachment",
                "displayName": "Add Attachment",
                "displayDescription": "Attach a file or document to the draft",
                "icon": "attach_file",
                "description": (
                    "Add an attachment reference to the email draft. Use 'ref' "
                    "to reference a chat attachment by filename or a document "
                    "created in the chat by its document ID."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                        "ref": {
                            "type": "string",
                            "description": (
                                "Reference: a chat attachment filename "
                                "(e.g. 'report.pdf') or a document ID from the chat"
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": "Display name for the attachment (optional, defaults to ref)",
                        },
                    },
                    "required": ["document_id", "ref"],
                },
            },
            {
                "name": "email_remove_attachment",
                "displayName": "Remove Attachment",
                "displayDescription": "Remove an attachment from the draft",
                "icon": "link_off",
                "description": (
                    "Remove an attachment from the email draft by its ref."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document ID of the email draft",
                        },
                        "ref": {
                            "type": "string",
                            "description": "The attachment ref to remove",
                        },
                    },
                    "required": ["document_id", "ref"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        doc_id = arguments.get("document_id", "")

        if name == "email_get_draft":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            return json.dumps({
                "document_id": doc.id,
                "name": doc.name,
                "version": doc.version,
                "subject": data.get("subject", ""),
                "to": data.get("to", []),
                "cc": data.get("cc", []),
                "bcc": data.get("bcc", []),
                "body_html": data.get("body_html", ""),
                "attachments": data.get("attachments", []),
            })

        elif name == "email_update_subject":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            data["subject"] = arguments["subject"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "subject_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "email_update_recipients":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            if "to" in arguments:
                data["to"] = arguments["to"]
            if "cc" in arguments:
                data["cc"] = arguments["cc"]
            if "bcc" in arguments:
                data["bcc"] = arguments["bcc"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "recipients_updated",
                "document_id": doc.id,
                "version": updated.version,
                "to": data.get("to", []),
                "cc": data.get("cc", []),
                "bcc": data.get("bcc", []),
            })

        elif name == "email_update_body":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            data["body_html"] = arguments["body_html"]
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "body_updated",
                "document_id": doc.id,
                "version": updated.version,
            })

        elif name == "email_add_attachment":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            atts = data.get("attachments", [])
            ref = arguments["ref"]
            display_name = arguments.get("name", ref)
            # Avoid duplicates
            if any(a["ref"] == ref for a in atts):
                return json.dumps({"status": "already_attached", "ref": ref})
            atts.append({"ref": ref, "name": display_name})
            data["attachments"] = atts
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "attachment_added",
                "document_id": doc.id,
                "version": updated.version,
                "attachments": atts,
            })

        elif name == "email_remove_attachment":
            doc = self._store.get(doc_id)
            if not doc:
                return json.dumps({"error": "Document not found"})
            if doc.type != "email_draft":
                return json.dumps({"error": f"Document is type '{doc.type}', not 'email_draft'"})
            data = doc.data or {}
            atts = data.get("attachments", [])
            ref = arguments["ref"]
            new_atts = [a for a in atts if a["ref"] != ref]
            if len(new_atts) == len(atts):
                return json.dumps({"status": "not_found", "ref": ref})
            data["attachments"] = new_atts
            updated = self._store.update(doc.id, data=data)
            return json.dumps({
                "status": "attachment_removed",
                "document_id": doc.id,
                "version": updated.version,
                "attachments": new_atts,
            })

        return json.dumps({"error": f"Unknown tool: {name}"})
