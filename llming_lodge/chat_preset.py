"""Pydantic models for Projects and Nudges (chat presets)."""

from typing import Literal, Optional

from pydantic import BaseModel


class PresetFile(BaseModel):
    file_id: str
    name: str
    size: int
    mime_type: str
    content: str = ""          # base64-encoded raw file bytes
    text_content: str = ""     # pre-extracted plain text (for context injection)


class Project(BaseModel):
    """A project groups conversations under shared settings."""

    id: str
    type: Literal["project", "nudge"] = "project"
    name: str
    icon: Optional[str] = None
    system_prompt: str = ""
    model: Optional[str] = None
    language: str = "auto"
    files: list[PresetFile] = []
    doc_plugins: list[str] | None = None  # enabled doc plugin types; None = all, [] = none
    created_at: str
    updated_at: str


class Nudge(Project):
    """A shareable conversation template. Inherits all Project fields."""

    type: Literal["nudge"] = "nudge"
    creator_name: str = ""
    creator_email: str = ""
    description: str = ""
    suggestions: str = ""  # newline-separated or '------'-separated suggestion texts
    capabilities: dict[str, bool | None] = {}  # per-nudge tool toggles: true=on, false=off, null=keep user setting
    uid: str = ""               # concept UUID shared between dev/live documents
    mode: str = "dev"           # "dev" or "live"
    category: str = ""          # matches a NudgeCategory.key (e.g. "general", "acme")
    sub_category: str = ""      # free-form sub-category set by the creator
    visibility: list[str] = []  # fnmatch globs, e.g. ["*@acme.com"]; empty = private
