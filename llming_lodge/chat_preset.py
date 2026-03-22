"""Pydantic models for Projects and Nudges (chat presets)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class PresetFile(BaseModel):
    file_id: str
    name: str
    size: int
    mime_type: str
    content: str = ""          # base64-encoded raw file bytes
    text_content: str = ""     # pre-extracted plain text (for context injection)


class BoltAction(BaseModel):
    """Action to execute when a bolt fires."""
    type: str                          # "url" | "prompt" | "mcp_tool" | "app" | "local_eval" | "worker_decode"
    url: str = ""                      # for type="url": URL with {input} placeholder
    target: str = "_blank"             # for type="url": "_blank" | "_self"
    template: str = ""                 # for type="prompt": prompt template with {input}
    auto_send: bool = True             # for type="prompt": send immediately or pre-fill
    tool_name: str = ""                # for type="mcp_tool": MCP tool name
    arg_mapping: dict[str, str] = {}   # for type="mcp_tool": param → template mapping
    app: str = ""                      # for type="app": app name in window._boltApps
    args: dict[str, Any] = {}          # for type="app": extra args passed to render
    handler: str = ""                  # for type="worker_decode": bolt handler name in Worker
    fallback_template: str = ""        # for type="worker_decode": fallback prompt if Worker fails
    followup_template: str = ""        # for type="worker_decode": LLM prompt after decode card


class BoltDef(BaseModel):
    """A bolt (fast command) defined within a nudge."""
    command: str                       # e.g. "product"
    aliases: list[str] = []            # e.g. ["produkt"] (i18n)
    label: str = ""                    # display name, e.g. "Product Lookup"
    icon: str = "bolt"                 # Material icon name
    description: str = ""
    description_i18n: dict[str, str] = {}  # {"de": "Produkt nachschlagen"}
    devices: list[str] = ["desktop", "tablet", "mobile"]
    regex: str | None = None           # auto-detect pattern
    match_anywhere: bool = False       # if True, regex can match anywhere in input (not just start)
    counter_check: str | None = None   # JS expression for 2nd-pass validation
    action: BoltAction = BoltAction(type="prompt")


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
    capabilities: dict[str, bool | str | None] = {}  # per-nudge tool toggles + server_mcp class path
    uid: str = ""               # concept UUID shared between dev/live documents
    mode: str = "dev"           # "dev" or "live"
    category: str = ""          # matches a NudgeCategory.key (e.g. "general", "acme")
    sub_category: str = ""      # free-form sub-category set by the creator
    visibility: list[str] = []  # fnmatch globs, e.g. ["*@acme.com"]; empty = private
    version: str = "0.0.1"     # semver-ish version, auto-incremented on each save
    translations: dict = {}    # {"de-de": {"name": "...", "description": "...", "suggestions": "..."}} per-locale overrides
    bolts: list[BoltDef] = []  # fast commands defined by this nudge
