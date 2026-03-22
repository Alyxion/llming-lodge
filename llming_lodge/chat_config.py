"""Pydantic config models for the llming-lodge chat page.

ChatAppConfig — app-level settings shared across all users.
ChatUserConfig — per-user runtime settings (MCPs, budget, identity).
QuickAction — a suggestion card shown on the empty-chat screen.
ThemeConfig — pre-computed CSS accent variables.
ChatFrontendConfig — the JSON object injected as ``window.__CHAT_CONFIG__``.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from llming_lodge.constants import DEFAULT_LANGUAGES as _DEFAULT_LANGUAGES


# ── Permission constants ────────────────────────────────────────────
# Pass these in ChatUserConfig.permissions to grant capabilities.
PERM_NUDGE_ADMIN = "nudge_admin"      # edit/publish/unpublish any nudge, see master nudges, transfer ownership
PERM_DEV_TOOLS = "dev_tools"          # /dev commands, shift+click gear, API keys dialog, prompt inspector
PERM_HUB_ADMIN = "hub_admin"          # admin link in hub header


class QuickAction(BaseModel):
    """A suggestion card shown on the empty-chat welcome screen."""
    id: str
    label: str
    description: str = Field(default="", serialization_alias="desc")
    icon: str = "lightbulb"
    engagement: str = ""
    prompt: str = ""
    text_prefix: str = Field(default="", serialization_alias="textPrefix")
    callback: bool = False


class ThemeConfig(BaseModel):
    """Pre-computed CSS accent variables passed to the JS frontend."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    accent: str
    accent_rgb: str
    accent_hover: str
    accent_light: str

    @classmethod
    def from_hex(cls, hex_color: str) -> "ThemeConfig":
        """Compute all CSS accent values from a single hex color."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        hr = max(0, int(r * 0.75))
        hg = max(0, int(g * 0.75))
        hb = max(0, int(b * 0.75))
        return cls(
            accent=hex_color,
            accent_rgb=f"{r}, {g}, {b}",
            accent_hover=f"#{hr:02x}{hg:02x}{hb:02x}",
            accent_light=f"rgba({r}, {g}, {b}, 0.12)",
        )


class NudgeCategory(BaseModel):
    """A category tab shown in the Nudge Explorer."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    key: str              # e.g. "acme" — must match category keys in nudge presets
    label: str            # display label
    icon: str = ""        # optional icon path relative to static_base
    large_icon: str = ""  # optional large banner/logo shown on the explorer page
    priority: int = 0     # lower = further left in tab bar


class VisibilityGroup(BaseModel):
    """A selectable group for nudge visibility (e.g. a subsidiary)."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    label: str            # e.g. "ACME Corp (HQ)"
    pattern: str          # glob pattern, e.g. "*@acme.com"
    location: str = ""    # e.g. "Berlin, Germany"


class NudgeFile(BaseModel):
    """A file attached to a nudge."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    name: str = ""
    content: str = ""  # base64 data URI
    mime_type: str = ""
    text_content: str = ""


class NudgeModel(BaseModel):
    """Full nudge document as stored in MongoDB."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    uid: str = ""
    mode: str = "dev"  # "dev" | "live"
    name: str = ""
    description: str = ""
    icon: str = ""
    category: str = ""
    sub_category: str = ""
    system_prompt: str = ""
    model: str = ""
    language: str = "auto"
    creator_email: str = ""
    creator_name: str = ""
    team_id: str | None = None
    visibility: list[str] = Field(default_factory=list)
    suggestions: list[Any] = Field(default_factory=list)
    capabilities: list[Any] = Field(default_factory=list)
    files: list[NudgeFile] = Field(default_factory=list)
    doc_plugins: list[str] | None = None  # enabled doc plugin types; None = all, [] = none
    translations: dict[str, Any] = Field(default_factory=dict)  # {"de-de": {"name": "...", "description": "...", "suggestions": [...]}}
    updated_at: str = ""
    created_at: str = ""


class PlaceholderDef(BaseModel):
    """Definition of a single placeholder within a slide layout."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    name: str           # "title", "subtitle", "body", "content", "content_left", "picture"
    ph_idx: int         # python-pptx placeholder index (0, 1, 10, 11, 12, 16, 17...)
    x: float = 0.0      # position in inches
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    accepts: list[str] = Field(default_factory=lambda: ["text"])  # "text", "list", "table", "chart", "image"


class LayoutDef(BaseModel):
    """Definition of a slide layout within a presentation template."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    name: str              # "title", "text", "two_columns", "end"
    label: str             # "Title", "Text", "2-Columns", "End"
    layout_index: int      # index into prs.slide_layouts[]
    placeholders: list[PlaceholderDef] = Field(default_factory=list)
    bg_image: str = ""     # pre-rendered background PNG URL for preview
    is_title: bool = False
    is_end: bool = False


class PresentationTemplate(BaseModel):
    """A branded presentation template (colors, fonts, logo) for slide decks."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    name: str                          # unique key, e.g. "corporate"
    label: str                         # display name, e.g. "Corporate Template"
    accent_color: str = ""             # hex, e.g. "#003366" — overrides --chat-accent
    title_color: str = ""              # title text color
    text_color: str = ""               # body text color
    subtitle_color: str = ""           # subtitle/author color
    title_bg: str = ""                 # CSS background for title slides
    content_bg: str = ""               # CSS background for content slides
    heading_font: str = ""             # font family for headings
    body_font: str = ""                # font family for body text
    logo_url: str = ""                 # URL/path to logo image
    logo_position: str = "bottom-right"
    template_path: str = ""            # server-side path to .pptx template file
    slide_width: float = 10.0          # inches
    slide_height: float = 5.625        # inches
    layouts: list[LayoutDef] = Field(default_factory=list)


class ChatAppConfig(BaseModel):
    """App-level config — shared across all users."""
    accent_color: str = ""
    app_logo: str = ""
    app_logo_link: str = ""
    app_title: str = ""
    app_mascot: str = ""
    app_mascot_incognito: str = ""  # mascot image shown in incognito mode (optional)
    show_budget: bool = False
    nudge_section_icon: str = "icons/phosphor/regular/drop.svg"
    nudge_categories: list[NudgeCategory] = Field(default_factory=list)
    nudge_mongo_uri: str = ""
    nudge_mongo_db: str = ""
    visibility_groups: list[VisibilityGroup] = Field(default_factory=list)
    default_system_prompt: str = ""  # default system prompt for all chats (replaces "You are a helpful assistant")
    nudge_base_system_prompt: str = ""  # prepended to every nudge's system_prompt (anti-hallucination, rules, etc.)
    tts_voice: str = "nova"  # default TTS voice id (e.g. "cedar", "nova"). Empty = use speech_service defaults.
    tts_model: str = ""  # TTS model override (e.g. "gpt-4o-mini-tts"). Empty = use speech_service default.
    speech_max_tokens: int = 2000  # max_output_tokens when speech response is enabled
    enable_voice_input: bool = True  # enable voice-to-text input mode (mic → transcribe → send)
    enable_live_voice: bool = True  # enable live voice mode (Realtime API / WebRTC)
    max_file_size: int = 20 * 1024 * 1024  # max size per uploaded file (bytes)
    max_session_size: int = 10 * 1024 * 1024  # max total file size per conversation (bytes)
    supported_languages: list[dict] = Field(default_factory=lambda: list(_DEFAULT_LANGUAGES))  # override to customize
    system_nudges: list[str] | None = None  # enabled system nudge keys (None = all registered, [] = none)
    bolt_label: str = "Bolts"  # UI label for bolts (host app can rename, e.g. "Jets")


class ChatUserConfig(BaseModel):
    """Per-user config — includes runtime objects (MCPs, budget)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str
    user_name: str
    full_name: str = ""
    user_email: str = ""
    user_avatar: str = ""
    budget_limits: list[Any] = Field(default_factory=list)
    budget_handler: Any = None
    context_preamble: str = ""
    mcp_servers: list[Any] = Field(default_factory=list)
    initial_model: str | None = None
    quick_actions: list[QuickAction] | None = None
    locale: str = "en-us"
    app_title: str = ""  # override ChatAppConfig.app_title with translated value
    fake_time: str = ""  # HH:MM override for testing time-dependent greetings
    on_language_change: Any = None  # async callback(lang) → {quick_actions, context_preamble}
    on_action_callback: Any = None  # async callback(action_id, text) → {notification?, ...}
    tool_toggle_notifications: dict[str, str] = Field(default_factory=dict)  # tool_name → message shown when enabled
    banner_html: str = ""  # optional HTML banner shown below the chat input
    translation_overrides: dict[str, Any] = Field(default_factory=dict)  # key → value overrides merged on top of locale translations
    user_teams: list[dict] = Field(default_factory=list)  # [{team_id, name, icon, role}, ...] for nudge ACL
    doc_plugins: list[str] | None = None  # enabled doc plugin types; None = all, [] = none
    enforced_theme: str = ""  # if set, forces this theme (e.g. "joche") and ignores localStorage
    permissions: set[str] = Field(default_factory=set)  # Set of PERM_* constants (e.g. {PERM_NUDGE_ADMIN, PERM_DEV_TOOLS})
    presentation_templates: list[PresentationTemplate] = Field(default_factory=list)
    directory_service: Any = None  # DirectoryService instance (people search)
    email_service: Any = None  # EmailService instance (draft / send)
    email_signature: str = ""  # HTML email signature to insert via toolbar
    tool_requires_providers: list[str] | None = None  # If set, restrict all MCP tools to these providers (respects PROVIDER_COMPAT)
    bg_logo_svg: str = ""  # optional inline SVG content for brand watermark in background (rendered inside the bg SVG viewBox 1200x800)
    prefetched_master_nudges: list[dict] | None = None  # pre-fetched master nudges (skip MongoDB in render)
    prefetched_discoverable_nudges: list[dict] | None = None  # pre-fetched discoverable nudges (skip MongoDB in render)
    budget_limits_for_user: Any = None  # (email: str) -> list[BudgetLimit] — create budget limits for another user (admin /dev budget command)
    on_message_intercept: Any = None  # async (text, controller) -> str|None — intercept before LLM
    on_new_chat: Any = None  # async (controller) -> None — cleanup on new chat
    app_extensions: list[Any] = Field(default_factory=list)  # list of AppExtension instances (lazy-loaded on demand)


class TeamInfo(BaseModel):
    """A team the user belongs to, for the frontend owner selector."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    team_id: str
    name: str
    icon: str = ""
    role: str  # "owner" | "editor" | "member"


class ChatFrontendConfig(BaseModel):
    """JSON config injected as ``window.__CHAT_CONFIG__``."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    session_id: str
    ws_path: str
    user_name: str
    full_name: str = ""
    user_email: str
    user_id: str
    user_avatar: str
    static_base: str = "/llming-static"
    show_budget: bool = False
    app_logo: str = ""
    app_logo_link: str = ""
    app_title: str = ""
    app_mascot: str = ""
    app_mascot_incognito: str = ""
    theme: ThemeConfig | None = None
    locale: str = "en-us"
    fake_time: str = ""
    nudge_section_icon: str = "icons/phosphor/regular/drop.svg"
    banner_html: str = ""
    speech_max_tokens: int = 2000
    enable_voice_input: bool = True
    enable_live_voice: bool = True
    translations: dict[str, Any] = Field(default_factory=dict)
    supported_languages: list[dict] = Field(default_factory=list)  # [{code, label, flag}, ...] for i18n flag bar
    nudge_categories: list[dict] = Field(default_factory=list)
    visibility_groups: list[dict] = Field(default_factory=list)
    teams: list[dict] = Field(default_factory=list)
    doc_plugins: list[str] = Field(default_factory=list)  # enabled doc plugin types for frontend
    presentation_templates: list[dict] = Field(default_factory=list)
    enforced_theme: str = ""  # if set, forces this theme and ignores localStorage
    permissions: list[str] = Field(default_factory=list)  # PERM_* constants for frontend capability checks
    bg_logo_svg: str = ""  # optional inline SVG content for brand watermark in bg
    email_signature: str = ""  # HTML email signature for toolbar insertion
    bolt_label: str = "Bolts"  # UI label for bolts (host app can rename, e.g. "Jets")
    app_extensions: list[dict] = Field(default_factory=list)  # [{name, label, icon, scriptUrl}] — available extensions (loaded on demand)
