"""Per-language writing style definitions for AI editing prompts."""

WRITING_STYLES: dict[str, dict[str, str]] = {
    "de": {
        "formal_greeting": "Sehr geehrte Damen und Herren,",
        "informal_greeting": "Hallo,",
        "formal_closing": "Mit freundlichen Grüßen",
        "informal_closing": "Viele Grüße",
        "style_notes": "German business writing is direct but polite. Use Sie (formal you). Avoid anglicisms when a German word exists.",
    },
    "en": {
        "formal_greeting": "Dear Sir or Madam,",
        "informal_greeting": "Hi,",
        "formal_closing": "Kind regards",
        "informal_closing": "Best",
        "style_notes": "British/American business English. Prefer active voice, concise sentences.",
    },
    "fr": {
        "formal_greeting": "Madame, Monsieur,",
        "informal_greeting": "Bonjour,",
        "formal_closing": "Veuillez agréer l'expression de mes salutations distinguées",
        "informal_closing": "Cordialement",
        "style_notes": "French business correspondence uses vous (formal). Closings are typically elaborate in formal contexts.",
    },
    "it": {
        "formal_greeting": "Gentilissimo/a,",
        "informal_greeting": "Ciao,",
        "formal_closing": "Cordiali saluti",
        "informal_closing": "A presto",
        "style_notes": "Italian business style uses Lei (formal you). Polite and warm tone.",
    },
    "es": {
        "formal_greeting": "Estimado/a señor/a,",
        "informal_greeting": "Hola,",
        "formal_closing": "Atentamente",
        "informal_closing": "Un saludo",
        "style_notes": "Spanish business writing uses usted (formal). Warm but respectful.",
    },
    "pt": {
        "formal_greeting": "Prezado(a) Senhor(a),",
        "informal_greeting": "Olá,",
        "formal_closing": "Atenciosamente",
        "informal_closing": "Abraços",
        "style_notes": "Portuguese business writing. Use o senhor/a senhora in formal contexts.",
    },
    "nl": {
        "formal_greeting": "Geachte heer/mevrouw,",
        "informal_greeting": "Hallo,",
        "formal_closing": "Met vriendelijke groet",
        "informal_closing": "Groetjes",
        "style_notes": "Dutch business style is direct and pragmatic. Use u (formal you).",
    },
    "pl": {
        "formal_greeting": "Szanowni Państwo,",
        "informal_greeting": "Cześć,",
        "formal_closing": "Z poważaniem",
        "informal_closing": "Pozdrawiam",
        "style_notes": "Polish business correspondence uses Pan/Pani (formal). Polite and structured.",
    },
    "cs": {
        "formal_greeting": "Vážená paní / Vážený pane,",
        "informal_greeting": "Dobrý den,",
        "formal_closing": "S pozdravem",
        "informal_closing": "Zdravím",
        "style_notes": "Czech business writing uses vy (formal plural). Polite but concise.",
    },
    "tr": {
        "formal_greeting": "Sayın Yetkili,",
        "informal_greeting": "Merhaba,",
        "formal_closing": "Saygılarımla",
        "informal_closing": "İyi günler",
        "style_notes": "Turkish business style uses siz (formal you). Respectful and structured.",
    },
}


def get_style_context(lang: str) -> str:
    """Build a compact LLM instruction string for the given language."""
    code = lang.split("-")[0].lower() if lang else "en"
    style = WRITING_STYLES.get(code)
    if not style:
        return ""
    return (
        f"Writing style for {code.upper()}: "
        f"Formal greeting: \"{style['formal_greeting']}\" | "
        f"Informal greeting: \"{style['informal_greeting']}\" | "
        f"Formal closing: \"{style['formal_closing']}\" | "
        f"Informal closing: \"{style['informal_closing']}\" | "
        f"{style['style_notes']}"
    )
