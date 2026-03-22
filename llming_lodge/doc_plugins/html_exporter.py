"""Server-side HTML export — assembles a standalone HTML document from spec."""

import logging

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{style_block}
</head>
<body>
{body}
{script_block}
</body>
</html>"""


def export_html(spec: dict) -> bytes:
    """Generate a standalone HTML document from an HTML sandbox spec.

    Args:
        spec: HTML spec with optional keys: ``html``, ``css``, ``js``, ``title``,
              ``content`` (fallback for ``html``).

    Returns:
        UTF-8 encoded HTML document as bytes.
    """
    title = _escape_html(spec.get("title", "Document"))
    html_body = spec.get("html") or spec.get("content") or ""
    css = spec.get("css", "")
    js = spec.get("js", "")

    style_block = f"<style>\n{css}\n</style>" if css else ""
    script_block = f"<script>\n{js}\n</script>" if js else ""

    doc = _HTML_TEMPLATE.format(
        title=title,
        style_block=style_block,
        body=html_body,
        script_block=script_block,
    )
    return doc.encode("utf-8")


def _escape_html(text: str) -> str:
    """Escape HTML special characters in text (for title, etc.)."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
