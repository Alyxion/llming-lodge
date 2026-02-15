import re
from typing import Dict, Iterable, Callable, Generator

# Matches a run of one or more consecutive HTML tags (comments, DOCTYPE, or tags)
_TAG_RUN_RE = re.compile(r'(?:<!--.*?-->|<![^>]*>|<[^>]+?>)+', re.DOTALL)

class HTMLStreamTranslator:
    """
    1. Replaces every HTML tag (opening, closing, self‑closing, comments, DOCTYPE)
       with a unique placeholder <llmtN>.
    2. Exposes the text-with‑markers, so you can feed it to *any* translator
       (LLM, API, local model …) that should **only** translate human‑readable
       content.
    3. When translated text becomes available – whether all at once or token‑by‑token –
       detokenise() restores the original markup in the correct positions.
    """

    def __init__(self) -> None:
        self._marker2tag: Dict[str, str] = {}   # e.g. '<llmt7>' → '<div class="x">'
        self._counter: int = 0

    # ---------- Phase 1: tokenisation ----------

    def tokenise(self, html: str) -> str:
        """
        Replace every RUN of consecutive HTML tags with a unique placeholder and cache the mapping.
        Returns the marker-filled text that can be sent for translation.
        """
        def _sub(match: re.Match) -> str:
            self._counter += 1
            marker = f"<LQT{self._counter}/>"
            self._marker2tag[marker] = match.group(0)  # Store the full run of tags
            return marker

        return _TAG_RUN_RE.sub(_sub, html)

    # ---------- Phase 2: de‑tokenisation ----------

    def detokenise(self, translated_text: str) -> str:
        """
        Replace all placeholders with their original HTML tags.
        If the end of the string contains an incomplete marker like "<LQT99",
        trim it off to avoid premature substitution.
        """
        # Check the last 10 characters
        if len(translated_text) > 10:
            ending = translated_text[-10:]
            if ending.endswith("<"):
                translated_text = translated_text[:-1]
            elif ending.endswith("<L"):
                translated_text = translated_text[:-2]
            elif ending.endswith("<LQ"):
                translated_text = translated_text[:-3]
            elif ending.endswith("<LQT"):
                translated_text = translated_text[:-4]
            elif "<LQT" in ending and "/>" not in ending:
                # find last occurrence of <LQT
                pos = ending.rfind("<LQT")
                # find last occurrence of ">"
                end_pos = ending.rfind(">")
                if end_pos == -1 or end_pos < pos:
                    translated_text = translated_text[:-(10 - pos)]

        out = translated_text
        for marker in sorted(self._marker2tag, key=len, reverse=True):
            out = out.replace(marker, self._marker2tag[marker])
        return out
