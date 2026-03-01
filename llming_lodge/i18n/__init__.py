"""Internationalization engine for llming-lodge chat UI.

Self-contained translation system — no external dependencies.

Usage::

    from llming_lodge.i18n import get_translations, t_chat

    translations = get_translations("de-de")  # full merged dict
    label = t_chat("chat.greeting", "de-de", name="Max")
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_TRANSLATIONS_DIR = Path(__file__).parent / "translations"

FALLBACK_CHAINS: dict[str, list[str]] = {
    "de-swg": ["de-swg", "de-de", "en-us"],
    "de-de": ["de-de", "en-us"],
    "en-us": ["en-us"],
    "en-in": ["en-in", "en-us"],
    "fr-fr": ["fr-fr", "en-us"],
    "it-it": ["it-it", "en-us"],
    "es-es": ["es-es", "en-us"],
    "zh-cn": ["zh-cn", "en-us"],
    "hi-in": ["hi-in", "en-us"],
    "nl-nl": ["nl-nl", "en-us"],
}


@lru_cache(maxsize=16)
def _load_file(lang: str) -> dict[str, str]:
    """Load a single translation JSON file. Cached."""
    path = _TRANSLATIONS_DIR / f"{lang}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[i18n] Failed to load {path}: {e}")
        return {}


def _get_chain(lang: str) -> list[str]:
    return FALLBACK_CHAINS.get(lang, [lang, "en-us"])


def _resolve(key: str, chain: list[str]) -> str:
    for lang in chain:
        data = _load_file(lang)
        if key in data:
            return data[key]
    return key


def t_chat(key: str, lang: str, **kwargs) -> str:
    """Translate a single key for the given language."""
    chain = _get_chain(lang)
    value = _resolve(key, chain)
    if kwargs:
        try:
            value = value.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return value


def get_translations(lang: str) -> dict:
    """Return fully-merged dict (all keys resolved through fallback chain).

    Suitable for injecting into the JS frontend as ``translations``.
    """
    chain = _get_chain(lang)
    merged: dict[str, str] = {}
    for fallback_lang in reversed(chain):
        data = _load_file(fallback_lang)
        merged.update(data)
    return merged
