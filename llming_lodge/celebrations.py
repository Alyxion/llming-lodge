"""Cultural celebrations — date computation, audience targeting, and greetings.

Pre-computed dates for major celebrations across cultures (2026-2055),
audience-based filtering (email, country, department), and multilingual
greetings resolved from i18n translation files.

Celebrations with algorithmic dates (Easter, Thanksgiving) are computed for
the full 30-year range.  Lunar/lunisolar holidays use authoritative calendar
sources and are hardcoded for 2026-2035 — extend the tuples to add years.

Greetings live in ``llming_lodge/i18n/translations/*.json`` under keys like
``celebration.holi``, ``celebration.christmas``, etc.  The format matches
``chat.greetings``: ``[{"text": "Happy Holi, {name}! 🎨🌈"}]``.

Usage::

    from llming_lodge.celebrations import (
        default_celebrations, UserContext,
        get_active, get_avatar, get_greetings, get_greeting_text,
    )

    celebrations = default_celebrations()
    celebrations["holi"].avatars = ["/static/logo/holi.png"]
    celebrations["holi"].audience.countries = ["India"]

    user = UserContext(email="dev@company.in", country="India")
    avatar = get_avatar(list(celebrations.values()), default="/logo.png", user=user)
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass, field
from fnmatch import fnmatch

# ── Range ────────────────────────────────────────────────────────────────────

_YEAR_START = 2026
_YEAR_END = 2056  # exclusive — covers 30 years

# ── Algorithmic date computation ─────────────────────────────────────────────


def _easter(year: int) -> datetime.date:
    """Easter Sunday via the Anonymous Gregorian algorithm (Computus)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return datetime.date(year, month, day + 1)


def _thanksgiving_us(year: int) -> datetime.date:
    """4th Thursday of November."""
    nov1 = datetime.date(year, 11, 1)
    first_thu = 1 + (3 - nov1.weekday()) % 7
    return datetime.date(year, 11, first_thu + 21)


# ── Hardcoded variable dates ────────────────────────────────────────────────
# Sources: hebcal.com, epicastrology.com, qppstudio.net, chinahighlights.com
# Extend the tuples below to cover additional years.

# Holi — Rangwali Holi (main color festival day, day after Holika Dahan)
_HOLI = [
    (2026, 3, 4), (2027, 3, 22), (2028, 3, 11), (2029, 3, 1), (2030, 3, 20),
    (2031, 3, 9), (2032, 3, 27), (2033, 3, 16), (2034, 3, 5), (2035, 3, 24),
]

# Diwali — main Lakshmi Puja day
_DIWALI = [
    (2026, 11, 8), (2027, 10, 29), (2028, 10, 17), (2029, 11, 5), (2030, 10, 26),
    (2031, 11, 14), (2032, 11, 2), (2033, 10, 22), (2034, 11, 10), (2035, 10, 30),
]

# Chinese New Year — first day
_CHINESE_NY = [
    (2026, 2, 17), (2027, 2, 6), (2028, 1, 26), (2029, 2, 13), (2030, 2, 3),
    (2031, 1, 23), (2032, 2, 11), (2033, 1, 31), (2034, 2, 19), (2035, 2, 8),
]

# Eid al-Fitr — expected dates (may vary ±1 day by moon sighting)
_EID_FITR = [
    (2026, 3, 20), (2027, 3, 9), (2028, 2, 26), (2029, 2, 14), (2030, 2, 4),
    (2031, 1, 24), (2032, 1, 14), (2033, 1, 2), (2033, 12, 23),
    (2034, 12, 12), (2035, 12, 1),
]

# Eid al-Adha — expected dates (may vary ±1 day by moon sighting)
_EID_ADHA = [
    (2026, 5, 27), (2027, 5, 16), (2028, 5, 5), (2029, 4, 24), (2030, 4, 13),
    (2031, 4, 2), (2032, 3, 22), (2033, 3, 11), (2034, 3, 1), (2035, 2, 18),
]

# Rosh Hashanah — first full day (celebration spans 2 days)
_ROSH_HASHANAH = [
    (2026, 9, 12), (2027, 10, 2), (2028, 9, 21), (2029, 9, 10), (2030, 9, 28),
    (2031, 9, 18), (2032, 9, 6), (2033, 9, 24), (2034, 9, 14), (2035, 10, 4),
]

# Hanukkah — first full day (celebration spans 8 days)
_HANUKKAH = [
    (2026, 12, 5), (2027, 12, 25), (2028, 12, 13), (2029, 12, 2), (2030, 12, 21),
    (2031, 12, 10), (2032, 11, 28), (2033, 12, 17), (2034, 12, 7), (2035, 12, 26),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _from_tuples(tuples: list[tuple[int, int, int]]) -> set[datetime.date]:
    return {datetime.date(y, m, d) for y, m, d in tuples}


def _fixed(month: int, day: int) -> set[datetime.date]:
    return {datetime.date(y, month, day) for y in range(_YEAR_START, _YEAR_END)}


def _multi_day(tuples: list[tuple[int, int, int]], extra: int) -> set[datetime.date]:
    result: set[datetime.date] = set()
    for y, m, d in tuples:
        base = datetime.date(y, m, d)
        for off in range(extra + 1):
            result.add(base + datetime.timedelta(days=off))
    return result


# ── Build date sets ──────────────────────────────────────────────────────────

def _build() -> dict[str, set[datetime.date]]:
    dates: dict[str, set[datetime.date]] = {}

    # Algorithmic — Easter + derived
    easter, good_friday, carnival = set(), set(), set()
    for y in range(_YEAR_START, _YEAR_END):
        e = _easter(y)
        easter.add(e)
        good_friday.add(e - datetime.timedelta(days=2))
        carnival.add(e - datetime.timedelta(days=47))  # Mardi Gras / Karneval
    dates["easter"] = easter
    dates["good_friday"] = good_friday
    dates["carnival"] = carnival

    # Algorithmic — Thanksgiving US
    dates["thanksgiving_us"] = {_thanksgiving_us(y) for y in range(_YEAR_START, _YEAR_END)}

    # Fixed-date
    dates["new_year"] = _fixed(1, 1)
    dates["valentines_day"] = _fixed(2, 14)
    dates["womens_day"] = _fixed(3, 8)
    dates["st_patricks_day"] = _fixed(3, 17)
    dates["earth_day"] = _fixed(4, 22)
    dates["workers_day"] = _fixed(5, 1)
    dates["halloween"] = _fixed(10, 31)
    dates["christmas_eve"] = _fixed(12, 24)
    dates["christmas"] = _fixed(12, 25)
    dates["new_years_eve"] = _fixed(12, 31)

    # Variable — Hindu
    dates["holi"] = _from_tuples(_HOLI)
    dates["diwali"] = _from_tuples(_DIWALI)

    # Variable — Chinese
    dates["chinese_new_year"] = _from_tuples(_CHINESE_NY)

    # Variable — Islamic
    dates["eid_al_fitr"] = _from_tuples(_EID_FITR)
    dates["eid_al_adha"] = _from_tuples(_EID_ADHA)

    # Variable — Jewish (multi-day)
    dates["rosh_hashanah"] = _multi_day(_ROSH_HASHANAH, extra=1)  # 2 days
    dates["hanukkah"] = _multi_day(_HANUKKAH, extra=7)            # 8 days

    return dates


CELEBRATION_DATES: dict[str, set[datetime.date]] = _build()
"""Pre-computed date sets keyed by celebration identifier."""


# ── Model ────────────────────────────────────────────────────────────────────

@dataclass
class UserContext:
    """User attributes for audience matching (typically from MS Graph)."""
    email: str = ""
    country: str = ""
    department: str = ""


@dataclass
class CelebrationAudience:
    """Who sees this celebration.  All-empty = universal (everyone).

    When multiple fields are set, a user matches if ANY field matches (OR).
    Email patterns use fnmatch glob syntax (case-insensitive).
    Country and department are case-insensitive exact matches.
    """
    email_patterns: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)


@dataclass
class Celebration:
    """A celebration with audience targeting, optional avatars, and i18n greetings.

    Greetings are resolved from translation files via ``greeting_key``
    (default: ``celebration.{key}``).  The translation value uses the
    same ``[{"text": "..."}]`` format as ``chat.greetings``.
    """
    key: str
    dates: set[datetime.date]
    audience: CelebrationAudience = field(default_factory=CelebrationAudience)
    avatars: list[str] = field(default_factory=list)
    greeting_key: str = ""  # i18n key; empty → "celebration.{key}"


# ── Audience matching ────────────────────────────────────────────────────────

def _matches_audience(aud: CelebrationAudience, user: UserContext | None) -> bool:
    if not aud.email_patterns and not aud.countries and not aud.departments:
        return True
    if user is None:
        return True
    email_lc = user.email.lower()
    country_lc = user.country.lower()
    dept_lc = user.department.lower()
    if aud.email_patterns and any(fnmatch(email_lc, p.lower()) for p in aud.email_patterns):
        return True
    if aud.countries and country_lc and any(c.lower() == country_lc for c in aud.countries):
        return True
    if aud.departments and dept_lc and any(d.lower() == dept_lc for d in aud.departments):
        return True
    return False


# ── Query helpers ────────────────────────────────────────────────────────────

def get_active(
    celebrations: list[Celebration],
    d: datetime.date | None = None,
    user: UserContext | None = None,
) -> list[Celebration]:
    """Return celebrations active on *d* that match *user*'s audience."""
    d = d or datetime.date.today()
    return [c for c in celebrations if d in c.dates and _matches_audience(c.audience, user)]


def get_avatar(
    celebrations: list[Celebration],
    default: str,
    d: datetime.date | None = None,
    user: UserContext | None = None,
) -> str:
    """Pick a random avatar from active celebrations, or *default*."""
    pool = [a for c in get_active(celebrations, d, user) for a in c.avatars]
    return random.choice(pool) if pool else default


def _resolve_greeting_entries(
    celebration: Celebration,
    lang: str,
) -> list[dict[str, str]]:
    """Resolve greeting entries for a single celebration from i18n."""
    from llming_lodge.i18n import get_translations

    translations = get_translations(lang)
    key = celebration.greeting_key or f"celebration.{celebration.key}"
    val = translations.get(key)
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [{"text": val}]
    return []


def get_greetings(
    celebrations: list[Celebration],
    lang: str,
    d: datetime.date | None = None,
    user: UserContext | None = None,
) -> list[dict[str, str]] | None:
    """Greeting dicts for active celebrations (for ``chat.greetings`` override).

    Resolves greetings from i18n translation files using each celebration's
    ``greeting_key``.  Returns ``[{"text": "..."}, ...]`` or ``None``.
    """
    active = get_active(celebrations, d, user)
    if not active:
        return None

    entries: list[dict[str, str]] = []
    for c in active:
        entries.extend(_resolve_greeting_entries(c, lang))

    return entries or None


def get_greeting_text(
    celebrations: list[Celebration],
    lang: str,
    name: str,
    default: str,
    d: datetime.date | None = None,
    user: UserContext | None = None,
) -> str:
    """Single greeting string with ``{name}`` substituted.

    Picks randomly when multiple greetings match.
    Returns *default* when no celebration greeting is active.
    """
    entries = get_greetings(celebrations, lang, d, user)
    if not entries:
        return default
    text = random.choice(entries)["text"]
    return text.format(name=name)


# ── Default celebrations ─────────────────────────────────────────────────────

def default_celebrations() -> dict[str, Celebration]:
    """All known celebrations — no audience or avatars set.

    Greetings are resolved at query time from i18n translation files
    (``celebration.{key}`` keys).  The consuming app should set
    ``audience`` and ``avatars`` on the celebrations it wants to activate.
    """
    D = CELEBRATION_DATES  # noqa: N806
    return {
        # Hindu
        "holi": Celebration(key="holi", dates=D["holi"]),
        "diwali": Celebration(key="diwali", dates=D["diwali"]),
        # Chinese
        "chinese_new_year": Celebration(key="chinese_new_year", dates=D["chinese_new_year"]),
        # Islamic
        "eid_al_fitr": Celebration(key="eid_al_fitr", dates=D["eid_al_fitr"]),
        "eid_al_adha": Celebration(key="eid_al_adha", dates=D["eid_al_adha"]),
        # Jewish
        "rosh_hashanah": Celebration(key="rosh_hashanah", dates=D["rosh_hashanah"]),
        "hanukkah": Celebration(key="hanukkah", dates=D["hanukkah"]),
        # Christian / Western
        "christmas": Celebration(key="christmas", dates=D["christmas"] | D["christmas_eve"]),
        "easter": Celebration(key="easter", dates=D["easter"]),
        "carnival": Celebration(key="carnival", dates=D["carnival"]),
        # Secular / International
        "new_year": Celebration(key="new_year", dates=D["new_year"] | D["new_years_eve"]),
        "halloween": Celebration(key="halloween", dates=D["halloween"]),
        "valentines_day": Celebration(key="valentines_day", dates=D["valentines_day"]),
        "womens_day": Celebration(key="womens_day", dates=D["womens_day"]),
        "thanksgiving_us": Celebration(key="thanksgiving_us", dates=D["thanksgiving_us"]),
        "st_patricks_day": Celebration(key="st_patricks_day", dates=D["st_patricks_day"]),
        # Date-only (no default greeting key in i18n)
        "good_friday": Celebration(key="good_friday", dates=D["good_friday"]),
        "earth_day": Celebration(key="earth_day", dates=D["earth_day"]),
        "workers_day": Celebration(key="workers_day", dates=D["workers_day"]),
    }
