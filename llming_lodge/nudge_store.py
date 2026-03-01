"""MongoDB-backed store for nudges with ACL and favorites."""

import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# In-memory file cache (shared across all sessions)
# ------------------------------------------------------------------

@dataclass
class CachedFile:
    """A single nudge file held in memory."""
    name: str
    mime_type: str
    size: int
    raw: bytes                    # decoded binary (the original file)
    text_content: str = ""        # extracted plain text for LLM context


@dataclass
class FileCacheEntry:
    """Cache entry for one nudge's files."""
    files: list[CachedFile]
    updated_at: str | None        # from MongoDB, used for freshness check
    mode: str                     # dev / live
    last_access: float = field(default_factory=time.monotonic)


class NudgeFileCache:
    """Process-wide, async-safe in-memory cache for nudge file data.

    - Files are loaded once from MongoDB and kept in RAM (binary + extracted text).
    - On every access the ``updated_at`` field is checked against MongoDB;
      if stale the entry is evicted and re-fetched.
    - A background task periodically evicts entries that haven't been
      accessed for ``MAX_IDLE`` seconds.
    """

    EVICTION_INTERVAL = 300   # sweep every 5 min
    MAX_IDLE = 1800           # evict after 30 min without access

    def __init__(self) -> None:
        self._entries: dict[str, FileCacheEntry] = {}
        self._lock: asyncio.Lock | None = None
        self._cleanup_task: asyncio.Task | None = None

    # -- lazy lock (safe to construct before event loop exists) --

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _ensure_cleanup(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.EVICTION_INTERVAL)
            now = time.monotonic()
            async with self._get_lock():
                stale = [uid for uid, e in self._entries.items()
                         if now - e.last_access > self.MAX_IDLE]
                for uid in stale:
                    del self._entries[uid]
                if stale:
                    logger.info("[FILE_CACHE] Evicted %d idle entries", len(stale))

    # -- public API --

    def evict(self, uid: str) -> None:
        """Immediately remove a uid from the cache (e.g. after save/delete)."""
        self._entries.pop(uid, None)

    async def get_files(
        self,
        uid: str,
        store: "NudgeStore",
        user_email: str,
        user_teams: list[dict] | None = None,
        *,
        nudge: dict | None = None,
    ) -> list[CachedFile]:
        """Return cached files for *uid*, re-fetching from MongoDB if stale.

        *nudge* can be passed to avoid a redundant ``get_for_user`` call
        when the caller already has the document.
        """
        lock = self._get_lock()
        self._ensure_cleanup()

        # 1. Check cache
        async with lock:
            entry = self._entries.get(uid)
            if entry:
                entry.last_access = time.monotonic()

        # 2. Freshness check (lightweight query)
        if entry:
            current_ts = await _fetch_updated_at(store, uid, entry.mode)
            if current_ts == entry.updated_at:
                return entry.files
            # Stale — drop it
            async with lock:
                self._entries.pop(uid, None)

        # 3. Cache miss — fetch full document and extract
        if nudge is None:
            nudge = await store.get_for_user(uid, user_email, user_teams)
        if not nudge:
            return []

        raw_files = nudge.get("files") or []
        if not raw_files:
            logger.debug("[FILE_CACHE] No files for nudge %s", uid)
            return []

        cached = _extract_to_cache(raw_files)
        logger.info("[FILE_CACHE] Loaded %d files for nudge %s (%s)",
                     len(cached), uid,
                     ", ".join(f"{f.name}={len(f.text_content)}chars" for f in cached))

        async with lock:
            self._entries[uid] = FileCacheEntry(
                files=cached,
                updated_at=nudge.get("updated_at"),
                mode=nudge.get("mode", "dev"),
            )

        return cached


# Module-level singleton
_file_cache = NudgeFileCache()


def get_file_cache() -> NudgeFileCache:
    return _file_cache


# -- helpers (module-level to keep the class lean) --

async def _fetch_updated_at(store: "NudgeStore", uid: str, mode: str) -> str | None:
    coll, _ = store._ensure_colls()
    doc = await coll.find_one({"uid": uid, "mode": mode}, {"updated_at": 1, "_id": 0})
    return doc.get("updated_at") if doc else None


def _extract_to_cache(files: list[dict]) -> list[CachedFile]:
    """Decode base64 → keep raw bytes in RAM, extract text."""
    from llming_lodge.documents import extract_text

    result: list[CachedFile] = []
    for f in files:
        if not isinstance(f, dict):
            continue
        content_b64 = f.get("content", "")
        if not content_b64:
            continue

        name = f.get("name", "file")
        mime = f.get("mime_type", "text/plain")

        try:
            raw = base64.b64decode(content_b64.split(",", 1)[-1])
        except Exception as e:
            logger.warning("[FILE_CACHE] base64 decode failed for %s: %s", name, e)
            continue

        # Extract text from raw bytes in memory (no disk I/O)
        text = ""
        if not mime.startswith("image/"):
            try:
                text = extract_text(raw, mime)
            except Exception as e:
                logger.warning("[FILE_CACHE] Text extraction failed for %s: %s", name, e)

        result.append(CachedFile(
            name=name,
            mime_type=mime,
            size=f.get("size", len(raw)),
            raw=raw,
            text_content=text,
        ))
    return result


# ------------------------------------------------------------------
# NudgeStore
# ------------------------------------------------------------------

class NudgeStore:
    """CRUD + search + favorites for nudges stored in MongoDB.

    Follows the lazy-init pattern from MongoDBBudgetLimit — the async
    MongoDB client is created on first use via ``nice_droplets.utils.mongo_helpers``.
    """

    COLLECTION = "nudges"
    FAVORITES_COLLECTION = "nudge_favorites"
    CACHE_TTL = 300  # seconds

    def __init__(self, mongo_uri: str, mongo_db: str):
        if not mongo_uri or not mongo_db:
            raise ValueError("mongo_uri and mongo_db are required")
        self._mongo_uri = mongo_uri
        self._mongo_db = mongo_db
        self._coll = None
        self._fav_coll = None
        self._cache: dict[str, tuple[float, dict]] = {}
        self._indexes_created = False

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _ensure_colls(self):
        if self._coll is None:
            from nice_droplets.utils.mongo_helpers import get_async_mongo_client
            client = get_async_mongo_client(self._mongo_uri)
            db = client[self._mongo_db]
            self._coll = db[self.COLLECTION]
            self._fav_coll = db[self.FAVORITES_COLLECTION]
        return self._coll, self._fav_coll

    async def _ensure_indexes(self):
        if self._indexes_created:
            return
        coll, fav_coll = self._ensure_colls()

        await coll.create_index([("uid", 1), ("mode", 1)], unique=True)
        await coll.create_index("creator_email")
        await coll.create_index("category")
        await coll.create_index("team_id")
        await coll.create_index([("is_master", 1), ("mode", 1)])
        await fav_coll.create_index(
            [("user_email", 1), ("nudge_uid", 1)], unique=True,
        )
        self._indexes_created = True

    # ------------------------------------------------------------------
    # Document cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, uid: str, mode: str) -> str:
        return f"{uid}:{mode}"

    def _cache_get(self, uid: str, mode: str) -> dict | None:
        key = self._cache_key(uid, mode)
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry[0]) < self.CACHE_TTL:
            return entry[1]
        self._cache.pop(key, None)
        return None

    def _cache_set(self, uid: str, mode: str, doc: dict):
        self._cache[self._cache_key(uid, mode)] = (time.monotonic(), doc)

    def _cache_evict(self, uid: str):
        self._cache.pop(self._cache_key(uid, "dev"), None)
        self._cache.pop(self._cache_key(uid, "live"), None)

    # ------------------------------------------------------------------
    # ACL
    # ------------------------------------------------------------------

    @staticmethod
    def _user_can_see(doc: dict, user_email: str, user_teams: list[dict] | None = None) -> bool:
        email = user_email.lower()
        if doc.get("creator_email", "").lower() == email and not doc.get("team_id"):
            return True
        if doc.get("team_id") and user_teams:
            if any(t["team_id"] == doc["team_id"] for t in user_teams):
                return True
        visibility = doc.get("visibility", [])
        if not visibility:
            return False
        return any(fnmatch(email, pat) for pat in visibility)

    @staticmethod
    def _user_can_edit(doc: dict, user_email: str, user_teams: list[dict] | None = None) -> bool:
        email = user_email.lower()
        if doc.get("creator_email", "").lower() == email and not doc.get("team_id"):
            return True
        if doc.get("team_id") and user_teams:
            return any(
                t["team_id"] == doc["team_id"] and t["role"] in ("owner", "editor")
                for t in user_teams
            )
        return False

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def save(self, data: dict, user_email: str, user_teams: list[dict] | None = None, *, is_admin: bool = False) -> dict:
        """Upsert a nudge. Only the creator (or team editor/owner) may update."""
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()

        uid = data.get("uid") or str(uuid4())
        mode = data.get("mode", "dev")
        data["uid"] = uid
        data["mode"] = mode
        data["creator_email"] = data.get("creator_email") or user_email

        # Validate: is_master requires team_id
        if data.get("is_master") and not data.get("team_id"):
            raise ValueError("Master nudges must belong to a team (team_id required)")

        # Check ownership for existing docs (admin bypasses)
        existing = await coll.find_one({"uid": uid, "mode": mode})
        if existing and not is_admin and not self._user_can_edit(existing, user_email, user_teams):
            raise PermissionError("Only the creator or team editor can update this nudge")

        # Strip text_content from files before persisting — it will be
        # extracted on-the-fly when the nudge is loaded into a chat session.
        for f in data.get("files") or []:
            if isinstance(f, dict):
                f.pop("text_content", None)

        # Upsert
        doc = {k: v for k, v in data.items() if k != "_id"}
        await coll.replace_one(
            {"uid": uid, "mode": mode}, doc, upsert=True,
        )
        self._cache_evict(uid)
        get_file_cache().evict(uid)
        return self._meta(doc)

    async def get(self, uid: str, mode: str, user_email: str, user_teams: list[dict] | None = None) -> dict | None:
        """Get a full nudge document. Enforces visibility."""
        cached = self._cache_get(uid, mode)
        if cached is not None:
            if self._user_can_see(cached, user_email, user_teams):
                return cached
            return None

        await self._ensure_indexes()
        coll, _ = self._ensure_colls()
        doc = await coll.find_one({"uid": uid, "mode": mode})
        if not doc:
            return None
        doc.pop("_id", None)
        self._cache_set(uid, mode, doc)
        if not self._user_can_see(doc, user_email, user_teams):
            return None
        return doc

    async def get_for_user(self, uid: str, user_email: str, user_teams: list[dict] | None = None) -> dict | None:
        """Return dev version for the creator/team editor, live for others."""
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()

        # Try dev first — if user can edit, return it
        dev = await coll.find_one({"uid": uid, "mode": "dev"})
        if dev:
            dev.pop("_id", None)
            self._cache_set(uid, "dev", dev)
            if self._user_can_edit(dev, user_email, user_teams):
                return dev

        # Otherwise return live if visible
        live = await coll.find_one({"uid": uid, "mode": "live"})
        if live:
            live.pop("_id", None)
            self._cache_set(uid, "live", live)
            if self._user_can_see(live, user_email, user_teams):
                return live

        return None

    async def flush_to_live(self, uid: str, user_email: str, user_teams: list[dict] | None = None) -> bool:
        """Copy all fields from dev to live. Creator or team editor/owner."""
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()

        dev = await coll.find_one({"uid": uid, "mode": "dev"})
        if not dev:
            return False
        if not self._user_can_edit(dev, user_email, user_teams):
            raise PermissionError("Only the creator or team editor can publish")

        live_doc = {k: v for k, v in dev.items() if k not in ("_id", "mode")}
        live_doc["mode"] = "live"
        await coll.replace_one(
            {"uid": uid, "mode": "live"}, live_doc, upsert=True,
        )
        self._cache_evict(uid)
        get_file_cache().evict(uid)
        return True

    async def delete(self, uid: str, user_email: str, user_teams: list[dict] | None = None) -> bool:
        """Delete both dev and live documents. Creator or team editor/owner."""
        await self._ensure_indexes()
        coll, fav_coll = self._ensure_colls()

        existing = await coll.find_one({"uid": uid})
        if not existing:
            return False
        if not self._user_can_edit(existing, user_email, user_teams):
            raise PermissionError("Only the creator or team editor can delete")

        await coll.delete_many({"uid": uid})
        await fav_coll.delete_many({"nudge_uid": uid})
        self._cache_evict(uid)
        get_file_cache().evict(uid)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        user_email: str,
        *,
        query: str = "",
        category: str = "",
        mine: bool = False,
        page: int = 0,
        page_size: int = 20,
        user_teams: list[dict] | None = None,
        include_master: bool = False,
        all_users: bool = False,
    ) -> list[dict]:
        """Search nudges with ACL + pagination.

        ``mine=True`` returns the creator's dev versions + team nudges.
        Otherwise returns visible live versions *plus* the user's own
        dev nudges (so creators see their WIP in the "All" tab).

        ``include_master=False`` (default) hides master nudges from
        non-admin users.  Admins pass ``True`` to see them.
        """
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()

        user_team_ids = [t["team_id"] for t in (user_teams or []) if t.get("role") in ("owner", "editor")]
        filt: dict[str, Any] = {}
        if not include_master:
            filt["is_master"] = {"$ne": True}
        text_or = None
        if query:
            text_or = [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
            ]
        if all_users:
            if text_or:
                filt["$or"] = text_or
        elif mine:
            owner_or: list[dict] = [{"creator_email": user_email, "team_id": None}]
            if user_team_ids:
                owner_or.append({"team_id": {"$in": user_team_ids}})
            if text_or:
                filt["$and"] = [{"$or": owner_or}, {"$or": text_or}]
            else:
                filt["$or"] = owner_or
            filt["mode"] = "dev"
        else:
            # Live docs visible to anyone + user's own dev docs + team dev docs
            mode_filter: list[dict] = [
                {"mode": "live"},
                {"mode": "dev", "creator_email": user_email},
            ]
            if user_team_ids:
                mode_filter.append({"mode": "dev", "team_id": {"$in": user_team_ids}})
            if text_or:
                filt["$and"] = [{"$or": mode_filter}, {"$or": text_or}]
            else:
                filt["$or"] = mode_filter
        if category:
            filt["category"] = category

        cursor = (
            coll.find(filt, {"files": 0})  # exclude heavy file data
            .sort("updated_at", -1)
            .skip(page * page_size)
            .limit((page_size + 1) * 2)  # over-fetch to allow dedup
        )
        results = []
        seen_uids: set[str] = set()
        async for doc in cursor:
            doc.pop("_id", None)
            uid = doc.get("uid", "")
            if uid in seen_uids:
                # Dedup: prefer dev for creator (already seen), skip live
                continue
            if mine or all_users or self._user_can_see(doc, user_email, user_teams):
                seen_uids.add(uid)
                results.append(self._meta(doc))
            if len(results) > page_size:
                break

        has_more = len(results) > page_size
        return results[:page_size], has_more

    # ------------------------------------------------------------------
    # Favorites
    # ------------------------------------------------------------------

    async def get_favorites(self, user_email: str, user_teams: list[dict] | None = None) -> list[dict]:
        """Return metadata for the user's favorite nudges."""
        await self._ensure_indexes()
        coll, fav_coll = self._ensure_colls()

        fav_docs = await fav_coll.find(
            {"user_email": user_email},
        ).to_list(length=100)

        uids = [d["nudge_uid"] for d in fav_docs]
        if not uids:
            return []

        results = []
        async for doc in coll.find({"uid": {"$in": uids}}, {"files": 0}):
            doc.pop("_id", None)
            if not self._user_can_see(doc, user_email, user_teams):
                continue
            # Prefer dev for creator/team editor, live for others
            can_edit = self._user_can_edit(doc, user_email, user_teams)
            if (can_edit and doc.get("mode") == "dev") or \
               (not can_edit and doc.get("mode") == "live"):
                results.append(self._meta(doc))
        return results

    async def set_favorite(self, user_email: str, uid: str, favorite: bool):
        """Toggle favorite status for a nudge."""
        await self._ensure_indexes()
        _, fav_coll = self._ensure_colls()

        if favorite:
            await fav_coll.update_one(
                {"user_email": user_email, "nudge_uid": uid},
                {"$set": {"user_email": user_email, "nudge_uid": uid}},
                upsert=True,
            )
        else:
            await fav_coll.delete_one(
                {"user_email": user_email, "nudge_uid": uid},
            )

    async def validate_visible(self, uids: list[str], user_email: str, user_teams: list[dict] | None = None) -> list[str]:
        """Return subset of UIDs that the user can still see (for favorites ACL)."""
        if not uids:
            return []
        coll, _ = self._ensure_colls()
        valid = set()
        async for doc in coll.find({"uid": {"$in": uids}}, {"uid": 1, "creator_email": 1, "team_id": 1, "visibility": 1, "mode": 1}):
            doc.pop("_id", None)
            if self._user_can_see(doc, user_email, user_teams):
                valid.add(doc["uid"])
        return list(valid)

    # ------------------------------------------------------------------
    # Master nudges
    # ------------------------------------------------------------------

    async def get_master_nudges(
        self,
        user_email: str,
        user_teams: list[dict] | None = None,
    ) -> list[dict]:
        """Return all live master nudges visible to this user."""
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()
        results = []
        async for doc in coll.find({"is_master": True, "mode": "live"}):
            doc.pop("_id", None)
            if self._user_can_see(doc, user_email, user_teams):
                results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Discoverable nudges (auto-discover)
    # ------------------------------------------------------------------

    async def get_discoverable_nudges(
        self,
        user_email: str,
        user_teams: list[dict] | None = None,
    ) -> list[dict]:
        """Return auto-discover nudges visible to this user.

        Returns the live version for regular users and the dev version
        for team editors/owners (so they can test before publishing).
        Excludes the heavy ``files`` field — content is fetched lazily
        when the ``consult_nudge`` tool is actually called.
        """
        await self._ensure_indexes()
        coll, _ = self._ensure_colls()
        results = []
        seen_uids: set[str] = set()
        async for doc in coll.find({"auto_discover": True}, {"files.content": 0, "files.text_content": 0}).sort("mode", 1):
            doc.pop("_id", None)
            uid = doc.get("uid", "")
            if uid in seen_uids:
                continue
            if doc.get("mode") == "dev":
                # Dev versions only visible to editors/owners
                if not self._user_can_edit(doc, user_email, user_teams):
                    continue
            else:
                if not self._user_can_see(doc, user_email, user_teams):
                    continue
            seen_uids.add(uid)
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _meta(doc: dict) -> dict:
        """Strip heavy fields, return metadata-only dict for listings."""
        return {
            k: v for k, v in doc.items()
            if k not in ("files", "_id")
        }
