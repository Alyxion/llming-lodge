"""Dev dashboard — event loop health, CPU, memory, process, disk, asyncio monitoring.

Routes:
  GET  /dev          — serves static index.html
  GET  /dev-static/* — static CSS/JS assets
  GET  /dev/state    — JSON time-series (auth via X-Dev-Key header)
  GET  /dev/info     — JSON system/process info (auth via X-Dev-Key header)

Requires LLMING_DEV_PASSWORD env var. If not set, build_dev_router() returns None.
"""

import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "dev"


def build_dev_router() -> "APIRouter | None":
    """Build the /dev dashboard router.

    Returns None if LLMING_DEV_PASSWORD is not set — the dashboard
    must never be accessible without a password.
    """
    _dev_key = os.environ.get("LLMING_DEV_PASSWORD", "")
    if not _dev_key:
        logger.info("[DEV] LLMING_DEV_PASSWORD not set — dev dashboard disabled")
        return None

    from llming_lodge.monitoring.heartbeat import HeartbeatMonitor

    router = APIRouter()

    def _check_key(request: Request) -> bool:
        return request.headers.get("x-dev-key", "") == _dev_key

    _index_html = (_STATIC_DIR / "index.html").read_text()

    @router.get("/dev", response_class=HTMLResponse)
    async def dev_page():
        return _index_html

    @router.get("/dev/state")
    async def dev_state(request: Request, since: float = 0):
        if not _check_key(request):
            await asyncio.sleep(0.2)  # rate-limit brute-force
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        monitor = HeartbeatMonitor.get()
        await monitor.start()
        return monitor.get_state(since=since)

    @router.get("/dev/info")
    async def dev_info(request: Request):
        """Snapshot asyncio tasks on the event loop, then run the rest in a thread."""
        if not _check_key(request):
            await asyncio.sleep(0.2)  # rate-limit brute-force
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        monitor = HeartbeatMonitor.get()
        await monitor.start()
        tasks_snapshot = monitor.snapshot_asyncio_tasks()
        return await asyncio.to_thread(monitor.get_info, tasks_snapshot)

    @router.get("/dev/llm")
    async def dev_llm(request: Request):
        """Return LLM provider cascade & model resolution info."""
        if not _check_key(request):
            await asyncio.sleep(0.2)
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        try:
            from llming_lodge.chat_controller import llm_manager
        except Exception:
            return {"error": "llm_manager not available"}

        cascade = llm_manager.provider_cascade
        active = [p for p in cascade if p in llm_manager.providers]
        inactive = [p for p in cascade if p not in llm_manager.providers]

        # Default model categories
        categories = {}
        for cat in ("small", "medium", "large", "reasoning_small", "reasoning_medium", "reasoning_large"):
            model_name = llm_manager.get_default_model(cat)
            if model_name:
                try:
                    provider = llm_manager.get_provider_for_model(model_name)
                except ValueError:
                    provider = None
                categories[cat] = {"model": model_name, "resolved_provider": provider}

        # Per-model cascade detail
        models = []
        seen = set()
        for provider_name, provider in llm_manager.providers.items():
            for info in provider.get_models():
                if info.model not in seen:
                    seen.add(info.model)
                    # Find all providers for this model
                    try:
                        providers_for = llm_manager.get_providers_for_model(info.model)
                    except ValueError:
                        providers_for = []
                    models.append({
                        "model": info.model,
                        "name": info.name,
                        "label": info.label,
                        "active_provider": providers_for[0] if providers_for else None,
                        "all_providers": providers_for,
                        "size": info.size.name,
                        "hosting_icon": info.hosting_icon,
                    })

        return {
            "cascade_order": cascade,
            "active_providers": active,
            "inactive_providers": inactive,
            "default_categories": categories,
            "models": models,
        }

    return router


def mount_dev_static(app) -> None:
    """Mount the /dev-static files. Call separately after include_router."""
    app.mount("/dev-static", StaticFiles(directory=str(_STATIC_DIR)), name="dev-static")
