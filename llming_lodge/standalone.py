"""Standalone HTTPS chat server — run llming-lodge without any host app.

Usage::

    cd samples/chat
    poetry run python app.py

    # Or via script entry point from anywhere:
    poetry run llming-lodge

    # Custom port / HTTP-only:
    PORT=9000 poetry run llming-lodge
    HTTPS=0 poetry run llming-lodge

Environment variables:
    PORT            — Server port (default: 8443)
    HTTPS           — Enable HTTPS with self-signed cert (default: 1)
    SYSTEM_PROMPT   — Default system prompt (optional)
    SSL_CERTFILE    — Path to TLS certificate (auto-generated if missing)
    SSL_KEYFILE     — Path to TLS private key (auto-generated if missing)

Loads .env from the current working directory or any parent directory.
"""

import asyncio
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


# ── Self-signed certificate generation ───────────────────────────

def _ensure_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed TLS certificate if files don't exist."""
    if cert_path.exists() and key_path.exists():
        return

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "llming-lodge dev"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    logger.info(f"Generated self-signed certificate: {cert_path}")


# ── FastAPI app factory ──────────────────────────────────────────

def create_app():
    """Create the FastAPI application.

    All heavy imports happen here so that .env is loaded before
    any provider code runs.  Also called by uvicorn in reload mode
    via the factory=True flag.
    """
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    from llming_lodge.api.chat_session_api import SessionRegistry, WebSocketChatController
    from llming_lodge.chat_page import ChatPage, build_chat_html, start_dev_file_watcher
    from llming_lodge.chat_config import ChatAppConfig
    from llming_lodge.server import (
        get_static_path, get_chat_static_path, get_ws_router,
        API_PREFIX, STATIC_PREFIX,
    )
    from llming_com.auth import get_auth as _auth, AUTH_COOKIE_NAME, SESSION_COOKIE_NAME

    @asynccontextmanager
    async def lifespan(_a):
        # Start session cleanup loop
        async def _cleanup():
            while True:
                await asyncio.sleep(60)
                SessionRegistry.get().cleanup_expired(ttl=600)
        task = asyncio.create_task(_cleanup())

        # Start dev file watcher for hot-reload
        await start_dev_file_watcher()

        yield
        task.cancel()

    _app = FastAPI(title="llming-lodge Chat", docs_url=None, redoc_url=None, lifespan=lifespan)

    _app.mount(STATIC_PREFIX, StaticFiles(directory=get_static_path()), name="llming-static")
    _app.mount("/chat-static", StaticFiles(directory=get_chat_static_path()), name="chat-static")
    _app.include_router(get_ws_router())

    @_app.get("/", response_class=HTMLResponse)
    async def index():
        session_id = str(uuid4())
        system_prompt = os.environ.get("SYSTEM_PROMPT")

        controller = WebSocketChatController(
            session_id=session_id,
            user_id="standalone",
            system_prompt=system_prompt,
        )

        registry = SessionRegistry.get()
        entry = registry.register_session(
            session_id=session_id,
            controller=controller,
            user_id="standalone",
            user_name="User",
        )

        asyncio.create_task(controller.discover_tools())

        config = {
            "sessionId": session_id,
            "wsPath": f"{API_PREFIX}/ws/{session_id}",
            "userName": "User",
            "userEmail": "",
            "userId": "standalone",
            "staticBase": STATIC_PREFIX,
        }

        config_json = json.dumps(config)

        # Store config on entry for the chat page route
        entry._frontend_config_json = config_json
        entry._frontend_renderers_json = "[]"
        entry._app_title = "Chat"

        html = build_chat_html(config_json, "[]", "Chat")

        # Set auth + session cookies on response
        token = _auth().sign_auth_token(session_id)
        response = HTMLResponse(html)
        response.set_cookie(
            AUTH_COOKIE_NAME, token,
            path="/", max_age=86400, samesite="lax",
        )
        response.set_cookie(
            SESSION_COOKIE_NAME, session_id,
            path="/", max_age=86400, samesite="lax", httponly=True,
        )
        return response

    return _app


# ── Entry point ──────────────────────────────────────────────────

def main():
    """Load .env, configure HTTPS, and start the server."""
    # Load .env BEFORE any llming_lodge imports that trigger LLMManager
    # find_dotenv() searches upward through parent directories
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    import uvicorn

    use_https = os.environ.get("HTTPS", "1") != "0"
    default_port = 8443 if use_https else 8000
    port = int(os.environ.get("PORT", str(default_port)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    ssl_kwargs = {}
    if use_https:
        cert_dir = Path.home() / ".llming-lodge" / "certs"
        cert_path = Path(os.environ.get("SSL_CERTFILE", str(cert_dir / "localhost.pem")))
        key_path = Path(os.environ.get("SSL_KEYFILE", str(cert_dir / "localhost-key.pem")))
        _ensure_self_signed_cert(cert_path, key_path)
        ssl_kwargs = {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}
        proto = "https"
    else:
        proto = "http"

    print(f"\n  llming-lodge chat → {proto}://localhost:{port}\n")
    uvicorn.run(
        "llming_lodge.standalone:create_app",
        factory=True,
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=[str(Path(__file__).resolve().parent)],
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
