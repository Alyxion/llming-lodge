#!/usr/bin/env python3
"""Standalone chat app — run llming-lodge as an HTTPS server.

    cd samples/chat
    poetry run python app.py

Requires at least one API key in your environment (e.g. OPENAI_API_KEY).
Starts on https://localhost:8443 with a self-signed certificate.

Environment variables:
    PORT            — Server port (default: 8443)
    HTTPS           — Set to 0 for plain HTTP (default: 1)
    SYSTEM_PROMPT   — Default system prompt (optional)
    SSL_CERTFILE    — Custom TLS certificate path
    SSL_KEYFILE     — Custom TLS key path
"""
from llming_lodge.standalone import main

if __name__ == "__main__":
    main()
