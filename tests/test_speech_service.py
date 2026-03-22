"""Integration tests for SpeechService — calls the real OpenAI API.

Requires OPENAI_API_KEY in .env.
"""

import os

import dotenv
import pytest
from pathlib import Path

dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

from llming_lodge.speech_service import SpeechService

FIXTURE = Path(__file__).parent / "fixtures" / "voice-test-german.webm"


def test_fixture_exists():
    """Sanity: voice test fixture is present."""
    assert FIXTURE.exists()
    assert FIXTURE.stat().st_size > 1000


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_transcribe_german_audio():
    """Transcribe the German voice fixture and verify we get text back."""
    service = SpeechService()
    audio_bytes = FIXTURE.read_bytes()
    text = await service.transcribe(audio_bytes, "voice-test-german.webm", "audio/webm")
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"Transcription: {text}")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_synthesize_returns_audio_and_timings():
    """Synthesize a short sentence and verify mp3 bytes + word timings."""
    service = SpeechService()
    mp3_bytes, word_timings = await service.synthesize("Hello, this is a test.")
    assert isinstance(mp3_bytes, bytes)
    assert len(mp3_bytes) > 1000  # mp3 should be at least a few KB
    assert isinstance(word_timings, list)
    assert len(word_timings) > 0
    for wt in word_timings:
        assert "word" in wt
        assert "start" in wt
        assert "end" in wt
    print(f"TTS audio: {len(mp3_bytes)} bytes, {len(word_timings)} words")
    print(f"Word timings: {word_timings}")
