"""Speech service: STT (transcription) and TTS (synthesis) via OpenAI API."""

import io
import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# Available TTS voices: (id, label, gender)
TTS_VOICES = [
    ("cedar", "Cedar", "male"),
    ("marin", "Marin", "female"),
    ("ash", "Ash", "male"),
    ("ballad", "Ballad", "male"),
    ("coral", "Coral", "female"),
    ("echo", "Echo", "male"),
    ("fable", "Fable", "male"),
    ("nova", "Nova", "female"),
    ("onyx", "Onyx", "male"),
    ("sage", "Sage", "female"),
    ("shimmer", "Shimmer", "female"),
    ("verse", "Verse", "female"),
    ("alloy", "Alloy", "neutral"),
]

DEFAULT_MALE_VOICE = "cedar"
DEFAULT_FEMALE_VOICE = "marin"


class SpeechService:
    STT_MODEL = "gpt-4o-transcribe"
    STT_TIMESTAMP_MODEL = "whisper-1"  # verbose_json + word timestamps only supported by whisper-1
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_DEFAULT_VOICE = DEFAULT_MALE_VOICE

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = AsyncOpenAI(api_key=api_key)

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "voice.webm",
        content_type: str = "audio/webm",
        language: str = "",
    ) -> str:
        """Transcribe audio to text using OpenAI."""
        if len(audio_bytes) < 100:
            logger.warning(f"[STT] Audio too small ({len(audio_bytes)} bytes), skipping")
            return ""
        logger.info(f"[STT] Transcribing: {len(audio_bytes)} bytes, filename={filename}, content_type={content_type}, model={self.STT_MODEL}")
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = filename
        kwargs = dict(
            model=self.STT_MODEL,
            file=audio_file,
            response_format="json",
        )
        # ISO-639-1 language hint (e.g. "de", "en", "fr")
        if language:
            kwargs["language"] = language
        try:
            result = await self._client.audio.transcriptions.create(**kwargs)
            return result.text
        except Exception as e:
            err_str = str(e).lower()
            # Catch various format-related errors and retry with whisper-1
            if any(s in err_str for s in ("corrupted", "unsupported", "invalid file format", "invalid_file_format", "could not process")):
                logger.warning(f"[STT] {self.STT_MODEL} format error ({e}), retrying with {self.STT_TIMESTAMP_MODEL}")
                audio_file.seek(0)
                kwargs["model"] = self.STT_TIMESTAMP_MODEL
                try:
                    result = await self._client.audio.transcriptions.create(**kwargs)
                    return result.text
                except Exception as e2:
                    logger.warning(f"[STT] Fallback also failed ({e2}), returning empty (likely truncated/empty audio)")
                    return ""
            logger.error(f"[STT] Transcription failed: {e}")
            raise

    # Locale code → language name for TTS instructions
    _LOCALE_NAMES = {
        "en-us": "English", "en-in": "English",
        "de-de": "German", "de-swg": "Swabian German",
        "fr-fr": "French", "it-it": "Italian", "hi-in": "Hindi",
    }

    # Models that support the 'instructions' parameter
    _INSTRUCTION_MODELS = {"gpt-4o-mini-tts"}

    def _tts_kwargs(self, text: str, locale: str = "", voice: str = "") -> dict:
        """Build common TTS kwargs."""
        kwargs = dict(
            model=self.TTS_MODEL,
            voice=voice or self.TTS_DEFAULT_VOICE,
            input=text,
            response_format="mp3",
        )
        # Only gpt-4o-mini-tts supports the 'instructions' param;
        # tts-1 / tts-1-hd auto-detect language from input text.
        if self.TTS_MODEL in self._INSTRUCTION_MODELS:
            lang_name = self._LOCALE_NAMES.get(locale, "")
            if lang_name:
                kwargs["instructions"] = f"Speak in {lang_name}."
        return kwargs

    async def synthesize_fast(self, text: str, locale: str = "", voice: str = "") -> bytes:
        """TTS only — no word-timing transcription. ~2x faster than synthesize()."""
        kwargs = self._tts_kwargs(text, locale, voice)
        tts_response = await self._client.audio.speech.create(**kwargs)
        return tts_response.content

    # Azure Realtime deployment name — set via env or override
    REALTIME_DEPLOYMENT = os.environ.get("AZURE_OPENAI_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview")

    def get_realtime_ws_url(self) -> str:
        """Build the Azure OpenAI Realtime WebSocket URL."""
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        deployment = self.REALTIME_DEPLOYMENT
        host = azure_endpoint.replace("https://", "").replace("http://", "")
        return f"wss://{host}/openai/realtime?api-version={api_version}&deployment={deployment}"

    @staticmethod
    def get_realtime_api_key() -> str:
        return os.environ.get("AZURE_OPENAI_API_KEY", "")

    async def synthesize(self, text: str, locale: str = "", voice: str = "") -> tuple[bytes, list[dict]]:
        """Synthesize text to speech and return (mp3_bytes, word_timings).

        Word timings are obtained by transcribing the generated audio back
        with timestamp_granularities=["word"].
        """
        kwargs = self._tts_kwargs(text, locale, voice)
        tts_response = await self._client.audio.speech.create(**kwargs)
        mp3_bytes = tts_response.content

        # Transcribe the generated audio to get word timestamps
        # whisper-1 is required for verbose_json + timestamp_granularities
        audio_file = io.BytesIO(mp3_bytes)
        audio_file.name = "speech.mp3"
        stt_result = await self._client.audio.transcriptions.create(
            model=self.STT_TIMESTAMP_MODEL,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )
        word_timings = [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in (stt_result.words or [])
        ]
        return mp3_bytes, word_timings
