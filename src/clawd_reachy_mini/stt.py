"""Speech-to-text backends for Reachy Mini interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from clawd_reachy_mini.config import Config

logger = logging.getLogger(__name__)


class STTBackend(ABC):
    """Abstract base class for speech-to-text backends."""

    def preload(self) -> None:
        """Preload the model to avoid delay on first transcription."""
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text."""
        pass

    @abstractmethod
    def transcribe_file(self, path: Path) -> str:
        """Transcribe audio file to text."""
        pass


class WhisperSTT(STTBackend):
    """Local Whisper model for speech-to-text."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def preload(self) -> None:
        """Preload the Whisper model."""
        self._load_model()

    def _load_model(self):
        if self._model is None:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = self._load_model()

        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        result = model.transcribe(
            audio,
            fp16=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        return result["text"].strip()

    def transcribe_file(self, path: Path) -> str:
        model = self._load_model()
        result = model.transcribe(str(path), fp16=False)
        return result["text"].strip()


class MlxWhisperSTT(STTBackend):
    """MLX Whisper — Metal-native, no OpenMP, designed for Apple Silicon."""

    # Maps from friendly model names to mlx-community HuggingFace repos
    _REPOS = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
    }

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self._repo = self._REPOS.get(model_name, f"mlx-community/whisper-{model_name}-mlx")

    def preload(self) -> None:
        import mlx_whisper  # noqa: F401 — triggers model download
        logger.info(f"MLX Whisper model ready: {self._repo}")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import tempfile
        import wave

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        # mlx_whisper has a Metal allocation bug when receiving numpy arrays directly;
        # write to a temp WAV file and transcribe via file path instead
        audio_int16 = (audio * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            logger.debug(f"MLX transcribing {len(audio)} samples via file: {tmp_path}")
            return self.transcribe_file(Path(tmp_path))
        finally:
            try:
                Path(tmp_path).unlink()
            except FileNotFoundError:
                pass

    def transcribe_file(self, path: Path) -> str:
        import mlx_whisper
        result = mlx_whisper.transcribe(str(path), path_or_hf_repo=self._repo)
        return result["text"].strip()


class FasterWhisperSTT(STTBackend):
    """Faster-Whisper for optimized local transcription."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None

    def preload(self) -> None:
        """Preload the Faster-Whisper model."""
        self._load_model()

    def _load_model(self):
        if self._model is None:
            import os
            # Prevent OpenMP from spawning multiple threads — causes SIGSEGV on macOS
            # when GStreamer/GLib threads are already running in the same process
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model: {self.model_name}")
            self._model = WhisperModel(self.model_name, compute_type="float32", cpu_threads=1)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        model = self._load_model()

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        logger.debug(f"Transcribing {len(audio)} samples (dtype={audio.dtype}, max={audio.max():.4f})")
        # Eagerly consume the lazy generator — transcription runs here
        segments, info = model.transcribe(audio, beam_size=1)
        logger.debug(f"Transcription language: {info.language} ({info.language_probability:.2f})")
        texts = []
        for segment in segments:
            logger.debug(f"Segment [{segment.start:.1f}s → {segment.end:.1f}s]: {segment.text!r}")
            texts.append(segment.text)
        return " ".join(texts).strip()

    def transcribe_file(self, path: Path) -> str:
        model = self._load_model()
        segments, _ = model.transcribe(str(path))
        return " ".join(segment.text for segment in segments).strip()


class OpenAISTT(STTBackend):
    """OpenAI Whisper API for cloud transcription."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import tempfile
        import wave

        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            # Convert to int16
            audio_int = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int.tobytes())

        try:
            return self.transcribe_file(temp_path)
        finally:
            temp_path.unlink()

    def transcribe_file(self, path: Path) -> str:
        client = self._get_client()
        logger.info("☁️ Sending audio to OpenAI Cloud Whisper...")
        with open(path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",  # Force English transcription
            )
        logger.info("☁️ OpenAI transcription complete")
        return response.text.strip()


def create_stt_backend(config: Config) -> STTBackend:
    """Create STT backend based on configuration."""
    backend = config.stt_backend.lower()

    if backend == "mlx-whisper":
        logger.info(f"Using MLX Whisper STT (model: {config.whisper_model})")
        return MlxWhisperSTT(model_name=config.whisper_model)
    elif backend == "whisper":
        logger.info(f"Using local Whisper STT (model: {config.whisper_model})")
        return WhisperSTT(model_name=config.whisper_model)
    elif backend == "faster-whisper":
        logger.info(f"Using local Faster-Whisper STT (model: {config.whisper_model})")
        return FasterWhisperSTT(model_name=config.whisper_model)
    elif backend == "openai":
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI STT backend")
        logger.info("☁️ Using OpenAI Cloud Whisper STT")
        return OpenAISTT(api_key=config.openai_api_key)
    else:
        raise ValueError(f"Unknown STT backend: {backend}")
