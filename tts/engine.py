"""
TARS Voice Synthesis Engine — MLX Native (Qwen3-TTS Voice Design)

Local TTS on Apple Silicon via mlx-audio.
Model: Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 — text-described voice design.
Voice: Generated from text descriptions (e.g. "deep authoritative American male").

No cloud APIs. No latency. No API keys.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import TARSConfig

_SENTINEL = None  # poison pill to stop the playback thread


class TARSVoice:
    """
    TARS voice synthesizer — fully local via mlx-audio Qwen3-TTS Voice Design.

    Qwen3-TTS generates voices from text descriptions instead of fixed presets.
    Each character gets a distinct voice based on their voice_desc string.
    Audio plays in a background thread so the prompt returns immediately.
    """

    def __init__(self, config: TARSConfig):
        self.config = config
        self.model = None
        self._model_loaded = False
        self.output_dir = Path(config.audio_output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._utterance_count = 0

        # Background playback queue + thread
        self._playback_queue: queue.Queue[tuple[Path, bool] | None] = queue.Queue()
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()

    def _ensure_model(self) -> None:
        """Lazy-load the TTS model on first use (saves startup time)."""
        if self._model_loaded:
            return

        import logging
        import warnings
        from mlx_audio.tts.utils import load_model

        # Suppress noisy transformers/huggingface warnings during load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev_levels = {}
            for name in ("transformers", "huggingface_hub", "transformers.modeling_utils"):
                logger = logging.getLogger(name)
                prev_levels[name] = logger.level
                logger.setLevel(logging.ERROR)
            try:
                self.model = load_model(self.config.mlx.tts_model)
            finally:
                for name, level in prev_levels.items():
                    logging.getLogger(name).setLevel(level)

        self._model_loaded = True

    def synthesize(
        self,
        text: str,
        voice_desc: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> Optional[Path]:
        """
        Convert text to voice audio via Qwen3-TTS Voice Design.

        Args:
            text: Text to synthesize.
            voice_desc: Voice description override (defaults to config value).
            speed: TTS speed override (defaults to config value).

        Returns path to the generated WAV file, or None on empty input.
        """
        if not text.strip():
            return None

        self._ensure_model()

        use_desc = voice_desc or self.config.mlx.tts_voice_desc
        use_speed = speed or self.config.mlx.tts_speed

        self._utterance_count += 1

        # Generate audio via Qwen3-TTS Voice Design
        audio_chunks = []
        for result in self.model.generate_voice_design(
            text=text,
            instruct=use_desc,
        ):
            chunk = np.array(result.audio, copy=False)
            if chunk.ndim > 1:
                chunk = chunk.flatten()
            audio_chunks.append(chunk)

        if not audio_chunks:
            return None

        # Concatenate all chunks into a single waveform
        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        # Normalize to [-1, 1] range if needed
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak

        # Save as WAV
        import soundfile as sf

        timestamp = int(time.time())
        filename = f"tts_{timestamp}_{self._utterance_count:04d}.wav"
        filepath = self.output_dir / filename

        sf.write(str(filepath), audio, self.config.mlx.tts_sample_rate)

        return filepath

    def _playback_worker(self) -> None:
        """Background thread: play queued audio files in order."""
        while True:
            item = self._playback_queue.get()
            if item is _SENTINEL:
                break
            audio_path, delete_after = item
            try:
                self._play_blocking(audio_path)
            except Exception as e:
                print(f"  [TARS] Playback error: {e}")
            finally:
                if delete_after:
                    audio_path.unlink(missing_ok=True)
                self._playback_queue.task_done()

    def _play_blocking(self, audio_path: Path) -> None:
        """Play an audio file synchronously (runs inside the worker thread)."""
        if not audio_path.exists():
            return

        try:
            self._play_with_sounddevice(audio_path)
        except Exception:
            self._play_with_system(audio_path)

    def play(self, audio_path: Path, delete_after: bool = False) -> None:
        """
        Enqueue audio for background playback.

        Files play in order without overlap. Returns immediately.
        """
        self._playback_queue.put((audio_path, delete_after))

    def _play_with_sounddevice(self, audio_path: Path) -> None:
        """Play using sounddevice + soundfile (cross-platform)."""
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(str(audio_path))
        sd.play(data, samplerate)
        sd.wait()

    def _play_with_system(self, audio_path: Path) -> None:
        """Fallback: use macOS afplay (primary target platform)."""
        import platform
        import subprocess

        system = platform.system()
        if system == "Darwin":
            subprocess.run(["afplay", str(audio_path)], check=True)
        elif system == "Linux":
            for player in ["mpv", "ffplay", "aplay", "paplay"]:
                try:
                    cmd = [player]
                    if player == "ffplay":
                        cmd.extend(["-nodisp", "-autoexit"])
                    cmd.append(str(audio_path))
                    subprocess.run(cmd, check=True, capture_output=True)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            raise RuntimeError("No audio player found. Install mpv or ffplay.")
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

    def speak(
        self,
        text: str,
        voice_desc: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> Optional[Path]:
        """
        Full pipeline: synthesize -> enqueue for background playback.

        Returns immediately after synthesis. Audio plays in a background thread.
        """
        audio_path = self.synthesize(text, voice_desc=voice_desc, speed=speed)
        if audio_path and self.config.play_audio:
            delete_after = not self.config.save_audio
            self.play(audio_path, delete_after=delete_after)
        elif audio_path and not self.config.save_audio:
            audio_path.unlink(missing_ok=True)
            return None
        return audio_path

    def wait(self) -> None:
        """Block until all queued audio has finished playing."""
        self._playback_queue.join()

    def cleanup(self) -> None:
        """Wait for playback to finish, then remove all generated audio files."""
        self._playback_queue.join()
        for f in self.output_dir.glob("*.wav"):
            f.unlink(missing_ok=True)
