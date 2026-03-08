"""
TARS-AI Configuration — MLX Native on Apple Silicon

All inference runs locally via MLX. No cloud APIs, no API keys.
LLM: mlx-lm (GPT-OSS 120B)    |    TTS: mlx-audio (Qwen3-TTS Voice Design)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.yaml"

def _load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return yaml.safe_load(f)


# ── TARS Personality Knobs ───────────────────────────────────────────────

@dataclass
class TARSPersonality:
    """
    Adjustable personality parameters, just like in the movie.
    Cooper: "Hey TARS, what's your honesty parameter?"
    TARS: "90 percent."
    Cooper: "90? ... Adjust it to 95."
    """
    humor: int = 75          # 0-100: dry wit / sarcasm level
    honesty: int = 90        # 0-100: bluntness vs diplomacy
    discretion: int = 85     # 0-100: information sharing restraint
    trust: int = 70          # 0-100: willingness to follow unusual orders

    def __post_init__(self):
        for attr in ("humor", "honesty", "discretion", "trust"):
            val = getattr(self, attr)
            if not 0 <= val <= 100:
                raise ValueError(f"{attr} must be 0-100, got {val}")

    def summary(self) -> str:
        return (
            f"Humor: {self.humor}% | Honesty: {self.honesty}% | "
            f"Discretion: {self.discretion}% | Trust: {self.trust}%"
        )

    def to_prompt_block(self) -> str:
        """Convert personality settings into system prompt directives."""
        scales = _load_prompts()["personality_scales"]
        blocks = []
        for attr in ("humor", "honesty", "discretion"):
            value = getattr(self, attr)
            for tier in scales[attr]:
                if value >= tier["min"]:
                    blocks.append(tier["directive"])
                    break
        return "\n".join(blocks)


# ── MLX Configuration ───────────────────────────────────────────────────

@dataclass
class MLXConfig:
    """All-local MLX inference settings for Apple Silicon."""

    # ── LLM (GPT-OSS 120B MoE — 117B total, 5.1B active per token) ──
    llm_model: str = "mlx-community/gpt-oss-120b-4bit"
    llm_max_tokens: int = 768
    llm_temperature: float = 0.7
    llm_repetition_penalty: float = 1.1
    llm_top_p: float = 0.9

    # ── MLX Server (OpenAI-compatible local inference) ──
    server_url: str = "http://localhost:8080/v1"

    # ── TTS (Qwen3-TTS Voice Design via mlx-audio) ──
    tts_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    tts_voice_desc: str = "A deep, measured, authoritative American male voice. Military precision. Deliberate pauses."
    tts_speed: float = 1.0
    tts_sample_rate: int = 24000


# ── Master Config ────────────────────────────────────────────────────────

@dataclass
class TARSConfig:
    personality: TARSPersonality = field(default_factory=TARSPersonality)
    mlx: MLXConfig = field(default_factory=MLXConfig)
    use_server: bool = False          # Use MLX server instead of direct mlx-lm
    audio_output_dir: str = "audio_output"
    play_audio: bool = True           # Auto-play generated audio
    save_audio: bool = True           # Keep audio files
    conversation_memory_k: int = 20   # Messages to keep in context
    multi_character_max_tokens: int = 256  # Shorter responses for guest characters

    def validate(self):
        """Sanity-check config before launch."""
        # Check that we're on macOS / Apple Silicon (MLX requirement)
        import platform
        if platform.system() != "Darwin":
            raise EnvironmentError(
                "TARS-AI MLX mode requires macOS on Apple Silicon (M1/M2/M3/M4). "
                f"Detected: {platform.system()} {platform.machine()}"
            )
