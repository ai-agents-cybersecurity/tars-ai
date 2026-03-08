# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TARS-AI is a fully local, Apple Silicon-only conversational AI agent embodying TARS from Interstellar. It uses MLX for both LLM inference (mlx-lm or a local MLX server) and text-to-speech (mlx-audio Kokoro), with LangGraph for stateful conversation management. No cloud dependencies.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with defaults
python main.py

# Run text-only (no TTS)
python main.py --no-voice

# Run with custom personality
python main.py --humor 100 --honesty 95

# Run with a different model
python main.py --llm mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Run with different voice/speed
python main.py --voice bm_george --speed 0.88

# Run using local MLX server (instead of direct mlx-lm)
python main.py --server

# Run with MLX server at a custom URL
python main.py --server --server-url http://localhost:9090/v1

# Or use env vars
USE_MLX_SERVER=1 python main.py
MLX_SERVER_URL=http://localhost:9090/v1 USE_MLX_SERVER=1 python main.py
```

There are no test or lint commands configured.

## Architecture

```
main.py                  CLI entry point, boot sequence, conversation loop
config/settings.py       Three dataclasses: TARSPersonality, MLXConfig, TARSConfig
core/agent.py            TARSAgent (LangGraph graph) + MLXLLM (mlx-lm) / ServerLLM (OpenAI-compat server)
core/prompt.py           build_tars_system_prompt() - dynamic system prompt from personality knobs
tts/engine.py            TARSVoice - Kokoro TTS synthesis, playback, audio file management
```

### Data Flow

User input → `TARSAgent.chat()` → settings command interception (fast path) or LangGraph invoke → `MLXLLM.generate()` (direct mlx-lm) or `ServerLLM.generate()` (local MLX server via OpenAI SDK) → response text to console → `TARSVoice.speak()` (if enabled) → synthesize WAV → play via sounddevice/afplay

### Key Patterns

- **Personality-as-prompt**: Four knobs (humor, honesty, discretion, trust, each 0-100) are converted to prompt directives via `TARSPersonality.to_prompt_block()` and injected into the system prompt. Adjustable at runtime via natural language commands intercepted in `_handle_settings_command()`.
- **Lazy TTS loading**: Kokoro model loads on first `speak()` call, not at startup (~164 MB saved until needed).
- **Conversation memory**: Full message history kept in LangGraph state, trimmed to last K messages (default 20) before each inference. System prompt at index 0 is always preserved.
- **Settings fast path**: Commands like "Set humor to 50" or "what's your settings?" are intercepted before LLM inference in `_handle_settings_command()`.

### Platform Constraint

macOS on Apple Silicon only. `TARSConfig.validate()` enforces this. All ML inference runs through Apple's MLX framework.

### Default Models (auto-downloaded from HuggingFace on first run)

- **LLM**: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit (~4.5 GB)
- **TTS**: mlx-community/Kokoro-82M-bf16 (~164 MB)
