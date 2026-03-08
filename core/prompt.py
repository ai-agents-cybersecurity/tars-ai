"""
TARS Personality Prompt Engineering

Builds the system prompt that makes the LLM behave like TARS from Interstellar.
The prompt is dynamically adjusted based on personality knob settings.
"""

from config.settings import TARSPersonality, _load_prompts


def build_tars_system_prompt(personality: TARSPersonality) -> str:
    """
    Construct TARS's system prompt with dynamic personality injection.
    """
    prompts = _load_prompts()
    template = prompts["system_prompt"]

    return template.format(
        personality_summary=personality.summary(),
        personality_directives=personality.to_prompt_block(),
    )
