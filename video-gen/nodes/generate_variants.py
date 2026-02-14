"""Node: Generate 8 high-level script variants using LLM with web search."""

import json
import logging

from llm import invoke_with_web_search
from persistence import save_thoughts
from prompts import (
    GENERATE_VARIANTS_FEEDBACK_SECTION,
    GENERATE_VARIANTS_SYSTEM,
    GENERATE_VARIANTS_USER,
)
from state import AgentState


def generate_variants(state: AgentState) -> dict:
    """Generate 8 variant ideas for the user's topic using web-search-grounded LLM."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_variants ===")

    topic = state["topic"]
    feedback = state.get("variant_feedback")
    previous_variants = state.get("variants", [])

    # Build feedback section if regenerating
    feedback_section = ""
    if feedback and previous_variants:
        prev_display = json.dumps(previous_variants, indent=2)
        feedback_section = GENERATE_VARIANTS_FEEDBACK_SECTION.format(
            feedback=feedback,
            previous_variants=prev_display,
        )
        log.info(f"Regenerating variants with feedback: {feedback}")
    else:
        log.info(f"Generating fresh variants for topic: {topic}")

    user_prompt = GENERATE_VARIANTS_USER.format(
        topic=topic,
        feedback_section=feedback_section,
    )

    log.debug(f"LLM+WebSearch call — prompt length: {len(user_prompt)} chars")
    log.info("Searching the web for topic research before generating variants...")

    raw = invoke_with_web_search(GENERATE_VARIANTS_SYSTEM, user_prompt)

    # Handle markdown code blocks
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    variants = json.loads(raw)
    log.info(f"Generated {len(variants)} variants (web-search grounded)")
    log.debug(f"Variants: {json.dumps(variants, indent=2)}")

    new_state = {
        "variants": variants,
        "variant_feedback": None,  # Clear feedback after use
        "current_step": "generate_variants",
    }

    save_thoughts(sid, "02_generate_variants", {**state, **new_state})
    log.debug("Saved thoughts for generate_variants")

    return new_state
