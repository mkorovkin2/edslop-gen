"""Node: Generate detailed 100-200 word scripts for selected variants using web search."""

import json
import logging

from llm import invoke_with_web_search
from persistence import save_thoughts
from prompts import (
    GENERATE_SCRIPT_JUDGE_FEEDBACK,
    GENERATE_SCRIPT_SYSTEM,
    GENERATE_SCRIPT_USER,
)
from state import AgentState


def generate_scripts(state: AgentState) -> dict:
    """Generate a detailed script for each selected variant, using web-search-grounded LLM."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_scripts ===")

    topic = state["topic"]
    variants = state["variants"]
    selected_ids = state["selected_variant_ids"]
    existing_scripts = state.get("scripts", [])
    judge_results = state.get("judge_results", [])

    # Build a map of which scripts need (re)generation
    # Scripts that passed judging don't need regen
    passed_variant_ids = set()
    failed_feedback = {}
    for jr in judge_results:
        if jr.get("passed"):
            passed_variant_ids.add(jr["variant_id"])
        else:
            failed_feedback[jr["variant_id"]] = jr.get("feedback", "")

    # Also check for user revision feedback from user_approve_scripts
    revision_feedback = {}
    for s in existing_scripts:
        if s.get("revision_feedback"):
            revision_feedback[s["variant_id"]] = s["revision_feedback"]

    # Find existing script text for failed scripts (to include as context)
    existing_script_map = {s["variant_id"]: s for s in existing_scripts}

    # Get selected variants
    selected_variants = [v for v in variants if v["id"] in selected_ids]

    new_scripts = []

    for variant in selected_variants:
        vid = variant["id"]

        # Skip if already passed judging and no revision feedback
        if vid in passed_variant_ids and vid not in revision_feedback:
            existing = existing_script_map.get(vid)
            if existing:
                log.info(f"Variant {vid} already passed judging, keeping existing script")
                new_scripts.append(existing)
                continue

        # Build judge feedback section if this is a regen
        judge_feedback_section = ""
        if vid in failed_feedback and failed_feedback[vid]:
            prev_script = existing_script_map.get(vid, {}).get("script_text", "")
            judge_feedback_section = GENERATE_SCRIPT_JUDGE_FEEDBACK.format(
                feedback=failed_feedback[vid],
                previous_script=prev_script,
            )
            log.info(f"Regenerating variant {vid} with judge feedback")
        elif vid in revision_feedback:
            prev_script = existing_script_map.get(vid, {}).get("script_text", "")
            judge_feedback_section = GENERATE_SCRIPT_JUDGE_FEEDBACK.format(
                feedback=revision_feedback[vid],
                previous_script=prev_script,
            )
            log.info(f"Regenerating variant {vid} with user revision feedback")
        else:
            log.info(f"Generating fresh script for variant {vid}: {variant['title']}")

        user_prompt = GENERATE_SCRIPT_USER.format(
            topic=topic,
            variant_title=variant["title"],
            variant_description=variant["description"],
            judge_feedback_section=judge_feedback_section,
        )

        log.debug(f"LLM+WebSearch call for variant {vid} — prompt length: {len(user_prompt)} chars")
        log.info(f"Searching the web to research variant {vid}: {variant['title']}...")

        raw = invoke_with_web_search(GENERATE_SCRIPT_SYSTEM, user_prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        script_data = json.loads(raw)
        script_data["variant_id"] = vid
        # Remove any leftover revision feedback
        script_data.pop("revision_feedback", None)

        log.info(
            f"Script for variant {vid}: {script_data['title']} "
            f"({script_data['word_count']} words) [web-search grounded]"
        )
        new_scripts.append(script_data)

    new_state = {
        "scripts": new_scripts,
        "judge_results": [],  # Clear previous judge results for fresh evaluation
        "current_step": "generate_scripts",
    }

    save_thoughts(sid, "04_generate_scripts", {**state, **new_state})
    log.debug("Saved thoughts for generate_scripts")

    return new_state
