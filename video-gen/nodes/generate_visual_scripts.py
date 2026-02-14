"""Node: Generate detailed visual cue scripts for each approved script using web search."""

import json
import logging

from llm import invoke_with_web_search
from persistence import save_thoughts
from prompts import (
    GENERATE_VISUAL_SCRIPT_JUDGE_FEEDBACK,
    GENERATE_VISUAL_SCRIPT_SYSTEM,
    GENERATE_VISUAL_SCRIPT_USER,
)
from state import AgentState


def generate_visual_scripts(state: AgentState) -> dict:
    """Generate a detailed visual cue script for each approved script, using web-search-grounded LLM."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_visual_scripts ===")

    # Skip if we're past this phase (visual scripts already approved)
    if state.get("approved_visual_scripts"):
        log.info("Visual scripts already approved, skipping generation")
        return {}

    topic = state["topic"]
    approved_scripts = state["approved_scripts"]
    existing_visuals = state.get("visual_scripts", [])
    visual_judge_results = state.get("visual_judge_results", [])

    # Build maps for regen logic
    passed_variant_ids = set()
    failed_feedback = {}
    for vr in visual_judge_results:
        if vr.get("passed"):
            passed_variant_ids.add(vr["variant_id"])
        else:
            failed_feedback[vr["variant_id"]] = vr.get("feedback", "")

    # Check for user revision feedback
    revision_feedback = {}
    for vs in existing_visuals:
        if vs.get("revision_feedback"):
            revision_feedback[vs["variant_id"]] = vs["revision_feedback"]

    existing_visual_map = {vs["variant_id"]: vs for vs in existing_visuals}

    new_visual_scripts = []

    for script in approved_scripts:
        vid = script["variant_id"]

        # Skip if already passed judging and no revision feedback
        if vid in passed_variant_ids and vid not in revision_feedback:
            existing = existing_visual_map.get(vid)
            if existing:
                log.info(f"Visual script for variant {vid} already passed judging, keeping")
                new_visual_scripts.append(existing)
                continue

        # Build feedback section if this is a regen
        feedback_section = ""
        if vid in failed_feedback and failed_feedback[vid]:
            prev_visual = json.dumps(existing_visual_map.get(vid, {}), indent=2)
            feedback_section = GENERATE_VISUAL_SCRIPT_JUDGE_FEEDBACK.format(
                feedback=failed_feedback[vid],
                previous_visual_script=prev_visual,
            )
            log.info(f"Regenerating visual script for variant {vid} with judge feedback")
        elif vid in revision_feedback:
            prev_visual = json.dumps(existing_visual_map.get(vid, {}), indent=2)
            feedback_section = GENERATE_VISUAL_SCRIPT_JUDGE_FEEDBACK.format(
                feedback=revision_feedback[vid],
                previous_visual_script=prev_visual,
            )
            log.info(f"Regenerating visual script for variant {vid} with user feedback")
        else:
            log.info(f"Generating fresh visual script for variant {vid}: {script['title']}")

        user_prompt = GENERATE_VISUAL_SCRIPT_USER.format(
            topic=topic,
            variant_id=vid,
            variant_title=script["title"],
            script_text=script["script_text"],
            feedback_section=feedback_section,
        )

        log.info(f"Researching visual ideas for variant {vid}: {script['title']}...")
        print(f"  Generating visual script for: {script['title']}...")

        raw = invoke_with_web_search(GENERATE_VISUAL_SCRIPT_SYSTEM, user_prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        visual_data = json.loads(raw)
        visual_data["variant_id"] = vid
        # Remove any leftover revision feedback
        visual_data.pop("revision_feedback", None)

        num_segments = len(visual_data.get("segments", []))
        total_dur = visual_data.get("total_duration_seconds", 0)
        log.info(
            f"Visual script for variant {vid}: {num_segments} segments, "
            f"{total_dur}s total [web-search grounded]"
        )
        new_visual_scripts.append(visual_data)

    new_state = {
        "visual_scripts": new_visual_scripts,
        "visual_judge_results": [],  # Clear for fresh evaluation
        "current_step": "generate_visual_scripts",
    }

    save_thoughts(sid, "08_generate_visual_scripts", {**state, **new_state})
    log.debug("Saved thoughts for generate_visual_scripts")

    return new_state
