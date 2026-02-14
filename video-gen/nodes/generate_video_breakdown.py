"""Node: Generate production-ready per-segment Sora prompts from approved visual scripts."""

import json
import logging
import os

from llm import invoke_with_web_search
from persistence import save_thoughts
from prompts import (
    GENERATE_VIDEO_BREAKDOWN_SYSTEM,
    GENERATE_VIDEO_BREAKDOWN_USER,
)
from state import AgentState


def generate_video_breakdown(state: AgentState) -> dict:
    """Generate detailed Sora prompts for each segment of each approved visual script."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_video_breakdown ===")

    # Skip if breakdown already approved
    if state.get("current_step") in ("user_approve_breakdown_done", "generate_videos"):
        log.info("Breakdown already approved, skipping generation")
        return {}

    topic = state["topic"]
    approved_visuals = state["approved_visual_scripts"]
    sora_model = os.getenv("SORA_MODEL", "sora-2")

    # Check for revision feedback on existing breakdown
    existing_breakdown = state.get("video_breakdown", [])
    revision_feedback = {}
    for bd in existing_breakdown:
        if bd.get("revision_feedback"):
            revision_feedback[bd["variant_id"]] = bd["revision_feedback"]

    breakdowns = []

    print("\n" + "=" * 60)
    print("GENERATING VIDEO BREAKDOWN (Sora Prompts)")
    print("=" * 60)

    for vs in approved_visuals:
        vid = vs["variant_id"]
        title = vs.get("variant_title", f"Variant {vid}")

        # Build feedback section
        feedback_section = ""
        if vid in revision_feedback:
            feedback_section = (
                f"IMPORTANT — The user provided feedback on a previous breakdown:\n"
                f'"{revision_feedback[vid]}"\n\n'
                f"Address their feedback in your new version."
            )
            log.info(f"Regenerating breakdown for variant {vid} with user feedback")
        else:
            log.info(f"Generating breakdown for variant {vid}: {title}")

        visual_segments_json = json.dumps(vs.get("segments", []), indent=2)

        user_prompt = GENERATE_VIDEO_BREAKDOWN_USER.format(
            topic=topic,
            variant_id=vid,
            variant_title=title,
            visual_segments_json=visual_segments_json,
            feedback_section=feedback_section,
            duration_hint="4",
            sora_model=sora_model,
        )

        print(f"  Generating Sora prompts for: {title}...")
        log.info(f"Researching visual references for variant {vid} breakdown...")

        raw = invoke_with_web_search(GENERATE_VIDEO_BREAKDOWN_SYSTEM, user_prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        breakdown_data = json.loads(raw)
        breakdown_data["variant_id"] = vid
        breakdown_data.pop("revision_feedback", None)

        num_segments = len(breakdown_data.get("segments", []))
        log.info(f"Breakdown for variant {vid}: {num_segments} Sora prompts generated")
        breakdowns.append(breakdown_data)

    new_state = {
        "video_breakdown": breakdowns,
        "current_step": "generate_video_breakdown",
    }

    save_thoughts(sid, "11_generate_video_breakdown", {**state, **new_state})
    log.debug("Saved thoughts for generate_video_breakdown")

    return new_state
