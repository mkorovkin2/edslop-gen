"""Node: User reviews visual cue scripts and approves or requests revisions."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from persistence import save_thoughts
from prompts import PARSE_VISUAL_APPROVAL_SYSTEM, PARSE_VISUAL_APPROVAL_USER
from state import AgentState


def user_approve_visuals(state: AgentState) -> dict:
    """Display visual cue scripts and let user approve or request revisions."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: user_approve_visuals ===")

    # Skip if we're past this phase (visual scripts already approved)
    if state.get("approved_visual_scripts"):
        log.info("Visual scripts already approved, skipping approval")
        return {}

    visual_scripts = state["visual_scripts"]

    # Display visual scripts to user
    print("\n" + "=" * 60)
    print("VISUAL CUE SCRIPTS FOR REVIEW")
    print("=" * 60)

    for vs in visual_scripts:
        vid = vs["variant_id"]
        title = vs.get("variant_title", f"Variant {vid}")
        total_dur = vs.get("total_duration_seconds", 0)
        segments = vs.get("segments", [])

        print(f"\n{'=' * 50}")
        print(f"Variant {vid}: {title}")
        print(f"Total Duration: {total_dur}s | Segments: {len(segments)}")
        print(f"{'=' * 50}")

        for seg in segments:
            sid_num = seg.get("segment_id", "?")
            time_range = seg.get("time_range", "?")
            dur = seg.get("duration_seconds", "?")
            print(f"\n  Segment {sid_num} [{time_range}] ({dur}s)")
            print(f"  Visual: {seg.get('visual_description', '')}")
            print(f"  Mood: {seg.get('mood', '')} | Camera: {seg.get('camera', '')}")
            print(f"  Transition: {seg.get('transition', '')}")
            print(f"  Script overlay: \"{seg.get('script_text_overlay', '')}\"")

    print("\n" + "=" * 60)
    print("Approve all visual scripts? Or request changes to specific ones.")
    print("(e.g., 'Looks good!' or 'Variant 2 needs more dynamic camera work')")
    print("=" * 60 + "\n")

    user_response = input("> ").strip()
    while not user_response:
        print("Please provide your feedback:")
        user_response = input("> ").strip()

    log.info(f"User response: {user_response}")

    # Build display for LLM parsing
    visuals_display = ""
    for vs in visual_scripts:
        vid = vs["variant_id"]
        title = vs.get("variant_title", f"Variant {vid}")
        visuals_display += f"\n[Variant {vid}] {title}:\n"
        for seg in vs.get("segments", []):
            visuals_display += (
                f"  Segment {seg.get('segment_id')}: {seg.get('visual_description', '')} "
                f"({seg.get('duration_seconds', '?')}s, {seg.get('camera', '')})\n"
            )

    parse_prompt = PARSE_VISUAL_APPROVAL_USER.format(
        visuals_display=visuals_display,
        user_response=user_response,
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=PARSE_VISUAL_APPROVAL_SYSTEM),
        HumanMessage(content=parse_prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    parsed = json.loads(raw)
    log.info(f"Parsed visual approval intent: {json.dumps(parsed)}")

    if parsed["action"] == "approve":
        new_state = {
            "approved_visual_scripts": visual_scripts,
            "current_step": "user_approve_visuals_done",
        }
        log.info("User approved all visual scripts")
        print("\nVisual scripts approved! Proceeding to video breakdown...\n")
    else:
        revision_feedback = parsed.get("revision_feedback", {})
        updated_visuals = []
        for vs in visual_scripts:
            vid_str = str(vs["variant_id"])
            if vid_str in revision_feedback:
                vs_copy = dict(vs)
                vs_copy["revision_feedback"] = revision_feedback[vid_str]
                updated_visuals.append(vs_copy)
                log.info(f"Visual variant {vid_str} needs revision: {revision_feedback[vid_str]}")
            else:
                updated_visuals.append(vs)

        new_state = {
            "visual_scripts": updated_visuals,
            "visual_judge_results": [],
            "visual_judge_iteration": 0,
            "current_step": "user_approve_visuals_revise",
        }
        log.info("User requested visual script revisions")
        print("\nRevising visual scripts based on your feedback...\n")

    save_thoughts(sid, "10_user_approve_visuals", {**state, **new_state})
    log.debug("Saved thoughts for user_approve_visuals")

    return new_state
