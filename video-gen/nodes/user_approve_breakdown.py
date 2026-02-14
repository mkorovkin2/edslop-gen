"""Node: User reviews per-segment Sora prompt breakdown and approves or requests revisions."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from persistence import save_thoughts
from prompts import PARSE_BREAKDOWN_APPROVAL_SYSTEM, PARSE_BREAKDOWN_APPROVAL_USER
from state import AgentState


def user_approve_breakdown(state: AgentState) -> dict:
    """Display the full Sora prompt breakdown and let user approve or request revisions."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: user_approve_breakdown ===")

    # Skip if breakdown already approved
    if state.get("current_step") in ("user_approve_breakdown_done", "generate_videos"):
        log.info("Breakdown already approved, skipping approval")
        return {}

    breakdowns = state["video_breakdown"]

    # Display breakdown to user
    print("\n" + "=" * 60)
    print("VIDEO PRODUCTION BREAKDOWN — SORA PROMPTS")
    print("=" * 60)

    for bd in breakdowns:
        vid = bd["variant_id"]
        title = bd.get("variant_title", f"Variant {vid}")
        segments = bd.get("segments", [])

        print(f"\n{'=' * 50}")
        print(f"Variant {vid}: {title}")
        print(f"Segments: {len(segments)}")
        print(f"{'=' * 50}")

        for seg in segments:
            seg_id = seg.get("segment_id", "?")
            duration = seg.get("duration", "?")
            size = seg.get("size", "?")
            model = seg.get("model", "?")
            filename = seg.get("filename", "?")

            print(f"\n  --- Segment {seg_id} ---")
            print(f"  File: {filename} | Duration: {duration}s | Size: {size} | Model: {model}")
            print(f"  Sora Prompt: {seg.get('sora_prompt', '')}")
            print(f"  Rationale: {seg.get('rationale', '')}")

    print("\n" + "=" * 60)
    print("Approve this breakdown? These exact prompts will be sent to Sora.")
    print("(e.g., 'Looks good!' or 'Segment 3 of variant 2 needs more detail')")
    print("=" * 60 + "\n")

    user_response = input("> ").strip()
    while not user_response:
        print("Please provide your feedback:")
        user_response = input("> ").strip()

    log.info(f"User response: {user_response}")

    # Build display for LLM parsing
    breakdown_display = ""
    for bd in breakdowns:
        vid = bd["variant_id"]
        title = bd.get("variant_title", f"Variant {vid}")
        breakdown_display += f"\n[Variant {vid}] {title}:\n"
        for seg in bd.get("segments", []):
            breakdown_display += (
                f"  Segment {seg.get('segment_id')}: \"{seg.get('sora_prompt', '')}\" "
                f"({seg.get('duration', '?')}s)\n"
            )

    parse_prompt = PARSE_BREAKDOWN_APPROVAL_USER.format(
        breakdown_display=breakdown_display,
        user_response=user_response,
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=PARSE_BREAKDOWN_APPROVAL_SYSTEM),
        HumanMessage(content=parse_prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    parsed = json.loads(raw)
    log.info(f"Parsed breakdown approval intent: {json.dumps(parsed)}")

    if parsed["action"] == "approve":
        new_state = {
            "current_step": "user_approve_breakdown_done",
        }
        log.info("User approved video breakdown")
        print("\nBreakdown approved! Proceeding to video generation...\n")
    else:
        revision_feedback = parsed.get("revision_feedback", {})
        updated_breakdowns = []
        for bd in breakdowns:
            vid_str = str(bd["variant_id"])
            if vid_str in revision_feedback:
                bd_copy = dict(bd)
                bd_copy["revision_feedback"] = revision_feedback[vid_str]
                updated_breakdowns.append(bd_copy)
                log.info(f"Breakdown variant {vid_str} needs revision: {revision_feedback[vid_str]}")
            else:
                updated_breakdowns.append(bd)

        new_state = {
            "video_breakdown": updated_breakdowns,
            "current_step": "user_approve_breakdown_revise",
        }
        log.info("User requested breakdown revisions")
        print("\nRevising breakdown based on your feedback...\n")

    save_thoughts(sid, "12_user_approve_breakdown", {**state, **new_state})
    log.debug("Saved thoughts for user_approve_breakdown")

    return new_state
