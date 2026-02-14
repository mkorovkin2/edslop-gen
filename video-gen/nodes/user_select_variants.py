"""Node: User selects up to 4 variants or requests regeneration."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from persistence import save_thoughts
from prompts import PARSE_USER_SELECTION_SYSTEM, PARSE_USER_SELECTION_USER
from state import AgentState


def user_select_variants(state: AgentState) -> dict:
    """Display variants and let user select or request regen."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: user_select_variants ===")

    variants = state["variants"]

    # Display variants to user
    print("\n" + "=" * 60)
    print("SCRIPT VARIANTS")
    print("=" * 60)
    for v in variants:
        print(f"\n  [{v['id']}] {v['title']}")
        print(f"      {v['description']}")
    print("\n" + "-" * 60)
    print("Select up to 4 variants (e.g., 'I'll take 1, 3, and 5')")
    print("Or request changes (e.g., 'Redo #2 with more humor, keep the rest')")
    print("-" * 60 + "\n")

    user_response = input("> ").strip()
    while not user_response:
        print("Please provide your selection or feedback:")
        user_response = input("> ").strip()

    log.info(f"User response: {user_response}")

    # Use LLM to parse the user's natural language response
    variants_display = "\n".join(
        f"[{v['id']}] {v['title']}: {v['description']}" for v in variants
    )
    parse_prompt = PARSE_USER_SELECTION_USER.format(
        variants_display=variants_display,
        user_response=user_response,
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=PARSE_USER_SELECTION_SYSTEM),
        HumanMessage(content=parse_prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    parsed = json.loads(raw)
    log.info(f"Parsed user intent: {json.dumps(parsed)}")

    if parsed["action"] == "regenerate":
        new_state = {
            "variant_feedback": parsed["feedback"],
            "current_step": "user_select_variants_regen",
        }
        log.info("User requested variant regeneration")
    else:
        selected = parsed["selected_ids"][:4]  # Cap at 4
        new_state = {
            "selected_variant_ids": selected,
            "variant_feedback": None,
            "current_step": "user_select_variants_done",
        }
        log.info(f"User selected variants: {selected}")

    save_thoughts(sid, "03_user_select_variants", {**state, **new_state})
    log.debug("Saved thoughts for user_select_variants")

    return new_state
