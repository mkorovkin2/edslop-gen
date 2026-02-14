"""Node: Prompt user for a video topic."""

import logging

from persistence import save_thoughts
from state import AgentState

logger = logging.getLogger("video_agent")


def get_topic(state: AgentState) -> dict:
    """Ask the user for a 1-2 sentence topic description."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: get_topic ===")

    # If resuming and topic already exists, skip
    if state.get("topic"):
        log.info(f"Resuming with existing topic: {state['topic']}")
        return {}

    print("\n" + "=" * 60)
    print("EDUCATIONAL VIDEO SCRIPT GENERATOR")
    print("=" * 60)
    print("\nWhat topic would you like to create educational videos about?")
    print("(Provide a 1-2 sentence description)\n")

    topic = input("> ").strip()
    while not topic:
        print("Please enter a topic:")
        topic = input("> ").strip()

    log.info(f"User topic: {topic}")

    new_state = {
        "topic": topic,
        "current_step": "get_topic",
    }

    save_thoughts(sid, "01_get_topic", {**state, **new_state})
    log.debug("Saved thoughts for get_topic")

    return new_state
