"""Entry point — starts the video generation agent with optional crash recovery."""

import argparse
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph
from logger import setup_logger
from persistence import load_latest_thoughts


def create_initial_state(session_id: str) -> dict:
    """Create a fresh initial state."""
    return {
        "session_id": session_id,
        "topic": "",
        "variants": [],
        "selected_variant_ids": [],
        "variant_feedback": None,
        "scripts": [],
        "judge_results": [],
        "judge_iteration": 0,
        "approved_scripts": [],
        "audio_paths": [],
        "visual_scripts": [],
        "visual_judge_results": [],
        "visual_judge_iteration": 0,
        "approved_visual_scripts": [],
        "video_breakdown": [],
        "video_paths": [],
        "current_step": "start",
    }


# Map current_step values to the graph node to resume from
STEP_TO_NODE = {
    "start": "get_topic",
    "get_topic": "generate_variants",
    "generate_variants": "user_select_variants",
    "user_select_variants_regen": "generate_variants",
    "user_select_variants_done": "generate_scripts",
    "generate_scripts": "judge_scripts",
    "judge_scripts": "generate_scripts",  # If crashed during judging, re-judge
    "user_approve_scripts_done": "generate_audio",
    "user_approve_scripts_revise": "generate_scripts",
    "generate_audio": "generate_visual_scripts",
    "generate_visual_scripts": "judge_visual_scripts",
    "judge_visual_scripts": "generate_visual_scripts",  # Re-judge → regen
    "user_approve_visuals_done": "generate_video_breakdown",
    "user_approve_visuals_revise": "generate_visual_scripts",
    "generate_video_breakdown": "user_approve_breakdown",
    "user_approve_breakdown_done": "generate_videos",
    "user_approve_breakdown_revise": "generate_video_breakdown",
    "generate_videos": "generate_videos",  # Retry on crash
}


def main():
    parser = argparse.ArgumentParser(description="Educational Video Script Generator")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Session ID to resume from a previous run",
    )
    args = parser.parse_args()

    # Determine session ID
    if args.resume:
        session_id = args.resume
        print(f"\nResuming session: {session_id}")
    else:
        session_id = uuid.uuid4().hex[:12]
        print(f"\nNew session: {session_id}")

    # Setup logging
    log = setup_logger(session_id)
    log.info(f"Session started: {session_id}")

    # Build or resume state
    if args.resume:
        saved = load_latest_thoughts(session_id)
        if saved:
            state = saved
            # Ensure session_id is set correctly
            state["session_id"] = session_id
            resume_step = state.get("current_step", "start")
            log.info(f"Loaded saved state from step: {resume_step}")
            print(f"Resuming from step: {resume_step}")

            # Determine which node to resume from
            resume_node = STEP_TO_NODE.get(resume_step, "get_topic")
            log.info(f"Will resume at graph node: {resume_node}")
        else:
            log.warning("No saved state found, starting fresh")
            print("No saved state found. Starting fresh.")
            state = create_initial_state(session_id)
            resume_node = None
    else:
        state = create_initial_state(session_id)
        resume_node = None

    # Build and run graph
    graph = build_graph()

    try:
        if resume_node and resume_node != "get_topic":
            # For resume, we invoke the graph starting from the appropriate node
            # LangGraph doesn't natively support mid-graph resume, so we use
            # update_state to set state and then invoke from a specific node
            config = {"configurable": {"thread_id": session_id}}

            # We need to use a checkpointer for mid-graph resume.
            # Since we're using a simple compiled graph without a checkpointer,
            # the simplest approach is to run the remaining nodes manually.
            # Instead, let's just re-run the graph from the start with our loaded state.
            # The nodes are designed to be idempotent — get_topic skips if topic exists.
            log.info("Running graph from start with loaded state (nodes are idempotent)")
            result = graph.invoke(state)
        else:
            result = graph.invoke(state)

        log.info("Graph execution completed successfully")

    except KeyboardInterrupt:
        log.warning("Session interrupted by user")
        print("\n\nSession interrupted. Resume with:")
        print(f"  python main.py --resume {session_id}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error during execution: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"\nYou can resume this session with:")
        print(f"  python main.py --resume {session_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
