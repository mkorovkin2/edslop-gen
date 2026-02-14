"""Node: User reviews final scripts and approves or requests revisions."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_llm
from persistence import save_thoughts
from prompts import PARSE_USER_APPROVAL_SYSTEM, PARSE_USER_APPROVAL_USER
from state import AgentState


def user_approve_scripts(state: AgentState) -> dict:
    """Display final scripts and let user approve or request revisions."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: user_approve_scripts ===")

    scripts = state["scripts"]

    # Display scripts to user
    print("\n" + "=" * 60)
    print("FINAL SCRIPTS FOR REVIEW")
    print("=" * 60)

    for script in scripts:
        print(f"\n--- Variant {script['variant_id']}: {script['title']} ---")
        print(f"({script['word_count']} words)\n")
        print(script["script_text"])
        print()

    print("=" * 60)
    print("Approve all scripts? Or request changes to specific ones.")
    print("(e.g., 'Looks good!' or 'Script 2 needs a stronger hook, script 4 is too technical')")
    print("=" * 60 + "\n")

    user_response = input("> ").strip()
    while not user_response:
        print("Please provide your feedback:")
        user_response = input("> ").strip()

    log.info(f"User response: {user_response}")

    # Use LLM to parse user's approval/revision intent
    scripts_display = "\n\n".join(
        f"[Variant {s['variant_id']}] {s['title']}:\n{s['script_text']}"
        for s in scripts
    )
    parse_prompt = PARSE_USER_APPROVAL_USER.format(
        scripts_display=scripts_display,
        user_response=user_response,
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=PARSE_USER_APPROVAL_SYSTEM),
        HumanMessage(content=parse_prompt),
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    parsed = json.loads(raw)
    log.info(f"Parsed approval intent: {json.dumps(parsed)}")

    if parsed["action"] == "approve":
        new_state = {
            "approved_scripts": scripts,
            "current_step": "user_approve_scripts_done",
        }
        log.info("User approved all scripts")
        print("\nScripts approved! Proceeding to audio generation...\n")
    else:
        # Inject revision feedback into the scripts that need changes
        revision_feedback = parsed.get("revision_feedback", {})
        updated_scripts = []
        for s in scripts:
            vid_str = str(s["variant_id"])
            if vid_str in revision_feedback:
                s_copy = dict(s)
                s_copy["revision_feedback"] = revision_feedback[vid_str]
                updated_scripts.append(s_copy)
                log.info(f"Variant {vid_str} needs revision: {revision_feedback[vid_str]}")
            else:
                updated_scripts.append(s)

        new_state = {
            "scripts": updated_scripts,
            "judge_results": [],  # Reset judge results for new cycle
            "judge_iteration": 0,  # Reset judge iteration counter
            "current_step": "user_approve_scripts_revise",
        }
        log.info("User requested script revisions")
        print("\nRevising scripts based on your feedback...\n")

    save_thoughts(sid, "06_user_approve_scripts", {**state, **new_state})
    log.debug("Saved thoughts for user_approve_scripts")

    return new_state
