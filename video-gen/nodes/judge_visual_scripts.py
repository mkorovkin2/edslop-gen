"""Node: LLM-as-a-judge evaluates visual cue scripts with fresh context."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_judge_llm
from persistence import save_thoughts
from prompts import JUDGE_VISUAL_SCRIPT_SYSTEM, JUDGE_VISUAL_SCRIPT_USER
from state import AgentState

MAX_VISUAL_JUDGE_ITERATIONS = 5


def judge_visual_scripts(state: AgentState) -> dict:
    """Evaluate each visual cue script using LLM-as-a-judge with completely fresh context."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: judge_visual_scripts ===")

    # Skip if we're past this phase (visual scripts already approved)
    if state.get("approved_visual_scripts"):
        log.info("Visual scripts already approved, skipping judging")
        return {}

    topic = state["topic"]
    visual_scripts = state["visual_scripts"]
    approved_scripts = state["approved_scripts"]
    iteration = state.get("visual_judge_iteration", 0) + 1

    log.info(f"Visual judge iteration: {iteration}/{MAX_VISUAL_JUDGE_ITERATIONS}")

    # Build script text lookup
    script_text_map = {s["variant_id"]: s["script_text"] for s in approved_scripts}

    judge_llm = get_judge_llm()

    results = []
    all_passed = True

    for vs in visual_scripts:
        vid = vs["variant_id"]

        # Skip already-passed
        prev_results = state.get("visual_judge_results", [])
        already_passed = any(
            vr["variant_id"] == vid and vr.get("passed") for vr in prev_results
        )
        if already_passed:
            existing = next(vr for vr in prev_results if vr["variant_id"] == vid)
            results.append(existing)
            log.info(f"Visual script for variant {vid} already passed, skipping")
            continue

        script_text = script_text_map.get(vid, "")
        visual_script_json = json.dumps(vs, indent=2)

        judge_prompt = JUDGE_VISUAL_SCRIPT_USER.format(
            topic=topic,
            variant_title=vs.get("variant_title", f"Variant {vid}"),
            script_text=script_text,
            visual_script_json=visual_script_json,
        )

        log.debug(f"Judging visual script for variant {vid}")

        response = judge_llm.invoke([
            SystemMessage(content=JUDGE_VISUAL_SCRIPT_SYSTEM),
            HumanMessage(content=judge_prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        judge_result = json.loads(raw)
        judge_result["variant_id"] = vid

        scores = judge_result["scores"]
        avg = judge_result["average_score"]
        passed = judge_result["passed"]

        log.info(f"Variant {vid} visual — scores: {scores}, avg: {avg}, passed: {passed}")
        if not passed:
            log.info(f"Variant {vid} visual feedback: {judge_result['feedback']}")
            all_passed = False

        results.append(judge_result)

    # Force-pass after max iterations
    if not all_passed and iteration >= MAX_VISUAL_JUDGE_ITERATIONS:
        log.warning(
            f"Hit max visual judge iterations ({MAX_VISUAL_JUDGE_ITERATIONS}). "
            "Accepting current visual scripts as best effort."
        )
        for r in results:
            if not r["passed"]:
                r["passed"] = True
                r["feedback"] = f"[Auto-passed after {MAX_VISUAL_JUDGE_ITERATIONS} iterations]"
        all_passed = True

    new_state = {
        "visual_judge_results": results,
        "visual_judge_iteration": iteration,
        "current_step": "judge_visual_scripts",
    }

    save_thoughts(sid, f"09_judge_visual_scripts_iter{iteration}", {**state, **new_state})
    log.debug("Saved thoughts for judge_visual_scripts")

    # Print summary
    print("\n" + "-" * 40)
    print(f"Visual Quality Check (Round {iteration})")
    print("-" * 40)
    for r in results:
        status = "PASS" if r["passed"] else "NEEDS IMPROVEMENT"
        print(f"  Variant {r['variant_id']}: {status} (avg: {r['average_score']})")
    print("-" * 40)

    return new_state
