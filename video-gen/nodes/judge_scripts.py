"""Node: LLM-as-a-judge evaluates scripts with fresh context."""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from llm import get_judge_llm
from persistence import save_thoughts
from prompts import JUDGE_SCRIPT_SYSTEM, JUDGE_SCRIPT_USER
from state import AgentState

MAX_JUDGE_ITERATIONS = 5


def judge_scripts(state: AgentState) -> dict:
    """Evaluate each script using LLM-as-a-judge with completely fresh context."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: judge_scripts ===")

    topic = state["topic"]
    scripts = state["scripts"]
    iteration = state.get("judge_iteration", 0) + 1

    log.info(f"Judge iteration: {iteration}/{MAX_JUDGE_ITERATIONS}")

    # Use a separate LLM instance for unbiased judging
    judge_llm = get_judge_llm()

    results = []
    all_passed = True

    for script in scripts:
        vid = script["variant_id"]

        # Skip scripts that already passed in a previous iteration
        # (check existing judge_results carried forward)
        prev_results = state.get("judge_results", [])
        already_passed = any(
            jr["variant_id"] == vid and jr.get("passed") for jr in prev_results
        )
        if already_passed:
            # Keep the passing result
            existing = next(jr for jr in prev_results if jr["variant_id"] == vid)
            results.append(existing)
            log.info(f"Variant {vid} already passed, skipping re-evaluation")
            continue

        judge_prompt = JUDGE_SCRIPT_USER.format(
            topic=topic,
            title=script["title"],
            script_text=script["script_text"],
            word_count=script["word_count"],
        )

        log.debug(f"Judging variant {vid} — prompt length: {len(judge_prompt)} chars")

        # Fresh context — only system + this one evaluation
        response = judge_llm.invoke([
            SystemMessage(content=JUDGE_SCRIPT_SYSTEM),
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

        log.info(
            f"Variant {vid} — scores: {scores}, avg: {avg}, passed: {passed}"
        )
        if not passed:
            log.info(f"Variant {vid} feedback: {judge_result['feedback']}")
            all_passed = False

        results.append(judge_result)

    # If we've hit max iterations, force-pass remaining failures
    if not all_passed and iteration >= MAX_JUDGE_ITERATIONS:
        log.warning(
            f"Hit max judge iterations ({MAX_JUDGE_ITERATIONS}). "
            "Accepting current scripts as best effort."
        )
        for r in results:
            if not r["passed"]:
                r["passed"] = True
                r["feedback"] = f"[Auto-passed after {MAX_JUDGE_ITERATIONS} iterations]"
        all_passed = True

    new_state = {
        "judge_results": results,
        "judge_iteration": iteration,
        "current_step": "judge_scripts",
    }

    save_thoughts(sid, f"05_judge_scripts_iter{iteration}", {**state, **new_state})
    log.debug("Saved thoughts for judge_scripts")

    # Print summary to user
    print("\n" + "-" * 40)
    print(f"Quality Check (Round {iteration})")
    print("-" * 40)
    for r in results:
        status = "PASS" if r["passed"] else "NEEDS IMPROVEMENT"
        print(f"  Variant {r['variant_id']}: {status} (avg: {r['average_score']})")
    print("-" * 40)

    return new_state
