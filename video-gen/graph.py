"""LangGraph graph definition — nodes, edges, and conditional routing."""

from langgraph.graph import END, StateGraph

from nodes import (
    generate_audio,
    generate_scripts,
    generate_variants,
    generate_video_breakdown,
    generate_videos,
    generate_visual_scripts,
    get_topic,
    judge_scripts,
    judge_visual_scripts,
    user_approve_breakdown,
    user_approve_scripts,
    user_approve_visuals,
    user_select_variants,
)
from state import AgentState


def route_after_variant_selection(state: AgentState) -> str:
    """Route based on whether user selected variants or wants regen."""
    if state.get("variant_feedback"):
        return "generate_variants"
    return "generate_scripts"


def route_after_judging(state: AgentState) -> str:
    """Route based on whether all scripts passed judging."""
    results = state.get("judge_results", [])
    all_passed = all(r.get("passed", False) for r in results)

    if all_passed:
        return "user_approve_scripts"
    return "generate_scripts"


def route_after_approval(state: AgentState) -> str:
    """Route based on whether user approved or wants revisions."""
    if state.get("approved_scripts"):
        return "generate_audio"
    return "generate_scripts"


def route_after_visual_judging(state: AgentState) -> str:
    """Route based on whether all visual scripts passed judging."""
    results = state.get("visual_judge_results", [])
    all_passed = all(r.get("passed", False) for r in results)

    if all_passed:
        return "user_approve_visuals"
    return "generate_visual_scripts"


def route_after_visual_approval(state: AgentState) -> str:
    """Route based on whether user approved visual scripts or wants revisions."""
    if state.get("approved_visual_scripts"):
        return "generate_video_breakdown"
    return "generate_visual_scripts"


def route_after_breakdown_approval(state: AgentState) -> str:
    """Route based on whether user approved breakdown or wants revisions."""
    step = state.get("current_step", "")
    if step == "user_approve_breakdown_done":
        return "generate_videos"
    return "generate_video_breakdown"


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph workflow."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("get_topic", get_topic)
    graph.add_node("generate_variants", generate_variants)
    graph.add_node("user_select_variants", user_select_variants)
    graph.add_node("generate_scripts", generate_scripts)
    graph.add_node("judge_scripts", judge_scripts)
    graph.add_node("user_approve_scripts", user_approve_scripts)
    graph.add_node("generate_audio", generate_audio)
    graph.add_node("generate_visual_scripts", generate_visual_scripts)
    graph.add_node("judge_visual_scripts", judge_visual_scripts)
    graph.add_node("user_approve_visuals", user_approve_visuals)
    graph.add_node("generate_video_breakdown", generate_video_breakdown)
    graph.add_node("user_approve_breakdown", user_approve_breakdown)
    graph.add_node("generate_videos", generate_videos)

    # Set entry point
    graph.set_entry_point("get_topic")

    # Linear edges
    graph.add_edge("get_topic", "generate_variants")
    graph.add_edge("generate_variants", "user_select_variants")

    # Conditional: after variant selection → regen or proceed to scripts
    graph.add_conditional_edges(
        "user_select_variants",
        route_after_variant_selection,
        {
            "generate_variants": "generate_variants",
            "generate_scripts": "generate_scripts",
        },
    )

    # Linear: scripts → judge
    graph.add_edge("generate_scripts", "judge_scripts")

    # Conditional: after judging → regen failed scripts or show to user
    graph.add_conditional_edges(
        "judge_scripts",
        route_after_judging,
        {
            "generate_scripts": "generate_scripts",
            "user_approve_scripts": "user_approve_scripts",
        },
    )

    # Conditional: after user approval → audio or revise
    graph.add_conditional_edges(
        "user_approve_scripts",
        route_after_approval,
        {
            "generate_audio": "generate_audio",
            "generate_scripts": "generate_scripts",
        },
    )

    # After audio → visual scripts (instead of END)
    graph.add_edge("generate_audio", "generate_visual_scripts")

    # Linear: visual scripts → visual judge
    graph.add_edge("generate_visual_scripts", "judge_visual_scripts")

    # Conditional: after visual judging → regen or show to user
    graph.add_conditional_edges(
        "judge_visual_scripts",
        route_after_visual_judging,
        {
            "generate_visual_scripts": "generate_visual_scripts",
            "user_approve_visuals": "user_approve_visuals",
        },
    )

    # Conditional: after visual approval → breakdown or revise
    graph.add_conditional_edges(
        "user_approve_visuals",
        route_after_visual_approval,
        {
            "generate_video_breakdown": "generate_video_breakdown",
            "generate_visual_scripts": "generate_visual_scripts",
        },
    )

    # Linear: breakdown → user approve breakdown
    graph.add_edge("generate_video_breakdown", "user_approve_breakdown")

    # Conditional: after breakdown approval → generate videos or revise
    graph.add_conditional_edges(
        "user_approve_breakdown",
        route_after_breakdown_approval,
        {
            "generate_videos": "generate_videos",
            "generate_video_breakdown": "generate_video_breakdown",
        },
    )

    # Terminal edge
    graph.add_edge("generate_videos", END)

    return graph.compile()
