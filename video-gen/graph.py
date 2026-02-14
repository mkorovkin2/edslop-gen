"""LangGraph graph definition — nodes, edges, and conditional routing."""

from langgraph.graph import END, StateGraph

from nodes import (
    generate_audio,
    generate_scripts,
    generate_variants,
    get_topic,
    judge_scripts,
    user_approve_scripts,
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

    # Terminal edge
    graph.add_edge("generate_audio", END)

    return graph.compile()
