"""State schema for the video generation agent."""

from typing import TypedDict


class AgentState(TypedDict):
    session_id: str
    topic: str
    variants: list[dict]  # [{id, title, description}]
    selected_variant_ids: list[int]  # Up to 4 chosen by user
    variant_feedback: str | None  # NL feedback for regen
    scripts: list[dict]  # [{variant_id, title, script_text, word_count}]
    judge_results: list[dict]  # [{variant_id, passed, scores, feedback}]
    judge_iteration: int  # Track regen attempts per cycle
    approved_scripts: list[dict]  # User-approved final scripts
    audio_paths: list[str]  # Paths to generated .mp3 files
    current_step: str  # For crash recovery
