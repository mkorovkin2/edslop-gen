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
    visual_scripts: list[dict]  # [{variant_id, segments: [{segment_id, time_range, visual_description, mood, camera}]}]
    visual_judge_results: list[dict]  # [{variant_id, passed, scores, feedback}]
    visual_judge_iteration: int  # Track visual judge regen attempts (cap at 5)
    approved_visual_scripts: list[dict]  # User-approved visual scripts
    video_breakdown: list[dict]  # [{variant_id, segments: [{segment_id, sora_prompt, duration, size, ...}]}]
    video_paths: list[str]  # Paths to generated .mp4 part files
    current_step: str  # For crash recovery
