"""Script generation and parsing agents."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState
from ..prompts import (
    SCRIPT_EDITOR_SYSTEM_MESSAGE,
    SCRIPT_RETURN_ONLY_SUFFIX,
    SCRIPT_WRITER_SYSTEM_MESSAGE,
    script_base_prompt,
    script_feedback_prompt,
    script_judge_prompt,
    script_parse_prompt,
    script_polish_prompt,
    script_revision_prompt
)

logger = logging.getLogger(__name__)

def _build_sources_xml(research_data: List[Dict[str, Any]]) -> str:
    """Build XML block for research sources."""
    if not research_data:
        return ""
    return "\n".join([
        "<source>"
        f"<title>{r.get('title', 'Source')}</title>"
        f"<excerpt>{r.get('content', '')}</excerpt>"
        "</source>"
        for r in research_data
    ])


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Strict JSON extraction from model output."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    return json.loads(cleaned)


async def synthesize_script_node(
    state: WorkflowState,
    openai_client: OpenAIClient,
    min_words: int,
    max_words: int
) -> Dict[str, Any]:
    """
    Generate educational script from research data.

    Args:
        state: Current workflow state
        openai_client: OpenAI API client
        min_words: Minimum word count
        max_words: Maximum word count

    Returns:
        Dict with updated script, metadata, and retry_counts
    """
    # Validate input
    if not state.get('research_data'):
        raise ValueError("No research data available for script generation")

    if len(state['research_data']) < 2:
        raise ValueError(f"Insufficient research data (need at least 2 sources, got {len(state['research_data'])})")

    # Get research synthesis if available
    synthesis = state.get('metadata', {}).get('research_synthesis', '')
    outline = state.get('script_outline', '').strip()

    # Get current retry count (graph-level attempts)
    retry_count = state.get('retry_counts', {}).get('synthesize_script', 0)

    logger.info(
        "Script: generating draft (attempt %d) with target range %d-%d words",
        retry_count + 1,
        min_words,
        max_words
    )
    logger.debug(
        "Script: research sources=%d, synthesis_chars=%d",
        len(state['research_data']),
        len(synthesis or "")
    )

    topic_xml = f"<topic>{state['topic']}</topic>"
    research_summary_xml = f"<research_summary>{synthesis}</research_summary>"
    outline_xml = f"<outline>{outline}</outline>" if outline else ""
    sources_xml = "\n".join([
        "<source>"
        f"<title>{r.get('title', 'Source')}</title>"
        f"<excerpt>{r.get('content', '')}</excerpt>"
        "</source>"
        for r in state['research_data']
    ])
    word_count_xml = f"<word_count><min>{min_words}</min><max>{max_words}</max></word_count>"
    retry_notice_xml = (
        "<retry_notice>"
        "<message>Previous attempt had incorrect word count. This is a retry; comply with the word count requirements.</message>"
        f"<attempt>{retry_count + 1}</attempt>"
        "</retry_notice>"
        if retry_count > 0 else ""
    )
    system_message = SCRIPT_WRITER_SYSTEM_MESSAGE

    base_prompt = script_base_prompt(
        topic_xml=topic_xml,
        research_summary_xml=research_summary_xml,
        outline_xml=outline_xml,
        sources_xml=sources_xml,
        word_count_xml=word_count_xml,
        retry_notice_xml=retry_notice_xml
    )

    max_attempts = 3
    attempts = 0
    quality_passed = False
    quality_issues: List[str] = []
    quality_fluff_examples: List[str] = []
    quality_score = 0.0
    script = ""
    last_fix_instructions = ""

    while attempts < max_attempts:
        attempts += 1
        if attempts == 1:
            prompt = base_prompt + SCRIPT_RETURN_ONLY_SUFFIX
        else:
            prompt = script_revision_prompt(
                quality_issues=quality_issues,
                last_fix_instructions=last_fix_instructions,
                min_words=min_words,
                max_words=max_words,
                script=script
            )

        script = await openai_client.generate(
            prompt,
            max_tokens=1600,
            temperature=0.6,
            system_message=system_message
        )

        word_count = len(script.split())

        quality_json = await openai_client.generate(
            script_judge_prompt(min_words, max_words, script),
            max_tokens=500,
            temperature=0.0
        )
        quality_data = _extract_json(quality_json)
        if not isinstance(quality_data, dict):
            raise ValueError("Script: quality eval returned invalid JSON")

        quality_passed = bool(quality_data.get("pass", False))
        quality_issues = quality_data.get("issues", []) if isinstance(quality_data.get("issues", []), list) else []
        last_fix_instructions = str(quality_data.get("fix_instructions", "")).strip()
        quality_score = float(quality_data.get("quality_score", 0.0) or 0.0)

        if quality_passed:
            logger.info("Script: quality pass on attempt %d (%d words)", attempts, word_count)
            break
        logger.info("Script: quality fail on attempt %d (%d words); retrying", attempts, word_count)

    # Update state
    return {
        "script": script,
        "metadata": {
            **state.get("metadata", {}),
            "script_generated_at": datetime.now().isoformat(),
            "word_count": len(script.split()),
            "script_retry_count": retry_count,
            "script_quality_passed": quality_passed,
            "script_quality_issues": quality_issues,
            "script_quality_fluff_examples": quality_fluff_examples,
            "script_quality_score": quality_score,
            "script_quality_attempts": attempts,
            "script_quality_fix_instructions": last_fix_instructions,
            "script_quality_checked_at": datetime.now().isoformat()
        },
        "retry_counts": {
            **state.get("retry_counts", {}),
            "synthesize_script": retry_count + 1
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + (attempts * 2)
        }
    }


async def parse_script_node(
    state: WorkflowState,
    openai_client: OpenAIClient
) -> Dict[str, Any]:
    """
    Parse script into sections and sentences.

    Args:
        state: Current workflow state
        openai_client: OpenAI API client

    Returns:
        Dict with updated script_sections and metadata
    """
    # Validate input
    if not state.get('script'):
        raise ValueError("No script available for parsing")

    script = state['script']
    outline = state.get('script_outline', '').strip()

    parse_prompt = script_parse_prompt(script, outline)

    logger.info("Script: parsing into sections")
    sections_json = await openai_client.generate(
        parse_prompt,
        max_tokens=2000,
        temperature=0.3
    )

    # Parse JSON
    try:
        sections = json.loads(sections_json.strip())
        if not isinstance(sections, list):
            raise ValueError("Invalid sections format")

        # Add start/end indices
        current_pos = 0
        for section in sections:
            section_text = section['text']
            start_idx = script.find(section_text, current_pos)
            if start_idx == -1:
                start_idx = current_pos
            end_idx = start_idx + len(section_text)
            section['start_index'] = start_idx
            section['end_index'] = end_idx
            current_pos = end_idx

    except Exception as e:
        logger.warning("Script: failed to parse sections JSON, using fallback. Error: %s", e)
        # Fallback: split into paragraphs
        paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]
        sections = []
        current_pos = 0
        for i, para in enumerate(paragraphs):
            # Simple sentence split
            sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
            start_idx = script.find(para, current_pos)
            sections.append({
                "section_id": f"section_{i+1}",
                "title": None,
                "text": para,
                "sentences": sentences,
                "start_index": start_idx if start_idx != -1 else current_pos,
                "end_index": start_idx + len(para) if start_idx != -1 else current_pos + len(para)
            })
            current_pos = start_idx + len(para) if start_idx != -1 else current_pos + len(para)

    logger.info("Script: parsed %d sections", len(sections))
    return {
        "script_sections": sections,
        "metadata": {
            **state.get("metadata", {}),
            "script_parsed_at": datetime.now().isoformat(),
            "script_sections_count": len(sections)
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 1
        }
    }


async def revise_script_with_feedback(
    topic: str,
    script: str,
    feedback: str,
    openai_client: OpenAIClient,
    min_words: int,
    max_words: int,
    outline: str = "",
    research_summary: str = "",
    research_data: List[Dict[str, Any]] | None = None
) -> str:
    """
    Revise a script based on user feedback while preserving constraints.
    """
    sources_xml = _build_sources_xml(research_data or [])
    outline_xml = f"<outline>{outline}</outline>" if outline else ""
    research_summary_xml = f"<research_summary>{research_summary}</research_summary>" if research_summary else ""
    word_count_xml = f"<word_count><min>{min_words}</min><max>{max_words}</max></word_count>"

    prompt = script_feedback_prompt(
        topic=topic,
        feedback=feedback,
        script=script,
        research_summary_xml=research_summary_xml,
        outline_xml=outline_xml,
        sources_xml=sources_xml,
        word_count_xml=word_count_xml
    )
    revised = await openai_client.generate(
        prompt,
        max_tokens=1600,
        temperature=0.5,
        system_message=SCRIPT_EDITOR_SYSTEM_MESSAGE
    )
    return revised.strip()


async def polish_script(
    topic: str,
    script: str,
    openai_client: OpenAIClient,
    min_words: int,
    max_words: int,
    outline: str = "",
    research_summary: str = "",
    research_data: List[Dict[str, Any]] | None = None
) -> str:
    """
    Apply a final improvement pass to a script while preserving constraints.
    """
    sources_xml = _build_sources_xml(research_data or [])
    outline_xml = f"<outline>{outline}</outline>" if outline else ""
    research_summary_xml = f"<research_summary>{research_summary}</research_summary>" if research_summary else ""
    word_count_xml = f"<word_count><min>{min_words}</min><max>{max_words}</max></word_count>"

    prompt = script_polish_prompt(
        topic=topic,
        script=script,
        research_summary_xml=research_summary_xml,
        outline_xml=outline_xml,
        sources_xml=sources_xml,
        word_count_xml=word_count_xml
    )
    improved = await openai_client.generate(
        prompt,
        max_tokens=1600,
        temperature=0.4,
        system_message=SCRIPT_EDITOR_SYSTEM_MESSAGE
    )
    return improved.strip()


def validate_script_word_count(state: WorkflowState) -> str:
    """
    Conditional edge function to validate script quality via LLM judge.

    Args:
        state: Current workflow state

    Returns:
        "continue" if valid, "retry" if invalid (and retries remaining), "max_retries" if exhausted
    """
    retry_count = state.get('retry_counts', {}).get('synthesize_script', 0)
    quality_passed = bool(state.get("metadata", {}).get("script_quality_passed", False))

    if quality_passed:
        return "continue"
    elif retry_count >= 3:
        logger.warning(
            "Script: validation failed (quality=%s) after %d retries; continuing anyway.",
            quality_passed,
            retry_count
        )
        return "max_retries"
    else:
        logger.info(
            "Script: validation failed (quality=%s); retrying.",
            quality_passed
        )
        return "retry"
