"""Script generation and parsing agents."""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState

logger = logging.getLogger(__name__)

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

    # Get current retry count
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
    system_message = (
        "You are an educational video script writer. "
        "Sound conversational and direct, but stick to the point. "
        "Prioritize conveying all required information over flow or polish. "
        "Transitions can be rough; it is fine to jump between points. "
        "Use common acronyms as-is (e.g., DARPA) without expanding them. "
        "Every sentence must introduce a concrete fact, mechanism, or necessary transition; no filler. "
        "Address the reader directly and follow any provided outline while keeping the output as continuous paragraphs."
    )

    # Generate script
    script_prompt = f"""
Create an engaging, informative script about the topic in <topic>.

{topic_xml}

{research_summary_xml}

{outline_xml}

<sources>
{sources_xml}
</sources>

<requirements>
  {word_count_xml}
  <style>Informative, straightforward, and conversational; zero fluff.</style>
  <language>Clear, accessible language.</language>
  <addressing>Address the reader directly as if presenting an informative case in a video.</addressing>
  <content>Include key technical concepts.</content>
  <outline>Follow the outline if provided, but write continuous paragraphs without headings.</outline>
  <structure>Paragraphs only; no headings, lists, or other formatting.</structure>
  <format>Plain paragraph text only.</format>
  <constraints>
    No stage directions or speaker notes.
    Do not include: greetings, scene-setting, motivational lines, rhetorical questions,
    "in this video" phrasing, or generic wrap-up statements.
    Avoid repetition and vague claims.
    Start immediately with a concrete definition or key technical claim.
    End with a final concrete point (not a summary or wrap-up).
  </constraints>
  <tone>Natural, conversational narration.</tone>
</requirements>

{retry_notice_xml}

Return ONLY the script text, nothing else.
"""

    script = await openai_client.generate(
        script_prompt,
        max_tokens=1500,
        temperature=0.7,
        system_message=system_message
    )

    # Count words
    word_count = len(script.split())
    logger.info("Script: generated %d words", word_count)

    # Evaluate script for concision / fluff
    quality_passed = False
    quality_issues = []
    quality_fluff_examples = []
    banned_phrases = [
        "in this video",
        "today we're going to",
        "today we are going to",
        "let's dive in",
        "let's jump in",
        "welcome",
        "stick around",
        "without further ado",
        "in conclusion",
        "to wrap up",
        "thanks for watching",
        "that's it for",
        "as you can see",
    ]
    lower_script = script.lower()
    banned_hits = [p for p in banned_phrases if p in lower_script]
    if banned_hits:
        quality_passed = False
        quality_issues.append(f"banned_phrases: {', '.join(banned_hits)}")
    if "?" in script:
        quality_passed = False
        quality_issues.append("contains_question_marks")
    try:
        quality_prompt = f"""
You are a strict editor. Evaluate the script for concision and zero-fluff requirements.

Rules for PASS:
- Every sentence delivers a concrete fact, mechanism, or necessary transition.
- No filler, generic framing, motivational lines, rhetorical questions, or scene-setting.
- No repetition or vague claims.
- Start is immediate and substantive; end is a concrete final point (not a wrap-up).

If any rule is violated, FAIL. If you are unsure, FAIL.

Return ONLY a JSON object:
{{
  "pass": true/false,
  "issues": ["short issue", ...],
  "fluff_examples": ["short excerpt", ...]
}}

Script:
<script>
{script}
</script>
"""
        quality_json = await openai_client.generate(
            quality_prompt,
            max_tokens=400,
            temperature=0.0
        )
        quality_data = json.loads(quality_json.strip())
        if isinstance(quality_data, dict):
            quality_passed = bool(quality_data.get("pass", False))
            quality_issues = quality_data.get("issues", []) if isinstance(quality_data.get("issues", []), list) else []
            quality_fluff_examples = (
                quality_data.get("fluff_examples", [])
                if isinstance(quality_data.get("fluff_examples", []), list)
                else []
            )
            if banned_hits:
                quality_passed = False
                if not any("banned_phrases" in str(item) for item in quality_issues):
                    quality_issues.append(f"banned_phrases: {', '.join(banned_hits)}")
            if "?" in script and "contains_question_marks" not in quality_issues:
                quality_issues.append("contains_question_marks")
    except Exception as e:
        logger.warning("Script: quality eval failed, treating as not passed. Error: %s", e)
        quality_passed = False
        quality_issues = ["quality_eval_failed"]

    # Update state
    return {
        "script": script,
        "metadata": {
            **state.get("metadata", {}),
            "script_generated_at": datetime.now().isoformat(),
            "word_count": word_count,
            "script_retry_count": retry_count,
            "script_quality_passed": quality_passed,
            "script_quality_issues": quality_issues,
            "script_quality_fluff_examples": quality_fluff_examples,
            "script_quality_checked_at": datetime.now().isoformat()
        },
        "retry_counts": {
            **state.get("retry_counts", {}),
            "synthesize_script": retry_count + 1
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 2
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

    outline_block = f"\nOutline (for guidance):\n{outline}\n" if outline else ""
    parse_prompt = f"""
Parse the following educational script into logical sections.
For each section, identify:
1. A descriptive title (if the section has one)
2. The full text of the section
3. Individual sentences within that section
{outline_block}
If an outline is provided, use it to guide section boundaries and titles where it fits the script.

Return your response as a JSON array of sections like this:
[
  {{
    "section_id": "section_1",
    "title": "Introduction to Topic",
    "text": "Full section text here...",
    "sentences": ["First sentence.", "Second sentence.", "Third sentence."]
  }},
  ...
]

Script:
<script>
{script}
</script>

Return ONLY the JSON array, no other text.
"""

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


def validate_script_word_count(state: WorkflowState) -> str:
    """
    Conditional edge function to validate script word count and concision.

    Args:
        state: Current workflow state

    Returns:
        "continue" if valid, "retry" if invalid (and retries remaining), "max_retries" if exhausted
    """
    min_words = int(os.getenv('SCRIPT_MIN_WORDS', '200'))
    max_words = int(os.getenv('SCRIPT_MAX_WORDS', '500'))

    word_count = len(state['script'].split())
    retry_count = state.get('retry_counts', {}).get('synthesize_script', 0)
    quality_passed = bool(state.get("metadata", {}).get("script_quality_passed", False))

    if min_words <= word_count <= max_words and quality_passed:
        return "continue"
    elif retry_count >= 3:
        logger.warning(
            "Script: validation failed (words=%d, quality=%s) after %d retries; continuing anyway.",
            word_count,
            quality_passed,
            retry_count
        )
        return "max_retries"
    else:
        logger.info(
            "Script: validation failed (words=%d, quality=%s); retrying.",
            word_count,
            quality_passed
        )
        return "retry"
