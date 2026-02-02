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

    # Build research context
    research_text = "\n\n".join([
        f"- {r.get('title', 'Source')}: {r.get('content', '')[:300]}"
        for r in state['research_data'][:8]  # Limit to 8 sources
    ])

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

    # Generate script
    script_prompt = f"""
You are an educational video script writer. Create an engaging, informative script about "{state['topic']}".

Research Summary:
{synthesis}

Detailed Sources:
{research_text}

Requirements:
- Write EXACTLY between {min_words} and {max_words} words
- Make it educational and engaging
- Use clear, accessible language
- Include key technical concepts
- Structure with clear sections/paragraphs
- Do NOT include stage directions or speaker notes
- Write in a natural, conversational tone suitable for narration

{f"IMPORTANT: Previous attempt had incorrect word count. This is attempt #{retry_count + 1}. Please carefully count words and ensure you meet the {min_words}-{max_words} word requirement." if retry_count > 0 else ""}

Return ONLY the script text, nothing else.
"""

    script = await openai_client.generate(
        script_prompt,
        max_tokens=1500,
        temperature=0.7
    )

    # Count words
    word_count = len(script.split())
    logger.info("Script: generated %d words", word_count)

    # Update state
    return {
        "script": script,
        "metadata": {
            **state.get("metadata", {}),
            "script_generated_at": datetime.now().isoformat(),
            "word_count": word_count,
            "script_retry_count": retry_count
        },
        "retry_counts": {
            **state.get("retry_counts", {}),
            "synthesize_script": retry_count + 1
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 1
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

    parse_prompt = f"""
Parse the following educational script into logical sections.
For each section, identify:
1. A descriptive title (if the section has one)
2. The full text of the section
3. Individual sentences within that section

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
{script}

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
    Conditional edge function to validate script word count.

    Args:
        state: Current workflow state

    Returns:
        "continue" if valid, "retry" if invalid (and retries remaining), "max_retries" if exhausted
    """
    min_words = int(os.getenv('SCRIPT_MIN_WORDS', '200'))
    max_words = int(os.getenv('SCRIPT_MAX_WORDS', '500'))

    word_count = len(state['script'].split())
    retry_count = state.get('retry_counts', {}).get('synthesize_script', 0)

    if min_words <= word_count <= max_words:
        return "continue"
    elif retry_count >= 3:
        logger.warning(
            "Script: word count %d outside range after %d retries; continuing anyway.",
            word_count,
            retry_count
        )
        return "max_retries"
    else:
        logger.info(
            "Script: word count %d outside range [%d, %d]; retrying.",
            word_count,
            min_words,
            max_words
        )
        return "retry"
