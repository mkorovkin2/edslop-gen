"""Outline generation and revision helpers."""

import logging

from ..utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


async def generate_outline(
    topic: str,
    openai_client: OpenAIClient,
    min_words: int,
    max_words: int,
    min_sections: int = 5,
    max_sections: int = 7
) -> str:
    """
    Generate a rough script outline for a topic.

    Args:
        topic: Video topic
        openai_client: OpenAI API client
        min_sections: Minimum number of sections
        max_sections: Maximum number of sections

    Returns:
        Outline text
    """
    logger.info("Outline: generating initial draft for topic=%s", topic)
    system_message = (
        "You are an educational video script planner. "
        "Produce concise, well-structured outlines."
    )
    prompt = f"""
Create a rough outline for an educational video script about the topic below.

Topic: {topic}
Script length: {min_words}-{max_words} words (keep it very concise).

Requirements:
- {min_sections}-{max_sections} sections in a logical flow (intro to wrap-up).
- Each section has a short title and 2-4 bullet points.
- Keep it concise and practical for a voiceover script.

Format exactly like this:
1. Section Title
   - point
   - point
2. Section Title
   - point
   - point

Return ONLY the outline text, nothing else.
"""
    outline = await openai_client.generate(
        prompt,
        max_tokens=800,
        temperature=0.4,
        system_message=system_message
    )
    return outline.strip()


async def revise_outline(
    topic: str,
    outline: str,
    feedback: str,
    openai_client: OpenAIClient,
    min_words: int,
    max_words: int
) -> str:
    """
    Revise an outline using user feedback.

    Args:
        topic: Video topic
        outline: Current outline text
        feedback: User feedback/instructions
        openai_client: OpenAI API client

    Returns:
        Revised outline text
    """
    logger.info("Outline: revising draft for topic=%s", topic)
    system_message = (
        "You are an educational video script planner. "
        "Revise outlines based on feedback while preserving the requested format."
    )
    prompt = f"""
Revise the outline based on the feedback. Keep the same outline format.

Topic: {topic}
Script length: {min_words}-{max_words} words (keep it very concise).

Current outline:
{outline}

Feedback:
{feedback}

Return ONLY the revised outline text, nothing else.
"""
    revised = await openai_client.generate(
        prompt,
        max_tokens=800,
        temperature=0.4,
        system_message=system_message
    )
    return revised.strip()
