"""Plan generation and revision helpers."""

import logging

from ..utils.openai_client import OpenAIClient
from ..prompts import (
    OUTLINE_REVISION_SYSTEM_MESSAGE,
    OUTLINE_SYSTEM_MESSAGE,
    outline_prompt,
    outline_revision_prompt
)

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
    Generate a focused plan paragraph for a topic.

    Args:
        topic: Video topic
        openai_client: OpenAI API client
        min_sections: Minimum number of sections
        max_sections: Maximum number of sections

    Returns:
        Plan paragraph text
    """
    logger.info("Outline: generating initial draft for topic=%s", topic)
    system_message = OUTLINE_SYSTEM_MESSAGE
    prompt = outline_prompt(topic, min_words, max_words)
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
    Revise a plan paragraph using user feedback.

    Args:
        topic: Video topic
        outline: Current plan paragraph text
        feedback: User feedback/instructions
        openai_client: OpenAI API client

    Returns:
        Revised plan paragraph text
    """
    logger.info("Outline: revising draft for topic=%s", topic)
    system_message = OUTLINE_REVISION_SYSTEM_MESSAGE
    prompt = outline_revision_prompt(topic, outline, feedback, min_words, max_words)
    revised = await openai_client.generate(
        prompt,
        max_tokens=800,
        temperature=0.4,
        system_message=system_message
    )
    return revised.strip()
