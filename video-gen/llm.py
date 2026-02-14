"""LLM client factory — returns OpenAI or Anthropic chat model based on env config."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5.2"


def get_llm():
    """Return a LangChain chat model based on LLM_PROVIDER env var.

    Used for parsing, routing, and non-research LLM calls.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            max_tokens=4096,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=4096,
        )


def get_judge_llm():
    """Return a separate LLM instance for judging (fresh context, lower temperature)."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.2,
            max_tokens=4096,
        )
    else:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.2,
            max_tokens=4096,
        )


def get_search_llm():
    """Return an OpenAI Responses API client with web_search forced on.

    Used for all content generation that needs real-world research
    (variant ideation, script writing). Uses the Responses API directly
    since LangChain doesn't natively support OpenAI's web_search_preview tool.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


def invoke_with_web_search(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call OpenAI Responses API with web_search_preview forced on.

    Args:
        system_prompt: System-level instructions.
        user_prompt: The user query / generation request.
        temperature: Sampling temperature.

    Returns:
        The model's text response (with web search grounding).
    """
    client = get_search_llm()

    response = client.responses.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        tools=[{"type": "web_search_preview"}],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Extract text from the response output items
    text_parts = []
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    text_parts.append(content_block.text)

    return "\n".join(text_parts)
