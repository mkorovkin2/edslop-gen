"""LLM client factory — returns OpenAI or Anthropic chat model based on env config."""

import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = "gpt-5.2"

log = logging.getLogger("video_agent.llm")


def retry_with_backoff(fn, max_retries=5, base_delay=2.0, max_delay=120.0):
    """Retry a callable with exponential backoff on rate limit (429) and server (5xx) errors.

    Works with both OpenAI and Anthropic SDK errors.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            err_str = str(e).lower()
            is_rate_limit = status == 429 or "rate" in err_str and "limit" in err_str
            is_server_error = isinstance(status, int) and 500 <= status < 600

            if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                # Use Retry-After header if available
                retry_after = getattr(e, "headers", {})
                if hasattr(retry_after, "get"):
                    retry_after = retry_after.get("retry-after")
                else:
                    retry_after = None

                if retry_after:
                    delay = float(retry_after)
                else:
                    delay = min(base_delay * (2 ** attempt), max_delay)

                log.warning(
                    f"Rate limit / server error (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay:.1f}s... Error: {e}"
                )
                time.sleep(delay)
            else:
                raise


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


def get_sora_client():
    """Return an OpenAI client configured for Sora video generation API."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


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

    def _call():
        return client.responses.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            tools=[{"type": "web_search_preview"}],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    response = retry_with_backoff(_call)

    # Extract text from the response output items
    text_parts = []
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    text_parts.append(content_block.text)

    return "\n".join(text_parts)
