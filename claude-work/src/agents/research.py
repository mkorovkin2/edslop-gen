"""Research agent using OpenAI web search for sources."""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState
from ..prompts import (
    research_query_generation_prompt,
    research_search_prompt,
    research_synthesis_judge_prompt,
    research_synthesis_prompt,
    research_synthesis_rewrite_prompt
)

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Best-effort JSON extraction from model output."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip fenced blocks if present
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Try to extract JSON object substring
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


async def _synthesize_research(
    topic: str,
    outline: str,
    query_summaries: List[Dict[str, Any]],
    openai_client: OpenAIClient
) -> tuple[str, bool, List[str], int]:
    """Synthesize research into a concise summary with a strict QA loop."""
    summaries_text = "\n\n".join([
        f"Query: {item.get('query')}\nSummary:\n{item.get('summary', '')}"
        for item in query_summaries
    ])
    synthesis_prompt = research_synthesis_prompt(topic, outline, summaries_text)

    attempts = 0
    passed = False
    issues: List[str] = []
    summary_text = ""
    while attempts < 2:
        attempts += 1
        summary_text = await openai_client.generate(
            synthesis_prompt,
            max_tokens=800,
            temperature=0.4
        )
        wc = len(summary_text.split())
        issues = []

        quality_json = await openai_client.generate(
            research_synthesis_judge_prompt(summary_text),
            max_tokens=400,
            temperature=0.0
        )
        quality_data = _extract_json(quality_json) or {}
        passed = bool(quality_data.get("pass", False))
        issues = quality_data.get("issues", []) if isinstance(quality_data.get("issues", []), list) else []
        if not (200 <= wc <= 300):
            passed = False
            if "word_count_out_of_range" not in issues:
                issues.append("word_count_out_of_range")

        if passed:
            break

        fix_instructions = quality_data.get("fix_instructions", "")
        synthesis_prompt = research_synthesis_rewrite_prompt(
            topic,
            outline,
            summaries_text,
            issues,
            fix_instructions
        )

    return summary_text, passed, issues, attempts


async def research_node(
    state: WorkflowState,
    openai_client: OpenAIClient
) -> Dict[str, Any]:
    """
    Research node: Perform web research on the topic using OpenAI web search.

    Uses the LLM to generate search queries, calls the web search tool for each query,
    and synthesizes results into structured summary data for downstream use.
    """
    topic = state['topic']
    outline = state.get('script_outline', '').strip()

    logger.info("Research: generating search queries for topic=%s", topic)
    query_generation_prompt = research_query_generation_prompt(topic, outline)

    queries_json = await openai_client.generate(
        query_generation_prompt,
        max_tokens=400,
        temperature=0.6
    )

    try:
        queries = json.loads(queries_json.strip())
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("Invalid queries format")
    except Exception as e:
        logger.warning("Research: failed to parse queries JSON, using fallback. Error: %s", e)
        queries = [topic, f"{topic} explained", f"{topic} applications"]

    logger.info("Research: using %d queries", len(queries))

    all_sources: List[Dict[str, Any]] = []
    query_summaries: List[Dict[str, Any]] = []
    openai_calls = 1  # query generation

    for query in queries:
        logger.debug("Research: web search query=%r", query)
        search_prompt = research_search_prompt(query)
        search_result = await openai_client.web_search(
            search_prompt,
            max_output_tokens=900,
            temperature=0.2
        )
        openai_calls += 1

        parsed = _extract_json(search_result.get("text", ""))
        summary = ""
        sources = []
        if parsed and isinstance(parsed, dict):
            summary = parsed.get("summary", "") if isinstance(parsed.get("summary", ""), str) else ""
            sources = parsed.get("sources", []) if isinstance(parsed.get("sources", []), list) else []

        if not sources:
            sources = search_result.get("sources", [])

        if not summary:
            summary = search_result.get("text", "").strip()

        query_summaries.append({
            "query": query,
            "summary": summary,
            "source_count": len(sources)
        })

        for src in sources:
            if isinstance(src, dict):
                all_sources.append({
                    "title": src.get("title", "Source"),
                    "url": src.get("url", ""),
                    "content": src.get("snippet") or src.get("summary") or summary,
                    "query": query
                })
            else:
                all_sources.append({
                    "title": "Source",
                    "url": str(src),
                    "content": summary,
                    "query": query
                })

    # De-duplicate by URL
    deduped_sources: List[Dict[str, Any]] = []
    seen_urls = set()
    for src in all_sources:
        url = src.get("url")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        deduped_sources.append(src)

    if len(deduped_sources) < 2:
        logger.error("Research: insufficient sources collected (%d)", len(deduped_sources))
        raise ValueError(
            f"Insufficient research data collected. "
            f"Only {len(deduped_sources)} sources found (need at least 2)."
        )

    logger.info("Research: collected %d sources", len(deduped_sources))

    synthesis, synthesis_passed, synthesis_issues, synthesis_attempts = await _synthesize_research(
        topic,
        outline,
        query_summaries,
        openai_client
    )
    openai_calls += 2 * synthesis_attempts  # synthesis + judge per attempt

    return {
        "research_data": deduped_sources,
        "metadata": {
            **state.get("metadata", {}),
            "research_completed_at": datetime.now().isoformat(),
            "research_source_count": len(deduped_sources),
            "research_queries_used": queries,
            "research_synthesis": synthesis,
            "research_synthesis_passed": synthesis_passed,
            "research_synthesis_issues": synthesis_issues,
            "research_synthesis_attempts": synthesis_attempts
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + openai_calls
        }
    }
