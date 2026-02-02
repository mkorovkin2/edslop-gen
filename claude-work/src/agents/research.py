"""Research agent using Tavily for web search."""

import logging
from typing import Dict, Any
from datetime import datetime
from ..utils.tavily_client import TavilyClient
from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState

logger = logging.getLogger(__name__)

async def research_node(
    state: WorkflowState,
    tavily_client: TavilyClient,
    openai_client: OpenAIClient
) -> Dict[str, Any]:
    """
    Research node: Perform web research on the topic using Tavily.

    Uses LLM to generate diverse search queries, then synthesizes
    research results into structured data.

    Args:
        state: Current workflow state
        tavily_client: Tavily API client
        openai_client: OpenAI API client

    Returns:
        Dict with updated research_data, metadata, and api_call_counts
    """
    topic = state['topic']

    # Step 1: Use LLM to generate 3-5 diverse search queries
    logger.info("Research: generating search queries for topic=%s", topic)
    query_generation_prompt = f"""
You are a research assistant. Given a technical topic, generate 3-5 diverse search queries
that will help gather comprehensive information about this topic.

Topic: {topic}

Generate queries that cover:
1. Basic concepts and definitions
2. Technical details and mechanisms
3. Real-world applications
4. Recent developments or research
5. Related concepts or comparisons

Return ONLY a JSON array of query strings, like:
["query 1", "query 2", "query 3"]
"""

    queries_json = await openai_client.generate(
        query_generation_prompt,
        max_tokens=500,
        temperature=0.7
    )

    # Parse queries from JSON
    import json
    try:
        queries = json.loads(queries_json.strip())
        if not isinstance(queries, list) or len(queries) == 0:
            raise ValueError("Invalid queries format")
    except Exception as e:
        logger.warning("Research: failed to parse queries JSON, using fallback. Error: %s", e)
        # Fallback: use topic directly
        queries = [topic, f"{topic} explained", f"{topic} applications"]

    logger.info("Research: using %d queries", len(queries))
    logger.debug("Research queries: %s", queries)

    # Step 2: Execute Tavily searches for each query
    all_results = []
    for query in queries:
        try:
            logger.debug("Research: searching query=%r", query)
            results = await tavily_client.search(query, search_depth="advanced")
            logger.debug("Research: %d results for query=%r", len(results), query)
            all_results.extend(results)
        except Exception as e:
            logger.warning("Research: search failed for %r: %s", query, e)
            continue

    if len(all_results) < 2:
        logger.error("Research: insufficient sources collected (%d)", len(all_results))
        raise ValueError(
            f"Insufficient research data collected. "
            f"Only {len(all_results)} sources found (need at least 2)."
        )

    logger.info("Research: collected %d sources", len(all_results))

    # Step 3: Use LLM to synthesize research into structured summary
    research_text = "\n\n".join([
        f"Source {i+1} - {r.get('title', 'Untitled')}:\n{r.get('content', '')[:500]}"
        for i, r in enumerate(all_results[:10])  # Limit to top 10 sources
    ])

    synthesis_prompt = f"""
Synthesize the following research results about "{topic}" into a structured summary.
Focus on key concepts, technical details, and important facts that would be useful
for creating an educational video script.

{research_text}

Return a concise summary (200-300 words) covering the most important points.
"""

    logger.info("Research: synthesizing summary from %d sources", min(len(all_results), 10))
    synthesis = await openai_client.generate(
        synthesis_prompt,
        max_tokens=800,
        temperature=0.5
    )
    logger.debug("Research: synthesis length=%d chars", len(synthesis))

    # Step 4: Update state
    return {
        "research_data": all_results,
        "metadata": {
            **state.get("metadata", {}),
            "research_completed_at": datetime.now().isoformat(),
            "research_source_count": len(all_results),
            "research_queries_used": queries,
            "research_synthesis": synthesis
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "tavily": state.get("api_call_counts", {}).get("tavily", 0) + len(queries),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + 2  # 2 LLM calls
        }
    }
