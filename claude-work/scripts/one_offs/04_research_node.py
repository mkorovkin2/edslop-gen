import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from src.config import load_config
from src.utils.openai_client import OpenAIClient
from src.utils.tavily_client import TavilyClient
from src.agents.research import research_node
from src.models import create_initial_state, generate_run_id
from scripts.one_offs._shared import save_state, get_topic, is_placeholder_key


async def main() -> None:
    cfg = load_config()
    if is_placeholder_key(cfg.openai_api_key) or is_placeholder_key(cfg.tavily_api_key):
        print("API keys look like placeholders. Update .env first.")
        raise SystemExit(1)

    topic = get_topic()
    run_id = generate_run_id()
    state = create_initial_state(topic, run_id)

    tavily_client = TavilyClient(
        api_key=cfg.tavily_api_key,
        max_concurrent=cfg.max_concurrent_tavily,
        max_per_minute=cfg.max_rate_tavily_per_min,
    )
    openai_client = OpenAIClient(
        api_key=cfg.openai_api_key,
        model=cfg.model_name,
        max_concurrent=cfg.max_concurrent_openai,
        max_per_minute=cfg.max_rate_openai_per_min,
    )

    updates = await research_node(state, tavily_client, openai_client)
    state.update(updates)
    save_state(state)

    print(f"Research OK. run_id: {run_id}")
    print(f"sources: {len(state.get('research_data', []))}")
    print(f"queries: {state.get('metadata', {}).get('research_queries_used', [])}")


if __name__ == "__main__":
    asyncio.run(main())
