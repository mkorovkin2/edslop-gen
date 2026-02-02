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
from src.agents.images import collect_images_node
from scripts.one_offs._shared import load_state, save_state, is_placeholder_key


async def main() -> None:
    cfg = load_config()
    if is_placeholder_key(cfg.openai_api_key) or is_placeholder_key(cfg.tavily_api_key):
        print("API keys look like placeholders. Update .env first.")
        raise SystemExit(1)

    state = load_state()

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

    updates = await collect_images_node(state, tavily_client, openai_client)
    state.update(updates)
    save_state(state)

    print("Collect images OK")
    print(f"images_collected: {len(state.get('images', []))}")


if __name__ == "__main__":
    asyncio.run(main())
