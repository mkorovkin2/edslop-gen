import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from src.config import load_config
from src.utils.tavily_client import TavilyClient
from scripts.one_offs._shared import is_placeholder_key


async def main() -> None:
    cfg = load_config()
    if is_placeholder_key(cfg.tavily_api_key):
        print("Tavily key looks like a placeholder. Update .env first.")
        raise SystemExit(1)

    client = TavilyClient(
        api_key=cfg.tavily_api_key,
        max_concurrent=cfg.max_concurrent_tavily,
        max_per_minute=cfg.max_rate_tavily_per_min,
    )

    results = await client.search("neural networks basics", search_depth="basic")
    print(f"Tavily results: {len(results)}")
    if results:
        print("First title:", results[0].get("title", "(no title)"))


if __name__ == "__main__":
    asyncio.run(main())
