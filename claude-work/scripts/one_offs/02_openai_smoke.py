import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from src.config import load_config
from src.utils.openai_client import OpenAIClient
from scripts.one_offs._shared import is_placeholder_key


async def main() -> None:
    cfg = load_config()
    if is_placeholder_key(cfg.openai_api_key):
        print("OpenAI key looks like a placeholder. Update .env first.")
        raise SystemExit(1)

    client = OpenAIClient(
        api_key=cfg.openai_api_key,
        model=cfg.model_name,
        max_concurrent=cfg.max_concurrent_openai,
        max_per_minute=cfg.max_rate_openai_per_min,
    )

    resp = await client.generate(
        "Respond with exactly: OK",
        max_tokens=5,
        temperature=0.0,
    )
    print("OpenAI response:", resp.strip())


if __name__ == "__main__":
    asyncio.run(main())
