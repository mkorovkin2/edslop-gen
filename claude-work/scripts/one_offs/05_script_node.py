import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from src.config import load_config
from src.utils.openai_client import OpenAIClient
from src.agents.script import synthesize_script_node
from scripts.one_offs._shared import load_state, save_state, is_placeholder_key


async def main() -> None:
    cfg = load_config()
    if is_placeholder_key(cfg.openai_api_key):
        print("OpenAI key looks like a placeholder. Update .env first.")
        raise SystemExit(1)

    state = load_state()

    openai_client = OpenAIClient(
        api_key=cfg.openai_api_key,
        model=cfg.model_name,
        max_concurrent=cfg.max_concurrent_openai,
        max_per_minute=cfg.max_rate_openai_per_min,
    )

    updates = await synthesize_script_node(
        state,
        openai_client,
        cfg.script_min_words,
        cfg.script_max_words,
    )
    state.update(updates)
    save_state(state)

    word_count = len(state.get("script", "").split())
    print("Script OK")
    print(f"word_count: {word_count}")


if __name__ == "__main__":
    asyncio.run(main())
