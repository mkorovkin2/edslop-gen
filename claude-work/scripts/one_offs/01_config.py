import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from scripts.one_offs._shared import is_placeholder_key


def main() -> None:
    cfg = load_config()

    if is_placeholder_key(cfg.openai_api_key) or is_placeholder_key(cfg.tavily_api_key):
        print("Config looks like placeholder keys. Update .env with real API keys first.")
        raise SystemExit(1)

    print("Config OK")
    print(f"model_name: {cfg.model_name}")
    print(f"script_min_words: {cfg.script_min_words}")
    print(f"script_max_words: {cfg.script_max_words}")
    print(f"images_min_total: {cfg.images_min_total}")
    print(f"images_per_section: {cfg.images_per_section}")
    print(f"tts_model: {cfg.tts_model}")
    print(f"tts_voice: {cfg.tts_voice}")
    print(f"max_concurrent_tavily: {cfg.max_concurrent_tavily}")
    print(f"max_concurrent_openai: {cfg.max_concurrent_openai}")


if __name__ == "__main__":
    main()
