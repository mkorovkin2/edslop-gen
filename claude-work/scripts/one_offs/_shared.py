import json
import os
import sys
from pathlib import Path

STATE_PATH = Path("output/one_off_state.json")


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_state() -> dict:
    if not STATE_PATH.exists():
        print(f"State file not found: {STATE_PATH}. Run scripts/one_offs/04_research_node.py first.")
        sys.exit(1)
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_state(state: dict, updates: dict) -> dict:
    state.update(updates)
    return state


def get_topic() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return os.getenv("ONE_OFF_TOPIC", "neural networks")


def is_placeholder_key(value: str) -> bool:
    v = (value or "").lower()
    return (
        "your-openai-api-key" in v
        or "your-tavily-api-key" in v
        or v.startswith("sk-your")
        or v.startswith("tvly-your")
    )
