"""Crash recovery persistence — save/load full state snapshots as thoughts files."""

import json
import os
from typing import Any


def save_thoughts(session_id: str, step_name: str, state: dict[str, Any]) -> str:
    """Save a full state snapshot to a thoughts JSON file. Returns the file path."""
    thoughts_dir = os.path.join("output", session_id, "thoughts")
    os.makedirs(thoughts_dir, exist_ok=True)

    file_path = os.path.join(thoughts_dir, f"{step_name}.json")

    # Serialize state — filter out non-serializable items
    serializable_state = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            serializable_state[key] = value
        except (TypeError, ValueError):
            serializable_state[key] = str(value)

    with open(file_path, "w") as f:
        json.dump(serializable_state, f, indent=2)

    return file_path


def load_latest_thoughts(session_id: str) -> dict[str, Any] | None:
    """Load the most recent thoughts file for a session. Returns None if no thoughts exist."""
    thoughts_dir = os.path.join("output", session_id, "thoughts")
    if not os.path.exists(thoughts_dir):
        return None

    files = sorted(os.listdir(thoughts_dir))
    if not files:
        return None

    # Find the latest file by modification time
    latest_file = max(
        files,
        key=lambda f: os.path.getmtime(os.path.join(thoughts_dir, f)),
    )

    file_path = os.path.join(thoughts_dir, latest_file)
    with open(file_path, "r") as f:
        return json.load(f)


def load_thoughts_for_step(session_id: str, step_name: str) -> dict[str, Any] | None:
    """Load thoughts for a specific step. Returns None if not found."""
    file_path = os.path.join("output", session_id, "thoughts", f"{step_name}.json")
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        return json.load(f)
