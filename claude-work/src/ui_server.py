"""Lightweight local web UI for the educational video workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web
from dotenv import load_dotenv

from .agents.images import (
    collect_images_node,
    download_images_node,
    map_images_node,
    validate_image_count,
)
from .agents.outline import generate_outline, revise_outline
from .agents.research import research_node
from .agents.script import (
    parse_script_node,
    polish_script,
    revise_script_with_feedback,
    synthesize_script_node,
    validate_script_word_count,
)
from .agents.voice import generate_voice_node
from .config import load_config
from .models import WorkflowState, create_initial_state, generate_run_id
from .prompts import script_judge_prompt
from .utils.openai_client import OpenAIClient
from .utils.output_manager import OutputManager
from .utils.tavily_client import TavilyClient
from .workflow import save_outputs_node

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


class UIState:
    def __init__(self, repo_root: Path):
        load_dotenv(repo_root / ".env")
        self.config = load_config()
        self.openai_client = OpenAIClient(
            api_key=self.config.openai_api_key,
            model=self.config.model_name,
            max_concurrent=self.config.max_concurrent_openai,
            max_per_minute=self.config.max_rate_openai_per_min,
        )
        self.tavily_client = TavilyClient(
            api_key=self.config.tavily_api_key,
            max_concurrent=self.config.max_concurrent_tavily,
            max_per_minute=self.config.max_rate_tavily_per_min,
        )
        self.output_manager = OutputManager(base_dir=str(repo_root / "output"))
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def log_event(self, run_id: str, message: str, level: str = "info") -> None:
        run = self.runs.get(run_id)
        if not run:
            return
        event = {"ts": _iso_now(), "level": level, "message": message}
        run["events"].append(event)
        if len(run["events"]) > 200:
            run["events"] = run["events"][-200:]

    def create_task(self, coro: asyncio.Future, task_type: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        self.tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "status": "running",
            "created_at": _iso_now(),
        }

        async def _runner() -> None:
            try:
                result = await coro
                self.tasks[task_id].update({
                    "status": "done",
                    "result": result,
                    "finished_at": _iso_now(),
                })
            except Exception as exc:
                self.tasks[task_id].update({
                    "status": "error",
                    "error": str(exc),
                    "finished_at": _iso_now(),
                })

        asyncio.create_task(_runner())
        return task_id

    def outputs_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        run_dir = self.output_manager.get_run_dir(run_id)
        if not run_dir.exists():
            return None
        files = []
        candidates = [
            "script.md",
            "outline.md",
            "images.json",
            "meta.json",
            str(Path("voice") / "narration.mp3"),
        ]
        for rel in candidates:
            path = run_dir / rel
            if path.exists():
                files.append(str(path.relative_to(run_dir)))
        images_dir = run_dir / "images"
        image_count = 0
        if images_dir.exists():
            image_count = len(list(images_dir.glob("*")))
        return {
            "run_dir": str(run_dir),
            "files": files,
            "image_count": image_count,
        }


async def _run_pre_script(ui: UIState, run_id: str) -> None:
    run = ui.runs[run_id]
    state: WorkflowState = run["state"]
    try:
        run["status"] = "running_pre_script"
        run["stage"] = "research"
        ui.log_event(run_id, "Research started")
        update = await research_node(state, ui.openai_client)
        state.update(update)
        ui.log_event(run_id, f"Research complete ({len(state.get('research_data', []))} sources)")

        run["stage"] = "synthesize_script"
        ui.log_event(run_id, "Script generation started")

        while True:
            update = await synthesize_script_node(
                state,
                ui.openai_client,
                ui.config.script_min_words,
                ui.config.script_max_words,
            )
            state.update(update)
            decision = validate_script_word_count(state)
            if decision == "retry":
                ui.log_event(run_id, "Script validation failed; retrying")
                continue
            if decision == "max_retries":
                ui.log_event(run_id, "Script validation failed; continuing with latest draft", level="warn")
            break

        word_count = len(state.get("script", "").split())
        ui.log_event(run_id, f"Script ready ({word_count} words)")
        run["status"] = "awaiting_review"
        run["stage"] = "review"
    except Exception as exc:
        run["status"] = "error"
        run["stage"] = "error"
        run["error"] = str(exc)
        ui.log_event(run_id, f"Run failed: {exc}", level="error")


async def _run_post_script(ui: UIState, run_id: str) -> None:
    run = ui.runs[run_id]
    state: WorkflowState = run["state"]
    try:
        run["status"] = "running_post_script"

        run["stage"] = "parse_script"
        ui.log_event(run_id, "Parsing script")
        update = await parse_script_node(state, ui.openai_client)
        state.update(update)

        run["stage"] = "collect_images"
        ui.log_event(run_id, "Collecting images")
        while True:
            update = await collect_images_node(state, ui.tavily_client, ui.openai_client)
            state.update(update)
            decision = validate_image_count(state)
            if decision == "retry":
                ui.log_event(run_id, "Image count low; retrying collection")
                continue
            if decision == "fallback":
                ui.log_event(run_id, "Image count low; continuing with available results", level="warn")
            break

        run["stage"] = "map_images"
        ui.log_event(run_id, "Mapping images to script")
        update = await map_images_node(state, ui.openai_client)
        state.update(update)

        run["stage"] = "download_images"
        ui.log_event(run_id, "Downloading images")
        update = await download_images_node(
            state,
            ui.openai_client,
            output_dir=str(ui.output_manager.base_dir),
        )
        state.update(update)

        run["stage"] = "generate_voice"
        ui.log_event(run_id, "Generating voice")
        update = await generate_voice_node(state, ui.openai_client)
        state.update(update)

        run["stage"] = "save_outputs"
        ui.log_event(run_id, "Saving outputs")
        update = await save_outputs_node(state, ui.output_manager)
        state.update(update)

        run["status"] = "complete"
        run["stage"] = "complete"
        ui.log_event(run_id, "Run complete")
    except Exception as exc:
        run["status"] = "error"
        run["stage"] = "error"
        run["error"] = str(exc)
        ui.log_event(run_id, f"Run failed: {exc}", level="error")


async def handle_index(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    static_dir = request.app["static_dir"]
    index_path = static_dir / "index.html"
    return web.FileResponse(index_path)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_run_start(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    topic = (data.get("topic") or "").strip()
    outline = (data.get("outline") or "").strip()
    if not topic:
        return web.json_response({"error": "topic is required"}, status=400)

    run_id = generate_run_id()
    state = create_initial_state(topic, run_id, script_outline=outline)
    ui.output_manager.create_run_directory(run_id)

    ui.runs[run_id] = {
        "run_id": run_id,
        "state": state,
        "status": "queued",
        "stage": "queued",
        "events": [],
        "error": None,
        "created_at": _iso_now(),
        "lock": asyncio.Lock(),
    }
    ui.log_event(run_id, "Run created")
    asyncio.create_task(_run_pre_script(ui, run_id))

    return web.json_response({"run_id": run_id})


async def handle_run_continue(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    run_id = data.get("run_id")
    if not run_id or run_id not in ui.runs:
        return web.json_response({"error": "run_id not found"}, status=404)

    run = ui.runs[run_id]
    if run["status"] != "awaiting_review":
        return web.json_response({"error": f"run status is {run['status']}"}, status=400)

    asyncio.create_task(_run_post_script(ui, run_id))
    return web.json_response({"run_id": run_id, "status": "running_post_script"})


async def handle_run_status(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    run_id = request.match_info.get("run_id")
    if not run_id or run_id not in ui.runs:
        return web.json_response({"error": "run_id not found"}, status=404)

    run = ui.runs[run_id]
    state = run["state"]
    payload = {
        "run_id": run_id,
        "status": run["status"],
        "stage": run["stage"],
        "error": run.get("error"),
        "topic": state.get("topic"),
        "outline": state.get("script_outline", ""),
        "script": state.get("script", ""),
        "metadata": state.get("metadata", {}),
        "events": run.get("events", []),
    }
    if run["status"] == "complete":
        payload["outputs"] = ui.outputs_summary(run_id)
    return web.json_response(payload)


async def handle_run_update_script(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    run_id = data.get("run_id")
    script = (data.get("script") or "").strip()

    if not run_id or run_id not in ui.runs:
        return web.json_response({"error": "run_id not found"}, status=404)
    run = ui.runs[run_id]
    if run["status"] not in ("awaiting_review",):
        return web.json_response({"error": f"run status is {run['status']}"}, status=400)

    async with run["lock"]:
        run["state"]["script"] = script
        meta = run["state"].get("metadata", {})
        meta = {
            **meta,
            "word_count": len(script.split()),
            "script_updated_at": _iso_now(),
        }
        run["state"]["metadata"] = meta
    ui.log_event(run_id, "Script updated by user")
    return web.json_response({"ok": True, "word_count": len(script.split())})


async def handle_outline_generate(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    topic = (data.get("topic") or "").strip()
    if not topic:
        return web.json_response({"error": "topic is required"}, status=400)

    task_id = ui.create_task(
        generate_outline(
            topic,
            ui.openai_client,
            min_words=ui.config.script_min_words,
            max_words=ui.config.script_max_words,
        ),
        "outline.generate",
    )
    return web.json_response({"task_id": task_id})


async def handle_outline_revise(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    topic = (data.get("topic") or "").strip()
    outline = (data.get("outline") or "").strip()
    feedback = (data.get("feedback") or "").strip()
    if not topic or not outline or not feedback:
        return web.json_response({"error": "topic, outline, and feedback are required"}, status=400)

    task_id = ui.create_task(
        revise_outline(
            topic,
            outline,
            feedback,
            ui.openai_client,
            min_words=ui.config.script_min_words,
            max_words=ui.config.script_max_words,
        ),
        "outline.revise",
    )
    return web.json_response({"task_id": task_id})


async def handle_script_validate(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    script = (data.get("script") or "").strip()
    if not script:
        return web.json_response({"error": "script is required"}, status=400)

    min_words = int(data.get("min_words") or ui.config.script_min_words)
    max_words = int(data.get("max_words") or ui.config.script_max_words)

    async def _validate() -> Dict[str, Any]:
        prompt = script_judge_prompt(min_words, max_words, script)
        response = await ui.openai_client.generate(
            prompt,
            max_tokens=500,
            temperature=0.0,
        )
        parsed = _extract_json(response)
        if not parsed:
            parsed = {"pass": False, "issues": ["invalid_json"], "fix_instructions": ""}
        parsed["word_count"] = len(script.split())
        return parsed

    task_id = ui.create_task(_validate(), "script.validate")
    return web.json_response({"task_id": task_id})


async def handle_script_revise(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    script = (data.get("script") or "").strip()
    feedback = (data.get("feedback") or "").strip()
    run_id = data.get("run_id")
    if not script or not feedback:
        return web.json_response({"error": "script and feedback are required"}, status=400)

    topic = (data.get("topic") or "").strip()
    outline = (data.get("outline") or "").strip()
    research_summary = ""
    research_data = []

    if run_id and run_id in ui.runs:
        run_state = ui.runs[run_id]["state"]
        topic = run_state.get("topic", topic)
        outline = run_state.get("script_outline", outline)
        research_summary = run_state.get("metadata", {}).get("research_synthesis", "")
        research_data = run_state.get("research_data", [])

    if not topic:
        return web.json_response({"error": "topic is required"}, status=400)

    task_id = ui.create_task(
        revise_script_with_feedback(
            topic,
            script,
            feedback,
            ui.openai_client,
            min_words=ui.config.script_min_words,
            max_words=ui.config.script_max_words,
            outline=outline,
            research_summary=research_summary,
            research_data=research_data,
        ),
        "script.revise",
    )
    return web.json_response({"task_id": task_id})


async def handle_script_polish(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    data = await request.json()
    script = (data.get("script") or "").strip()
    run_id = data.get("run_id")
    if not script:
        return web.json_response({"error": "script is required"}, status=400)

    topic = (data.get("topic") or "").strip()
    outline = (data.get("outline") or "").strip()
    research_summary = ""
    research_data = []

    if run_id and run_id in ui.runs:
        run_state = ui.runs[run_id]["state"]
        topic = run_state.get("topic", topic)
        outline = run_state.get("script_outline", outline)
        research_summary = run_state.get("metadata", {}).get("research_synthesis", "")
        research_data = run_state.get("research_data", [])

    if not topic:
        return web.json_response({"error": "topic is required"}, status=400)

    task_id = ui.create_task(
        polish_script(
            topic,
            script,
            ui.openai_client,
            min_words=ui.config.script_min_words,
            max_words=ui.config.script_max_words,
            outline=outline,
            research_summary=research_summary,
            research_data=research_data,
        ),
        "script.polish",
    )
    return web.json_response({"task_id": task_id})


async def handle_task_status(request: web.Request) -> web.Response:
    ui: UIState = request.app["ui"]
    task_id = request.match_info.get("task_id")
    task = ui.tasks.get(task_id)
    if not task:
        return web.json_response({"error": "task_id not found"}, status=404)
    return web.json_response(task)


def build_app(repo_root: Path) -> web.Application:
    app = web.Application(client_max_size=5 * 1024 * 1024)
    app["ui"] = UIState(repo_root)
    static_dir = repo_root / "src" / "ui" / "static"
    app["static_dir"] = static_dir

    app.router.add_get("/", handle_index)
    app.router.add_get("/api/health", handle_health)
    app.router.add_post("/api/run/start", handle_run_start)
    app.router.add_post("/api/run/continue", handle_run_continue)
    app.router.add_get("/api/run/status/{run_id}", handle_run_status)
    app.router.add_post("/api/run/update_script", handle_run_update_script)

    app.router.add_post("/api/outline/generate", handle_outline_generate)
    app.router.add_post("/api/outline/revise", handle_outline_revise)

    app.router.add_post("/api/script/validate", handle_script_validate)
    app.router.add_post("/api/script/revise", handle_script_revise)
    app.router.add_post("/api/script/polish", handle_script_polish)

    app.router.add_get("/api/task/{task_id}", handle_task_status)
    app.router.add_static("/static", static_dir, show_index=False)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local web UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8787, type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    app = build_app(repo_root)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
