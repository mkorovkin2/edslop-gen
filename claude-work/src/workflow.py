"""LangGraph workflow definition."""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
try:
    # Older/newer langgraph releases may place SqliteSaver here.
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
    _HAS_SQLITE_SAVER = True
except Exception:
    SqliteSaver = None  # type: ignore
    _HAS_SQLITE_SAVER = False

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None  # type: ignore

from .models import WorkflowState
from .agents.research import research_node
from .agents.script import synthesize_script_node, parse_script_node, validate_script_word_count
from .agents.images import (
    collect_images_node,
    map_images_node,
    download_images_node,
    validate_image_count
)
from .agents.voice import generate_voice_node
from .utils.openai_client import OpenAIClient
from .utils.tavily_client import TavilyClient
from .utils.output_manager import OutputManager
from .utils.progress import ProgressCallback, wrap_node_with_progress
from .config import Config

logger = logging.getLogger(__name__)

async def save_outputs_node(
    state: WorkflowState,
    output_manager: OutputManager
) -> Dict[str, Any]:
    """
    Final node: Save all outputs to disk.

    Args:
        state: Current workflow state
        output_manager: Output manager instance

    Returns:
        Dict with completed metadata
    """
    run_id = state['run_id']

    # Write script.md
    await output_manager.write_script(run_id, state['script'])

    # Write outline.md if available
    outline_text = state.get('script_outline', '').strip()
    if outline_text:
        await output_manager.write_outline(run_id, outline_text)

    # Write images.json
    await output_manager.write_images_json(
        run_id,
        state['images_mapping'],
        state['images']
    )

    # Write meta.json with all metadata
    metadata = state['metadata']
    metadata.update({
        "run_id": run_id,
        "topic": state['topic'],
        "model_name": os.getenv('MODEL_NAME', 'gpt-5.2'),
        "outline_used": bool(outline_text),
        "script_word_count": len(state['script'].split()),
        "script_sections_count": len(state['script_sections']),
        "images_collected": len(state['images']),
        "images_downloaded": metadata.get('images_downloaded', 0),
        "images_failed": metadata.get('images_failed', 0),
        "research_sources": metadata.get('research_source_count', 0),
        "research_queries_used": metadata.get('research_queries_used', []),
        "api_calls_tavily": state.get('api_call_counts', {}).get('tavily', 0),
        "api_calls_openai": state.get('api_call_counts', {}).get('openai', 0),
        "voice_chunks": metadata.get('voice_chunks', 1),
        "voice_duration_seconds": metadata.get('voice_duration_seconds'),
        "tts_model": metadata.get('tts_model', 'tts-1-hd'),
        "tts_voice": metadata.get('tts_voice', 'alloy'),
        "tts_speed": metadata.get('tts_speed', 1.2)
    })

    await output_manager.write_metadata(run_id, metadata)

    return {
        "metadata": metadata
    }


def create_workflow(
    config: Config,
    progress_callback: Optional[ProgressCallback] = None
) -> StateGraph:
    """
    Create and configure the LangGraph workflow.

    Args:
        config: Application configuration

    Returns:
        Compiled StateGraph workflow
    """
    # Initialize API clients
    tavily_client = TavilyClient(
        api_key=config.tavily_api_key,
        max_concurrent=config.max_concurrent_tavily,
        max_per_minute=config.max_rate_tavily_per_min
    )

    openai_client = OpenAIClient(
        api_key=config.openai_api_key,
        model=config.model_name,
        max_concurrent=config.max_concurrent_openai,
        max_per_minute=config.max_rate_openai_per_min
    )

    output_manager = OutputManager()

    # Initialize checkpointing for crash recovery (fallback to in-memory if sqlite saver unavailable)
    memory = None
    if _HAS_SQLITE_SAVER and SqliteSaver is not None:
        logger.info("Checkpointing enabled (sqlite)")
        memory = SqliteSaver.from_conn_string("checkpoints.db")
    elif MemorySaver is not None:
        logger.info("Checkpointing enabled (memory)")
        memory = MemorySaver()
    else:
        logger.info("Checkpointing disabled (no saver available)")

    def _wrap(node_func, node_name: str):
        if progress_callback is None:
            return node_func
        return wrap_node_with_progress(node_func, node_name, progress_callback)

    # Create workflow graph
    graph = StateGraph(WorkflowState)

    # Define node functions with dependency injection
    async def research(state: WorkflowState) -> Dict[str, Any]:
        return await research_node(state, tavily_client, openai_client)

    async def synthesize_script(state: WorkflowState) -> Dict[str, Any]:
        return await synthesize_script_node(
            state,
            openai_client,
            config.script_min_words,
            config.script_max_words
        )

    async def parse_script(state: WorkflowState) -> Dict[str, Any]:
        return await parse_script_node(state, openai_client)

    async def collect_images(state: WorkflowState) -> Dict[str, Any]:
        return await collect_images_node(state, tavily_client, openai_client)

    async def map_images(state: WorkflowState) -> Dict[str, Any]:
        return await map_images_node(state, openai_client)

    async def download_images(state: WorkflowState) -> Dict[str, Any]:
        return await download_images_node(state, openai_client)

    async def generate_voice(state: WorkflowState) -> Dict[str, Any]:
        return await generate_voice_node(state, openai_client)

    async def save_outputs(state: WorkflowState) -> Dict[str, Any]:
        return await save_outputs_node(state, output_manager)

    # Add all nodes
    graph.add_node("research", _wrap(research, "research"))
    graph.add_node("synthesize_script", _wrap(synthesize_script, "synthesize_script"))
    graph.add_node("parse_script", _wrap(parse_script, "parse_script"))
    graph.add_node("collect_images", _wrap(collect_images, "collect_images"))
    graph.add_node("map_images", _wrap(map_images, "map_images"))
    graph.add_node("download_images", _wrap(download_images, "download_images"))
    graph.add_node("generate_voice", _wrap(generate_voice, "generate_voice"))
    graph.add_node("save_outputs", _wrap(save_outputs, "save_outputs"))

    # Define workflow edges
    graph.set_entry_point("research")
    graph.add_edge("research", "synthesize_script")

    # Conditional edge for script validation
    graph.add_conditional_edges(
        "synthesize_script",
        validate_script_word_count,
        {
            "continue": "parse_script",
            "retry": "synthesize_script",
            "max_retries": "parse_script"  # Continue anyway after max retries
        }
    )

    graph.add_edge("parse_script", "collect_images")

    # Conditional edge for image validation
    graph.add_conditional_edges(
        "collect_images",
        validate_image_count,
        {
            "sufficient": "map_images",
            "retry": "collect_images",
            "fallback": "map_images"  # Continue with what we have
        }
    )

    graph.add_edge("map_images", "download_images")
    graph.add_edge("download_images", "generate_voice")
    graph.add_edge("generate_voice", "save_outputs")
    graph.add_edge("save_outputs", END)

    # Compile workflow with checkpointing if available
    if memory is not None:
        return graph.compile(checkpointer=memory)
    return graph.compile()


async def run_workflow(topic: str, config: Config, script_outline: str = "") -> Dict[str, Any]:
    """
    Run the complete workflow for a given topic.

    Args:
        topic: User's input topic
        config: Application configuration

    Returns:
        Final workflow state
    """
    from .models import create_initial_state, generate_run_id

    # Generate run ID and create initial state
    run_id = generate_run_id()
    initial_state = create_initial_state(topic, run_id, script_outline=script_outline)

    # Create output directory
    output_manager = OutputManager()
    output_manager.create_run_directory(run_id)

    # Create and run workflow
    progress_callback = ProgressCallback()
    workflow = create_workflow(config, progress_callback=progress_callback)

    logger.info("Workflow: starting run_id=%s topic=%s", run_id, topic)
    print(f"\n🚀 Starting workflow for: {topic}")
    print(f"📁 Run ID: {run_id}\n")

    try:
        final_state = await workflow.ainvoke(
            initial_state,
            {"configurable": {"thread_id": run_id}}
        )
        progress_callback.on_workflow_complete(final_state)
        logger.info("Workflow: completed run_id=%s", run_id)
        return final_state
    except Exception as e:
        print(f"\n❌ Workflow failed: {str(e)}")
        logger.exception("Workflow failed")
        raise
