"""Progress tracking for workflow execution."""

import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ProgressCallback:
    """Callback handler for workflow progress updates."""

    def __init__(self):
        """Initialize progress tracker."""
        self.start_time = datetime.now()
        self.node_times = {}

    def on_node_start(self, node_name: str, state: Dict[str, Any]):
        """
        Called when a node starts execution.

        Args:
            node_name: Name of the node
            state: Current workflow state
        """
        self.node_times[node_name] = datetime.now()

        logger.info("Starting: %s", node_name)

        # Show context-specific info
        if node_name == "research":
            logger.info("   Topic: %s", state.get('topic', 'Unknown'))
        elif node_name == "synthesize_script":
            sources = len(state.get('research_data', []))
            logger.info("   Research sources: %s", sources)
        elif node_name == "collect_images":
            sections = len(state.get('script_sections', []))
            logger.info("   Script sections: %s", sections)
        elif node_name == "download_images":
            images = len(state.get('images', []))
            logger.info("   Images to download: %s", images)

    def on_node_end(self, node_name: str, result: Dict[str, Any]):
        """
        Called when a node completes execution.

        Args:
            node_name: Name of the node
            result: Node result (updated state fields)
        """
        # Calculate duration
        if node_name in self.node_times:
            start = self.node_times[node_name]
            duration = (datetime.now() - start).total_seconds()
            logger.info("Completed: %s (%.1fs)", node_name, duration)
        else:
            logger.info("Completed: %s", node_name)

        # Show relevant metrics from metadata
        if 'metadata' in result:
            meta = result['metadata']

            if 'word_count' in meta:
                logger.info("   Script: %s words", meta['word_count'])

            if 'images_collected' in meta:
                logger.info("   Images collected: %s", meta['images_collected'])

            if 'images_downloaded' in meta:
                failed = meta.get('images_failed', 0)
                logger.info("   Images downloaded: %s (failed: %s)", meta['images_downloaded'], failed)

        if 'api_call_counts' in result:
            logger.debug("   API calls: %s", result['api_call_counts'])

    def on_workflow_complete(self, state: Dict[str, Any]):
        """
        Called when entire workflow completes.

        Args:
            state: Final workflow state
        """
        duration = (datetime.now() - self.start_time).total_seconds()

        logger.info("Workflow complete in %.1fs", duration)
        logger.info("   Run ID: %s", state['run_id'])
        logger.info("   Output: output/%s/", state['run_id'])

        # Show summary stats
        meta = state.get('metadata', {})
        if 'word_count' in meta:
            logger.info("   Script: %s words", meta['word_count'])
        if 'images_downloaded' in meta:
            logger.info("   Images: %s", meta['images_downloaded'])
        if 'voice_path' in state and state['voice_path']:
            logger.info("   Voice: %s", state['voice_path'])

    def on_error(self, node_name: str, error: Exception):
        """
        Called when a node encounters an error.

        Args:
            node_name: Name of the node
            error: Exception that occurred
        """
        logger.error("Error in %s: %s", node_name, str(error))


def wrap_node_with_progress(
    node_func,
    node_name: str,
    progress_callback: ProgressCallback
):
    """
    Wrap a node function with progress tracking.

    Args:
        node_func: The node function to wrap
        node_name: Name of the node
        progress_callback: Progress callback instance

    Returns:
        Wrapped function
    """
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            progress_callback.on_node_start(node_name, state)
            result = node_func(state)
            progress_callback.on_node_end(node_name, result)
            return result
        except Exception as e:
            progress_callback.on_error(node_name, e)
            raise

    # For async nodes
    async def async_wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            progress_callback.on_node_start(node_name, state)
            result = await node_func(state)
            progress_callback.on_node_end(node_name, result)
            return result
        except Exception as e:
            progress_callback.on_error(node_name, e)
            raise

    # Return async version if node_func is async
    import inspect
    if inspect.iscoroutinefunction(node_func):
        return async_wrapped
    return wrapped
