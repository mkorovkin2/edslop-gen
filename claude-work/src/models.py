"""Data models and state definitions for the workflow."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
import operator
from datetime import datetime


class WorkflowState(TypedDict):
    """
    State passed between workflow nodes.

    Uses Annotated with 'add' reducer for accumulating lists.
    """
    # Core data
    topic: str                                            # User's input topic
    run_id: str                                           # Unique run identifier
    script_outline: str                                   # Optional script outline
    research_data: Annotated[List[Dict[str, Any]], operator.add]   # Tavily research results (accumulated)
    script: str                                           # Generated script (200-500 words)
    script_sections: List[Dict[str, Any]]                 # Parsed script sections
    images: Annotated[List[Dict[str, Any]], operator.add]          # Collected image metadata (accumulated)
    images_mapping: Dict[str, List[int]]                  # Script part -> image indices
    voice_path: str                                       # Path to generated audio
    metadata: Dict[str, Any]                              # Run metadata (updated incrementally)

    # Error handling & retry tracking
    error_state: Optional[Dict[str, Any]]                 # Current error state if any
    retry_counts: Annotated[Dict[str, int], operator.or_]          # Retry attempts per node
    node_metadata: Dict[str, Dict[str, Any]]              # Per-node execution metadata

    # API rate limiting
    api_call_counts: Dict[str, int]                       # API calls per service (tavily, openai)


class ImageMetadata(BaseModel):
    """Metadata for a collected image."""
    url: str = Field(..., description="Image URL")
    description: str = Field(..., description="Image description from Tavily")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    query_used: str = Field(..., description="Search query that found this image")
    local_path: Optional[str] = Field(None, description="Local file path after download")


class ScriptSection(BaseModel):
    """A section of the script with sentences."""
    section_id: str = Field(..., description="Unique section identifier")
    title: Optional[str] = Field(None, description="Section title/heading")
    text: str = Field(..., description="Full section text")
    sentences: List[str] = Field(..., description="Individual sentences in section")
    start_index: int = Field(..., ge=0, description="Character start index in full script")
    end_index: int = Field(..., ge=0, description="Character end index in full script")


class ResearchResult(BaseModel):
    """Result from Tavily research."""
    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Source URL")
    content: str = Field(..., description="Extracted content")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    published_date: Optional[str] = Field(None, description="Publication date if available")


class RunMetadata(BaseModel):
    """Complete metadata for a workflow run."""
    run_id: str = Field(..., description="Unique run identifier")
    topic: str = Field(..., description="User's input topic")
    started_at: str = Field(..., description="ISO timestamp when run started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when run completed")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Total duration in seconds")

    # Script metadata
    script_word_count: int = Field(..., ge=0, description="Final script word count")
    script_sections_count: int = Field(..., ge=0, description="Number of script sections")
    script_retry_count: int = Field(default=0, ge=0, description="Script generation retries")

    # Image metadata
    images_collected: int = Field(..., ge=0, description="Total images collected")
    images_downloaded: int = Field(..., ge=0, description="Successfully downloaded images")
    images_failed: int = Field(default=0, ge=0, description="Failed image downloads")
    image_queries_used: int = Field(..., ge=0, description="Number of image search queries")

    # Research metadata
    research_sources: int = Field(..., ge=0, description="Number of research sources")
    research_queries_used: List[str] = Field(..., description="Search queries used for research")

    # API usage metadata
    api_calls_tavily: int = Field(..., ge=0, description="Total Tavily API calls")
    api_calls_openai: int = Field(..., ge=0, description="Total OpenAI API calls")
    estimated_cost_usd: Optional[float] = Field(None, ge=0, description="Estimated run cost in USD")

    # Voice metadata
    voice_chunks: int = Field(default=1, ge=1, description="Number of TTS chunks generated")
    voice_duration_seconds: Optional[float] = Field(None, ge=0, description="Audio duration in seconds")

    # Model versions
    model_name: str = Field(..., description="LLM model used")
    tts_model: str = Field(..., description="TTS model used")
    tts_voice: str = Field(..., description="TTS voice used")

    # Error tracking
    errors_encountered: List[Dict[str, Any]] = Field(default_factory=list, description="Errors during run")
    warnings: List[str] = Field(default_factory=list, description="Warnings during run")


def create_initial_state(topic: str, run_id: str, script_outline: str = "") -> WorkflowState:
    """
    Create initial workflow state for a new run.

    Args:
        topic: User's input topic
        run_id: Unique run identifier

    Returns:
        WorkflowState: Initial state with empty collections
    """
    return WorkflowState(
        topic=topic,
        run_id=run_id,
        script_outline=script_outline,
        research_data=[],
        script="",
        script_sections=[],
        images=[],
        images_mapping={},
        voice_path="",
        metadata={
            "started_at": datetime.now().isoformat(),
            "topic": topic,
            "run_id": run_id
        },
        error_state=None,
        retry_counts={},
        node_metadata={},
        api_call_counts={"tavily": 0, "openai": 0}
    )


def generate_run_id() -> str:
    """
    Generate a unique run identifier.

    Format: run_YYYYMMDD_HHMMSS_XXX
    where XXX is a 3-digit random number for uniqueness.

    Returns:
        str: Unique run identifier
    """
    import random
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(100, 999)
    return f"run_{timestamp}_{random_suffix}"
