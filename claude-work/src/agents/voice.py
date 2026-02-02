"""Voice synthesis agent using OpenAI TTS."""

import os
import logging
from typing import Dict, Any, List
from datetime import datetime
from pydub import AudioSegment
from pathlib import Path
import aiofiles

from ..utils.openai_client import OpenAIClient
from ..models import WorkflowState

logger = logging.getLogger(__name__)


def split_script_at_sentences(script: str, max_chars: int = 4000) -> List[str]:
    """
    Split script into chunks at sentence boundaries.

    Args:
        script: Full script text
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Split into sentences (simple implementation)
    sentences = []
    current = ""

    for char in script:
        current += char
        if char in '.!?' and len(current) > 50:  # End of sentence
            sentences.append(current.strip())
            current = ""

    if current.strip():
        sentences.append(current.strip())

    # Group sentences into chunks under max_chars
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def generate_voice_node(
    state: WorkflowState,
    openai_client: OpenAIClient,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Generate speech from script using OpenAI TTS.

    Handles chunking for scripts exceeding 4096 character limit.

    Args:
        state: Current workflow state
        openai_client: OpenAI API client
        output_dir: Base output directory

    Returns:
        Dict with updated voice_path and metadata
    """
    # Validate input
    if not state.get('script'):
        raise ValueError("No script available for voice generation")

    script = state['script']
    run_id = state['run_id']

    tts_model = os.getenv('TTS_MODEL', 'tts-1-hd')
    tts_voice = os.getenv('TTS_VOICE', 'alloy')
    logger.info(
        "Voice: generating audio (model=%s, voice=%s, script_chars=%d)",
        tts_model,
        tts_voice,
        len(script)
    )

    # Create output directory
    voice_dir = Path(output_dir) / run_id / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    # Check if chunking is needed
    if len(script) > 4000:
        logger.info("Voice: script length %d exceeds TTS limit; chunking", len(script))

        # Split script into chunks
        chunks = split_script_at_sentences(script, max_chars=4000)
        logger.info("Voice: split into %d chunks", len(chunks))

        # Generate audio for each chunk
        audio_segments = []

        for i, chunk in enumerate(chunks):
            logger.info("Voice: generating chunk %d/%d", i + 1, len(chunks))

            audio_data = await openai_client.generate_speech(
                text=chunk,
                voice=tts_voice,
                model=tts_model
            )

            # Save temporary chunk
            temp_path = voice_dir / f"chunk_{i:02d}.mp3"
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(audio_data)

            # Load as AudioSegment
            audio_segments.append(AudioSegment.from_mp3(str(temp_path)))

        # Concatenate all chunks
        logger.info("Voice: concatenating audio segments")
        final_audio = sum(audio_segments)  # Sum concatenates AudioSegments

        # Export final audio
        final_path = voice_dir / "narration.mp3"
        final_audio.export(str(final_path), format="mp3")

        # Clean up temporary chunks
        for i in range(len(chunks)):
            temp_path = voice_dir / f"chunk_{i:02d}.mp3"
            if temp_path.exists():
                temp_path.unlink()

        num_chunks = len(chunks)

    else:
        # Single API call - no chunking needed
        logger.info("Voice: generating audio in single request")

        audio_data = await openai_client.generate_speech(
            text=script,
            voice=tts_voice,
            model=tts_model
        )

        final_path = voice_dir / "narration.mp3"
        async with aiofiles.open(final_path, 'wb') as f:
            await f.write(audio_data)

        num_chunks = 1

    # Calculate audio duration
    try:
        audio = AudioSegment.from_mp3(str(final_path))
        duration_seconds = len(audio) / 1000.0  # Convert ms to seconds
    except Exception:
        duration_seconds = None

    duration_label = f"{duration_seconds:.1f}s" if duration_seconds is not None else "unknown"
    logger.info(
        "Voice: audio generated (%s, duration=%s)",
        str(final_path),
        duration_label
    )

    return {
        "voice_path": str(final_path),
        "metadata": {
            **state.get("metadata", {}),
            "voice_generated_at": datetime.now().isoformat(),
            "voice_chunks": num_chunks,
            "voice_duration_seconds": duration_seconds,
            "tts_model": tts_model,
            "tts_voice": tts_voice
        },
        "api_call_counts": {
            **state.get("api_call_counts", {}),
            "openai": state.get("api_call_counts", {}).get("openai", 0) + num_chunks
        }
    }
