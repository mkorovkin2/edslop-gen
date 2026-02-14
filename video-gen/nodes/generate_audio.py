"""Node: Generate audio files from approved scripts using ElevenLabs TTS."""

import logging
import os

from dotenv import load_dotenv

from persistence import save_thoughts
from state import AgentState

load_dotenv()


def generate_audio(state: AgentState) -> dict:
    """Call ElevenLabs API to generate .mp3 audio for each approved script."""
    sid = state["session_id"]
    log = logging.getLogger(f"video_agent.{sid}")
    log.info("=== NODE: generate_audio ===")

    # Skip if audio already generated
    if state.get("audio_paths"):
        log.info("Audio already generated, skipping")
        return {}

    approved = state["approved_scripts"]
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is not set in environment")
    if not voice_id:
        raise ValueError("ELEVENLABS_VOICE_ID is not set in environment")

    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=api_key)

    audio_dir = os.path.join("output", sid, "audio")
    scripts_dir = os.path.join("output", sid, "scripts")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    audio_paths = []

    print("\n" + "=" * 60)
    print("GENERATING AUDIO")
    print("=" * 60)

    for script in approved:
        vid = script["variant_id"]
        title = script["title"]
        text = script["script_text"]

        # Save script text file
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        safe_title = safe_title.strip().replace(" ", "_")
        script_path = os.path.join(scripts_dir, f"variant_{vid}_{safe_title}.txt")
        with open(script_path, "w") as f:
            f.write(f"Title: {title}\n")
            f.write(f"Word Count: {script['word_count']}\n")
            f.write(f"Variant ID: {vid}\n")
            f.write("-" * 40 + "\n\n")
            f.write(text)

        log.info(f"Saved script to {script_path}")

        # Generate audio via ElevenLabs
        log.info(f"Generating audio for variant {vid}: {title}")
        print(f"  Generating audio for: {title}...")

        audio_generator = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
        )

        audio_path = os.path.join(audio_dir, f"variant_{vid}_{safe_title}.mp3")
        with open(audio_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        audio_paths.append(audio_path)
        log.info(f"Saved audio to {audio_path}")
        print(f"  Saved: {audio_path}")

    new_state = {
        "audio_paths": audio_paths,
        "current_step": "generate_audio",
    }

    save_thoughts(sid, "07_generate_audio", {**state, **new_state})
    log.debug("Saved thoughts for generate_audio")

    # Final summary
    output_dir = os.path.join("output", sid)
    abs_output = os.path.abspath(output_dir)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nAll files saved to: {abs_output}")
    print(f"  Scripts: {os.path.abspath(scripts_dir)}")
    print(f"  Audio:   {os.path.abspath(audio_dir)}")
    print(f"  Logs:    {os.path.abspath(os.path.join(output_dir, 'agent.log'))}")
    print(f"  Thoughts: {os.path.abspath(os.path.join(output_dir, 'thoughts'))}")
    print(f"\n{len(audio_paths)} audio files generated.")
    print("=" * 60 + "\n")

    return new_state
