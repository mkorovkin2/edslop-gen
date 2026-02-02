"""CLI entry point for educational video generation."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .agents.outline import generate_outline, revise_outline
from .workflow import run_workflow
from .utils.openai_client import OpenAIClient
from .utils.logging_utils import configure_logging


def _parse_args(argv):
    skip_outline = False
    topic_parts = []
    for arg in argv:
        if arg in ("--no-outline", "--skip-outline"):
            skip_outline = True
            continue
        topic_parts.append(arg)
    topic = " ".join(topic_parts) if topic_parts else None
    return topic, skip_outline


def _read_multiline(prompt: str) -> str:
    print(prompt)
    print("(finish with a single '.' on its own line)")
    lines = []
    while True:
        line = input()
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


async def _outline_flow(topic: str, openai_client: OpenAIClient) -> str:
    outline = await generate_outline(topic, openai_client)
    auto_accept = not sys.stdin.isatty()

    while True:
        print("\n" + "-" * 60)
        print("Outline draft:")
        print(outline)
        print("-" * 60)

        if auto_accept:
            return outline

        print("Outline options: [a]ccept, [e]dit, [r]egenerate, [m]anual, [q]uit")
        choice = input("Choice (a/e/r/m/q or type feedback): ").strip()

        if choice == "" or choice.lower() in ("a", "accept"):
            return outline
        if choice.lower() in ("q", "quit", "exit"):
            print("Exiting at your request.")
            sys.exit(1)
        if choice.lower() in ("r", "regenerate"):
            outline = await generate_outline(topic, openai_client)
            continue
        if choice.lower() in ("m", "manual"):
            manual = _read_multiline("Paste a replacement outline:")
            if manual:
                outline = manual
            continue
        if choice.lower() in ("e", "edit", "revise"):
            feedback = input("Describe changes: ").strip()
            if not feedback:
                continue
        else:
            feedback = choice

        outline = await revise_outline(topic, outline, feedback, openai_client)


async def main_async(topic: str = None, skip_outline: bool = False):
    """
    Async main function.

    Args:
        topic: Optional topic (if not provided, will prompt user)
    """
    configure_logging()

    print("=" * 60)
    print("  Educational Video Content Generator")
    print("  Powered by LangGraph + GPT-5.2 + Tavily")
    print("=" * 60)

    # Load configuration
    try:
        config = load_config()
        print("\n✓ Configuration loaded successfully")
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease ensure you have:")
        print("  1. Created a .env file (copy from .env.example)")
        print("  2. Added your OPENAI_API_KEY and TAVILY_API_KEY")
        sys.exit(1)

    # Get topic from user if not provided
    if not topic:
        print("\n" + "=" * 60)
        topic = input("Enter a technical topic for your video: ").strip()

        if not topic:
            print("❌ No topic provided. Exiting.")
            sys.exit(1)

    print(f"\n📚 Topic: {topic}")
    print(f"📊 Script target: {config.script_min_words}-{config.script_max_words} words")
    print(f"🖼️  Images target: {config.images_min_total}+ images")
    print(f"🤖 Model: {config.model_name}")

    # Optional outline flow
    script_outline = ""
    if not skip_outline:
        try:
            print("\nGenerating a rough outline...")
            outline_client = OpenAIClient(
                api_key=config.openai_api_key,
                model=config.model_name,
                max_concurrent=config.max_concurrent_openai,
                max_per_minute=config.max_rate_openai_per_min
            )
            script_outline = await _outline_flow(topic, outline_client)
            if script_outline:
                print("Outline accepted. Continuing the workflow...\n")
        except Exception as e:
            print(f"\n❌ Outline generation failed: {e}")
            if sys.stdin.isatty():
                proceed = input("Continue without an outline? [y/N]: ").strip().lower()
                if proceed not in ("y", "yes"):
                    sys.exit(1)
            else:
                sys.exit(1)

    # Run workflow
    start_time = datetime.now()

    try:
        final_state = await run_workflow(topic, config, script_outline=script_outline)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Display results
        print("\n" + "=" * 60)
        print("✅ VIDEO CONTENT GENERATION COMPLETE!")
        print("=" * 60)

        meta = final_state['metadata']
        run_id = final_state['run_id']

        print(f"\n📁 Output directory: output/{run_id}/")
        print(f"\n📄 Script:")
        print(f"   - Words: {meta.get('script_word_count', 0)}")
        print(f"   - Sections: {meta.get('script_sections_count', 0)}")

        if meta.get('script_retry_count', 0) > 0:
            print(f"   - Retries: {meta['script_retry_count']}")

        print(f"\n🖼️  Images:")
        print(f"   - Collected: {meta.get('images_collected', 0)}")
        print(f"   - Downloaded: {meta.get('images_downloaded', 0)}")

        if meta.get('images_failed', 0) > 0:
            print(f"   - Failed: {meta['images_failed']}")

        print(f"\n🎙️  Voice:")
        print(f"   - Model: {meta.get('tts_model', 'N/A')}")
        print(f"   - Voice: {meta.get('tts_voice', 'N/A')}")

        if meta.get('voice_duration_seconds'):
            duration_min = int(meta['voice_duration_seconds'] // 60)
            duration_sec = int(meta['voice_duration_seconds'] % 60)
            print(f"   - Duration: {duration_min}m {duration_sec}s")

        if meta.get('voice_chunks', 1) > 1:
            print(f"   - Chunks: {meta['voice_chunks']}")

        print(f"\n🔧 API Usage:")
        print(f"   - Tavily calls: {meta.get('api_calls_tavily', 0)}")
        print(f"   - OpenAI calls: {meta.get('api_calls_openai', 0)}")

        print(f"\n⏱️  Total time: {duration:.1f}s")

        print(f"\n📂 Generated files:")
        output_dir = Path("output") / run_id
        if output_dir.exists():
            print(f"   ✓ {output_dir / 'script.md'}")
            outline_path = output_dir / 'outline.md'
            if outline_path.exists():
                print(f"   ✓ {outline_path}")
            print(f"   ✓ {output_dir / 'images.json'}")
            print(f"   ✓ {output_dir / 'meta.json'}")
            print(f"   ✓ {output_dir / 'voice' / 'narration.mp3'}")
            images_dir = output_dir / 'images'
            if images_dir.exists():
                num_images = len(list(images_dir.glob('*')))
                print(f"   ✓ {images_dir}/ ({num_images} images)")

        print("\n" + "=" * 60)
        print("🎉 Ready for video compilation!")
        print("=" * 60 + "\n")

        return final_state

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Synchronous entry point for CLI.

    Usage:
        python -m src.main
        python src/main.py
    """
    # Get topic from command line args if provided
    topic, skip_outline = _parse_args(sys.argv[1:])

    # Run async main
    asyncio.run(main_async(topic, skip_outline=skip_outline))


if __name__ == "__main__":
    main()
