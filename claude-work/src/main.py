"""CLI entry point for educational video generation."""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from .config import load_config
from .workflow import run_workflow
from .utils.logging_utils import configure_logging


async def main_async(topic: str = None):
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

    # Run workflow
    start_time = datetime.now()

    try:
        final_state = await run_workflow(topic, config)

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
    topic = None
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])

    # Run async main
    asyncio.run(main_async(topic))


if __name__ == "__main__":
    main()
