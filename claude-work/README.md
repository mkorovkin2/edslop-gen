# Educational Video Content Generator

An end-to-end LLM-driven system for generating educational video content using **LangGraph**, **GPT-5.2**, and **Tavily**. Generates scripts, collects relevant images, and synthesizes voice narration—all automatically from a simple topic prompt.

## Features

✅ **Fully LLM-Driven**: No hardcoded templates or logic—everything generated dynamically
✅ **Sequential LangGraph Workflow**: 8 nodes with conditional edges for validation
✅ **Web Research**: Automatic research using Tavily API
✅ **Script Generation**: 200-500 word educational scripts with automatic validation
✅ **Image Collection**: Automatic image search and intelligent mapping to script sections
✅ **Voice Synthesis**: High-quality text-to-speech with automatic chunking for long scripts
✅ **Rate Limiting**: Built-in concurrency and rate limit controls
✅ **Error Recovery**: Retry logic with exponential backoff and checkpointing
✅ **Progress Tracking**: Real-time progress updates during generation

## System Architecture

```
User Topic → Research → Script Generation → Script Parsing → Image Collection
→ Image Mapping → Image Download → Voice Synthesis → Output Files
```

### Workflow Nodes

1. **Research**: Uses LLM to generate search queries, executes Tavily searches
2. **Script Generation**: Creates 200-500 word educational script with retry validation
3. **Script Parsing**: LLM parses script into sections and sentences
4. **Image Collection**: Generates diverse image queries and searches Tavily
5. **Image Mapping**: LLM maps collected images to script sections
6. **Image Download**: Downloads images with failure handling
7. **Voice Synthesis**: OpenAI TTS with automatic chunking for long scripts
8. **Save Outputs**: Writes all files to structured output directory

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (with access to GPT-5.2 and TTS models)
- Tavily API key
- ffmpeg (required for pydub audio processing)

### Quickstart (Recommended)

```bash
./setup.sh
source .venv/bin/activate
cp .env.example .env
```

Edit `.env`, then run:

```bash
python -m src.main "neural networks"
```

### Install ffmpeg

**macOS** (using Homebrew):
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt-get install ffmpeg
```

**Windows**:
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Install Python Dependencies

```bash
cd claude-work
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

### 1. Create `.env` file

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

### 2. Edit `.env`

```env
# API Keys (REQUIRED)
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here

# LLM Configuration
MODEL_NAME=gpt-5.2

# Script Generation Limits
SCRIPT_MIN_WORDS=200
SCRIPT_MAX_WORDS=500

# Image Collection Configuration
IMAGES_MIN_TOTAL=10
IMAGES_PER_SECTION=5

# Text-to-Speech Configuration
TTS_MODEL=tts-1-hd
TTS_VOICE=alloy  # Options: alloy, echo, fable, onyx, nova, shimmer

# Rate Limiting
MAX_CONCURRENT_TAVILY=5
MAX_CONCURRENT_OPENAI=10
MAX_RATE_TAVILY_PER_MIN=100
MAX_RATE_OPENAI_PER_MIN=500
```

## Usage

### Basic Usage

Run the interactive CLI:

```bash
source .venv/bin/activate
python -m src.main
```

You'll be prompted to enter a topic:
```
Enter a technical topic for your video: quantum entanglement
```

### Command Line Usage

Provide the topic directly:

```bash
source .venv/bin/activate
python -m src.main "photosynthesis in plants"
```

### One-off Node Tests (Debug/Validation)

These run each node in isolation and persist state to `output/one_off_state.json`.
Run them in order:

```bash
source .venv/bin/activate
python scripts/one_offs/01_config.py
python scripts/one_offs/02_openai_smoke.py
python scripts/one_offs/03_tavily_smoke.py
python scripts/one_offs/04_research_node.py "your topic"
python scripts/one_offs/05_script_node.py
python scripts/one_offs/06_parse_script_node.py
python scripts/one_offs/07_collect_images_node.py
python scripts/one_offs/08_map_images_node.py
python scripts/one_offs/09_download_images_node.py
python scripts/one_offs/10_voice_node.py
```

Note: These scripts make real API calls and will incur costs.

### Example Session

```
============================================================
  Educational Video Content Generator
  Powered by LangGraph + GPT-5.2 + Tavily
============================================================

✓ Configuration loaded successfully

============================================================
Enter a technical topic for your video: neural networks

📚 Topic: neural networks
📊 Script target: 200-500 words
🖼️  Images target: 10+ images
🤖 Model: gpt-5.2

🚀 Starting workflow for: neural networks
📁 Run ID: run_20260201_143022_742

🔄 Starting: research
   Topic: neural networks
✓ Completed: research (12.3s)

🔄 Starting: synthesize_script
   Research sources: 15
✓ Completed: synthesize_script (8.1s)
   Script: 387 words

🔄 Starting: parse_script
✓ Completed: parse_script (4.2s)

🔄 Starting: collect_images
   Script sections: 4
✓ Completed: collect_images (45.8s)
   Images collected: 32

🔄 Starting: map_images
✓ Completed: map_images (6.5s)

🔄 Starting: download_images
   Images to download: 24
✓ Completed: download_images (18.3s)
   Images downloaded: 22 (failed: 2)

🔄 Starting: generate_voice
Generating audio for script...
✓ Completed: generate_voice (14.7s)

🔄 Starting: save_outputs
✓ Completed: save_outputs (2.1s)

============================================================
✅ VIDEO CONTENT GENERATION COMPLETE!
============================================================

📁 Output directory: output/run_20260201_143022_742/

📄 Script:
   - Words: 387
   - Sections: 4

🖼️  Images:
   - Collected: 32
   - Downloaded: 22

🎙️  Voice:
   - Model: tts-1-hd
   - Voice: alloy
   - Duration: 2m 18s

🔧 API Usage:
   - Tavily calls: 23
   - OpenAI calls: 31

⏱️  Total time: 112.0s

📂 Generated files:
   ✓ output/run_20260201_143022_742/script.md
   ✓ output/run_20260201_143022_742/images.json
   ✓ output/run_20260201_143022_742/meta.json
   ✓ output/run_20260201_143022_742/voice/narration.mp3
   ✓ output/run_20260201_143022_742/images/ (22 images)

============================================================
🎉 Ready for video compilation!
============================================================
```

## Output Structure

Each run creates a unique directory with all generated content:

```
output/
└── run_20260201_143022_742/
    ├── script.md              # Educational script (200-500 words)
    ├── images.json            # Image mappings to script sections
    ├── meta.json              # Complete run metadata
    ├── images/                # Downloaded images
    │   ├── 001_neural-network-diagram.jpg
    │   ├── 002_activation-function-graph.png
    │   └── ...
    └── voice/
        └── narration.mp3      # TTS-generated narration
```

### images.json Format

```json
{
  "mapping": {
    "section_1": [0, 3, 7],
    "section_2": [1, 5, 9, 12],
    "section_3": [2, 8, 14]
  },
  "images": [
    {
      "url": "https://example.com/image.jpg",
      "description": "Neural network architecture diagram",
      "query_used": "neural network layers diagram",
      "local_path": "images/001_neural-network-diagram.jpg"
    },
    ...
  ],
  "total_images": 22,
  "generated_at": "2026-02-01T14:32:45.123456"
}
```

### meta.json Contents

Complete metadata including:
- Topic and run information
- Script statistics (word count, sections, retries)
- Image collection details (collected, downloaded, failed)
- Research sources and queries used
- API usage (Tavily and OpenAI call counts)
- Voice synthesis details (chunks, duration, model)
- Timestamps and duration
- Errors and warnings

## Configuration Options

### Script Generation

- `SCRIPT_MIN_WORDS` (default: 200): Minimum acceptable word count
- `SCRIPT_MAX_WORDS` (default: 500): Maximum acceptable word count
- Scripts outside this range trigger automatic retry (up to 3 attempts)

### Image Collection

- `IMAGES_MIN_TOTAL` (default: 10): Minimum total images to collect
- `IMAGES_PER_SECTION` (default: 5): Number of search queries per script section
- System collects 2-3x more images than needed for LLM selection

### Text-to-Speech

- `TTS_MODEL`: `tts-1` (faster) or `tts-1-hd` (higher quality)
- `TTS_VOICE`: Choose from `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- Automatic chunking for scripts > 4000 characters

### Rate Limiting

- `MAX_CONCURRENT_TAVILY`: Maximum simultaneous Tavily API calls (default: 5)
- `MAX_CONCURRENT_OPENAI`: Maximum simultaneous OpenAI API calls (default: 10)
- `MAX_RATE_TAVILY_PER_MIN`: Tavily requests per minute (default: 100)
- `MAX_RATE_OPENAI_PER_MIN`: OpenAI requests per minute (default: 500)

## Cost Estimation

Typical cost per generation:

- **OpenAI GPT-5.2**: 15-20 calls × $0.10 = **$1.50-2.00**
- **OpenAI TTS**: 500 words × $0.015/1K chars = **$0.02**
- **Tavily**: 20-30 searches × $0.005 = **$0.10-0.15**

**Total: ~$1.65-2.20 per video generation**

Runtime: 1-2 minutes

## Troubleshooting

### "Model 'gpt-5.2' not available"

The system expects GPT-5.2 to be available. If you encounter this error:
- Verify your OpenAI API key has access to the model
- Check the model name in your `.env` file
- Ensure your API key is correctly formatted (starts with `sk-`)

### Rate Limit Errors

If you hit rate limits:
- Reduce `MAX_CONCURRENT_TAVILY` and `MAX_CONCURRENT_OPENAI`
- Reduce `MAX_RATE_TAVILY_PER_MIN` and `MAX_RATE_OPENAI_PER_MIN`
- Check your API tier limits

### Image Download Failures

Some image downloads may fail (404, timeout, etc.). This is expected:
- System collects 2-3x more images than needed
- Failed downloads are logged in `meta.json`
- Workflow continues with successfully downloaded images

### ffmpeg Not Found

If you get an ffmpeg error during voice generation:
- Install ffmpeg (see Installation section)
- Restart your terminal after installation
- Verify: `ffmpeg -version`

### Configuration Errors

If the system can't find your `.env` file:
```bash
# Verify .env exists
ls -la .env

# Check it's in the correct directory
pwd  # Should be in claude-work/

# Verify API keys are set
cat .env | grep API_KEY
```

### Checkpointer Requires `thread_id`

If you invoke the LangGraph workflow manually, provide a `thread_id` in the configurable context:
```python
final_state = await workflow.ainvoke(initial_state, {"configurable": {"thread_id": run_id}})
```

## Architecture Details

### LangGraph Workflow

The system uses LangGraph's `StateGraph` with conditional edges for validation:

```python
research → synthesize_script → [VALIDATE] → parse_script → collect_images
        → [VALIDATE] → map_images → download_images → generate_voice
        → save_outputs
```

**Conditional Edges**:
1. **Script Validation**: Checks word count (200-500), retries up to 3 times
2. **Image Validation**: Checks minimum image count, retries up to 2 times

### State Management

- Uses `TypedDict` with `Annotated` reducers for accumulating data
- State includes error tracking, retry counts, and per-node metadata
- Checkpointing enabled (SQLite when available, otherwise in-memory)

### Rate Limiting

- Semaphore-based concurrent request limiting
- Per-minute rate limiting via `aiolimiter`
- Separate limits for Tavily and OpenAI APIs

### Error Handling

- Exponential backoff retry (3 attempts, 4-10 second wait)
- Graceful degradation (continues with partial data when possible)
- Comprehensive error logging in metadata

## Development

### Project Structure

```
src/
├── main.py                 # CLI entry point
├── config.py              # Configuration management
├── models.py              # Data models and state
├── workflow.py            # LangGraph workflow definition
├── agents/
│   ├── research.py        # Research agent
│   ├── script.py          # Script generation
│   ├── images.py          # Image collection
│   └── voice.py           # Voice synthesis
└── utils/
    ├── rate_limiter.py    # Rate limiting base class
    ├── tavily_client.py   # Tavily API client
    ├── openai_client.py   # OpenAI API client
    ├── context_manager.py # Token management
    ├── progress.py        # Progress tracking
    └── output_manager.py  # File I/O operations
```

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

[Specify your license]

## Credits

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [OpenAI API](https://openai.com/) - GPT-5.2 and TTS
- [Tavily](https://tavily.com/) - Web search and image discovery
- [pydub](https://github.com/jiaaro/pydub) - Audio processing

---

**Ready to create educational video content at scale!** 🎓🎥
